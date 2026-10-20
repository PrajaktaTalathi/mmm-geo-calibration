"""A small, transparent media mix model with optional geo-test calibration.

Model
-----
applications_t = baseline_t + sum_c  beta_c * hill(adstock(spend_c,t / scale_c, theta_c), K_c) + error

- baseline: intercept, linear trend and two yearly Fourier terms (seasonality)
- adstock (theta_c): how much of a channel's effect carries into later weeks
- hill (K_c): diminishing returns; K_c is the spend level giving half the max effect
- beta_c: the channel's maximum weekly effect (kept >= 0)

For fixed (theta, K) the model is linear in beta and the baseline terms, so those
are solved exactly by bounded least squares. Only the 10 non-linear parameters
(theta and K for five channels) are searched.

Calibration
-----------
A geo test says "radio added I incremental applications over this window, +/- se".
The model's own estimate for that window is beta_radio * (sum of radio's transformed
spend over the window), which is linear in beta_radio. So the experiment enters the
fit as one extra, weighted observation. Its weight (sigma / se) means the model
honours the experiment about as much as its precision deserves: a tight test pulls
hard, a noisy one pulls gently.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, lsq_linear, minimize

from .simulate import CHANNELS, adstock, hill

THETA_BOUNDS = (0.0, 0.85)
K_BOUNDS = (0.2, 3.0)


def baseline_features(n: int, offset: int = 0) -> np.ndarray:
    t = np.arange(offset, offset + n)
    return np.column_stack([
        np.ones(n),
        t / 52,
        np.sin(2 * np.pi * t / 52), np.cos(2 * np.pi * t / 52),
        np.sin(4 * np.pi * t / 52), np.cos(4 * np.pi * t / 52),
    ])


@dataclass
class FittedMMM:
    theta: dict
    half_sat: dict
    beta: dict
    base_coef: np.ndarray
    scales: dict
    sigma: float
    calibrated: bool
    params: np.ndarray = field(repr=False)

    # ----- prediction -----
    def channel_contribution(self, channel: str, spend: np.ndarray) -> np.ndarray:
        x = adstock(np.asarray(spend, float) / self.scales[channel], self.theta[channel])
        return self.beta[channel] * hill(x, self.half_sat[channel])

    def predict(self, df: pd.DataFrame, offset: int = 0) -> tuple[np.ndarray, dict]:
        base = baseline_features(len(df), offset) @ self.base_coef
        contribs = {c: self.channel_contribution(c, df[f"spend_{c}"].values) for c in CHANNELS}
        return base + sum(contribs.values()), {"baseline": base, **contribs}


def _design(df: pd.DataFrame, params: np.ndarray, scales: dict, offset: int = 0) -> np.ndarray:
    cols = []
    for i, c in enumerate(CHANNELS):
        theta, k = params[2 * i], params[2 * i + 1]
        cols.append(hill(adstock(df[f"spend_{c}"].values / scales[c], theta), k))
    return np.column_stack([baseline_features(len(df), offset), np.column_stack(cols)])


def _solve(X: np.ndarray, y: np.ndarray, calib_row=None):
    """Bounded least squares: baseline terms free, channel betas >= 0."""
    n_base = X.shape[1] - len(CHANNELS)
    if calib_row is not None:
        X = np.vstack([X, calib_row[0]])
        y = np.append(y, calib_row[1])
    lb = np.r_[np.full(n_base, -np.inf), np.zeros(len(CHANNELS))]
    res = lsq_linear(X, y, bounds=(lb, np.full(X.shape[1], np.inf)), method="bvls")
    return res.x, float(np.sum((X @ res.x - y) ** 2))


def _calibration_row(df, params, scales, experiment, sigma):
    """Turn the experiment into one weighted row of the regression."""
    X = _design(df, params, scales)
    j = X.shape[1] - len(CHANNELS) + CHANNELS.index(experiment["channel"])
    window = slice(experiment["start"], experiment["end"])
    w = sigma / experiment["se"]
    row = np.zeros(X.shape[1])
    row[j] = w * X[window, j].sum()
    return row, w * experiment["incremental"]


def _objective(params, df, y, scales, experiment, sigma):
    X = _design(df, params, scales)
    calib = _calibration_row(df, params, scales, experiment, sigma) if experiment else None
    return _solve(X, y, calib)[1]


def fit(df: pd.DataFrame, experiment: dict | None = None, sigma: float | None = None,
        start: np.ndarray | None = None, seed: int = 0, y: np.ndarray | None = None) -> FittedMMM:
    """Fit the MMM. Pass `experiment` (and the residual `sigma` of an uncalibrated
    fit) to calibrate. Pass `start` to do a fast local refit (used for bootstrap)."""
    y = df["applications"].values.astype(float) if y is None else y
    scales = {c: float(df[f"spend_{c}"][df[f"spend_{c}"] > 0].mean()) for c in CHANNELS}
    bounds = [THETA_BOUNDS, K_BOUNDS] * len(CHANNELS)
    if experiment is not None and sigma is None:
        raise ValueError("calibration needs sigma from an uncalibrated fit")
    args = (df, y, scales, experiment, sigma)

    if start is None:
        res = differential_evolution(_objective, bounds, args=args, seed=seed, tol=1e-7,
                                     popsize=20, maxiter=400, polish=True)
    else:
        res = minimize(_objective, start, args=args, bounds=bounds, method="L-BFGS-B")
    params = res.x

    X = _design(df, params, scales)
    calib = _calibration_row(df, params, scales, experiment, sigma) if experiment else None
    coef, _ = _solve(X, y, calib)
    n_base = X.shape[1] - len(CHANNELS)
    resid = y - X @ coef
    return FittedMMM(
        theta={c: params[2 * i] for i, c in enumerate(CHANNELS)},
        half_sat={c: params[2 * i + 1] for i, c in enumerate(CHANNELS)},
        beta={c: coef[n_base + i] for i, c in enumerate(CHANNELS)},
        base_coef=coef[:n_base],
        scales=scales,
        sigma=float(resid.std(ddof=X.shape[1])),
        calibrated=experiment is not None,
        params=params,
    )


# ---------- reporting helpers ----------

def channel_summary(model: FittedMMM, df: pd.DataFrame, last_n: int = 52) -> pd.DataFrame:
    """Spend, contribution, cost per incremental application (CPIA) and marginal CPIA
    over the last `last_n` weeks."""
    rows = []
    window = slice(len(df) - last_n, len(df))
    for c in CHANNELS:
        spend = df[f"spend_{c}"].values.astype(float)
        contrib = model.channel_contribution(c, spend)[window].sum()
        bumped = spend.copy()
        bumped[window] *= 1.01
        extra = model.channel_contribution(c, bumped)[window].sum() - contrib
        s = spend[window].sum()
        rows.append({
            "channel": c,
            "spend": s,
            "incremental_apps": contrib,
            "cpia": s / contrib if contrib > 0 else np.inf,
            "marginal_cpia": 0.01 * s / extra if extra > 0 else np.inf,
        })
    return pd.DataFrame(rows).set_index("channel")


def optimise_budget(contribution_fn, df: pd.DataFrame, last_n: int = 52,
                    lo: float = 0.6, hi: float = 1.4) -> pd.Series:
    """Re-split the same total budget across channels to maximise incremental
    applications. Each channel keeps its weekly pattern but is scaled by a
    multiplier, held within [lo, hi] so the plan stays close to spend levels the
    model has actually seen (no extrapolating far outside the data)."""
    window = slice(len(df) - last_n, len(df))
    spend = {c: df[f"spend_{c}"].values.astype(float) for c in CHANNELS}
    totals = np.array([spend[c][window].sum() for c in CHANNELS])

    def total_contrib(m):
        out = 0.0
        for i, c in enumerate(CHANNELS):
            s = spend[c].copy()
            s[window] *= m[i]
            out += contribution_fn(c, s)[window].sum()
        return out

    cons = {"type": "eq", "fun": lambda m: m @ totals - totals.sum()}
    res = minimize(lambda m: -total_contrib(m), np.ones(len(CHANNELS)), method="SLSQP",
                   bounds=[(lo, hi)] * len(CHANNELS), constraints=[cons])
    return pd.Series(res.x, index=CHANNELS)


def evaluate_plan(contribution_fn, df: pd.DataFrame, multipliers: pd.Series, last_n: int = 52) -> float:
    window = slice(len(df) - last_n, len(df))
    total = 0.0
    for c in CHANNELS:
        s = df[f"spend_{c}"].values.astype(float).copy()
        s[window] *= multipliers[c]
        total += contribution_fn(c, s)[window].sum()
    return total


def block_bootstrap(model: FittedMMM, df: pd.DataFrame, experiment: dict | None,
                    n_boot: int = 100, block: int = 8, seed: int = 0) -> pd.DataFrame:
    """Uncertainty: resample residuals in 8-week blocks (keeps their time
    structure), redraw the experiment result within its own standard error, refit,
    and record each channel's CPIA. Returns one row per bootstrap draw."""
    rng = np.random.default_rng(seed)
    fitted, _ = model.predict(df)
    resid = df["applications"].values - fitted
    n = len(df)
    draws = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block, size=n // block + 1)
        r = np.concatenate([resid[s:s + block] for s in starts])[:n]
        y_star = fitted + r
        exp_star = None
        if experiment is not None:
            exp_star = {**experiment, "incremental": experiment["incremental"] + rng.normal(0, experiment["se"])}
        m = fit(df, exp_star, model.sigma if experiment else None, start=model.params, y=y_star)
        draws.append(channel_summary(m, df)["cpia"])
    return pd.DataFrame(draws).reset_index(drop=True)
