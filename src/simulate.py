"""Generate a synthetic weekly dataset for a media mix model.

Three years of national weekly data: spend in five channels and the number of
applications (the outcome). The shape mirrors a typical consumer-finance
acquisition mix: always-on search, social and affiliate spend, plus radio and
out-of-home (OOH) brand bursts that tend to run at the same time.

That last point is deliberate. When two channels are always on together, a
model fitted on observational data alone struggles to tell them apart. A geo
experiment on one of them gives the model a fixed point to anchor on.

All numbers here are synthetic. No real company data is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_WEEKS = 156
CHANNELS = ["search", "social", "affiliate", "radio", "ooh"]

# True response parameters (what the model is trying to recover).
# adstock: share of last week's effect that carries into this week
# half_sat: spend level (as a multiple of the channel's typical weekly spend) at which
#           the channel delivers half of its maximum effect
# max_effect: weekly applications the channel would add at full saturation
TRUE = {
    "search":    {"adstock": 0.10, "half_sat": 1.2, "max_effect": 1500},
    "social":    {"adstock": 0.30, "half_sat": 1.0, "max_effect": 500},
    "affiliate": {"adstock": 0.15, "half_sat": 0.9, "max_effect": 700},
    "radio":     {"adstock": 0.60, "half_sat": 0.8, "max_effect": 900},
    "ooh":       {"adstock": 0.70, "half_sat": 1.5, "max_effect": 120},
}

# Brand bursts (radio + OOH run together); start week and length.
BURSTS = [(14, 8), (40, 6), (66, 8), (92, 6), (118, 8), (144, 6)]
CALIBRATION_BURST = (118, 8)   # the burst that was also run as a geo test
CARRYOVER_WEEKS = 6            # weeks after the burst counted in the test readout


def adstock(x: np.ndarray, theta: float) -> np.ndarray:
    """Geometric carryover, normalised so a steady spend level maps to itself."""
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for t, v in enumerate(x):
        carry = v + theta * carry
        out[t] = carry
    return out * (1 - theta)


_adstock = adstock


def hill(x: np.ndarray, half_sat: float) -> np.ndarray:
    """Diminishing returns: 0 at zero spend, 0.5 at half_sat, approaches 1."""
    return x / (x + half_sat)


def channel_effect(spend: np.ndarray, scale: float, adstock: float,
                   half_sat: float, max_effect: float) -> np.ndarray:
    return max_effect * hill(_adstock(spend / scale, adstock), half_sat)


def simulate(seed: int = 11) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    t = np.arange(N_WEEKS)
    dates = pd.Timestamp("2023-01-02") + pd.to_timedelta(t * 7, unit="D")

    # Demand: two peaks a year (an enrolment-style cycle) and gentle growth
    season = 0.22 * np.sin(2 * np.pi * (t - 20) / 52) + 0.08 * np.sin(4 * np.pi * t / 52)
    baseline = 4000 * (1 + season) * (1 + 0.0012 * t)

    spend = {}
    # Search follows demand (budgets flex with the season), plus noise
    search = 60_000 * (1 + 0.4 * season) * rng.lognormal(0, 0.22, N_WEEKS)
    for start, length, factor in ((25, 5, 0.45), (75, 4, 1.6), (128, 5, 0.5)):  # budget cuts / pushes
        search[start:start + length] *= factor
    spend["search"] = search
    # Social: always-on with a few pushes
    social = 25_000 * rng.lognormal(0, 0.25, N_WEEKS)
    for start in (8, 30, 60, 85, 110, 135):
        social[start:start + 4] *= 1.8
    spend["social"] = social
    # Affiliate: steady, slowly growing
    # Affiliate: steady, slowly growing, with partner pauses and payout changes
    affiliate = 30_000 * (1 + 0.002 * t) * rng.lognormal(0, 0.20, N_WEEKS)
    for start, length, factor in ((48, 6, 0.35), (100, 5, 1.7), (138, 4, 0.4)):
        affiliate[start:start + length] *= factor
    spend["affiliate"] = affiliate
    # Radio and OOH: bursts that mostly overlap
    radio = np.zeros(N_WEEKS)
    ooh = np.zeros(N_WEEKS)
    for start, length in BURSTS:
        radio[start:start + length] = 45_000 * rng.lognormal(0, 0.10, length)
        o_start = start + rng.integers(-1, 2)
        ooh[o_start:o_start + length] = 35_000 * rng.lognormal(0, 0.10, length)
    spend["radio"], spend["ooh"] = radio, ooh

    scales = {c: float(spend[c][spend[c] > 0].mean()) for c in CHANNELS}
    effects = {c: channel_effect(spend[c], scales[c], **TRUE[c]) for c in CHANNELS}

    noise = rng.normal(0, 0.025, N_WEEKS)
    applications = (baseline + sum(effects.values())) * (1 + noise)

    df = pd.DataFrame({"week": dates.date.astype(str)})
    for c in CHANNELS:
        df[f"spend_{c}"] = spend[c].round(0)
    df["applications"] = applications.round(0).astype(int)

    truth = {"effects": effects, "scales": scales, "baseline": baseline}
    return df, truth


def geo_test_result(truth: dict, seed: int = 1) -> dict:
    """What a geo test on the calibration burst would report.

    The geo test measures radio's incremental applications during the burst and
    the carryover weeks after it, scaled to national. It comes with its own
    uncertainty (a standard error of about 15%), just like a real experiment.
    """
    rng = np.random.default_rng(seed)
    start, length = CALIBRATION_BURST
    window = slice(start, start + length + CARRYOVER_WEEKS)
    true_incremental = float(truth["effects"]["radio"][window].sum())
    se = 0.15 * true_incremental
    measured = true_incremental + rng.normal(0, se)
    return {"channel": "radio", "start": start, "end": start + length + CARRYOVER_WEEKS,
            "incremental": measured, "se": se}


if __name__ == "__main__":
    df, _ = simulate()
    df.to_csv("data/weekly_media.csv", index=False)
    print(f"wrote data/weekly_media.csv: {len(df)} weeks, {len(CHANNELS)} channels")
