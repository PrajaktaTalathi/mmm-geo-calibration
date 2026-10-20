"""End-to-end media mix model, with and without geo-test calibration.

Run:  python run_analysis.py
Writes data to ./data, charts and readout.txt to ./charts. Takes a few minutes
(most of it is the bootstrap).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.simulate import CHANNELS, TRUE, channel_effect, geo_test_result, simulate
from src import mmm

N_BOOT = 60
LAST = 52  # report on the most recent year
LABEL = {"search": "Paid search", "social": "Paid social", "affiliate": "Affiliate / partner",
         "radio": "Radio", "ooh": "Out-of-home"}

Path("data").mkdir(exist_ok=True)
Path("charts").mkdir(exist_ok=True)

df, truth = simulate()
df.to_csv("data/weekly_media.csv", index=False)
experiment = geo_test_result(truth)

# ---------- fit ----------
print("fitting uncalibrated model ...")
uncal = mmm.fit(df)
print("fitting calibrated model ...")
cal = mmm.fit(df, experiment, uncal.sigma)


def true_contribution(c, spend):
    return channel_effect(np.asarray(spend, float), truth["scales"][c], **TRUE[c])


window = slice(len(df) - LAST, len(df))
spend_last = pd.Series({c: df[f"spend_{c}"].values[window].sum() for c in CHANNELS})
true_cpia = pd.Series({c: spend_last[c] / truth["effects"][c][window].sum() for c in CHANNELS})
sum_u = mmm.channel_summary(uncal, df, LAST)
sum_c = mmm.channel_summary(cal, df, LAST)

print("bootstrapping uncertainty ...")
boot_u = mmm.block_bootstrap(uncal, df, None, n_boot=N_BOOT, seed=1)
boot_c = mmm.block_bootstrap(cal, df, experiment, n_boot=N_BOOT, seed=2)
CAP = 5_000  # a draw with no measurable effect has infinite CPIA; cap it for the interval
ci_u = boot_u.clip(upper=CAP).quantile([0.1, 0.9]).T
ci_c = boot_c.clip(upper=CAP).quantile([0.1, 0.9]).T

# ---------- out-of-sample check ----------
HOLD = 26
train = df.iloc[:-HOLD].reset_index(drop=True)
hold_model = mmm.fit(train)
pred_all, _ = hold_model.predict(df)
actual = df["applications"].values
mape_in = np.mean(np.abs(pred_all[:-HOLD] - actual[:-HOLD]) / actual[:-HOLD])
mape_out = np.mean(np.abs(pred_all[-HOLD:] - actual[-HOLD:]) / actual[-HOLD:])

# ---------- budget re-split ----------
plan_u = mmm.optimise_budget(uncal.channel_contribution, df, LAST)
plan_c = mmm.optimise_budget(cal.channel_contribution, df, LAST)
current = pd.Series(1.0, index=CHANNELS)
true_now = mmm.evaluate_plan(true_contribution, df, current, LAST)
true_u = mmm.evaluate_plan(true_contribution, df, plan_u, LAST)
true_c = mmm.evaluate_plan(true_contribution, df, plan_c, LAST)
promised_u = mmm.evaluate_plan(uncal.channel_contribution, df, plan_u, LAST) - \
    mmm.evaluate_plan(uncal.channel_contribution, df, current, LAST)
promised_c = mmm.evaluate_plan(cal.channel_contribution, df, plan_c, LAST) - \
    mmm.evaluate_plan(cal.channel_contribution, df, current, LAST)

# ---------- readout ----------
def money(x):
    return f"${x:,.0f}"

def rng_txt(lo, hi):
    return f"{lo:,.0f}-{hi:,.0f}" if hi < CAP else f"{lo:,.0f}-no lift"

# Where would another experiment pay off most? The untested channel where the
# plan moves the most money, i.e. where a wrong cost estimate would hurt most.
at_stake = (plan_c - 1).abs() * spend_last
next_test = at_stake.drop(experiment["channel"]).idxmax()
true_window = truth["effects"]["radio"][experiment["start"]:experiment["end"]].sum()

lines = [
    "MEDIA MIX MODEL READOUT (synthetic data)",
    f"{len(df)} weeks, {len(CHANNELS)} channels. Reporting on the last {LAST} weeks "
    f"(total media spend {money(spend_last.sum())}).",
    f"Radio and OOH spend correlation: {np.corrcoef(df.spend_radio, df.spend_ooh)[0, 1]:.2f} "
    "(they mostly run in the same bursts)",
    f"Geo test fed in: radio added {experiment['incremental']:,.0f} +/- {experiment['se']:,.0f} "
    f"incremental applications over weeks {experiment['start']}-{experiment['end'] - 1} (burst + carryover); "
    f"simulated truth was {true_window:,.0f}",
    "",
    "Cost per incremental application (CPIA), last 52 weeks. Range = 80% bootstrap interval.",
    f"{'Channel':<22}{'Spend':>12}{'True':>9}{'Uncalibrated':>24}{'Calibrated':>24}",
]
for c in CHANNELS:
    u = f"{sum_u.cpia[c]:,.0f} ({rng_txt(ci_u.loc[c, 0.1], ci_u.loc[c, 0.9])})"
    k = f"{sum_c.cpia[c]:,.0f} ({rng_txt(ci_c.loc[c, 0.1], ci_c.loc[c, 0.9])})"
    lines.append(f"{LABEL[c]:<22}{money(spend_last[c]):>12}{true_cpia[c]:>9.0f}{u:>24}{k:>24}")
lines += [
    "",
    "Marginal CPIA (cost of the next application at current spend), calibrated model:",
    "  " + ", ".join(f"{LABEL[c]} {sum_c.marginal_cpia[c]:,.0f}" for c in CHANNELS),
    "",
    f"Fit check: trained on the first {len(df) - HOLD} weeks, predicted the last {HOLD}. "
    f"MAPE in-sample {mape_in:.1%}, out-of-sample {mape_out:.1%}.",
    "",
    "Budget re-split (same total, each channel held within -40% / +40% of current):",
    f"{'Channel':<22}{'Uncalibrated plan':>20}{'Calibrated plan':>18}",
]
lines += [f"{LABEL[c]:<22}{plan_u[c] - 1:>+20.0%}{plan_c[c] - 1:>+18.0%}" for c in CHANNELS]
lines += [
    "",
    "What each plan actually delivers against the simulated truth (incremental applications, 52 weeks):",
    f"  current split            {true_now:>10,.0f}",
    f"  uncalibrated model plan  {true_u:>10,.0f}  ({true_u - true_now:+,.0f}; the model promised {promised_u:+,.0f})",
    f"  calibrated model plan    {true_c:>10,.0f}  ({true_c - true_now:+,.0f}; the model promised {promised_c:+,.0f})",
    "",
    f"Next experiment: {LABEL[next_test]}. The plan moves {plan_c[next_test] - 1:+.0%} of its spend "
    f"({money(abs(plan_c[next_test] - 1) * spend_last[next_test])}) on a cost estimate the model can only place "
    f"between {rng_txt(ci_c.loc[next_test, 0.1], ci_c.loc[next_test, 0.9])}. Test it before making that move.",
]
readout = "\n".join(lines)
print(readout)
Path("charts/readout.txt").write_text(readout)

# ---------- charts ----------
INK, MUTED, ACCENT, SHADE, BLUE = "#1f2a37", "#8a94a3", "#c2410c", "#f1e7df", "#2f5d8a"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
dates = pd.to_datetime(df["week"])

# 1. CPIA: truth vs uncalibrated vs calibrated
fig, ax = plt.subplots(figsize=(9, 4.2))
y = np.arange(len(CHANNELS))
for off, s, ci, col, lab in ((-0.18, sum_u.cpia, ci_u, MUTED, "Uncalibrated MMM"),
                             (0.18, sum_c.cpia, ci_c, ACCENT, "Calibrated with geo test")):
    s = s.clip(upper=CAP)
    ax.errorbar(s.values, y + off, xerr=[s.values - ci[0.1].values, ci[0.9].values - s.values],
                fmt="o", color=col, ms=7, capsize=3, lw=1.5, label=lab)
ax.scatter(true_cpia.values, y, marker="|", s=500, color=INK, lw=2.5, label="Simulated truth", zorder=5)
ax.set_yticks(y, [LABEL[c] for c in CHANNELS]); ax.invert_yaxis()
ax.set_xscale("log"); ax.set_xlabel("Cost per incremental application ($, log scale)")
ax.set_title("One geo test fixes the radio vs. out-of-home mix-up", loc="left", color=INK)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig("charts/cpia_comparison.png", dpi=150); plt.close(fig)

# 2. Decomposition (calibrated)
_, parts = cal.predict(df)
fig, ax = plt.subplots(figsize=(9, 4))
colors = ["#d9dde3", "#2f5d8a", "#6c8ebf", "#9fb6d6", ACCENT, "#e8a37c"]
ax.stackplot(dates, [parts[k] for k in ["baseline", *CHANNELS]], colors=colors,
             labels=["Baseline (organic demand)", *[LABEL[c] for c in CHANNELS]], alpha=0.95)
ax.plot(dates, actual, color=INK, lw=1, label="Actual applications")
ax.set_ylabel("Weekly applications")
ax.set_title("Where weekly applications come from (calibrated model)", loc="left", color=INK)
ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper left")
ax.set_ylim(0, actual.max() * 1.25)
fig.tight_layout(); fig.savefig("charts/decomposition.png", dpi=150); plt.close(fig)

# 3. Response curves (steady weekly spend), calibrated vs truth
fig, axes = plt.subplots(1, len(CHANNELS), figsize=(12, 3), sharey=False)
for ax, c in zip(axes, CHANNELS):
    s = df[f"spend_{c}"]
    avg = s[s > 0].mean()
    x = np.linspace(0, 2.5 * avg, 100)
    ax.plot(x / 1000, cal.beta[c] * x / cal.scales[c] / (x / cal.scales[c] + cal.half_sat[c]), color=ACCENT, lw=2, label="Calibrated")
    ax.plot(x / 1000, TRUE[c]["max_effect"] * (x / truth["scales"][c]) / (x / truth["scales"][c] + TRUE[c]["half_sat"]),
            color=INK, lw=1.2, ls="--", label="Truth")
    ax.axvline(avg / 1000, color=MUTED, lw=1, ls=":")
    ax.set_title(LABEL[c], fontsize=10, color=INK)
    ax.set_xlabel("Weekly spend ($k)")
axes[0].set_ylabel("Weekly applications added")
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Diminishing returns by channel (dotted line = typical weekly spend)", x=0.01, ha="left", color=INK)
fig.tight_layout(); fig.savefig("charts/response_curves.png", dpi=150); plt.close(fig)

# 4. Budget re-split
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.bar(y - 0.2, (plan_u - 1) * 100, 0.38, color=MUTED, label=f"Uncalibrated plan (real gain {true_u - true_now:+,.0f})")
ax.bar(y + 0.2, (plan_c - 1) * 100, 0.38, color=ACCENT, label=f"Calibrated plan (real gain {true_c - true_now:+,.0f})")
ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks(y, [LABEL[c] for c in CHANNELS], fontsize=9)
ax.set_ylabel("Change vs. current spend (%)")
ax.set_title("Same budget, different advice: the uncalibrated model moves money the wrong way", loc="left", color=INK, fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig("charts/budget_resplit.png", dpi=150); plt.close(fig)

# 5. Out-of-sample fit
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.axvspan(dates.iloc[-HOLD], dates.iloc[-1], color=SHADE, zorder=0, label="Held-out weeks")
ax.plot(dates, actual, color=INK, lw=1.4, label="Actual")
ax.plot(dates, pred_all, color=BLUE, lw=1.4, ls="--", label="Model prediction")
ax.set_ylabel("Weekly applications")
ax.set_title(f"Out-of-sample check: {mape_out:.1%} error on the 26 weeks the model never saw", loc="left", color=INK)
ax.legend(frameon=False, loc="upper left", ncol=3)
fig.tight_layout(); fig.savefig("charts/holdout_fit.png", dpi=150); plt.close(fig)

print("\ncharts written to ./charts")
