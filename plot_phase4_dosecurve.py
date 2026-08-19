"""Turn phase4_dosecurve.csv into three publication figures (runs on the real output)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats

CSV = "/scratch/delmiari/project/data/synthseg_output/phase4_dosecurve.csv"
REGIONS = ["post_cingulate","precuneus","inf_parietal","inf_temporal","mid_temporal"]
df = pd.read_csv(CSV); df = df[df.status=="ok"]
full = df[df.condition=="full"].set_index("RID")

def pct_err(sub, col):
    m = sub.merge(full[[col]].rename(columns={col:"f"}), on="RID")
    return 100*(m[col]-m.f).abs()/m.f

# ---- FIG 1: dose-response (composite) ----
fig, ax = plt.subplots(figsize=(7,5))
for cond,color in [("low","#B4B2A9"),("denoised","#185FA5")]:
    g = df[df.condition==cond].groupby("dose").apply(lambda s: pct_err(s,"composite").mean())
    ax.plot(g.index*100, g.values, "o-", color=color, lw=2, label=cond)
ax.axhline(5, ls="--", color="#A32D2D", label="5% tolerance")
ax.set_xlabel("dose (% of full)"); ax.set_ylabel("mean |SUVR error| (%)")
ax.invert_xaxis(); ax.legend(frameon=False); ax.set_title("Fidelity vs dose (meta-ROI)")
fig.tight_layout(); fig.savefig("fig1_dose_curve.png", dpi=150, bbox_inches="tight")

# ---- FIG 2: region-wise fidelity at lowest dose (denoised) ----
lowest = df.dose.replace(1.0, np.nan).min()
sub = df[(df.condition=="denoised") & (df.dose==lowest)]
errs = {r: pct_err(sub, r).mean() for r in REGIONS}
fig, ax = plt.subplots(figsize=(7,5))
order = sorted(errs, key=errs.get)
ax.barh(range(len(order)), [errs[r] for r in order], color="#185FA5")
ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
ax.set_xlabel("mean |SUVR error| (%)"); ax.set_title(f"Which region degrades first (denoised @ {lowest*100:.0f}% dose)")
fig.tight_layout(); fig.savefig("fig2_region_fidelity.png", dpi=150, bbox_inches="tight")

# ---- FIG 3: does the SUVR-NfL association survive? ----
def corr_at(cond, dose):
    s = df[(df.condition==cond)&(df.dose==dose)].groupby("RID").composite.mean().reset_index()
    s = s.merge(full.reset_index()[["RID","PLASMA_NFL"]], on="RID").dropna()
    return stats.spearmanr(s.composite, s.PLASMA_NFL)[0] if len(s)>3 else np.nan
base_r = corr_at("full", 1.0)
fig, ax = plt.subplots(figsize=(7,5))
doses = sorted(df[df.condition=="denoised"].dose.unique())
ax.axhline(base_r, ls="--", color="#2E7D32", label=f"full-dose (r={base_r:.2f})")
ax.plot([d*100 for d in doses], [corr_at("denoised",d) for d in doses], "o-",
        color="#185FA5", lw=2, label="denoised")
ax.set_xlabel("dose (% of full)"); ax.set_ylabel("SUVR-NfL Spearman r")
ax.invert_xaxis(); ax.legend(frameon=False); ax.set_title("Does the biological signal survive?")
fig.tight_layout(); fig.savefig("fig3_nfl_survival.png", dpi=150, bbox_inches="tight")
print("saved fig1_dose_curve.png, fig2_region_fidelity.png, fig3_nfl_survival.png")
