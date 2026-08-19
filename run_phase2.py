"""
Phase 2 — FDG hypometabolism vs plasma NfL concordance & discordance analysis (Age-Controlled).
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# SETUP & SETUP SEED
# -------------------------------------------------------------------------
rng = np.random.default_rng(0)
ORDER = ["CN", "MCI", "Dementia"]

# Read dataset with age
df = pd.read_csv("phase1_analysis_table_age.csv")
df["fdg_suvr"] = pd.to_numeric(df["fdg_suvr"], errors="coerce")
df["nfl_value"] = pd.to_numeric(df["nfl_value"], errors="coerce")
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df = df.dropna(subset=["fdg_suvr", "nfl_value", "age"]).copy()

df["lognfl"] = np.log(df["nfl_value"])
df["z_lognfl"] = (df.lognfl - df.lognfl.mean()) / df.lognfl.std()
df["DIAGNOSIS"] = pd.Categorical(df["DIAGNOSIS"], categories=ORDER, ordered=True)

print("=" * 65)
print(f"       PHASE 2 ANALYSIS: FDG PET VS PLASMA NfL (AGE-CONTROLLED, N = {len(df)})")
print("=" * 65 + "\n")

# -------------------------------------------------------------------------
# 1. DESCRIPTIVE TABLE BY DIAGNOSIS (Including Age)
# -------------------------------------------------------------------------
print("1. DESCRIPTIVE TABLE BY DIAGNOSIS")
print("-" * 65)
for g in ORDER:
    s = df[df["DIAGNOSIS"] == g]
    if not len(s): 
        continue
    print(f"  {g:>8}  n={len(s):3d} | Age {s.age.mean():.1f}±{s.age.std():.1f} yrs | FDG {s.fdg_suvr.mean():.3f}±{s.fdg_suvr.std():.3f} | "
          f"NfL median {s.nfl_value.median():.1f} (IQR {s.nfl_value.quantile(.25):.0f}-{s.nfl_value.quantile(.75):.0f}) | "
          f"MMSE {s.MMSCORE.mean():.1f} | APOE4+ {100*s.APOE4_CARRIER.mean():.0f}%")

print(f"\n  SANITY CHECK:")
print(f"    FDG CN > Dementia? {df[df.DIAGNOSIS=='CN'].fdg_suvr.mean():.3f} vs "
      f"{df[df.DIAGNOSIS=='Dementia'].fdg_suvr.mean():.3f}")
print(f"    NfL CN < Dementia? {df[df.DIAGNOSIS=='CN'].nfl_value.median():.1f} vs "
      f"{df[df.DIAGNOSIS=='Dementia'].nfl_value.median():.1f}")

# -------------------------------------------------------------------------
# 2. CONCORDANCE: RAW & AGE-ADJUSTED PARTIAL CORRELATION
# -------------------------------------------------------------------------
def boot_ci(x, y, n=2000):
    rs = []
    idx = np.arange(len(x))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        rs.append(stats.spearmanr(x.iloc[b], y.iloc[b]).correlation)
    return np.nanpercentile(rs, [2.5, 97.5])

print("\n2. CONCORDANCE (Raw vs Age-Adjusted Partial Spearman)")
print("-" * 65)
raw_r, raw_p = stats.spearmanr(df.fdg_suvr, df.nfl_value)
lo, hi = boot_ci(df.fdg_suvr, df.nfl_value)
print(f"  Raw Overall:            r={raw_r:+.3f} [{lo:+.3f}, {hi:+.3f}] p={raw_p:.1e}  (n={len(df)})")

# Partial Spearman correlation (Residualized on age)
res_fdg = smf.ols("fdg_suvr ~ age", data=df).fit().resid
res_nfl = smf.ols("lognfl ~ age", data=df).fit().resid
part_r, part_p = stats.spearmanr(res_fdg, res_nfl)
print(f"  Partial (Age-Adjusted): r={part_r:+.3f} p={part_p:.1e}")

for g in ORDER:
    s = df[df.DIAGNOSIS == g]
    if len(s) > 10:
        rg, pg = stats.spearmanr(s.fdg_suvr, s.nfl_value)
        print(f"  {g:>8}:   r={rg:+.3f} p={pg:.3f}  (n={len(s)})")

# -------------------------------------------------------------------------
# 3. GROUP COMPARISONS & AGE-ADJUSTED ANCOVA
# -------------------------------------------------------------------------
print("\n3. GROUP DIFFERENCES ACROSS DIAGNOSTIC STAGES")
print("-" * 65)
print("  [A] Unadjusted Non-parametric (Kruskal-Wallis):")
for var, name in [("fdg_suvr", "FDG"), ("nfl_value", "NfL")]:
    groups = [df[df.DIAGNOSIS == g][var].dropna() for g in ORDER]
    H, p = stats.kruskal(*groups)
    eps2 = (H - len(ORDER) + 1) / (len(df) - len(ORDER))
    medians = [round(g.median(), 2) for g in groups]
    print(f"    {name:>4}: H={H:.1f}, p={p:.1e}, eps^2={eps2:.3f} | Medians [CN, MCI, Dem]: {medians}")

print("\n  [B] Age & Covariate Adjusted (ANCOVA):")
anc_fdg = smf.ols("fdg_suvr ~ C(DIAGNOSIS) + age + C(PTGENDER) + APOE4_CARRIER", data=df).fit()
anc_nfl = smf.ols("lognfl ~ C(DIAGNOSIS) + age + C(PTGENDER) + APOE4_CARRIER", data=df).fit()
print(f"    FDG ANCOVA p-value (DIAGNOSIS): {anc_fdg.pvalues.filter(like='DIAGNOSIS').min():.1e}")
print(f"    NfL ANCOVA p-value (DIAGNOSIS): {anc_nfl.pvalues.filter(like='DIAGNOSIS').min():.1e}")

# -------------------------------------------------------------------------
# 4. ADJUSTED OLS REGRESSION (WITH AGE CONTROL)
# -------------------------------------------------------------------------
print("\n4. ADJUSTED MODEL: fdg_suvr ~ z(logNfL) + age + C(PTGENDER) + APOE4_CARRIER")
print("-" * 65)

# Without age
m0 = smf.ols("fdg_suvr ~ z_lognfl + C(PTGENDER) + APOE4_CARRIER", data=df).fit()
b0, p0 = m0.params["z_lognfl"], m0.pvalues["z_lognfl"]

# With age
m1 = smf.ols("fdg_suvr ~ z_lognfl + age + C(PTGENDER) + APOE4_CARRIER", data=df).fit()
b1, ci1 = m1.params["z_lognfl"], m1.conf_int().loc["z_lognfl"]
b_age, p_age = m1.params["age"], m1.pvalues["age"]

print(f"  Without Age: beta = {b0:+.4f} SUVR/SD logNfL (p = {p0:.1e})")
print(f"  With Age:    beta = {b1:+.4f} SUVR/SD logNfL [{ci1[0]:+.4f}, {ci1[1]:+.4f}] (p = {m1.pvalues['z_lognfl']:.1e})")
print(f"  Age Effect:  beta = {b_age:+.5f} SUVR/yr (p = {p_age:.1e})")

# -------------------------------------------------------------------------
# 5. DISCORDANCE VIEW (AGE-RESIDUALIZED Z-SCORES)
# -------------------------------------------------------------------------
# Residualize on age first so discordance isn't driven by age differences
res_zN = smf.ols("lognfl ~ age", data=df).fit().resid
res_zF = smf.ols("fdg_suvr ~ age", data=df).fit().resid

df["zN"] = (res_zN - res_zN.mean()) / res_zN.std()
df["zF"] = (-(res_zF) - (-res_zF).mean()) / res_zF.std() # Inverted so higher = worse

concordant = ((df.zN > 0) & (df.zF > 0)) | ((df.zN < 0) & (df.zF < 0))
disc_nfl_high = (df.zN > 0.5) & (df.zF < -0.5)   # High NfL, normal FDG
disc_fdg_low  = (df.zF > 0.5) & (df.zN < -0.5)   # Low FDG, normal NfL

print("\n5. BIOMARKER DISCORDANCE (AGE-RESIDUALIZED)")
print("-" * 65)
print(f"  Concordant subjects:      {concordant.mean()*100:.1f}% ({concordant.sum()}/{len(df)})")
print(f"  NfL-high / FDG-ok:        {disc_nfl_high.sum()} subjects")
print(f"  FDG-low  / NfL-ok:        {disc_fdg_low.sum()} subjects")

# -------------------------------------------------------------------------
# 6. FIGURE GENERATION
# -------------------------------------------------------------------------
cols = {"CN": "#2E7D32", "MCI": "#ED9C28", "Dementia": "#B4271F"}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

# Panel A: Scatter plot
for g in ORDER:
    s = df[df["DIAGNOSIS"] == g]
    ax[0].scatter(s.nfl_value, s.fdg_suvr, s=20, alpha=0.7, color=cols[g], label=g)
ax[0].set_xscale("log")
ax[0].set_xlabel("Plasma NfL (pg/mL, log scale)")
ax[0].set_ylabel("FDG SUVR (Meta-ROI)")
ax[0].set_title(f"Raw Concordance (Spearman r={raw_r:+.2f})")
ax[0].legend(frameon=False)

# Panel B: Boxplot across stages
data = [df[df["DIAGNOSIS"] == g].fdg_suvr for g in ORDER]
ax[1].boxplot(data, tick_labels=ORDER)
ax[1].set_ylabel("FDG SUVR")
ax[1].set_title("FDG Meta-ROI by Stage")

# Panel C: Discordance quadrants (Age-Adjusted)
for g in ORDER:
    s = df[df["DIAGNOSIS"] == g]
    ax[2].scatter(s.zN, s.zF, s=20, alpha=0.7, color=cols[g])
ax[2].axhline(0, color="#999", lw=0.8, ls="--")
ax[2].axvline(0, color="#999", lw=0.8, ls="--")
ax[2].set_xlabel("Age-Resid. NfL (z-score, higher = worse)")
ax[2].set_ylabel("Age-Resid. -FDG SUVR (z-score, higher = worse)")
ax[2].set_title("Age-Controlled Discordance Quadrants")

fig.suptitle(f"Phase 2 — FDG Hypometabolism vs Plasma NfL (ADNI, N = {len(df)})", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("phase2_concordance.png", dpi=140, bbox_inches="tight")
print("\nSaved figure to 'phase2_concordance.png'\n")