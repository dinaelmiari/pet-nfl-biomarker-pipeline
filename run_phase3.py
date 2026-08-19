"""
Phase 3 - discordance characterization (Aim 2).

Splits the cohort by whether NfL and FDG agree (both age-residualized, "higher =
worse"), then profiles the two discordant corners against the concordant group.

Groups:
  Concordant        - NfL and FDG point the same way
  NfL-high/FDG-ok   - blood says damage, metabolism looks preserved
  FDG-low/NfL-ok    - metabolism says damage, blood looks normal

Because each discordant corner is small (~15), this is DESCRIPTIVE: we report
group profiles and standardized differences (Cohen's d / proportion gaps), NOT
significance tests on tiny cells. All printed output is aggregate / safe to share.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TABLE = "phase1_analysis_table_age.csv"
ORDER = ["CN", "MCI", "Dementia"]
THR = 0.5

# -------------------------------------------------------------------------
# 1. LOAD DATA & RESIDUALIZE ON AGE
# -------------------------------------------------------------------------
df = pd.read_csv(TABLE)
for c in ["fdg_suvr", "nfl_value", "age", "MMSCORE", "CDRSB", "APOE4_CARRIER"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["fdg_suvr", "nfl_value", "age"]).copy()
df["lognfl"] = np.log(df["nfl_value"])

# Age-residualize both markers, put on "higher = worse" z-scale
rN = smf.ols("lognfl ~ age", data=df).fit().resid
rF = smf.ols("fdg_suvr ~ age", data=df).fit().resid
df["zN"] = (rN - rN.mean()) / rN.std()            # NfL: higher = worse
df["zF"] = (-rF - (-rF).mean()) / rF.std()        # -FDG: higher = worse

# -------------------------------------------------------------------------
# 2. ASSIGN DISCORDANCE GROUPS
# -------------------------------------------------------------------------
def label(r):
    if r.zN > THR and r.zF < -THR:  return "NfL-high/FDG-ok"
    if r.zF > THR and r.zN < -THR:  return "FDG-low/NfL-ok"
    if (r.zN > 0) == (r.zF > 0):     return "Concordant"
    return "Intermediate"

df["group"] = df.apply(label, axis=1)

GROUPS = ["Concordant", "NfL-high/FDG-ok", "FDG-low/NfL-ok", "Intermediate"]
print("=" * 65)
print("                       GROUP SIZES")
print("=" * 65)
for g in GROUPS:
    print(f"  {g:>16}: {int((df.group == g).sum())}")

# -------------------------------------------------------------------------
# 3. GROUP PROFILES
# -------------------------------------------------------------------------
def profile(s):
    return dict(
        n=len(s),
        pct_CN=100 * (s.DIAGNOSIS == "CN").mean(),
        pct_MCI=100 * (s.DIAGNOSIS == "MCI").mean(),
        pct_Dem=100 * (s.DIAGNOSIS == "Dementia").mean(),
        age=s.age.mean(),
        pct_female=100 * (s.PTGENDER == "Female").mean(),
        pct_APOE4=100 * s.APOE4_CARRIER.mean(),
        MMSE=s.MMSCORE.mean(),
        CDRSB=s.CDRSB.mean(),
        NfL_med=s.nfl_value.median(),
        FDG=s.fdg_suvr.mean(),
    )

print("\n" + "=" * 65)
print("                      GROUP PROFILES")
print("=" * 65)
prof = {g: profile(df[df.group == g]) for g in GROUPS if (df.group == g).any()}
pp = pd.DataFrame(prof).T
with pd.option_context("display.float_format", lambda v: f"{v:.1f}"):
    print(pp.to_string())

# -------------------------------------------------------------------------
# 4. STANDARDIZED DIFFERENCES (COHEN'S D) VS CONCORDANT
# -------------------------------------------------------------------------
con = df[df.group == "Concordant"]

def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.std()**2 + (nb - 1) * b.std()**2) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp else np.nan

print("\n" + "=" * 65)
print("STANDARDIZED DIFFERENCE vs CONCORDANT (Cohen's d; +/- = higher/lower)")
print("=" * 65)
for g in ["NfL-high/FDG-ok", "FDG-low/NfL-ok"]:
    s = df[df.group == g]
    if len(s) < 3:
        continue
    print(f"\n  {g} (n={len(s)}):")
    for var in ["age", "MMSCORE", "CDRSB"]:
        print(f"     {var:>8}: d={cohens_d(s[var].dropna(), con[var].dropna()):+.2f}")
    print(f"     APOE4+: {100*s.APOE4_CARRIER.mean():.0f}% vs {100*con.APOE4_CARRIER.mean():.0f}% concordant")
    print(f"     %Dementia: {100*(s.DIAGNOSIS=='Dementia').mean():.0f}% vs {100*(con.DIAGNOSIS=='Dementia').mean():.0f}% concordant")

# -------------------------------------------------------------------------
# 5. FIGURE GENERATION
# -------------------------------------------------------------------------
cmap = {
    "Concordant": "#9AA0A6",
    "NfL-high/FDG-ok": "#B4271F",
    "FDG-low/NfL-ok": "#185FA5",
    "Intermediate": "#D8D8D8",
}

fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))

# Panel 1: Scatter plot with thresholds
for g in GROUPS:
    s = df[df.group == g]
    ax[0].scatter(
        s.zN, s.zF, s=20, alpha=0.75, color=cmap[g], label=f"{g} ({len(s)})"
    )
ax[0].axhline(0, color="#bbb", lw=0.8)
ax[0].axvline(0, color="#bbb", lw=0.8)
ax[0].axhline(THR, color="#ddd", ls=":")
ax[0].axhline(-THR, color="#ddd", ls=":")
ax[0].axvline(THR, color="#ddd", ls=":")
ax[0].axvline(-THR, color="#ddd", ls=":")
ax[0].set_xlabel("NfL (age-adj z, higher=worse)")
ax[0].set_ylabel("-FDG (age-adj z, higher=worse)")
ax[0].set_title("Who the markers disagree about")
ax[0].legend(frameon=False, fontsize=8)

# Panel 2: Bar plot comparison
gg = [g for g in GROUPS if (df.group == g).any()]
x = np.arange(len(gg))
ax[1].bar(
    x - 0.2,
    [100 * (df[df.group == g].DIAGNOSIS == "Dementia").mean() for g in gg],
    0.4,
    label="% Dementia",
    color="#B4271F",
)
ax[1].bar(
    x + 0.2,
    [100 * df[df.group == g].APOE4_CARRIER.mean() for g in gg],
    0.4,
    label="% APOE4+",
    color="#ED9C28",
)
ax[1].set_xticks(x)
ax[1].set_xticklabels([g.replace("/", "/\n") for g in gg], fontsize=7)
ax[1].set_ylabel("%")
ax[1].set_title("Profile by group")
ax[1].legend(frameon=False)

fig.tight_layout()
fig.savefig("phase3_discordance.png", dpi=140, bbox_inches="tight")
print("\n" + "=" * 65)
print("Saved figure to 'phase3_discordance.png'")

# Save CSV output
df[["rid", "group", "zN", "zF", "DIAGNOSIS"]].to_csv("phase3_groups.csv", index=False)
print("Saved individual group mapping to 'phase3_groups.csv'")
print("=" * 65 + "\n")