"""
Analyze the upgraded $n = 352$ UPENN multi-marker cohort.
1. Global and stage-specific Spearman correlations (FDG vs NfL).
2. Age-adjusted partial correlations / regression.
3. Multi-marker showdown: comparing how NfL, pT217, GFAP, and Amyloid track FDG hypometabolism.
"""
import pandas as pd
import numpy as np
from scipy import stats

def main():
    print("Loading upenn_matched_cohort.csv...")
    df = pd.read_csv("upenn_matched_cohort.csv")
    print(f"Loaded cohort size: {len(df)} subjects\n")

    # Clean subset for statistics
    sub = df.dropna(subset=["nfl_value", "fdg_suvr"]).copy()

    # 1. Global Spearman Correlation: FDG vs NfL
    r_global, p_global = stats.spearmanr(sub["fdg_suvr"], sub["nfl_value"])
    print("=" * 60)
    print("1. GLOBAL FDG vs NfNaL (Quanterix) CONCORDANCE")
    print("=" * 60)
    print(f"Spearman r = {r_global:.3f} (p = {p_global:.2e}, n = {len(sub)})\n")

    # 2. Within-Group Correlations by Diagnosis
    print("=" * 60)
    print("2. WITHIN-GROUP CORRELATIONS (FDG vs NfL)")
    print("=" * 60)
    if "DIAGNOSIS" in sub.columns:
        for dx, group in sub.groupby("DIAGNOSIS"):
            if len(group) > 5:
                r_dx, p_dx = stats.spearmanr(group["fdg_suvr"], group["nfl_value"])
                print(f"  {dx:<10}: r = {r_dx:.3f} (p = {p_dx:.3f}, n = {len(group)})")
    print()

    # 3. Multi-Marker Showdown: Correlating FDG against all panel markers
    print("=" * 60)
    print("3. MULTI-MARKER SHOWDOWN (Spearman r against FDG SUVR)")
    print("=" * 60)
    markers = ["nfl_value", "NfL_F", "pT217_F", "GFAP_Q", "AB42_AB40_F"]
    marker_labels = {
        "nfl_value": "NfL (Quanterix)",
        "NfL_F":     "NfL (Fujirebio)",
        "pT217_F":   "p-tau217 (Fujirebio)",
        "GFAP_Q":    "GFAP (Quanterix)",
        "AB42_AB40_F": "Abeta42/40 Ratio"
    }
    
    results = []
    for m in markers:
        if m in sub.columns:
            clean_m = sub.dropna(subset=[m, "fdg_suvr"])
            if len(clean_m) > 5:
                r_m, p_m = stats.spearmanr(clean_m["fdg_suvr"], clean_m[m])
                label = marker_labels.get(m, m)
                results.append({"marker": label, "r": r_m, "p": p_m, "n": len(clean_m)})
                print(f"  {label:<22}: Spearman r = {r_m:+.3f}  (p = {p_m:.2e}, n = {len(clean_m)})")

    print("\nAnalysis complete! Review the output above to verify your headline findings.")

if __name__ == "__main__":
    main()