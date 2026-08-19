import os
import pandas as pd
import numpy as np
import scipy.stats as stats

print("=== NATIVE COHORT STATISTICAL VALIDATION & NFL OVERLAP CHECK ===")

# 1. Load Consolidated Master Cohort
master_path = "/scratch/delmiari/project/native_suvr_master_cohort.csv"
df = pd.read_csv(master_path)

print(f"Total Native Cohort: {len(df)} scans | Unique Subjects: {df['RID'].nunique()}")

# -------------------------------------------------------------------------
# CHECK 1: STATISTICAL SEPARATION (Kruskal-Wallis & Cohen's d)
# -------------------------------------------------------------------------
cn = df[df['DX_Group'] == 'CN']['ucb_suvr'].dropna()
mci = df[df['DX_Group'] == 'MCI']['ucb_suvr'].dropna()
dem = df[df['DX_Group'] == 'Dementia']['ucb_suvr'].dropna()

# Kruskal-Wallis H-test
kw_stat, kw_p = stats.kruskal(cn, mci, dem)

# Cohen's d between CN and Dementia
n1, n2 = len(cn), len(dem)
s1, s2 = np.var(cn, ddof=1), np.var(dem, ddof=1)
s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
cohens_d = (np.mean(cn) - np.mean(dem)) / s_pooled

print("\n--- Diagnostic Separation Statistics ---")
print(f"Kruskal-Wallis H-statistic: {kw_stat:.2f} (p-value: {kw_p:.3e})")
print(f"CN vs Dementia Cohen's d:   {cohens_d:.3f}")

# -------------------------------------------------------------------------
# CHECK 2: NFL BIOMARKER OVERLAP
# -------------------------------------------------------------------------
nfl_path = "/scratch/delmiari/project/data/ADNIMERGE2/data/BLENNOWNFL.rda"

try:
    import rdata
    parsed_nfl = rdata.parser.parse_file(nfl_path)
    converted_nfl = rdata.conversion.convert(parsed_nfl)
    df_nfl = list(converted_nfl.values())[0]

    df_nfl['RID'] = pd.to_numeric(df_nfl['RID'], errors='coerce')
    
    # Merge with Master Cohort on RID and VISCODE
    nfl_overlap = df.merge(df_nfl[['RID', 'VISCODE', 'PLASMA_NFL']].dropna(), on=['RID', 'VISCODE'], how='inner')
    nfl_overlap = nfl_overlap.drop_duplicates(subset=['RID', 'EXAMDATE'])

    print("\n--- NfL Biomarker Overlap ---")
    print(f"Matched NfL Scans:     {len(nfl_overlap)}")
    print(f"Unique NfL Subjects:  {nfl_overlap['RID'].nunique()}")
    print("\nDiagnostic Breakdown of NfL Overlap Subset:")
    print(nfl_overlap['DX_Group'].value_counts().to_string())

    # Save locked milestone cohort with NfL flags
    nfl_overlap.to_csv("/scratch/delmiari/project/native_nfl_overlap_cohort.csv", index=False)
    print("\nSaved locked NfL cohort to: /scratch/delmiari/project/native_nfl_overlap_cohort.csv")

except Exception as e:
    print(f"\nCould not load/process NfL data: {e}")
    print("Ensure rdata is available and file path is correct.")

