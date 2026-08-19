import os
import pandas as pd
import numpy as np
import rdata

print("=== BUILDING FULL NATIVE COHORT (DXSUM + FREESURFER + UCBERKELEY) ===")

# 1. Load DXSUM (Diagnosis)
dx_path = "/scratch/delmiari/project/data/ADNIMERGE2/data/DXSUM.rda"
parsed_dx = rdata.parser.parse_file(dx_path)
converted_dx = rdata.conversion.convert(parsed_dx)
df_dx = list(converted_dx.values())[0]

# 2. Load FreeSurfer
fs_path = "/scratch/delmiari/project/data/ADNIMERGE2/data/UCSFFSX.rda"
parsed_fs = rdata.parser.parse_file(fs_path)
converted_fs = rdata.conversion.convert(parsed_fs)
df_fs = list(converted_fs.values())[0]

# 3. Load UC Berkeley Ground Truth
ucb_path = "/scratch/delmiari/project/data/UCBERKELEYFDG_8mm_02_17_23_14Aug2026.csv"
df_ucb = pd.read_csv(ucb_path)

# Pivot UC Berkeley
ucb_pivot = df_ucb.pivot_table(
    index=['RID', 'EXAMDATE', 'VISCODE'], 
    columns='ROINAME', 
    values='MEAN', 
    aggfunc='first'
).reset_index()

if 'MetaROI' in ucb_pivot.columns and 'Top50PonsVermis' in ucb_pivot.columns:
    ucb_pivot['ucb_suvr'] = ucb_pivot['MetaROI'] / ucb_pivot['Top50PonsVermis']

# Clean Types
df_dx['RID'] = pd.to_numeric(df_dx['RID'], errors='coerce')
df_fs['RID'] = pd.to_numeric(df_fs['RID'], errors='coerce')
ucb_pivot['RID'] = pd.to_numeric(ucb_pivot['RID'], errors='coerce')

# Diagnostic mapping
dx_map = {1: 'CN', 2: 'MCI', 3: 'Dementia', 'CN': 'CN', 'MCI': 'MCI', 'Dementia': 'Dementia', 'AD': 'Dementia'}
if 'DIAGNOSIS' in df_dx.columns:
    df_dx['DX_Group'] = df_dx['DIAGNOSIS'].map(dx_map).fillna(df_dx['DIAGNOSIS'])

# Merge everything
cohort = ucb_pivot.merge(df_dx[['RID', 'VISCODE', 'DX_Group']].dropna(), on=['RID', 'VISCODE'], how='inner')
if 'OVERALLQC' in df_fs.columns:
    cohort = cohort.merge(df_fs[['RID', 'VISCODE', 'OVERALLQC']].dropna(), on=['RID', 'VISCODE'], how='inner')
cohort = cohort.drop_duplicates(subset=['RID', 'EXAMDATE'])

print(f"\nFinal Consolidated Native Cohort Size: {len(cohort)} Scans")

print("\n--- Diagnostic Breakdown & SUVR Statistics ---")
stats = cohort.groupby('DX_Group')['ucb_suvr'].agg(['count', 'mean', 'std', 'median'])
print(stats.to_string())

# Save master dataset
output_csv = "/scratch/delmiari/project/native_suvr_master_cohort.csv"
cohort.to_csv(output_csv, index=False)
print(f"\nSaved consolidated master cohort to: {output_csv}")
