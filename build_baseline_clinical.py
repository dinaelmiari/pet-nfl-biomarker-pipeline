import pyreadr
import pandas as pd

DATA = "/scratch/delmiari/project/data/ADNIMERGE2/data/"

def load_df(name):
    res = pyreadr.read_r(DATA + name + ".rda")
    return res[list(res.keys())[0]]

print("Loading raw files...")
dxsum = load_df("DXSUM")
ptdemog = load_df("PTDEMOG")
apoeres = load_df("APOERES")
mmse = load_df("MMSE")
cdr = load_df("CDR")

# 1. Filter Baseline Visit for Diagnostic & Cognitive Tables
dx_bl = dxsum[dxsum['VISCODE2'] == 'bl'][['RID', 'DIAGNOSIS', 'EXAMDATE']].drop_duplicates(subset=['RID'])
mmse_bl = mmse[mmse['VISCODE2'] == 'bl'][['RID', 'MMSCORE']].drop_duplicates(subset=['RID'])
cdr_bl = cdr[cdr['VISCODE2'] == 'bl'][['RID', 'CDGLOBAL', 'CDRSB']].drop_duplicates(subset=['RID'])

# 2. Extract Demographics (Subject level)
demog_sub = ptdemog[['RID', 'PTGENDER', 'PTEDUCAT', 'PTDOBYY']].drop_duplicates(subset=['RID'])

# 3. Extract APOE Status & derive e4 carrier flag
apoe_sub = apoeres[['RID', 'GENOTYPE']].drop_duplicates(subset=['RID'])
apoe_sub['APOE4_CARRIER'] = apoe_sub['GENOTYPE'].astype(str).str.contains('4').astype(int)

# 4. Merge all together on RID
cohort = dx_bl.merge(demog_sub, on='RID', how='left') \
              .merge(apoe_sub, on='RID', how='left') \
              .merge(mmse_bl, on='RID', how='left') \
              .merge(cdr_bl, on='RID', how='left')

print(f"\nSuccessfully created baseline clinical cohort: shape={cohort.shape}")
print(cohort.head())

# Save to CSV for easy downstream joining with PET/Plasma scripts
cohort.to_csv("baseline_clinical_cohort.csv", index=False)
print("\nSaved to 'baseline_clinical_cohort.csv'")
