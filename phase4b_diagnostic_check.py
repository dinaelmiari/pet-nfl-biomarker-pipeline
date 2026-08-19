import glob
import os
import pandas as pd
import numpy as np
import nibabel as nib
import ants

print("=== PHASE 4B DIAGNOSTIC & CN vs. DEMENTIA VALIDATION ===")

meta_mask_path = "/scratch/delmiari/project/mask_metaroi.nii.gz"
pons_mask_path = "/scratch/delmiari/project/mask_pons.nii.gz"

meta_ants = ants.image_read(meta_mask_path)
pons_ants = ants.image_read(pons_mask_path)

meta_arr = meta_ants.numpy() > 0.5
pons_arr = pons_ants.numpy() > 0.5

print(f"MetaROI Mask Voxels: {np.sum(meta_arr)}")
print(f"Pons Mask Voxels:    {np.sum(pons_arr)}")

# Load Cohort Data
coh = pd.read_csv("/scratch/delmiari/project/phase4_final_cohort.csv")
print("\nCohort Columns Available:", list(coh.columns))

# Detect Diagnosis column dynamically
dx_col = None
for candidate in ['DX', 'DIAGNOSIS', 'DX_bl', 'Group', 'DXGROUP', 'Research Group']:
    if candidate in coh.columns:
        dx_col = candidate
        break

if not dx_col:
    # Look for fuzzy match
    for c in coh.columns:
        if 'dx' in c.lower() or 'group' in c.lower() or 'diag' in c.lower():
            dx_col = c
            break

print(f"Using Diagnosis Column: '{dx_col}'")

pet_files = [f for f in glob.glob("/scratch/delmiari/project/**/*.nii*", recursive=True) if 'mask' not in os.path.basename(f).lower()][:30]

results = []
for pf in pet_files:
    fname = os.path.basename(pf)
    
    exam_date = ""
    for p in fname.split("_"):
        if len(p) >= 8 and p[:8].isdigit():
            exam_date = f"{p[:4]}-{p[4:6]}-{p[6:8]}"
            break
            
    if not exam_date:
        continue

    try:
        pet_img = ants.image_read(pf)
        # Resample PET to mask grid (Simple Phase 4b approach)
        resampled_pet = ants.resample_image_to_target(pet_img, meta_ants).numpy()
        
        m_val = np.mean(resampled_pet[meta_arr]) if np.sum(meta_arr) > 0 else np.nan
        p_val = np.mean(resampled_pet[pons_arr]) if np.sum(pons_arr) > 0 else np.nan
        suvr  = m_val / p_val if (p_val > 0 and not np.isnan(p_val)) else np.nan
        
        results.append({
            'nifti_file': fname,
            'exam_date': exam_date,
            'target_mean': m_val,
            'ref_mean': p_val,
            'suvr_simple': suvr
        })
    except Exception as e:
        pass

df_res = pd.DataFrame(results)

ucb = pd.read_csv("/scratch/delmiari/project/data/UCBERKELEYFDG_8mm_02_17_23_14Aug2026.csv")

coh['EXAMDATE'] = coh['EXAMDATE'].astype(str)
ucb['EXAMDATE'] = ucb['EXAMDATE'].astype(str)
ucb['RID']      = pd.to_numeric(ucb['RID'], errors='coerce')

merge_cols = ['EXAMDATE', 'RID']
if dx_col and dx_col not in merge_cols:
    merge_cols.append(dx_col)

mapped = df_res.merge(coh[merge_cols].drop_duplicates('EXAMDATE'), left_on='exam_date', right_on='EXAMDATE', how='inner')

ucb_pivot = ucb.pivot_table(index=['RID', 'EXAMDATE'], columns='ROINAME', values='MEAN', aggfunc='first').reset_index()
if 'MetaROI' in ucb_pivot.columns and 'Top50PonsVermis' in ucb_pivot.columns:
    ucb_pivot['ucb_suvr'] = ucb_pivot['MetaROI'] / ucb_pivot['Top50PonsVermis']

matched = mapped.merge(ucb_pivot, left_on=['RID', 'exam_date'], right_on=['RID', 'EXAMDATE'], how='inner')

print(f"\nMatched Diagnostic Scans: {len(matched)}")

if len(matched) > 0 and dx_col and dx_col in matched.columns:
    print(f"\n--- Diagnostic Group Statistics (Simple Native Grid SUVR by '{dx_col}') ---")
    print(matched.groupby(dx_col)['suvr_simple'].agg(['count', 'mean', 'std']).to_string())

if len(matched) > 0:
    r_val = np.corrcoef(matched['suvr_simple'], matched['ucb_suvr'])[0, 1]
    print(f"\nPhase 4b Correlation with UC Berkeley: r = {r_val:.4f}\n")
    print("--- Direct Comparison Sample ---")
    show_cols = [c for c in ['RID', 'exam_date', dx_col, 'suvr_simple', 'ucb_suvr', 'target_mean', 'ref_mean'] if c and c in matched.columns]
    print(matched[show_cols].head(10).to_string())
