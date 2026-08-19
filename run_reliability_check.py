import re, os
from pathlib import Path
import numpy as np, pandas as pd, ants

BASE_DIR = Path("/scratch/delmiari/project")
FINAL    = BASE_DIR / "phase4_final_cohort.csv"
T1_DIR   = BASE_DIR / "data/selected_t1"
OUT_DIR  = BASE_DIR / "data/synthseg_output"

# Regions (SynthSeg labels)
METAROI_LABELS = [1023, 2023, 1025, 2025, 1008, 2008, 1009, 2009, 1015, 2015] # Cortical composite
REF_LABELS     = [16, 7, 46]                                                # Brainstem / Cerebellar

final_df  = pd.read_csv(FINAL).drop_duplicates(subset=['RID'])
t1_files  = list(T1_DIR.glob("*.nii*"))
seg_files = list(OUT_DIR.glob("*.nii*"))

def find_file_by_rid(file_list, rid):
    rid_str = str(int(rid))
    padded_rid = f"{int(rid):04d}"
    pattern = re.compile(rf"_S_(?:0*{rid_str}|{padded_rid})[\._]")
    matches = [f for f in file_list if pattern.search(f.name)]
    return matches[0] if matches else None

def compute_suvr(pet_numpy, seg_numpy):
    tmask = np.isin(seg_numpy, METAROI_LABELS)
    rmask = np.isin(seg_numpy, REF_LABELS)
    target_m = float(pet_numpy[tmask].mean()) if tmask.any() else np.nan
    ref_m    = float(pet_numpy[rmask].mean()) if rmask.any() else np.nan
    return target_m / ref_m if (ref_m and ref_m > 0) else np.nan

results = []
print("==================================================================")
print("       RUNNING RELIABILITY AUDIT (TEST A, TEST B1, TEST B2)       ")
print("==================================================================\n")

target_count = 5
processed = 0

for idx, row in final_df.iterrows():
    if processed >= target_count:
        break

    rid = int(row['RID'])
    pet_path = Path(row['nifti_path'])
    
    t1_f  = find_file_by_rid(t1_files, rid)
    seg_f = find_file_by_rid(seg_files, rid)
    
    if not (pet_path.exists() and t1_f and seg_f):
        continue

    processed += 1

    fixed_t1   = ants.image_read(str(t1_f))
    moving_pet = ants.image_read(str(pet_path))
    seg_img    = ants.image_read(str(seg_f))
    seg_data   = seg_img.numpy()

    # --- RUN 1: Full Registration & Matrix Export ---
    reg1 = ants.registration(
        fixed=fixed_t1, moving=moving_pet, type_of_transform='DenseRigid', verbose=False
    )
    warped_pet1 = ants.apply_transforms(
        fixed=seg_img, moving=moving_pet, transformlist=reg1['fwdtransforms'], interpolator='linear'
    )
    suvr_run1 = compute_suvr(warped_pet1.numpy(), seg_data)

    # --- TEST A: Same Warped Volume, Re-masked (Array Indexing Check) ---
    suvr_test_a = compute_suvr(warped_pet1.numpy(), seg_data)
    var_a = abs(suvr_run1 - suvr_test_a)

    # --- TEST B1: Independent Re-Registration (Testing Optimization Jitter) ---
    reg2 = ants.registration(
        fixed=fixed_t1, moving=moving_pet, type_of_transform='DenseRigid', verbose=False
    )
    warped_pet2 = ants.apply_transforms(
        fixed=seg_img, moving=moving_pet, transformlist=reg2['fwdtransforms'], interpolator='linear'
    )
    suvr_test_b1 = compute_suvr(warped_pet2.numpy(), seg_data)
    pct_var_b1 = (abs(suvr_run1 - suvr_test_b1) / suvr_run1) * 100.0

    # --- TEST B2: Re-applying Saved Transform Matrix (Register-Once Proof) ---
    saved_transform = reg1['fwdtransforms']
    warped_pet_saved = ants.apply_transforms(
        fixed=seg_img, moving=moving_pet, transformlist=saved_transform, interpolator='linear'
    )
    suvr_test_b2 = compute_suvr(warped_pet_saved.numpy(), seg_data)
    pct_var_b2 = (abs(suvr_run1 - suvr_test_b2) / suvr_run1) * 100.0

    results.append({
        'RID': rid,
        'SUVR_Run1': suvr_run1,
        'SUVR_TestA': suvr_test_a,
        'Var_A_Abs': var_a,
        'SUVR_TestB1_Indep': suvr_test_b1,
        'PctVar_B1_Indep': pct_var_b1,
        'SUVR_TestB2_SavedTx': suvr_test_b2,
        'PctVar_B2_SavedTx': pct_var_b2
    })

    print(f"[RID {rid:4d}]")
    print(f"  └─ Run 1 Base SUVR:          {suvr_run1:.6f}")
    print(f"  └─ Test A  (Re-extract):    {suvr_test_a:.6f}  | Abs Diff: {var_a:.2e}")
    print(f"  └─ Test B1 (Indep Reg):     {suvr_test_b1:.6f}  | % Variance: {pct_var_b1:.4f}%")
    print(f"  └─ Test B2 (Saved Matrix):  {suvr_test_b2:.6f}  | % Variance: {pct_var_b2:.4f}%")
    print("-" * 65)

res_df = pd.DataFrame(results)
if len(res_df) > 0:
    print("\n=== SUMMARY METRICS OVER SUBJECTS ===")
    print(f"Total Subjects Evaluated:                    {len(res_df)}")
    print(f"Mean Abs Diff (Test A - Array Indexing):     {res_df['Var_A_Abs'].mean():.2e}")
    print(f"Mean % Variance (Test B1 - Independent Reg): {res_df['PctVar_B1_Indep'].mean():.4f}%")
    print(f"Mean % Variance (Test B2 - Saved Matrix):    {res_df['PctVar_B2_SavedTx'].mean():.4f}%")
else:
    print("\n[ERROR] No matching subjects with complete (PET, T1, SynthSeg) files were found!")

