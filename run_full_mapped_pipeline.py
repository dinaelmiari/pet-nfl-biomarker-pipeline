import re
from pathlib import Path
import numpy as np, pandas as pd, ants
from scipy.stats import pearsonr, spearmanr

BASE_DIR = Path("/scratch/delmiari/project")
COHORT   = BASE_DIR / "native_nfl_overlap_cohort.csv"
MATCHED  = BASE_DIR / "phase4_matched_images.csv"
T1_DIR   = BASE_DIR / "data/selected_t1"
OUT_DIR  = BASE_DIR / "data/synthseg_output"

METAROI_LABELS = [1023, 2023, 1025, 2025, 1008, 2008, 1009, 2009, 1015, 2015]
REF_LABELS     = [16, 7, 46]

cohort_df = pd.read_csv(COHORT)
matched_df = pd.read_csv(MATCHED)

# Ensure date formatting matches
cohort_df['PET_DATE_STR']  = pd.to_datetime(cohort_df['PET_DATE']).dt.strftime('%Y-%m-%d')
matched_df['EXAMDATE_STR'] = pd.to_datetime(matched_df['EXAMDATE']).dt.strftime('%Y-%m-%d')

t1_files  = list(T1_DIR.glob("*.nii*"))
seg_files = list(OUT_DIR.glob("*.nii*"))

def find_file_by_rid(file_list, rid):
    rid_str = str(int(rid))
    padded_rid = f"{int(rid):04d}"
    pattern = re.compile(rf"_S_(?:0*{rid_str}|{padded_rid})[\._]")
    matches = [f for f in file_list if pattern.search(f.name)]
    return matches[0] if matches else None

results = []
print(f"Processing {len(cohort_df)} cohort records via EXAMDATE mapping...\n")

for idx, row in cohort_df.iterrows():
    rid      = int(row['RID'])
    pet_date = row['PET_DATE_STR']
    
    # 1. Match PET image via EXAMDATE
    pet_match = matched_df[matched_df['EXAMDATE_STR'] == pet_date]
    if pet_match.empty:
        results.append({'RID': rid, 'PET_DATE': pet_date, 'status': 'missing_pet_date_match'})
        continue
    
    pet_path = Path(pet_match.iloc[0]['nifti_path'])
    if not pet_path.exists():
        results.append({'RID': rid, 'PET_DATE': pet_date, 'status': 'pet_file_not_found'})
        continue

    # 2. Match T1 and SynthSeg files by RID
    t1_f  = find_file_by_rid(t1_files, rid)
    seg_f = find_file_by_rid(seg_files, rid)
    
    if not t1_f or not seg_f:
        missing = []
        if not t1_f: missing.append("T1")
        if not seg_f: missing.append("SynthSeg")
        results.append({'RID': rid, 'PET_DATE': pet_date, 'status': f"missing_{'_'.join(missing)}"})
        continue

    try:
        fixed_t1   = ants.image_read(str(t1_f))
        moving_pet = ants.image_read(str(pet_path))
        seg_img    = ants.image_read(str(seg_f))

        # Two-stage registration: Translation (Centers of Mass) -> Rigid
        reg = ants.registration(
            fixed=fixed_t1,
            moving=moving_pet,
            type_of_transform='DenseRigid',
            initial_transform=None,
            verbose=False
        )

        warped_pet = ants.apply_transforms(
            fixed=seg_img,
            moving=moving_pet,
            transformlist=reg['fwdtransforms'],
            interpolator='linear'
        )

        pet_data = warped_pet.numpy()
        seg_data = seg_img.numpy()

        tmask = np.isin(seg_data, METAROI_LABELS)
        rmask = np.isin(seg_data, REF_LABELS)

        target_m = float(pet_data[tmask].mean()) if tmask.any() else np.nan
        ref_m    = float(pet_data[rmask].mean()) if rmask.any() else np.nan
        suvr     = target_m / ref_m if (ref_m and ref_m > 0) else np.nan

        results.append({
            'RID': rid,
            'PET_DATE': pet_date,
            'native_suvr': suvr, 
            'target_mean': target_m, 
            'ref_mean': ref_m, 
            'ucb_suvr': row.get('ucb_suvr', np.nan),
            'PLASMA_NFL': row.get('PLASMA_NFL', np.nan),
            'status': 'ok'
        })
        print(f"[RID {rid:4d} | {pet_date}] SUVR: {suvr:.4f} | Target: {target_m:.4f} | Ref: {ref_m:.4f}")
    except Exception as e:
        results.append({'RID': rid, 'PET_DATE': pet_date, 'status': f"error: {str(e)}"})

res_df = pd.DataFrame(results)
res_df.to_csv(BASE_DIR / "data/synthseg_output/clean_suvr_mapped_results.csv", index=False)

valid = res_df[res_df['status'] == 'ok'].drop_duplicates(subset=['RID'])

print("\n=== VALIDATION RESULTS ===")
print(f"Total Cohort Rows Processed: {len(res_df)}")
print(f"Valid Unique RIDs Extracted: {len(valid)}")

if len(valid) > 2:
    sub_ucb = valid.dropna(subset=['native_suvr', 'ucb_suvr'])
    if len(sub_ucb) > 2:
        r_ucb, p_ucb = pearsonr(sub_ucb['native_suvr'], sub_ucb['ucb_suvr'])
        s_ucb, sp_ucb = spearmanr(sub_ucb['native_suvr'], sub_ucb['ucb_suvr'])
        print(f"\n1. Native SUVR vs. UCB SUVR (N={len(sub_ucb)}):")
        print(f"   Pearson r  = {r_ucb:.4f} (p = {p_ucb:.4e})")
        print(f"   Spearman rho = {s_ucb:.4f} (p = {sp_ucb:.4e})")

    sub_nfl = valid.dropna(subset=['native_suvr', 'PLASMA_NFL'])
    if len(sub_nfl) > 2:
        r_nfl, p_nfl = pearsonr(sub_nfl['native_suvr'], sub_nfl['PLASMA_NFL'])
        s_nfl, sp_nfl = spearmanr(sub_nfl['native_suvr'], sub_nfl['PLASMA_NFL'])
        print(f"\n2. Native SUVR vs. Plasma NfL (N={len(sub_nfl)}):")
        print(f"   Pearson r  = {r_nfl:.4f} (p = {p_nfl:.4e})")
        print(f"   Spearman rho = {s_nfl:.4f} (p = {sp_nfl:.4e})")

