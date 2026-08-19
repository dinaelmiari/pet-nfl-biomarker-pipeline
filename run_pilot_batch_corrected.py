import os, sys, gc
from pathlib import Path
import numpy as np, pandas as pd, ants

BASE_DIR = Path("/scratch/delmiari/project")
COHORT   = BASE_DIR / "native_nfl_overlap_cohort.csv"
T1_DIR   = BASE_DIR / "data/selected_t1"
PET_DIR  = BASE_DIR / "data/nifti_fdg"
OUT_DIR  = BASE_DIR / "data/synthseg_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS  = OUT_DIR / "clean_suvr_43_corrected.csv"

METAROI_LABELS = [1023, 2023, 1025, 2025, 1008, 2008, 1009, 2009, 1015, 2015]
REF_LABELS     = [16, 7, 46]

def resolve_files(rid):
    rid_int = int(rid)
    rid_str_padded = f"{rid_int:04d}"
    rid_str_raw    = str(rid_int)
    
    t1_matches = list(T1_DIR.glob(f"*_S_{rid_str_padded}_T1.nii*")) + \
                 list(T1_DIR.glob(f"*_S_{rid_str_raw}_T1.nii*")) + \
                 list(T1_DIR.glob(f"*{rid_str_padded}*T1.nii*")) + \
                 list(T1_DIR.glob(f"*{rid_str_raw}*T1.nii*"))
    t1_matches = list(dict.fromkeys(t1_matches))
    
    pet_matches = list(PET_DIR.glob(f"*_S_{rid_str_padded}*.nii*")) + \
                  list(PET_DIR.glob(f"*_S_{rid_str_raw}*.nii*")) + \
                  list(PET_DIR.glob(f"*{rid_str_padded}*.nii*")) + \
                  list(PET_DIR.glob(f"*{rid_str_raw}*.nii*"))
    pet_matches = list(dict.fromkeys(pet_matches))
    
    seg_matches = list(OUT_DIR.glob(f"*{rid_str_padded}*synthseg*.nii*")) + \
                  list(OUT_DIR.glob(f"*{rid_str_raw}*synthseg*.nii*")) + \
                  list(OUT_DIR.glob(f"*{rid}*synthseg*.nii*"))
    seg_matches = list(dict.fromkeys(seg_matches))
    
    if not t1_matches: return None, None, None, "missing_t1"
    if not pet_matches: return t1_matches[0], None, None, "missing_pet"
    if not seg_matches: return t1_matches[0], pet_matches[0], None, "missing_synthseg"
    
    return t1_matches[0], pet_matches[0], seg_matches[0], "ok"

def process_subject(t1_path, pet_path, seg_path):
    fixed_t1   = ants.image_read(str(t1_path))
    moving_pet = ants.image_read(str(pet_path))
    seg_img    = ants.image_read(str(seg_path))

    reg = ants.registration(fixed=fixed_t1, moving=moving_pet, type_of_transform='Rigid', verbose=False)

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

    target_mean = float(pet_data[tmask].mean()) if tmask.any() else np.nan
    ref_mean    = float(pet_data[rmask].mean()) if rmask.any() else np.nan
    suvr        = target_mean / ref_mean if (ref_mean and ref_mean > 0) else np.nan

    return target_mean, ref_mean, suvr

def main():
    cohort_df = pd.read_csv(COHORT)
    results = []
    
    print(f"Starting corrected batch extraction on {len(cohort_df)} subjects...")
    
    for idx, row in cohort_df.iterrows():
        rid  = row['RID']
        ptid = row.get('PTID', '')
        
        t1, pet, seg, status = resolve_files(rid)
        
        if status != "ok":
            results.append({'RID': rid, 'PTID': ptid, 'native_suvr': np.nan, 'target_mean': np.nan, 'ref_mean': np.nan, 'status': status})
            continue
            
        try:
            target_m, ref_m, suvr = process_subject(t1, pet, seg)
            status_flag = "ok" if (0.85 <= suvr <= 1.85) else "suvr_out_of_range"
            
            results.append({'RID': rid, 'PTID': ptid, 'native_suvr': suvr, 'target_mean': target_m, 'ref_mean': ref_m, 'status': status_flag})
            print(f"[RID {rid}] SUVR: {suvr:.4f} | Target: {target_m:.4f} | Ref: {ref_m:.4f} | Status: {status_flag}")
        except Exception as e:
            results.append({'RID': rid, 'PTID': ptid, 'native_suvr': np.nan, 'target_mean': np.nan, 'ref_mean': np.nan, 'status': f"error: {str(e)}"})
            print(f"[RID {rid}] Failed: {e}")
            
    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS, index=False)
    print(f"\nFinished batch run! Saved results to: {RESULTS}")

if __name__ == '__main__':
    main()
