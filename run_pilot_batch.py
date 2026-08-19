import os, sys, gc
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib, ants

# Define paths
BASE_DIR = Path("/scratch/delmiari/project")
COHORT   = BASE_DIR / "native_nfl_overlap_cohort.csv"
T1_DIR   = BASE_DIR / "data/selected_t1"
PET_DIR  = BASE_DIR / "data/nifti_fdg"
OUT_DIR  = BASE_DIR / "data/synthseg_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS  = OUT_DIR / "native_suvr_pilot_matched_43.csv"

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

def process_subject(t1_path, pet_path, seg_path, out_warped_path):
    tx_t1  = ants.image_read(str(t1_path))
    tx_pet = ants.image_read(str(pet_path))

    # Coregister PET to T1
    reg = ants.registration(fixed=tx_t1, moving=tx_pet, type_of_transform='Rigid', verbose=False)
    
    # Resample warped PET directly onto fixed T1 matrix grid
    warped_pet = ants.apply_transforms(
        fixed=tx_t1,
        moving=tx_pet,
        transformlist=reg['fwdtransforms'],
        interpolator='linear'
    )
    
    ants.image_write(warped_pet, str(out_warped_path))

    # Load both via nibabel (now guaranteed identical matrix dimensions)
    pet_nib = nib.load(str(out_warped_path))
    seg_nib = nib.load(str(seg_path))

    pet_data = pet_nib.get_fdata()
    seg_data = seg_nib.get_fdata()

    assert pet_data.shape == seg_data.shape, f"Shape mismatch: {pet_data.shape} vs {seg_data.shape}"

    target_mask = np.isin(seg_data, METAROI_LABELS)
    ref_mask    = np.isin(seg_data, REF_LABELS)

    target_mean = np.mean(pet_data[target_mask]) if np.any(target_mask) else np.nan
    ref_mean    = np.mean(pet_data[ref_mask]) if np.any(ref_mask) else np.nan
    suvr        = target_mean / ref_mean if (ref_mean and ref_mean > 0) else np.nan

    return target_mean, ref_mean, suvr

def main():
    cohort_df = pd.read_csv(COHORT)
    results = []
    
    print(f"Starting extraction on {len(cohort_df)} subjects using pilot alignment logic...")
    
    for idx, row in cohort_df.iterrows():
        rid  = row['RID']
        ptid = row.get('PTID', '')
        
        t1, pet, seg, status = resolve_files(rid)
        
        if status != "ok":
            results.append({'RID': rid, 'PTID': ptid, 'native_suvr': np.nan, 'target_mean': np.nan, 'ref_mean': np.nan, 'status': status})
            continue
            
        warped_out = OUT_DIR / f"{rid}_pet_to_t1.nii.gz"
        try:
            target_m, ref_m, suvr = process_subject(t1, pet, seg, warped_out)
            status_flag = "ok" if (0.85 <= suvr <= 1.85) else "suvr_out_of_range"
            
            results.append({'RID': rid, 'PTID': ptid, 'native_suvr': suvr, 'target_mean': target_m, 'ref_mean': ref_m, 'status': status_flag})
            print(f"[RID {rid}] SUVR: {suvr:.4f} | Target: {target_m:.4f} | Ref: {ref_m:.4f} | Status: {status_flag}")
        except Exception as e:
            results.append({'RID': rid, 'PTID': ptid, 'native_suvr': np.nan, 'target_mean': np.nan, 'ref_mean': np.nan, 'status': f"error: {str(e)}"})
            print(f"[RID {rid}] Failed: {e}")
            
    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS, index=False)
    print(f"\nFinished! Results saved to: {RESULTS}")

if __name__ == '__main__':
    main()
