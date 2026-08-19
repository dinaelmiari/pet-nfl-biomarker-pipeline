"""
Phase 4 - Native SUVR Extraction via SynthSeg with Composite Ref (Pons + Cerebellar WM)
Scales the exact pilot recipe (052_S_1346) across the 37-subject cohort via Slurm.
"""
import subprocess, sys, gc
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib, ants
from scipy import stats

COHORT  = Path("/scratch/delmiari/project/native_nfl_overlap_cohort.csv")
T1_DIR  = Path("/home/delmiari/scratch/project/data/selected_t1")
PET_DIR = Path("/home/delmiari/scratch/project/data/nifti_t1")
OUT_DIR = Path("/home/delmiari/scratch/project/data/synthseg_output"); OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = OUT_DIR / "native_suvr_37.csv"

METAROI_LABELS = [1023, 2023, 1025, 2025, 1008, 2008, 1009, 2009, 1015, 2015]
REF_LABELS = [16, 7, 46]

def extract_suvr(pet_data, seg_data):
    assert pet_data.shape == seg_data.shape, f"Shape mismatch: {pet_data.shape} vs {seg_data.shape}"
    tmask = np.isin(seg_data, METAROI_LABELS)
    rmask = np.isin(seg_data, REF_LABELS)
    
    if not tmask.any() or not rmask.any():
        return np.nan, np.nan, np.nan, int(tmask.sum()), int(rmask.sum())
        
    tmean = float(pet_data[tmask].mean())
    ref_voxels = pet_data[rmask]
    cutoff = np.percentile(ref_voxels, 50)
    rmean_top50 = float(ref_voxels[ref_voxels >= cutoff].mean())
    
    suvr = tmean / rmean_top50
    return suvr, tmean, rmean_top50, int(tmask.sum()), int(rmask.sum())

def resolve_ptid_and_t1(rid):
    hits = list(T1_DIR.glob(f"*_{rid}_T1.nii.gz"))
    if hits:
        t1_path = hits[0]
        ptid = t1_path.name.replace("_T1.nii.gz", "")
        return ptid, t1_path
    return None, None

def find_pet(ptid):
    d = PET_DIR / ptid
    for pat in ("*Coreg_Avg_*.nii.gz", "*.nii.gz", "*.nii"):
        hits = sorted(d.rglob(pat))
        if hits: return hits[0]
    return None

def process(rid):
    ptid, t1 = resolve_ptid_and_t1(rid)
    if t1 is None:
        return dict(RID=rid, PTID=None, status="missing_t1")
    pet = find_pet(ptid)
    if pet is None:
        return dict(RID=rid, PTID=ptid, status="missing_pet")
        
    seg = OUT_DIR / f"{ptid}_synthseg_parc.nii.gz"
    if not seg.exists():
        r = subprocess.run(["mri_synthseg", "--i", str(t1), "--o", str(seg), "--parc", "--robust", "--cpu"],
                           capture_output=True, text=True)
        if r.returncode != 0 or not seg.exists():
            return dict(RID=rid, PTID=ptid, status="synthseg_fail")
            
    # Load and register volumes
    t1_img = ants.image_read(str(t1))
    pet_img = ants.image_read(str(pet))
    reg = ants.registration(fixed=t1_img, moving=pet_img, type_of_transform="Rigid")
    
    seg_img = ants.image_read(str(seg))
    seg_data = nib.load(str(seg)).get_fdata()
    
    pet_on_seg = ants.resample_image_to_target(reg["warpedmovout"], seg_img, interp_type="linear").numpy()
    
    suvr, tmean, rmean, nt, nr = extract_suvr(pet_on_seg, seg_data)
    
    # Explicit memory cleanup
    del t1_img, pet_img, reg, seg_img, seg_data, pet_on_seg
    gc.collect()
    
    return dict(RID=rid, PTID=ptid, native_suvr=round(suvr, 4) if np.isfinite(suvr) else np.nan,
                target_mean=round(tmean, 4), ref_mean=round(rmean, 4),
                meta_vox=nt, ref_vox=nr, status="ok" if np.isfinite(suvr) else "empty_mask")

def main():
    coh = pd.read_csv(COHORT).drop_duplicates("RID")
    done_rids = set()
    if RESULTS.exists() and RESULTS.stat().st_size > 0:
        res_df = pd.read_csv(RESULTS)
        if "RID" in res_df.columns:
            done_rids = set(res_df["RID"].dropna().astype(int))

    todo = coh[~coh["RID"].astype(int).isin(done_rids)]
    print(f"{len(coh)} total subjects | {len(done_rids)} completed | {len(todo)} remaining to process")
    
    for i, (_, r) in enumerate(todo.iterrows(), 1):
        rid = int(r.RID)
        res = process(rid)
        pd.DataFrame([res]).to_csv(RESULTS, mode="a", header=not (RESULTS.exists() and RESULTS.stat().st_size > 0), index=False)
        print(f"  [{i}/{len(todo)}] RID {rid} ({res.get('PTID','-')}): {res.get('native_suvr','-')} [{res['status']}]", flush=True)

    # ---- Cohort Validation Summary ----
    df = pd.read_csv(RESULTS).merge(coh, on="RID", how="left")
    ok = df[df.status == "ok"].dropna(subset=["native_suvr", "ucb_suvr"])
    print(f"\n=== COHORT VALIDATION (n={len(ok)}) ===")
    if len(ok) > 3:
        pr = stats.pearsonr(ok.native_suvr, ok.ucb_suvr)
        sr = stats.spearmanr(ok.native_suvr, ok.ucb_suvr)
        mean_abs_diff = (100 * (ok.native_suvr - ok.ucb_suvr).abs() / ok.ucb_suvr).mean()
        print(f"Correlation vs UC Berkeley SUVR:")
        print(f"  Pearson r  = {pr[0]:.3f} (p = {pr[1]:.1e})")
        print(f"  Spearman r = {sr[0]:.3f} (p = {sr[1]:.1e})")
        print(f"  Mean absolute diff: {mean_abs_diff:.2f}%")
        
    dx_col = "DX_Group" if "DX_Group" in ok.columns else "DX" if "DX" in ok.columns else None
    if dx_col:
        print(f"\nNative SUVR by Diagnostic Group ({dx_col}):")
        for g in sorted(ok[dx_col].dropna().unique()):
            s = ok[ok[dx_col] == g].native_suvr
            if len(s): print(f"  {g:>10}: Median = {s.median():.3f} | Mean = {s.mean():.3f} | n = {len(s)}")

if __name__ == "__main__":
    main()
