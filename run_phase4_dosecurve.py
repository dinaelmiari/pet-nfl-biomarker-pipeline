"""
PHASE 4 EXTENDED - denoising fidelity across DOSE x REGION x BIOLOGY.

Reuses the validated register-once design (0% jitter). For each subject:
  register full PET -> T1 ONCE, reuse that transform for every condition.
  For each dose in DOSES: low-dose (Poisson) -> denoise (Gaussian) -> extract SUVR
  for the composite meta-ROI AND each AD-signature sub-region.

Three questions, one run:
  1. DOSE curve   : how does |SUVR error| grow as dose drops? where's the breakpoint?
  2. REGION map   : which AD-signature region loses fidelity first?
  3. BIOLOGY      : does the SUVR-NfL association survive low-dose denoising?
"""
import numpy as np, pandas as pd, nibabel as nib
from pathlib import Path
from scipy.ndimage import gaussian_filter
import ants

BASE = Path("/scratch/delmiari/project")
COHORT = BASE / "native_nfl_overlap_cohort.csv"
T1_DIR, PET_DIR, SEG_DIR = BASE/"data/selected_t1", BASE/"data/nifti_fdg", BASE/"data/synthseg_output"
OUT = SEG_DIR / "phase4_dosecurve.csv"

DOSES = [0.50, 0.25, 0.125, 0.05]     # fractions of full dose
REPS  = 3                              # noise realizations per dose (averaged -> stabler curve)
SIGMA = 1.0
REF_LABELS = [16, 7, 46]
REGIONS = {                            # AD-signature sub-regions (L,R DK labels)
    "post_cingulate": [1023, 2023], "precuneus": [1025, 2025],
    "inf_parietal":   [1008, 2008], "inf_temporal": [1009, 2009],
    "mid_temporal":   [1015, 2015],
}
COMPOSITE = sum(REGIONS.values(), [])

def ref_mean(pet, seg):
    rv = pet[np.isin(seg, REF_LABELS)]
    return rv[rv >= np.percentile(rv, 50)].mean()

def suvr_all(pet, seg):
    r = ref_mean(seg=seg, pet=pet)
    out = {"composite": pet[np.isin(seg, COMPOSITE)].mean()/r}
    for name, lab in REGIONS.items():
        out[name] = pet[np.isin(seg, lab)].mean()/r
    return out

def low_dose(img, dose, pctl=99, rc=300.0):
    K = rc/np.percentile(img[img>0], pctl)
    return np.random.poisson(np.maximum(img,0)*K*dose)/(K*dose)

def resolve(rid):
    r=int(rid); pad=f"{r:04d}"
    g=lambda d,ps:next((h for p in ps for h in [next(iter(d.glob(p)),None)] if h),None)
    return (g(T1_DIR,[f"*{pad}*T1.nii*",f"*{r}*T1.nii*"]),
            g(PET_DIR,[f"*{pad}*.nii*",f"*{r}*.nii*"]),
            g(SEG_DIR,[f"*{pad}*synthseg*.nii*",f"*{r}*synthseg*.nii*"]))

def process(rid):
    t1,pet,seg = resolve(rid)
    if not (t1 and pet and seg): return [], "missing_files"
    t1a,peta,sega = ants.image_read(str(t1)),ants.image_read(str(pet)),ants.image_read(str(seg))
    segnp = sega.numpy(); fullnp = peta.numpy()
    tx = ants.registration(fixed=t1a, moving=peta, type_of_transform="Rigid")["fwdtransforms"]
    warp = lambda mov: ants.apply_transforms(fixed=sega, moving=mov, transformlist=tx,
                                             interpolator="linear").numpy()
    rows = []
    full_suvr = suvr_all(warp(peta), segnp)
    rows.append(dict(dose=1.0, rep=0, condition="full", **full_suvr))
    for dose in DOSES:
        for rep in range(REPS):
            lo = low_dose(fullnp, dose)
            dn = gaussian_filter(lo, sigma=SIGMA)
            rows.append(dict(dose=dose, rep=rep, condition="low",
                             **suvr_all(warp(peta.new_image_like(lo)), segnp)))
            rows.append(dict(dose=dose, rep=rep, condition="denoised",
                             **suvr_all(warp(peta.new_image_like(dn)), segnp)))
    return rows, "ok"

def main():
    coh = pd.read_csv(COHORT).drop_duplicates("RID")
    done = set(pd.read_csv(OUT).RID) if OUT.exists() else set()
    for i,(_,r) in enumerate(coh[~coh.RID.isin(done)].iterrows(),1):
        try: rows, st = process(r.RID)
        except Exception as e: rows, st = [], f"error:{type(e).__name__}"
        base = dict(RID=int(r.RID), PTID=r.get("PTID",""),
                    DX=r.get("DX", r.get("DX_Group","")), PLASMA_NFL=r.get("PLASMA_NFL", np.nan))
        recs = [dict(**base, status=st, **row) for row in rows] or [dict(**base, status=st)]
        pd.DataFrame(recs).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        print(f"  {i} RID{r.RID}: {st} ({len(rows)} rows)", flush=True)
    print(f"\ndone -> {OUT}\nnext: python plot_phase4_dosecurve.py")

if __name__ == "__main__":
    main()
