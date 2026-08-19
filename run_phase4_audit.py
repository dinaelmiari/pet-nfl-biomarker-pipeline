"""
PHASE 4 CORE - the denoising fidelity audit.

For each subject, on ONE saved registration (Test B2 design = 0% jitter):
  full-dose PET --> SUVR_full   (within-pipeline reference truth)
  low-dose (Poisson) --> SUVR_low
  denoised (Gaussian baseline) --> SUVR_denoised
Then: does denoising preserve SUVR (dSUVR ~ 0) or bias it systematically?

Key design (validated): register full PET->T1 ONCE, reuse that transform for all
three conditions, extract with the SAME fixed SynthSeg mask. Bias cancels; the
shift is real denoising effect.
"""
import sys, numpy as np, pandas as pd, nibabel as nib
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy import stats
import ants

BASE = Path("/scratch/delmiari/project")
COHORT = BASE / "native_nfl_overlap_cohort.csv"
T1_DIR = BASE / "data/selected_t1"
PET_DIR = BASE / "data/nifti_fdg"
SEG_DIR = BASE / "data/synthseg_output"
OUT = SEG_DIR / "phase4_dsuvr.csv"

METAROI_LABELS = [1023, 2023, 1025, 2025, 1008, 2008, 1009, 2009, 1015, 2015]
REF_LABELS = [16, 7, 46]
DOSE = 0.25
SIGMA = 1.0

# ---------- validated extraction recipe (50th-pctl reference) ----------
def extract_suvr(pet_data, seg_data):
    tmask = np.isin(seg_data, METAROI_LABELS)
    rmask = np.isin(seg_data, REF_LABELS)
    if not tmask.any() or not rmask.any():
        return np.nan
    tmean = pet_data[tmask].mean()
    refv = pet_data[rmask]; cutoff = np.percentile(refv, 50)
    rmean = refv[refv >= cutoff].mean()
    return float(tmean / rmean) if rmean > 0 else np.nan

def simulate_low_dose(img, dose=DOSE, pctl=99, ref_counts=300.0):
    pos = img[img > 0]
    K = ref_counts / np.percentile(pos, pctl)
    low = np.random.poisson(np.maximum(img, 0) * K * dose)
    return low / (K * dose)

def icc_3_1(x, y):
    d = np.column_stack([x, y]); n, k = d.shape
    gm = d.mean(); rows = d.mean(1); cols = d.mean(0)
    msr = k * ((rows - gm)**2).sum() / (n - 1)
    sse = ((d - rows[:, None] - cols[None, :] + gm)**2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse)

# ---------- file resolution ----------
def resolve(rid):
    r = int(rid); pad = f"{r:04d}"
    def first(d, pats):
        for p in pats:
            h = list(d.glob(p))
            if h: return h[0]
        return None
    t1 = first(T1_DIR, [f"*{pad}*T1.nii*", f"*{r}*T1.nii*"])
    pet = first(PET_DIR, [f"*{pad}*.nii*", f"*{r}*.nii*"])
    seg = first(SEG_DIR, [f"*{pad}*synthseg*.nii*", f"*{r}*synthseg*.nii*"])
    return t1, pet, seg

def process(rid):
    t1, pet, seg = resolve(rid)
    if not (t1 and pet and seg):
        return None, "missing_files"
    t1_a = ants.image_read(str(t1)); pet_a = ants.image_read(str(pet)); seg_a = ants.image_read(str(seg))
    seg_np = seg_a.numpy()
    full_np = pet_a.numpy()
    low_a = pet_a.new_image_like(simulate_low_dose(full_np))
    den_a = pet_a.new_image_like(gaussian_filter(low_a.numpy(), sigma=SIGMA))

    reg = ants.registration(fixed=t1_a, moving=pet_a, type_of_transform="Rigid")
    tx = reg["fwdtransforms"]                      # ONE transform, reused for all three
    def warp_extract(mov):
        w = ants.apply_transforms(fixed=seg_a, moving=mov, transformlist=tx, interpolator="linear")
        return extract_suvr(w.numpy(), seg_np)
    s_full = warp_extract(pet_a); s_low = warp_extract(low_a); s_den = warp_extract(den_a)
    return dict(suvr_full=round(s_full,4), suvr_low=round(s_low,4), suvr_denoised=round(s_den,4)), "ok"

def main():
    coh = pd.read_csv(COHORT).drop_duplicates("RID")
    done = set(pd.read_csv(OUT).RID) if OUT.exists() else set()
    rows = []
    for i, (_, r) in enumerate(coh[~coh.RID.isin(done)].iterrows(), 1):
        try:
            res, st = process(r.RID)
        except Exception as e:
            res, st = None, f"error:{type(e).__name__}"
        row = dict(RID=int(r.RID), PTID=r.get("PTID",""),
                   DX=r.get("DX", r.get("DX_Group","")), status=st)
        if res: row.update(res)
        rows.append(row)
        pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        print(f"  {i} RID{r.RID}: full={row.get('suvr_full','-')} low={row.get('suvr_low','-')} "
              f"den={row.get('suvr_denoised','-')} [{st}]", flush=True)

    # ---------- the result ----------
    df = pd.read_csv(OUT)
    ok = df[df.status == "ok"].dropna(subset=["suvr_full","suvr_denoised"])
    print(f"\n=== PHASE 4 RESULT (n={len(ok)}) ===")
    d_low = ok.suvr_low - ok.suvr_full
    d_den = ok.suvr_denoised - ok.suvr_full
    pe_den = 100 * d_den.abs() / ok.suvr_full
    print(f"low-dose  vs full: mean dSUVR {d_low.mean():+.4f} (sd {d_low.std():.4f})")
    print(f"denoised  vs full: mean dSUVR {d_den.mean():+.4f} (sd {d_den.std():.4f}) | "
          f"mean |%err| {pe_den.mean():.2f}%")
    if len(ok) > 3:
        print(f"full vs denoised: Pearson r={stats.pearsonr(ok.suvr_full, ok.suvr_denoised)[0]:.3f} | "
              f"ICC(3,1)={icc_3_1(ok.suvr_full.values, ok.suvr_denoised.values):.3f}")
        w = stats.wilcoxon(ok.suvr_full, ok.suvr_denoised)
        print(f"systematic shift? Wilcoxon p={w.pvalue:.3f} "
              f"({'systematic bias' if w.pvalue<0.05 else 'no significant systematic bias'})")
    print("\nread: dSUVR near 0 + high ICC = denoising preserves the biomarker;")
    print("      systematic dSUVR = denoising distorts it (cleaner image, shifted number).")

if __name__ == "__main__":
    main()
