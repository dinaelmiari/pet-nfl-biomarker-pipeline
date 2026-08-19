"""
Phase 4 - batch SUVR over all FDG scans.

- builds + saves the meta-ROI and pons masks ONCE (template space), then reuses them
- registers each scan to MNI in parallel workers, applies the shared masks
- writes one row per scan incrementally (crash-safe); re-running SKIPS done scans
- records QC columns so bad registrations are easy to spot and drop later

Usage:
  python phase4_batch_suvr.py --nifti-dir /scratch/delmiari/project/data/nifti_fdg \
      --out phase4_suvr.csv --workers 8 [--cohort phase4_final_cohort.csv] [--limit N]

Test the bookkeeping only (no ANTs needed):
  python phase4_batch_suvr.py --selftest
"""
import os, re, csv, glob, sys, argparse, datetime

META_MASK = "mask_metaroi.nii.gz"
PONS_MASK = "mask_pons.nii.gz"
FIELDS = ["nifti_file", "exam_date", "suvr", "meta_mean", "pons_mean",
          "meta_vox", "pons_vox", "status"]

METAROI_LABELS = ["Angular Gyrus", "Cingulate Gyrus, posterior division",
                  "Precuneous Cortex", "Inferior Temporal Gyrus, posterior division",
                  "Inferior Temporal Gyrus, temporooccipital part",
                  "Middle Temporal Gyrus, posterior division"]
PONS_MNI = np.array([0, -26, -34, 1]) if 'np' in globals() else (0, -26, -34)
PONS_R_MM = 8.0

# ---------- lightweight helpers (no heavy deps) ----------
def date_from_name(fn):
    m = re.search(r"(\d{8})\d{6}", os.path.basename(fn))
    if not m: return ""
    s = m.group(1)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"

def load_done(out_csv):
    if not os.path.exists(out_csv): return set()
    with open(out_csv) as f:
        return {r["nifti_file"] for r in csv.DictReader(f)}

# ---------- mask building (needs nilearn/ants) ----------
def build_and_save_masks(mni_ants):
    import numpy as np, nibabel as nib, ants
    from nilearn import datasets

    ho = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr50-2mm")
    atlas = ho.maps if isinstance(ho.maps, nib.Nifti1Image) else nib.load(ho.maps)
    labels = ho.labels; arr = atlas.get_fdata(); aff = atlas.affine

    meta = np.zeros(arr.shape, np.uint8)
    for name in METAROI_LABELS:
        if name in labels: meta[arr == labels.index(name)] = 1

    inv = np.linalg.inv(aff)
    pons_mni_vec = np.array([0, -26, -34, 1])
    ijk = (inv @ pons_mni_vec)[:3]

    xx, yy, zz = np.ogrid[:arr.shape[0], :arr.shape[1], :arr.shape[2]]
    vmm = np.abs([aff[0, 0], aff[1, 1], aff[2, 2]])
    dist = np.sqrt(((xx - ijk[0]) * vmm[0])**2 + ((yy - ijk[1]) * vmm[1])**2 + ((zz - ijk[2]) * vmm[2])**2)
    pons = (dist <= PONS_R_MM).astype(np.uint8)

    def nifti_to_ants(nii):
        return ants.from_numpy(
            nii.get_fdata().astype(np.float32),
            origin=tuple(nii.affine[:3, 3]),
            spacing=tuple(np.abs(np.diag(nii.affine[:3, :3]))),
            direction=np.sign(nii.affine[:3, :3])
        )

    for a_img, path in [(nib.Nifti1Image(meta, aff), META_MASK),
                        (nib.Nifti1Image(pons, aff), PONS_MASK)]:
        r = ants.resample_image_to_target(nifti_to_ants(a_img), mni_ants,
                                          interp_type="genericLabel")
        ants.image_write(r, path)
    print(f"masks saved: {META_MASK}, {PONS_MASK}")

# ---------- worker ----------
_G = {}
def _init_worker():
    import ants
    _G["mni"]  = ants.image_read(ants.get_ants_data("mni"))
    _G["meta"] = ants.image_read(META_MASK).numpy() > 0.5
    _G["pons"] = ants.image_read(PONS_MASK).numpy() > 0.5

def process_one(path):
    import ants, numpy as np
    row = {k: "" for k in FIELDS}
    row["nifti_file"] = os.path.basename(path)
    row["exam_date"] = date_from_name(path)
    try:
        fdg = ants.image_read(path)
        warped = ants.registration(fixed=_G["mni"], moving=fdg,
                                   type_of_transform="Affine")["warpedmovout"].numpy()
        mv = float(warped[_G["meta"]].mean()); pv = float(warped[_G["pons"]].mean())
        suvr = mv / pv if pv > 0 else float("nan")
        row.update(suvr=round(suvr, 4), meta_mean=round(mv, 4), pons_mean=round(pv, 4),
                   meta_vox=int(_G["meta"].sum()), pons_vox=int(_G["pons"].sum()))
        # QC flag: implausible SUVR or dead reference => needs review
        row["status"] = "ok" if (pv > 0.02 and 0.8 < suvr < 3.5) else "check"
    except Exception as e:
        row["status"] = f"ERROR:{type(e).__name__}"
    return row

# ---------- driver ----------
def run(args):
    import ants
    from concurrent.futures import ProcessPoolExecutor, as_completed
    if not (os.path.exists(META_MASK) and os.path.exists(PONS_MASK)):
        print("building masks (once)...")
        build_and_save_masks(ants.image_read(ants.get_ants_data("mni")))

    files = sorted(glob.glob(os.path.join(args.nifti_dir, "**", "*.nii*"), recursive=True))
    done = load_done(args.out)
    todo = [f for f in files if os.path.basename(f) not in done]
    if args.limit: todo = todo[:args.limit]
    print(f"{len(files)} scans found | {len(done)} already done | {len(todo)} to process")

    new_file = not os.path.exists(args.out)
    fh = open(args.out, "a", newline=""); w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file: w.writeheader(); fh.flush()

    n_ok = n_chk = n_err = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(process_one, f): f for f in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result(); w.writerow(row); fh.flush()
            st = row["status"]
            n_ok += st == "ok"; n_chk += st == "check"; n_err += st.startswith("ERROR")
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} | ok={n_ok} check={n_chk} err={n_err}", flush=True)
    fh.close()
    print(f"\ndone. ok={n_ok} check={n_chk} err={n_err}. wrote {args.out}")
    print("next: inspect the SUVR distribution and the 'check'/'ERROR' rows.")

def selftest():
    assert date_from_name("ADNI_..._Uniform_6mm_Res_20050929000000_5.nii.gz") == "2005-09-29"
    assert date_from_name("no_date.nii.gz") == ""
    print("date parsing ok:", date_from_name("x_20090925000000_5.nii.gz"))
    open("_t.csv", "w").write("nifti_file,suvr\na.nii.gz,1.5\n")
    assert load_done("_t.csv") == {"a.nii.gz"}; os.remove("_t.csv")
    print("resume/skip bookkeeping ok")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nifti-dir"); p.add_argument("--out", default="phase4_suvr.csv")
    p.add_argument("--workers", type=int, default=8); p.add_argument("--limit", type=int)
    p.add_argument("--cohort"); p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest: selftest()
    else: run(a)