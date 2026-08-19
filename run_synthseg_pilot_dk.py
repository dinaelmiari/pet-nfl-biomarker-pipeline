"""
Phase 4 pilot - Native SUVR via SynthSeg with Combined Pons + Cerebellar WM Ref.
"""
import numpy as np, pandas as pd, nibabel as nib
from pathlib import Path

SUB, RID = "052_S_1346", 1346
COHORT   = Path("/scratch/delmiari/project/native_nfl_overlap_cohort.csv")
WORK     = Path("/home/delmiari/scratch/project/data/pilot_output")
SEG      = WORK / f"{SUB}_synthseg_parc.nii.gz"
PET2T1   = WORK / f"{SUB}_pet_to_t1.nii.gz"

METAROI_LABELS = [1023, 2023, 1025, 2025, 1008, 2008, 1009, 2009, 1015, 2015]
# 16 = Brainstem, 7 = Left-Cerebellum-White-Matter, 46 = Right-Cerebellum-White-Matter
REF_LABELS = [16, 7, 46]

pet = nib.load(str(PET2T1)).get_fdata()
seg = nib.load(str(SEG)).get_fdata()

tmask = np.isin(seg, METAROI_LABELS)
rmask = np.isin(seg, REF_LABELS)

tmean = float(pet[tmask].mean())

ref_voxels = pet[rmask]
cutoff = np.percentile(ref_voxels, 50)
rmean_top50 = float(ref_voxels[ref_voxels >= cutoff].mean())

suvr = tmean / rmean_top50

print("--- Composite Ref (Pons + Cerebellar WM) Results ---")
print(f"Target Meta-ROI Mean:         {tmean:.4f}  (Benchmark MetaROI: 1.2990)")
print(f"Composite Ref Mean (Top50%):  {rmean_top50:.4f}  (Benchmark Top50PonsVermis: 1.1810)")
print(f"\n>> Composite Native SUVR:     {suvr:.4f}  (Benchmark ucb_suvr: 1.0999)")

df = pd.read_csv(COHORT); row = df[df.RID == RID].iloc[0]
bench = float(row["ucb_suvr"])
pct = 100 * abs(suvr - bench) / bench
print(f"Difference vs benchmark: {pct:.2f}%")
