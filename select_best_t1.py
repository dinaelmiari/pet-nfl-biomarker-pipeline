import shutil
import nibabel as nib
from pathlib import Path

nifti_dir = Path("/home/delmiari/scratch/project/data/nifti_t1/")
target_t1_dir = Path("/home/delmiari/scratch/project/data/selected_t1/")
target_t1_dir.mkdir(parents=True, exist_ok=True)

selected_count = 0

for sub_dir in sorted([d for d in nifti_dir.iterdir() if d.is_dir()]):
    sub_id = sub_dir.name
    nii_files = list(sub_dir.glob("*.nii.gz")) + list(sub_dir.glob("*.nii"))
    
    candidates = []
    for nii_path in nii_files:
        name = nii_path.name.lower()
        # Ignore non-3D / low-res derivative files if possible
        if any(x in name for x in ["localizer", "loc", "std_img_and_vox"]):
            continue
            
        try:
            img = nib.load(str(nii_path))
            shape = img.shape
            if len(shape) == 3 and min(shape) > 100:
                # Rank candidates: prioritize non-repeat MPRAGE -> repeat MPRAGE -> any 3D
                is_mprage = "mprage" in name
                is_repeat = "repeat" in name
                rank = (is_mprage and not is_repeat, is_mprage, shape[0] * shape[1] * shape[2])
                candidates.append((rank, nii_path, shape))
        except Exception:
            continue
            
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_nii = candidates[0][1]
        
        # Copy to clean standardized path: selected_t1/057_S_1269_T1.nii.gz
        out_file = target_t1_dir / f"{sub_id}_T1.nii.gz"
        shutil.copy2(best_nii, out_file)
        
        selected_count += 1
        print(f"[{selected_count}/87] {sub_id} -> {best_nii.name}")
    else:
        print(f"[{sub_id}] WARNING: No candidate found!")

print(f"\nSuccessfully created primary T1 dataset for {selected_count}/87 subjects in {target_t1_dir}")
