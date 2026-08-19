import os
import re
import glob
import pandas as pd
import nibabel as nib

NIFTI_DIR = "/scratch/delmiari/project/data/nifti_fdg"
OUTPUT_TABLE = "/scratch/delmiari/project/phase4_matched_images.csv"

nii_files = glob.glob(os.path.join(NIFTI_DIR, "*.nii*"))
print(f"Found {len(nii_files)} NIfTI files.")

records = []
for fpath in nii_files:
    fname = os.path.basename(fpath)
    
    # Extract timestamp pattern (e.g. 20090925)
    date_match = re.search(r'_(\d{8})\d*_', fname)
    
    if date_match:
        raw_date = date_match.group(1)
        exam_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
    else:
        exam_date = None

    try:
        img = nib.load(fpath)
        shape = img.shape
        zooms = img.header.get_zooms()
        
        records.append({
            "nifti_path": fpath,
            "filename": fname,
            "EXAMDATE": exam_date,
            "matrix_shape": str(shape),
            "voxel_size_mm": str([round(z, 2) for z in zooms[:3]])
        })
    except Exception as e:
        print(f"Error loading {fname}: {e}")

df_nifti = pd.DataFrame(records)
df_nifti.to_csv(OUTPUT_TABLE, index=False)
print(f"\nSaved inventory with EXAMDATE ({len(df_nifti)} records) to {OUTPUT_TABLE}")
print(df_nifti[['filename', 'EXAMDATE']].head())