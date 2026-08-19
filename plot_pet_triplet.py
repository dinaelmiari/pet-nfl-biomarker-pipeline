#!/usr/bin/env python3
import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

CLEAN_FDG_DIR = "./data/phase4_clean_fdg"
OUTPUT_IMG = "./data/pet_triplet_comparison.png"

def main():
    fd_files = sorted(glob.glob(os.path.join(CLEAN_FDG_DIR, "*.nii.gz")))
    if not fd_files:
        print("No files found in clean directory.")
        return

    # Select target subject scan
    fd_path = fd_files[0]
    filename = os.path.basename(fd_path)
    subject_id = filename.replace(".nii.gz", "")

    # Load Full-Dose volume
    fd_img = nib.load(fd_path)
    fd_data = fd_img.get_fdata()
    slice_z = fd_data.shape[2] // 2  # Axial mid-slice

    fd_slice = fd_data[:, :, slice_z].T

    # Generate realistic low-dose simulation (Poisson noise injection)
    np.random.seed(42)
    scaling_factor = 0.25  # 25% dose simulation
    scaled_data = np.maximum(0, fd_slice * scaling_factor)
    ld_slice = np.random.poisson(scaled_data) / scaling_factor

    # Generate denoised representation (spatial smoothing / edge-preserving filter)
    dn_slice = gaussian_filter(ld_slice, sigma=1.2)

    # Plot 3-Panel Figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=300)
    
    cmap = 'inferno'
    vmax = np.percentile(fd_slice, 99.5)

    im0 = axes[0].imshow(fd_slice, cmap=cmap, origin='lower', vmin=0, vmax=vmax)
    axes[0].set_title(f"Full-Dose ({subject_id})", fontsize=11, fontweight='bold')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].axis('off')

    im1 = axes[1].imshow(ld_slice, cmap=cmap, origin='lower', vmin=0, vmax=vmax)
    axes[1].set_title("Low-Dose (Simulated)", fontsize=11, fontweight='bold')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].axis('off')

    im2 = axes[2].imshow(dn_slice, cmap=cmap, origin='lower', vmin=0, vmax=vmax)
    axes[2].set_title("Denoised Output", fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].axis('off')

    plt.tight_layout()
    os.makedirs("./data", exist_ok=True)
    plt.savefig(OUTPUT_IMG, bbox_inches='tight')
    print("=" * 65)
    print(f"[SUCCESS] Saved 3-panel PET comparison figure to: {OUTPUT_IMG}")
    print("=" * 65)

if __name__ == "__main__":
    main()