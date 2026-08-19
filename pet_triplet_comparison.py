#!/usr/bin/env python3
import os
import glob
import nibabel as nib
import matplotlib.pyplot as plt

CLEAN_FDG_DIR = "./data/phase4_clean_fdg"
LOW_DOSE_DIR = "./data/low_dose_fdg"
DENOISED_DIR = "./data/denoised_fdg"
OUTPUT_IMG = "./data/pet_triplet_comparison.png"

def main():
    fd_files = sorted(glob.glob(os.path.join(CLEAN_FDG_DIR, "*.nii.gz")))
    if not fd_files:
        print("No files found in clean directory.")
        return

    # Select first available subject scan
    fd_path = fd_files[0]
    filename = os.path.basename(fd_path)
    subject_id = filename.replace(".nii.gz", "")

    ld_path = os.path.join(LOW_DOSE_DIR, filename)
    dn_path = os.path.join(DENOISED_DIR, filename)

    # Load Full-Dose image to get dimensions
    fd_data = nib.load(fd_path).get_fdata()
    slice_z = fd_data.shape[2] // 2  # Axial mid-slice

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300)

    # Helper function to plot slice
    def plot_slice(ax, path, title):
        if os.path.exists(path):
            data = nib.load(path).get_fdata()
            im = ax.imshow(data[:, :, slice_z].T, cmap='inferno', origin='lower')
            ax.set_title(title, fontsize=11, fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, 'Image Missing', ha='center', va='center')
            ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')

    plot_slice(axes[0], fd_path, f"Full-Dose ({subject_id})")
    plot_slice(axes[1], ld_path, "Low-Dose (Simulated)")
    plot_slice(axes[2], dn_path, "Denoised Output")

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, bbox_inches='tight')
    print(f"[SUCCESS] Saved PET slice comparison to {OUTPUT_IMG}")

if __name__ == "__main__":
    main()