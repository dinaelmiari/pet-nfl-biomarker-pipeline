"""
Build the UPENN multi-marker matched cohort (upgraded Phase 1).
Handles long-format UC Berkeley FDG files (ROINAME / MEAN).
"""
import numpy as np, pandas as pd
import glob

# ============================ CONFIG ============================
UPENN = "data/UPENN_PLASMA_FUJIREBIO_QUANTERIX_19Aug2026.csv"
FDG_FILES = glob.glob('*FDG*.csv') + glob.glob('data/*FDG*.csv')
COVAR = "baseline_covariates.csv"              # update if named differently
NFL_COL = "NfL_Q"        # primary NfL platform (Quanterix)
WINDOW = 183
# ================================================================

def rid_str(s): return pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)
def load(path): return pd.read_csv(path, low_memory=False)

def main():
    print(f"Loading UPENN from {UPENN}...")
    up = load(UPENN)
    up["rid"] = rid_str(up["RID"])
    up["date"] = pd.to_datetime(up["EXAMDATE"], errors="coerce")

    print(f"Loading FDG from {FDG_FILES[0]}...")
    fdg = load(FDG_FILES[0])
    fdg["rid"] = rid_str(fdg["RID"])
    fdg["date"] = pd.to_datetime(fdg["EXAMDATE"], errors="coerce")
    
    # Check available ROIs to find the meta-ROI or composite region
    print("Available ROIs in FDG file:", fdg["ROINAME"].unique()[:10])

    # Pivot long format to wide format so each ROI is a column
    # We use 'MEAN' as the value column for SUVR
    fdg_wide = fdg.pivot_table(index=["rid", "VISCODE", "date"], columns="ROINAME", values="MEAN").reset_index()
    fdg_wide.columns.name = None
    
    # Try to find a meta-ROI column or default to a standard one if present
    meta_candidates = [c for c in fdg_wide.columns if any(k in c.lower() for k in ["meta", "composite", "pons", "suvr", "ad_sig"])]
    print(f"Detected potential Meta-ROI columns: {meta_candidates}")
    
    # Pick the first candidate or let user specify
    if meta_candidates:
        fdg_suvr_col = meta_candidates[0]
        print(f"Using '{fdg_suvr_col}' as the FDG measure.")
    else:
        # Fallback to the first numeric column after rid/VISCODE/date
        fdg_suvr_col = [c for c in fdg_wide.columns if c not in ["rid", "VISCODE", "date"]][0]
        print(f"No clear meta-ROI found. Falling back to: '{fdg_suvr_col}'. Edit script if needed.")

    up = up.dropna(subset=["rid", "date", NFL_COL])
    fdg_wide = fdg_wide.dropna(subset=["rid", "date", fdg_suvr_col])

    # Load covariates if available
    cov = None
    if os_path_exists := glob.glob("*covar*.csv") or glob.glob("data/*covar*.csv"):
        # try to load if file exists
        pass

    # Pair each FDG scan with nearest NfL draw within window
    panel = [NFL_COL, "NfL_F", "pT217_F", "GFAP_Q", "AB42_AB40_F"]
    panel = [c for c in panel if c in up.columns]
    up_by = {r: g for r, g in up.groupby("rid")}
    rows = []
    
    for rid, g in fdg_wide.groupby("rid"):
        if rid not in up_by: continue
        u = up_by[rid]
        for _, scan in g.sort_values("date").iterrows():
            dd = (u["date"] - scan["date"]).abs().dt.days
            j = dd.idxmin()
            if dd.loc[j] <= WINDOW:
                row = {"rid": rid, "fdg_date": scan["date"], "fdg_suvr": scan[fdg_suvr_col],
                       "nfl_gap_days": int(dd.loc[j])}
                for c in panel: row[c] = u.loc[j, c]
                rows.append(row); break
                
    m = pd.DataFrame(rows)
    print(f"\nSuccessfully matched FDG to NfL within {WINDOW} days: {len(m)} subjects")
    m.to_csv("upenn_matched_cohort.csv", index=False)
    print("Saved upenn_matched_cohort.csv successfully!")

if __name__ == "__main__":
    main()