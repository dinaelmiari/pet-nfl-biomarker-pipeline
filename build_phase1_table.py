"""
Phase 1 - build the analysis table (RDA version).

Attaches nearest NfL draw within 183 days to each FDG scan,
then merges baseline covariates.
"""
import sys
import numpy as np
import pandas as pd
import pyreadr

# ============================ CONFIG ============================
DATA_DIR  = "/scratch/delmiari/project/data/ADNIMERGE2/data/"
FDG_RDA   = DATA_DIR + "UCBERKELEYFDG_8mm.rda"
NFL_RDA   = DATA_DIR + "BLENNOWPLASMANFL.rda"
COVAR_CSV = "data/baseline_covariates.csv"

FDG_ID, FDG_DATE = "RID", "EXAMDATE"
FDG_VALUE        = "MEAN"      # Mean SUVR
FDG_ROI_FILTER   = "MetaROI"   # Primary AD-signature ROI in UC Berkeley table

NFL_ID, NFL_DATE = "RID", "EXAMDATE"
NFL_VALUE        = "PLASMA_NFL"
COVAR_ID         = "RID"

WINDOW_DAYS = 183
# ================================================================


def load_rda(path):
    r = pyreadr.read_r(path)
    return r[list(r.keys())[0]]

def std(df, id_col, date_col):
    for c in (id_col, date_col):
        if c not in df.columns:
            raise KeyError(f"'{c}' not in {list(df.columns)}")
    out = df.copy()
    out["rid"]  = out[id_col].astype(float).astype("Int64").astype(str)
    out["date"] = pd.to_datetime(out[date_col], errors="coerce")
    return out.dropna(subset=["rid", "date"])

def nearest_nfl(fdg, nfl, value_col, window):
    """Attach nearest NfL value (and gap in days) to each FDG scan."""
    fdg_sorted = fdg.sort_values("date")
    nfl_sorted = nfl.sort_values("date")

    merged = pd.merge_asof(
        fdg_sorted,
        nfl_sorted[["rid", "date", value_col]].rename(
            columns={"date": "nfl_date", value_col: "nfl_value"}
        ),
        left_on="date",
        right_on="nfl_date",
        by="rid",
        direction="nearest",
        tolerance=pd.Timedelta(days=window)
    )

    merged["nfl_gap_days"] = (merged["date"] - merged["nfl_date"]).dt.days.abs()
    return merged


def main():
    print("COHORT FLOW\n" + "=" * 50)
    
    # Load raw data
    raw_fdg = load_rda(FDG_RDA)
    raw_nfl = load_rda(NFL_RDA)
    cov     = pd.read_csv(COVAR_CSV)
    
    # Filter FDG for target MetaROI if ROINAME column exists
    if "ROINAME" in raw_fdg.columns:
        rois = raw_fdg["ROINAME"].unique()
        target_roi = "MetaROI" if "MetaROI" in rois else rois[0]
        print(f"Filtering FDG PET for ROI: '{target_roi}'")
        raw_fdg = raw_fdg[raw_fdg["ROINAME"] == target_roi]

    fdg = std(raw_fdg, FDG_ID, FDG_DATE)
    nfl = std(raw_nfl, NFL_ID, NFL_DATE)
    cov["rid"] = cov[COVAR_ID].astype(float).astype("Int64").astype(str)

    print(f"FDG scans ({FDG_ROI_FILTER}):  {len(fdg):>6}  ({fdg['rid'].nunique()} subjects)")
    print(f"NfL draws:            {len(nfl):>6}  ({nfl['rid'].nunique()} subjects)")
    print(f"Covariate rows:       {len(cov):>6}")

    # 1. Attach nearest NfL to each FDG scan
    m = nearest_nfl(fdg, nfl, NFL_VALUE, WINDOW_DAYS)
    m = m.dropna(subset=["nfl_value"])
    print(f"FDG scans w/ NfL<={WINDOW_DAYS}d: {len(m):>6}  ({m['rid'].nunique()} subjects)")

    # 2. One row per subject: earliest matched FDG scan
    m = m.sort_values("date").groupby("rid", as_index=False).first()
    print(f"One row per subject:  {len(m):>6}")

    # 3. Merge covariates
    keep = ["rid", "DIAGNOSIS", "PTGENDER", "APOE4_CARRIER", "MMSCORE", "CDGLOBAL", "CDRSB"]
    keep = [c for c in keep if c in cov.columns]
    out = m.merge(cov[keep], on="rid", how="left")
    print(f"Merged w/ covariates: {len(out):>6}  ({out['DIAGNOSIS'].notna().sum()} have diagnosis)")

    # 4. Save final table
    final = out.rename(columns={FDG_VALUE: "fdg_suvr"})[
        ["rid", "date", "fdg_suvr", "nfl_value", "nfl_gap_days",
         *[c for c in keep if c != "rid"]]
    ]
    final.to_csv("phase1_analysis_table.csv", index=False)
    print("\nSaved 'phase1_analysis_table.csv' with shape:", final.shape)
    print("\nBy Diagnosis:")
    print(final["DIAGNOSIS"].value_counts(dropna=False).to_string())
    print(f"\nNfL-PET Gap (days): median {final['nfl_gap_days'].median():.0f}, "
          f"max {final['nfl_gap_days'].max():.0f}")


if __name__ == "__main__":
    main()
