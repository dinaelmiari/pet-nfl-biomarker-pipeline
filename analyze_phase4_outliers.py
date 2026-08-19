import pandas as pd, numpy as np
from scipy import stats
from pathlib import Path

COHORT  = Path("/scratch/delmiari/project/native_nfl_overlap_cohort.csv")
RESULTS = Path("/home/delmiari/scratch/project/data/synthseg_output/native_suvr_37.csv")

coh = pd.read_csv(COHORT).drop_duplicates("RID")
res = pd.read_csv(RESULTS)
df  = res.merge(coh, on="RID", how="left")

ok = df[df.status == "ok"].copy()
ok["suvr_diff"] = ok["native_suvr"] - ok["ucb_suvr"]
ok["abs_pct_err"] = 100 * ok["suvr_diff"].abs() / ok["ucb_suvr"]

print("=== SUBJECT-LEVEL METRICS (Sorted by Absolute Error) ===")
cols = ["RID", "PTID", "DX_Group", "native_suvr", "ucb_suvr", "target_mean", "ref_mean", "abs_pct_err"]
avail_cols = [c for c in cols if c in ok.columns]
print(ok[avail_cols].sort_values("abs_pct_err", ascending=False).to_string(index=False))

print("\n=== REFERENCE REGION ANALYSIS BY GROUP ===")
if "DX_Group" in ok.columns:
    grp = ok.groupby("DX_Group")[["target_mean", "ref_mean", "native_suvr", "ucb_suvr"]].agg(["mean", "std"])
    print(grp.to_string())

# Rank discordance check
ok["rank_native"] = ok["native_suvr"].rank()
ok["rank_ucb"]    = ok["ucb_suvr"].rank()
ok["rank_diff"]   = (ok["rank_native"] - ok["rank_ucb"]).abs()

print("\n=== TOP RANK DISCORDANCES (Spearman Drivers) ===")
print(ok[["RID", "PTID", "DX_Group", "native_suvr", "ucb_suvr", "rank_native", "rank_ucb", "rank_diff"]]
      .sort_values("rank_diff", ascending=False).head(5).to_string(index=False))
