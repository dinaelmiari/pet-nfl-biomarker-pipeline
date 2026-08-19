"""
ADNI — STEP ONE: how much do NfL and each PET tracer overlap?

This does NOT model anything yet. It answers the single question that decides
our starting point: for each PET tracer, how many people have both a PET scan
and an NfL blood draw close in time? Whichever tracer overlaps best with NfL is
the one we build the first analysis on.

THREE MODES
  python adni_step1.py --peek       # load each file, print its columns (run this FIRST)
  python adni_step1.py --selftest   # test the matching logic on tiny fake data (no real data)
  python adni_step1.py              # run the real overlap counts using CONFIG below

WORKFLOW
  1. Set data path to /scratch/delmiari/project/data/
  2. Run --peek to verify column names for all files.
  3. Run --selftest to confirm logic.
  4. Run with no arguments to get the overlap table + recommended tracer.
"""

import sys
import os
import pandas as pd
import numpy as np

# ============================ CONFIG ============================
# Path on Hinton cluster
DATA = "/scratch/delmiari/project/data/"
WINDOW_DAYS = 183   # "close in time" = within ~6 months

FILES = {
    "nfl": {
        "path": os.path.join(DATA, "ADNI_BLENNOWPLASMANFLLONG_10_03_18_10Aug2026.csv"),
        "id":   "RID",
        "date": "DRAW_DATE",     # CHANGED: 'EXAMDATE' -> 'DRAW_DATE'
        "value":"PLASMA_NFL",
    },
    "amyloid": {
        "path": os.path.join(DATA, "UCBERKELEY_AMY_6MM_14Aug2026.csv"),
        "id":   "RID",
        "date": "SCANDATE",
        "value":"SUMMARY_SUVR",  # ADDED: value column
    },
    "fdg": {
        "path": os.path.join(DATA, "UCBERKELEYFDG_8mm_02_17_23_14Aug2026.csv"),
        "id":   "RID",
        "date": "EXAMDATE",      # CHANGED: 'SCANDATE' -> 'EXAMDATE' (fdg has EXAMDATE)
        "value":"MEAN",          # ADDED: value column
    },
    "tau": {
        "path": os.path.join(DATA, "UCBERKELEY_TAU_6MM_14Aug2026.csv"),
        "id":   "RID",
        "date": "SCANDATE",
        "value":"META_TEMPORAL_SUVR", # ADDED: value column
    },
}
# =========================================================================


def load(path):
    return pd.read_csv(path, low_memory=False)


def std(df, id_col, date_col):
    """Return a copy with clean 'rid' (str) and 'date' (datetime) columns."""
    for col in (id_col, date_col):
        if col not in df.columns:
            raise KeyError(
                f"\n  Column '{col}' not found.\n"
                f"  Available columns: {list(df.columns)}\n"
                f"  -> fix this in CONFIG (run --peek to see the right names)."
            )
    out = df.copy()
    out["rid"] = out[id_col].astype(str).str.strip()
    out["date"] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=["rid", "date"])
    return out


def overlap(pet, nfl, window_days):
    """Count PET scans (and unique subjects) that have an NfL draw within window."""
    nfl_by = {rid: g["date"].values for rid, g in nfl.groupby("rid")}
    matched_scans = 0
    subj_pet = set(pet["rid"].unique())
    subj_both = set()
    for rid, g in pet.groupby("rid"):
        if rid not in nfl_by:
            continue
        ndates = nfl_by[rid]
        for d in g["date"].values:
            diff_days = np.abs((ndates - d) / np.timedelta64(1, "D"))
            if diff_days.min() <= window_days:
                matched_scans += 1
                subj_both.add(rid)
    return {
        "pet_scans": len(pet),
        "scans_with_nfl": matched_scans,
        "subjects_pet": len(subj_pet),
        "subjects_both": len(subj_both),
    }


# ------------------------------ modes ------------------------------

def peek():
    print("PEEK — columns in each file (fill CONFIG from these)\n" + "=" * 60)
    hint = ("RID", "PTID", "ID", "DATE", "EXAMDATE", "SCANDATE", "VISCODE", "NFL", "NEFL")
    for name, cfg in FILES.items():
        try:
            df = load(cfg["path"])
        except FileNotFoundError:
            print(f"\n[{name}] FILE NOT FOUND: {cfg['path']}")
            continue
        print(f"\n[{name}]  {cfg['path']}  shape={df.shape}")
        likely = [c for c in df.columns if any(h in c.upper() for h in hint)]
        print("  likely id/date/value cols:", likely)
        print("  all columns:", list(df.columns))


def selftest():
    print("SELF-TEST on tiny fake data (no real records)...")
    pet = pd.DataFrame({"RID": [1, 1, 2, 3],
                        "SCANDATE": ["2015-01-10", "2018-06-01",
                                     "2016-03-01", "2017-07-01"]})
    nfl = pd.DataFrame({"RID": [1, 2],
                        "EXAMDATE": ["2015-01-20", "2016-09-01"],  # subj2 draw ~184d away
                        "NFL": [30.0, 45.0]})
    pet = std(pet, "RID", "SCANDATE")
    nfl = std(nfl, "RID", "EXAMDATE")
    r = overlap(pet, nfl, WINDOW_DAYS)
    print(" ", r)
    assert r["scans_with_nfl"] == 1, r
    assert r["subjects_both"] == 1, r
    print("self-test passed: date matching + overlap counting work.")


def run():
    nfl = std(load(FILES["nfl"]["path"]), FILES["nfl"]["id"], FILES["nfl"]["date"])
    print(f"NfL draws: {len(nfl)} rows, {nfl['rid'].nunique()} subjects\n")
    print(f"{'tracer':>8} | {'PET scans':>9} | {'scans w/ NfL':>12} | "
          f"{'subj (PET)':>10} | {'subj (PET+NfL)':>14}")
    print("-" * 70)
    best, best_n = None, -1
    for tr in ("amyloid", "fdg", "tau"):
        cfg = FILES[tr]
        try:
            pet = std(load(cfg["path"]), cfg["id"], cfg["date"])
        except FileNotFoundError:
            print(f"{tr:>8} | file not found ({cfg['path']})")
            continue
        r = overlap(pet, nfl, WINDOW_DAYS)
        print(f"{tr:>8} | {r['pet_scans']:>9} | {r['scans_with_nfl']:>12} | "
              f"{r['subjects_pet']:>10} | {r['subjects_both']:>14}")
        if r["subjects_both"] > best_n:
            best, best_n = tr, r["subjects_both"]
    print("-" * 70)
    if best:
        print(f"\n>> Recommended starting tracer: {best.upper()} "
              f"({best_n} subjects have both PET and NfL within {WINDOW_DAYS} days)")
        print("   (highest NfL overlap = cleanest first analysis)")


if __name__ == "__main__":
    if "--peek" in sys.argv:
        peek()
    elif "--selftest" in sys.argv:
        selftest()
    else:
        run()