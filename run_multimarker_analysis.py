"""
Multi-marker concordance with FDG hypometabolism (upgraded cohort, n=352).

Question: which plasma marker tracks regional FDG SUVR most - NfL, ptau217,
GFAP, or amyloid ratio? And does the ranking survive age-adjustment and hold
WITHIN diagnostic groups (i.e. is it real, or just disease-stage separation)?

Outputs: printed table (raw, age-adjusted partial, within-group) + a ranking figure.
All markers rank-based (Spearman), so skew is handled without transformation.
"""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

COHORT = "upenn_matched_cohort.csv"
PTDEMOG_RDA = "data/ADNIMERGE2/data/PTDEMOG.rda"   # for age; optional (try/except)
FDG = "fdg_suvr"
MARKERS = {"NfL (Quanterix)":"nfl_value", "NfL (Fujirebio)":"NfL_F",
           "p-tau217":"pT217_F", "GFAP":"GFAP_Q", "Abeta42/40":"AB42_AB40_F"}
GROUPS = ["CN","MCI","Dementia"]

def partial_spearman(x, y, z):
    """Spearman partial correlation of x,y controlling z (via rank residuals)."""
    rx,ry,rz = (stats.rankdata(v) for v in (x,y,z))
    ex = rx - np.polyval(np.polyfit(rz,rx,1),rz)
    ey = ry - np.polyval(np.polyfit(rz,ry,1),rz)
    return stats.pearsonr(ex,ey)

def main():
    df = pd.read_csv(COHORT)
    df[FDG] = pd.to_numeric(df[FDG], errors="coerce")
    print(f"n = {len(df)}")

    # try to attach age (rigor: markers rise with age)
    have_age = False
    try:
        import pyreadr
        d = pyreadr.read_r(PTDEMOG_RDA); d = d[list(d.keys())[0]].copy()
        d["rid"] = pd.to_numeric(d["RID"],errors="coerce").astype("Int64")
        if "PTDOBYY" in d.columns:
            mm = pd.to_numeric(d.get("PTDOBMM",6),errors="coerce").fillna(6)
            dob = pd.to_datetime(dict(year=pd.to_numeric(d["PTDOBYY"],errors="coerce"),
                                      month=mm, day=15), errors="coerce")
            dob = pd.DataFrame({"rid":d["rid"],"dob":dob}).dropna().groupby("rid",as_index=False).first()
            df["rid"] = pd.to_numeric(df["rid"],errors="coerce").astype("Int64")
            df = df.merge(dob,on="rid",how="left")
            df["age"] = (pd.to_datetime(df["fdg_date"],errors="coerce") - df["dob"]).dt.days/365.25
            have_age = df["age"].notna().sum() > 0.8*len(df)
    except Exception as e:
        print(f"(age not attached: {e}) -- reporting raw + within-group only")
    if have_age: print(f"age attached for {df.age.notna().sum()} subjects")

    # ---- data-quality guard: drop constant/empty marker columns ----
    usable = {}
    for name,col in MARKERS.items():
        if col in df.columns and pd.to_numeric(df[col],errors="coerce").nunique() > 5:
            usable[name] = col
        else:
            print(f"  SKIP {name} ({col}): constant/empty column")
    print()

    # ---- main table ----
    print(f"{'marker':>16} | {'raw r':>8} {'p':>9} | {'age-adj r':>9} | {'CN':>6} {'MCI':>6} {'Dem':>6}")
    print("-"*78)
    ranks = {}
    for name,col in usable.items():
        m = df[[FDG,col,"DIAGNOSIS"]+(["age"] if have_age else [])].copy()
        m[col] = pd.to_numeric(m[col],errors="coerce"); m = m.dropna(subset=[FDG,col])
        r,p = stats.spearmanr(m[FDG], m[col]); ranks[name]=abs(r)
        adj = ""
        if have_age:
            ma = m.dropna(subset=["age"])
            pr,_ = partial_spearman(ma[FDG].values, ma[col].values, ma["age"].values)
            adj = f"{pr:+.3f}"
        wg = []
        for g in GROUPS:
            s = m[m.DIAGNOSIS==g]
            wg.append(f"{stats.spearmanr(s[FDG],s[col]).correlation:+.2f}" if len(s)>10 else "  -")
        print(f"{name:>16} | {r:+.3f} {p:.1e} | {adj:>9} | {wg[0]:>6} {wg[1]:>6} {wg[2]:>6}")

    # ---- ranking figure ----
    order = sorted(ranks, key=ranks.get)
    fig,ax = plt.subplots(figsize=(8,4.5))
    ax.barh(range(len(order)), [ranks[k] for k in order], color="#185FA5")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    ax.set_xlabel("|Spearman r| with FDG SUVR")
    ax.set_title("Which plasma marker tracks FDG hypometabolism most (n=%d)"%len(df))
    for i,k in enumerate(order): ax.text(ranks[k]+.005,i,f"{ranks[k]:.2f}",va="center")
    fig.tight_layout(); fig.savefig("fig_multimarker_ranking.png",dpi=150,bbox_inches="tight")
    print("\nsaved fig_multimarker_ranking.png")

if __name__ == "__main__":
    main()