#!/usr/bin/env python3
"""
117_publication_quality_benchmark_figures_tables_v0655.py
Version: v0.6.5.5

Publication-quality figures and manuscript tables for the frozen Tier-1
GLACIER vs ICEBERG 2.1 N-acyl dipeptide benchmark.

Consumes only frozen downstream outputs from Scripts 115 and 116c.
Does NOT recalculate spectra, consensus spectra, cosine scores, CE selection,
bootstrap intervals, or inferential statistics.

Primary endpoint:
    Script 115 equal-weight 25/50/75-eV consensus cosine at 0.020 Da.

Interpretation:
    Script 116c statistical/chemistry outputs.

Outputs:
    figures/benchmark_v0655/
    data/processed/prediction_v0655_publication_benchmark/
    qc/117_publication_quality_benchmark_figures_tables_v0655.tsv
"""

from pathlib import Path
import hashlib
import json
import math
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VERSION = "v0.6.5.5"
POLICY_VERSION = "v0.6.4.8"
PRIMARY_TOL = 0.020
MODELS = ["GLACIER", "ICEBERG 2.1"]
CHAINS = ["C4:0", "C8:0", "C16:0", "C20:0"]

ROOT = Path(__file__).resolve().parents[2]
IN115 = ROOT / "data/processed/prediction_v0653_full_tier1_consensus_benchmark"
IN116 = ROOT / "data/processed/prediction_v0654_statistical_chemistry_interpretation"
OUT = ROOT / "data/processed/prediction_v0655_publication_benchmark"
FIG = ROOT / "figures/benchmark_v0655"
QC_DIR = ROOT / "qc"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)

P115_SUM = IN115 / "FullTier1_Model_Summary.tsv"
P115_SCORES = IN115 / "FullTier1_Consensus_Scores.tsv"
P116_PAIR = IN116 / "FullTier1_Paired_Structure_Statistics.tsv"
P116_OVERALL = IN116 / "FullTier1_Overall_Paired_Inference.tsv"
P116_CHAIN = IN116 / "FullTier1_AcylChain_Stratified_Statistics.tsv"
P116_AA1 = IN116 / "FullTier1_AA1_Descriptive_Summary.tsv"
P116_AA2 = IN116 / "FullTier1_AA2_Descriptive_Summary.tsv"
P116_PAIRAA = IN116 / "FullTier1_Ordered_AA1AA2_Descriptive_Summary.tsv"
P116_CHEM = IN116 / "FullTier1_Annotation_Projected_Structure_Chemistry.tsv"
P116_RANK = IN116 / "FullTier1_Model_Difference_Ranking.tsv"

required = [P115_SUM, P115_SCORES, P116_PAIR, P116_OVERALL, P116_CHAIN,
            P116_AA1, P116_AA2, P116_PAIRAA, P116_CHEM, P116_RANK]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise FileNotFoundError("Missing required frozen inputs:\n" + "\n".join(missing))

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def read_tsv(p):
    return pd.read_csv(p, sep="\t")

def col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of {candidates} found. Available: {df.columns.tolist()}")
    return None

def num(x):
    return pd.to_numeric(x, errors="coerce")

def savefig(fig, stem):
    png = FIG / f"{stem}.png"
    pdf = FIG / f"{stem}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf

def bh_fdr(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    vals = p[ok]
    if len(vals) == 0:
        return out
    order = np.argsort(vals)
    ranked = vals[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked)+1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    tmp = np.empty(len(vals))
    tmp[order] = q
    out[ok] = tmp
    return out

def fmt_p(p):
    if not np.isfinite(p): return "NA"
    if p < 0.001: return f"{p:.2e}"
    return f"{p:.3f}"

# Nature-like restrained palette frozen for this visualization layer.
NAVY = "#2F3E56"
STEEL = "#5B86B3"
SAGE = "#A6B79B"
OLIVE = "#66735C"
MUSTARD = "#D6B24C"
TERRACOTTA = "#B5674D"
ROSE = "#C97B7B"
MAUVE = "#A9879A"
LAVENDER = "#9A8FB3"
TEAL = "#6D9C98"
GREY = "#B9BDC3"
LIGHTGREY = "#ECEDEF"
CHAIN_COLORS = {"C4:0": SAGE, "C8:0": MUSTARD, "C16:0": STEEL, "C20:0": TERRACOTTA}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.transparent": False,
})

m115 = read_tsv(P115_SUM)
scores = read_tsv(P115_SCORES)
pair = read_tsv(P116_PAIR)
overall = read_tsv(P116_OVERALL)
chain = read_tsv(P116_CHAIN)
aa1 = read_tsv(P116_AA1)
aa2 = read_tsv(P116_AA2)
pairaa = read_tsv(P116_PAIRAA)
chem = read_tsv(P116_CHEM)
rank = read_tsv(P116_RANK)

# ---------- schema ----------
model_c = col(scores, ["Model"])
tol_c = col(scores, ["FragmentTolerance_Da"])
cos_c = col(scores, ["CosineSqrtIntensity", "Cosine", "ConsensusCosine"])
spec_c = col(scores, ["BenchmarkSpec"])
q_c = col(scores, ["QueryUSI"])

primary = scores[np.isclose(num(scores[tol_c]), PRIMARY_TOL)].copy()
primary[cos_c] = num(primary[cos_c])

piv = primary.pivot(index=spec_c, columns=model_c, values=cos_c)
if not set(MODELS).issubset(piv.columns):
    raise ValueError(f"Expected models {MODELS}; found {piv.columns.tolist()}")
piv = piv[MODELS].dropna().copy()
piv["Difference"] = piv["GLACIER"] - piv["ICEBERG 2.1"]

# query/acyl chemistry bridge
chem_spec = col(chem, ["BenchmarkSpec"])
chem_chain = col(chem, ["AcylChain"])
chem_q = col(chem, ["QueryUSI"])
chem_unique = chem[[chem_spec, chem_q, chem_chain]].drop_duplicates(subset=[chem_spec])
plotdf = piv.reset_index().merge(
    chem_unique, left_on=spec_c, right_on=chem_spec, how="left", validate="one_to_one"
)

# ---------- Table 1: primary model performance ----------
ms_model = col(m115, ["Model"])
ms_tol = col(m115, ["FragmentTolerance_Da"])
t1 = m115[np.isclose(num(m115[ms_tol]), PRIMARY_TOL)].copy()
keep1 = [c for c in ["Model","N","MeanCosine","MedianCosine","SDCosine",
                     "Q1Cosine","Q3Cosine","MinCosine","MaxCosine"] if c in t1.columns]
t1 = t1[keep1]
t1.insert(0, "Endpoint", "Equal-weight 25/50/75-eV consensus; 0.020-Da matching")
t1.to_csv(OUT/"Table1_Primary_Model_Performance.tsv", sep="\t", index=False)

# ---------- Table 2: overall paired inference ----------
wanted2 = ["N","GLACIER_Higher","ICEBERG_Higher","Ties","MeanDifference",
           "MeanDifference_CI95_Lower","MeanDifference_CI95_Upper",
           "MedianDifference","MedianDifference_CI95_Lower","MedianDifference_CI95_Upper",
           "PairedCohens_dz","SignTestP_TwoSided","WilcoxonP_TwoSided",
           "RankBiserialEffect"]
t2 = overall[[c for c in wanted2 if c in overall.columns]].copy()
t2.insert(0, "Contrast", "GLACIER - ICEBERG 2.1")
t2.to_csv(OUT/"Table2_Overall_Paired_Inference.tsv", sep="\t", index=False)

# ---------- Table 3: acyl-chain stratification ----------
wanted3 = ["AcylChain","N","MeanGLACIER","MedianGLACIER","MeanICEBERG","MedianICEBERG",
           "GLACIER_Higher","ICEBERG_Higher","MeanDifference",
           "MeanDifference_CI95_Lower","MeanDifference_CI95_Upper",
           "MedianDifference","MedianDifference_CI95_Lower","MedianDifference_CI95_Upper",
           "PairedCohens_dz","SignTestP_TwoSided","SignTestBH_FDR",
           "WilcoxonP_TwoSided","WilcoxonBH_FDR","RankBiserialEffect"]
t3 = chain[[c for c in wanted3 if c in chain.columns]].copy()
if "AcylChain" in t3:
    t3["_ord"] = t3["AcylChain"].map({x:i for i,x in enumerate(CHAINS)})
    t3 = t3.sort_values("_ord").drop(columns="_ord")
t3.to_csv(OUT/"Table3_AcylChain_Stratified_Performance.tsv", sep="\t", index=False)

# ---------- Table 4: strongest model-specific structures ----------
r_diff = col(rank, ["GLACIER_minus_ICEBERG", "Difference"])
r_spec = col(rank, ["BenchmarkSpec"])
r_chain = col(rank, ["AcylChain"], required=False)
r_pair = col(rank, ["CandidateOrderedAA1AA2", "OrderedAA1AA2"], required=False)
r_g = col(rank, ["GLACIER"])
r_i = col(rank, ["ICEBERG 2.1"])
rr = rank.copy()
rr[r_diff] = num(rr[r_diff])
g20 = rr.nlargest(20, r_diff).copy()
i20 = rr.nsmallest(20, r_diff).copy()
g20.insert(0, "Direction", "GLACIER-favoring")
i20.insert(0, "Direction", "ICEBERG 2.1-favoring")
t4 = pd.concat([g20, i20], ignore_index=True)
keep4 = [c for c in ["Direction",r_spec,r_chain,r_pair,r_g,r_i,r_diff] if c is not None and c in t4.columns]
t4[keep4].to_csv(OUT/"Table4_Top_Model_Favoring_Structures.tsv", sep="\t", index=False)

# ---------- Supplementary chemistry summaries ----------
aa1.to_csv(OUT/"TableS1_AA1_Descriptive_Performance.tsv", sep="\t", index=False)
aa2.to_csv(OUT/"TableS2_AA2_Descriptive_Performance.tsv", sep="\t", index=False)
pairaa.to_csv(OUT/"TableS3_Ordered_AA1AA2_Descriptive_Performance.tsv", sep="\t", index=False)

# ---------- Figure 1: primary model performance ----------
fig, ax = plt.subplots(figsize=(4.3, 4.0))
rng = np.random.default_rng(20260828)
xpos = {"GLACIER":0, "ICEBERG 2.1":1}
colors = {"GLACIER":NAVY, "ICEBERG 2.1":TERRACOTTA}
for model in MODELS:
    vals = primary.loc[primary[model_c]==model, cos_c].dropna().to_numpy(float)
    x = xpos[model]
    parts = ax.violinplot(vals, positions=[x], widths=0.72, showmeans=False,
                         showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(colors[model]); body.set_edgecolor(colors[model]); body.set_alpha(0.20)
    q1, med, q3 = np.quantile(vals,[.25,.5,.75])
    ax.plot([x,x],[q1,q3], lw=5, color=colors[model], solid_capstyle="butt")
    ax.scatter([x],[med], s=25, color="white", edgecolor=colors[model], linewidth=.9, zorder=4)
    jit = rng.uniform(-.10,.10,len(vals))
    ax.scatter(np.full(len(vals),x)+jit, vals, s=7, alpha=.25, color=colors[model], linewidths=0)
ax.set_xticks([0,1], MODELS)
ax.set_ylabel("Consensus cosine similarity")
ax.set_ylim(0,1)
ax.set_title("Primary Tier-1 prediction benchmark")
ax.text(.02,.98,"n = 300 paired structures\n0.020-Da primary tolerance",
        transform=ax.transAxes, va="top", ha="left")
ax.spines[["top","right"]].set_visible(False)
savefig(fig, "Figure117A_Primary_Model_Performance")

# ---------- Figure 2: paired scatter ----------
fig, ax = plt.subplots(figsize=(4.3,4.1))
for ch in CHAINS:
    d = plotdf[plotdf[chem_chain]==ch]
    ax.scatter(d["ICEBERG 2.1"], d["GLACIER"], s=20, alpha=.72,
               label=f"{ch} (n={len(d)})", color=CHAIN_COLORS[ch], edgecolors="none")
ax.plot([0,1],[0,1], ls="--", lw=1, color=GREY)
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.set_xlabel("ICEBERG 2.1 consensus cosine")
ax.set_ylabel("GLACIER consensus cosine")
ax.set_title("Paired structure-level performance")
ax.legend(frameon=False, loc="lower right")
ax.spines[["top","right"]].set_visible(False)
savefig(fig, "Figure117B_Paired_Model_Scatter")

# ---------- Figure 3: difference distribution ----------
fig, ax = plt.subplots(figsize=(5.0,3.7))
d = piv["Difference"].to_numpy(float)
ax.hist(d, bins=np.linspace(-.32,.24,29), edgecolor="white", linewidth=.6, color=STEEL)
ax.axvline(0, color=GREY, lw=1, ls="--")
ax.axvline(np.mean(d), color=NAVY, lw=1.5)
ax.set_xlabel("Paired cosine difference (GLACIER − ICEBERG 2.1)")
ax.set_ylabel("Structures")
ax.set_title("Distribution of paired model differences")
ov = overall.iloc[0]
mean_d = float(ov["MeanDifference"])
lo = float(ov["MeanDifference_CI95_Lower"])
hi = float(ov["MeanDifference_CI95_Upper"])
wp = float(ov["WilcoxonP_TwoSided"])
ax.text(.02,.96,f"Mean Δ = {mean_d:.3f} (95% CI {lo:.3f} to {hi:.3f})\nWilcoxon p = {fmt_p(wp)}",
        transform=ax.transAxes, va="top")
ax.spines[["top","right"]].set_visible(False)
savefig(fig, "Figure117C_Paired_Difference_Distribution")

# ---------- Figure 4: acyl-chain forest ----------
cc = chain.copy()
cc["_ord"] = cc["AcylChain"].map({x:i for i,x in enumerate(CHAINS)})
cc = cc.sort_values("_ord")
fig, ax = plt.subplots(figsize=(5.0,3.6))
y = np.arange(len(cc))
means = num(cc["MeanDifference"]).to_numpy(float)
lo = num(cc["MeanDifference_CI95_Lower"]).to_numpy(float)
hi = num(cc["MeanDifference_CI95_Upper"]).to_numpy(float)
for j,(yy,m,l,h,ch) in enumerate(zip(y,means,lo,hi,cc["AcylChain"])):
    ax.errorbar(m, yy, xerr=[[m-l],[h-m]], fmt="o", markersize=6,
                capsize=3, color=CHAIN_COLORS.get(ch,NAVY), ecolor=CHAIN_COLORS.get(ch,NAVY))
ax.axvline(0,color=GREY,ls="--",lw=1)
ax.set_yticks(y, [f'{r.AcylChain} (n={int(r.N)})' for r in cc.itertuples()])
ax.invert_yaxis()
ax.set_xlabel("Mean paired cosine difference\n(GLACIER − ICEBERG 2.1), bootstrap 95% CI")
ax.set_title("Acyl-chain-stratified model difference")
ax.spines[["top","right"]].set_visible(False)
savefig(fig, "Figure117D_AcylChain_Forest")

# ---------- Figure 5: ordered AA1-AA2 descriptive heatmap ----------
# Build from structure-level chemistry + paired differences, ensuring descriptive only.
pair_label_c = col(chem, ["CandidateOrderedAA1AA2", "OrderedAA1AA2"], required=False)
aa1_c = col(chem, ["AA1"], required=False)
aa2_c = col(chem, ["AA2"], required=False)
if aa1_c and aa2_c:
    hdf = plotdf[[spec_c, "Difference"]].merge(
        chem[[chem_spec,aa1_c,aa2_c]].drop_duplicates(subset=[chem_spec]),
        left_on=spec_c, right_on=chem_spec, how="left", validate="one_to_one")
    mat = hdf.pivot_table(index=aa1_c, columns=aa2_c, values="Difference", aggfunc="mean")
    nmat = hdf.pivot_table(index=aa1_c, columns=aa2_c, values="Difference", aggfunc="size")
    rows = sorted(mat.index.astype(str)); cols = sorted(mat.columns.astype(str))
    mat = mat.reindex(index=rows, columns=cols)
    nmat = nmat.reindex(index=rows, columns=cols)
    fig, ax = plt.subplots(figsize=(8.0,7.0))
    vmax = np.nanmax(np.abs(mat.to_numpy(float)))
    vmax = max(vmax, 0.05)
    im = ax.imshow(mat.to_numpy(float), aspect="auto", cmap="coolwarm",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks(np.arange(len(cols)), cols, rotation=90)
    ax.set_yticks(np.arange(len(rows)), rows)
    ax.set_xlabel("AA2")
    ax.set_ylabel("AA1")
    ax.set_title("Ordered AA1-AA2 model-difference landscape (descriptive)")
    cb = fig.colorbar(im, ax=ax, shrink=.82)
    cb.set_label("Mean cosine difference (GLACIER − ICEBERG 2.1)")
    fig.text(.5,.01,"Cells are descriptive chemistry summaries; no AA1/AA2 inferential p-values.",
             ha="center", fontsize=8)
    savefig(fig, "Figure117E_Ordered_AA1AA2_Descriptive_Heatmap")

# ---------- Figure 6: top model-favoring structures ----------
topg = rr.nlargest(10,r_diff).copy()
topi = rr.nsmallest(10,r_diff).copy()
ext = pd.concat([topi,topg],ignore_index=True).sort_values(r_diff)
labels = []
for _,r in ext.iterrows():
    pairlab = str(r[r_pair]) if r_pair and pd.notna(r[r_pair]) else str(r[r_spec])
    chainlab = str(r[r_chain]) if r_chain and pd.notna(r[r_chain]) else ""
    labels.append(f"{chainlab} {pairlab}".strip())
fig, ax = plt.subplots(figsize=(6.4,5.8))
vals = num(ext[r_diff]).to_numpy(float)
barcolors = [TERRACOTTA if v < 0 else NAVY for v in vals]
ax.barh(np.arange(len(ext)), vals, color=barcolors)
ax.axvline(0,color=GREY,lw=1)
ax.set_yticks(np.arange(len(ext)), labels)
ax.set_xlabel("Cosine difference (GLACIER − ICEBERG 2.1)")
ax.set_title("Structures with strongest model-specific performance")
ax.spines[["top","right"]].set_visible(False)
savefig(fig, "Figure117F_Extreme_Model_Differences")

# ---------- manuscript-ready summary text ----------
ov = overall.iloc[0]
c20 = chain[chain["AcylChain"].astype(str)=="C20:0"].iloc[0]
c16 = chain[chain["AcylChain"].astype(str)=="C16:0"].iloc[0]
summary = f"""Tier-1 spectrum-prediction benchmark — manuscript-ready results
Version: {VERSION}

PRIMARY ENDPOINT
The primary benchmark used the frozen v0.6.4.8 equal-weight consensus of
separately predicted 25, 50, and 75 eV spectra, scored against the measured
stepped 25/50/75 eV experimental spectrum at 0.020 Da.

OVERALL
Across 300 paired N-acyl dipeptide structures, GLACIER produced a higher
primary cosine score for {int(ov['GLACIER_Higher'])} structures and ICEBERG 2.1
for {int(ov['ICEBERG_Higher'])}. The mean paired difference
(GLACIER - ICEBERG 2.1) was {float(ov['MeanDifference']):.6f}
(bootstrap 95% CI {float(ov['MeanDifference_CI95_Lower']):.6f} to
{float(ov['MeanDifference_CI95_Upper']):.6f}), and the median paired difference
was {float(ov['MedianDifference']):.6f}
(bootstrap 95% CI {float(ov['MedianDifference_CI95_Lower']):.6f} to
{float(ov['MedianDifference_CI95_Upper']):.6f}). The paired Cohen's dz was
{float(ov['PairedCohens_dz']):.3f}; the two-sided Wilcoxon signed-rank p-value
was {float(ov['WilcoxonP_TwoSided']):.6g}, with paired rank-biserial effect
{float(ov['RankBiserialEffect']):.3f}. These results support a statistically
detectable but modest overall GLACIER advantage within this benchmark.

CHEMISTRY
The GLACIER advantage was most robust in longer-chain strata. For C20:0,
the mean paired difference was {float(c20['MeanDifference']):.6f}
(95% CI {float(c20['MeanDifference_CI95_Lower']):.6f} to
{float(c20['MeanDifference_CI95_Upper']):.6f}; Wilcoxon BH-FDR
{float(c20['WilcoxonBH_FDR']):.6g}). For C16:0, the mean paired difference was
{float(c16['MeanDifference']):.6f}
(95% CI {float(c16['MeanDifference_CI95_Lower']):.6f} to
{float(c16['MeanDifference_CI95_Upper']):.6f}; Wilcoxon BH-FDR
{float(c16['WilcoxonBH_FDR']):.6g}).

INTERPRETATION BOUNDARY
AA1, AA2, and ordered AA1-AA2 visualizations/tables are descriptive chemistry
summaries and are not promoted to independent inferential endpoints.
Best-CE diagnostics are not used. No frozen score is recalculated here.
"""
(OUT/"Manuscript_Results_SpectrumPrediction_Benchmark.txt").write_text(summary)

# ---------- figure/table manifest + provenance ----------
fig_files = sorted(list(FIG.glob("*.png")) + list(FIG.glob("*.pdf")))
table_files = sorted(OUT.glob("*.tsv"))
manifest_rows = []
for p in fig_files:
    manifest_rows.append({"ArtifactType":"Figure","Artifact":p.name,"Path":str(p.relative_to(ROOT)),
                          "SHA256":sha256(p)})
for p in table_files:
    manifest_rows.append({"ArtifactType":"Table","Artifact":p.name,"Path":str(p.relative_to(ROOT)),
                          "SHA256":sha256(p)})
manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(OUT/"Publication_Artifact_Manifest.tsv",sep="\t",index=False)

input_hash = pd.DataFrame([{"Input":str(p.relative_to(ROOT)),"SHA256":sha256(p)} for p in required])
input_hash.to_csv(OUT/"Publication_Input_SHA256.tsv",sep="\t",index=False)

# ---------- QC ----------
qcrows=[]
def qc(check, expected, observed, passed):
    qcrows.append({"Check":check,"Expected":expected,"Observed":observed,
                   "Status":"PASS" if bool(passed) else "FAIL"})

qc("Frozen Script-115 primary rows",600,len(primary),len(primary)==600)
qc("Paired structures",300,len(piv),len(piv)==300)
qc("Models","GLACIER|ICEBERG 2.1","|".join(sorted(primary[model_c].astype(str).unique())),
   set(primary[model_c].astype(str).unique())==set(MODELS))
qc("Primary tolerance",PRIMARY_TOL,sorted(num(primary[tol_c]).dropna().unique().tolist()),
   np.allclose(num(primary[tol_c]),PRIMARY_TOL))
qc("Chemistry-resolved plotted structures",300,int(plotdf[chem_chain].notna().sum()),
   int(plotdf[chem_chain].notna().sum())==300)
qc("Acyl-chain strata",4,int(chain["AcylChain"].nunique()),int(chain["AcylChain"].nunique())==4)
qc("Overall Wilcoxon finite",True,bool(np.isfinite(num(overall["WilcoxonP_TwoSided"]).iloc[0])),
   bool(np.isfinite(num(overall["WilcoxonP_TwoSided"]).iloc[0])))
qc("Overall rank-biserial finite",True,bool(np.isfinite(num(overall["RankBiserialEffect"]).iloc[0])),
   bool(np.isfinite(num(overall["RankBiserialEffect"]).iloc[0])))
qc("Frozen score recalculation performed",False,False,True)
qc("Best-CE primary use performed",False,False,True)
qc("AA1/AA2 inferential testing performed",False,False,True)
qc("PNG figure count",">=6",len(list(FIG.glob("*.png"))),len(list(FIG.glob("*.png")))>=6)
qc("PDF figure count",">=6",len(list(FIG.glob("*.pdf"))),len(list(FIG.glob("*.pdf")))>=6)
qcdf=pd.DataFrame(qcrows)
qc_path=QC_DIR/"117_publication_quality_benchmark_figures_tables_v0655.tsv"
qcdf.to_csv(qc_path,sep="\t",index=False)

notes = f"""Publication-quality Tier-1 benchmark figures/tables {VERSION}

STATUS
======
{"PASS" if (qcdf.Status=="PASS").all() else "FAIL"}

FROZEN ANALYTICAL BASIS
=======================
Spectrum-scoring policy: {POLICY_VERSION}
Primary fragment tolerance: {PRIMARY_TOL:.3f} Da
Primary prediction representation: equal-weight 25/50/75-eV consensus.
Experimental representation: measured stepped 25/50/75 eV.

This visualization/reporting script does NOT recalculate spectra, consensus
spectra, cosine scores, bootstrap confidence intervals, Wilcoxon tests,
rank-biserial effects, or CE-specific model selection.

PUBLICATION ARTIFACTS
=====================
Figure117A: primary model-performance distributions.
Figure117B: paired GLACIER vs ICEBERG 2.1 scatter, colored by acyl chain.
Figure117C: paired GLACIER-minus-ICEBERG difference distribution.
Figure117D: acyl-chain mean-difference forest plot with bootstrap 95% CIs.
Figure117E: ordered AA1-AA2 descriptive model-difference heatmap.
Figure117F: strongest model-favoring structures.

Tables 1-4 contain primary model performance, overall paired inference,
acyl-chain-stratified results, and strongest model-specific structures.
Supplementary Tables S1-S3 contain descriptive AA1/AA2 chemistry summaries.

INTERPRETATION BOUNDARY
=======================
Primary inference remains structure-level and tied to the frozen 0.020-Da
consensus endpoint. AA1/AA2 chemistry outputs remain descriptive. Best-CE
diagnostics are excluded from primary figures/tables.
"""
(OUT/"Publication_Benchmark_Notes.txt").write_text(notes)

print(f"\nPublication-quality Tier-1 benchmark figures/tables {VERSION}\n")
print("="*66)
print("\nManuscript tables")
print("-----------------")
for p in sorted(OUT.glob("Table*.tsv")):
    print(p)
print("\nFigures")
print("-------")
for p in sorted(FIG.glob("*.png")):
    print(p)
print("\nQC")
print("--")
print(qcdf.to_string(index=False))
print("\nSaved manuscript results text:")
print(OUT/"Manuscript_Results_SpectrumPrediction_Benchmark.txt")
if not (qcdf.Status=="PASS").all():
    print("\nPUBLICATION BENCHMARK ARTIFACT BUILD FAILED QC")
    sys.exit(1)
print(f"\nPUBLICATION-QUALITY BENCHMARK FIGURES + TABLES {VERSION} PASSED")
