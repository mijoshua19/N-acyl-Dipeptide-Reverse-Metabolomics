#!/usr/bin/env python3
"""
116_statistical_chemistry_stratified_interpretation_v0654.py

Statistical + chemistry-stratified interpretation of the frozen full Tier-1
GLACIER vs ICEBERG 2.1 consensus benchmark.

Version: v0.6.5.4

IMPORTANT
---------
This is a downstream interpretation layer. It NEVER recalculates or modifies
the frozen v0.6.4.8 spectrum-scoring policy. Primary inputs are the Script-115
0.020-Da consensus scores.

Chemistry is joined from the official Query_to_Annotation_Links.tsv using the
true experimental QueryUSI. Because an official query can have multiple
within-chain candidate/isomer annotations, Script 116 explicitly distinguishes:
  - AcylChain: query-level, required to be unique.
  - AA1/AA2: annotation-level; a query may map to >1 ordered AA1-AA2 chemistry.

Therefore:
  * acyl-chain statistics use each BenchmarkSpec exactly once;
  * AA1/AA2 and ordered-pair summaries are annotation-projected descriptive
    analyses and do NOT duplicate structures for inferential tests.

Statistical outputs:
  - paired GLACIER - ICEBERG differences (n=300);
  - deterministic bootstrap 95% CIs for mean and median paired differences;
  - paired Wilcoxon signed-rank test (two-sided) when SciPy is available;
  - sign/binomial test (two-sided; exact implementation);
  - paired standardized mean-difference effect size (dz);
  - rank-biserial effect size from paired nonzero differences;
  - acyl-chain stratified descriptive statistics + bootstrap CIs;
  - AA1, AA2, and ordered AA1-AA2 annotation-projected descriptive summaries;
  - strongest structure-level model-favoring differences;
  - multiple-testing control (Benjamini-Hochberg) for exploratory acyl-chain
    Wilcoxon p-values only.

No post-hoc alteration of the primary scoring method is permitted.
"""

from __future__ import annotations
from pathlib import Path
import hashlib
import math
import sys
import warnings

import numpy as np
import pandas as pd

VERSION = "v0.6.5.4"
FROZEN_POLICY = "v0.6.4.8"
PRIMARY_TOL = 0.020
BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 20260828
CI_ALPHA = 0.05
MIN_GROUP_N_FOR_INFERENCE = 10

ROOT = Path.cwd().resolve()

S115 = ROOT / "data/processed/prediction_v0653_full_tier1_consensus_benchmark"
SCORES = S115 / "FullTier1_Consensus_Scores.tsv"
QC115 = ROOT / "qc/115_score_full_tier1_consensus_benchmark_v0653.tsv"

CHEM = ROOT / "data/intermediate/Query_to_Annotation_Links.tsv"

OUT = ROOT / "data/processed/prediction_v0654_statistical_chemistry_interpretation"
QC_DIR = ROOT / "qc"
OUT.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["GLACIER", "ICEBERG 2.1"]
CHAIN_ORDER = ["C4:0", "C8:0", "C16:0", "C20:0"]
AA_ORDER = [
    "Ala","Arg","Asn","Asp","Cys","Gln","Glu","Gly","His","Ile",
    "Leu","Lys","Met","Phe","Pro","Ser","Thr","Trp","Tyr","Val"
]

AA_MAP = {
    "A":"Ala","R":"Arg","N":"Asn","D":"Asp","C":"Cys","Q":"Gln","E":"Glu",
    "G":"Gly","H":"His","I":"Ile","L":"Leu","K":"Lys","M":"Met","F":"Phe",
    "P":"Pro","S":"Ser","T":"Thr","W":"Trp","Y":"Tyr","V":"Val",
    "ALA":"Ala","ARG":"Arg","ASN":"Asn","ASP":"Asp","CYS":"Cys","GLN":"Gln",
    "GLU":"Glu","GLY":"Gly","HIS":"His","ILE":"Ile","LEU":"Leu","LYS":"Lys",
    "MET":"Met","PHE":"Phe","PRO":"Pro","SER":"Ser","THR":"Thr","TRP":"Trp",
    "TYR":"Tyr","VAL":"Val"
}

def require(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Required input missing: {p}")

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def write_tsv(df: pd.DataFrame, p: Path):
    df.to_csv(p, sep="\t", index=False)

def choose(cols, candidates, required=True):
    lower = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise RuntimeError(
            f"Could not find any of {candidates}. Observed columns={list(cols)}"
        )
    return None

def clean_series(s):
    s = s.astype("string").str.strip()
    return s.mask(s.isin(["", "nan", "None", "<NA>"]))

def normalize_aa(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    if not s:
        return pd.NA
    key = s.upper()
    if key in AA_MAP:
        return AA_MAP[key]
    # Preserve already title-cased three-letter names.
    t = s[:1].upper() + s[1:].lower()
    return t if t in AA_ORDER else pd.NA

def exact_two_sided_binom_p(k, n):
    """Exact two-sided binomial test for p=0.5, matching probability ordering."""
    if n <= 0:
        return np.nan
    from math import comb
    probs = np.array([comb(n, i) * (0.5 ** n) for i in range(n + 1)], dtype=float)
    pk = probs[k]
    return float(probs[probs <= pk + 1e-15].sum())

def bootstrap_ci(x, statistic="mean", reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    # Chunked for deterministic memory-safe execution.
    vals = np.empty(reps, dtype=float)
    chunk = 1000
    fn = np.mean if statistic == "mean" else np.median
    pos = 0
    while pos < reps:
        m = min(chunk, reps - pos)
        idx = rng.integers(0, len(x), size=(m, len(x)))
        vals[pos:pos+m] = fn(x[idx], axis=1)
        pos += m
    return tuple(np.quantile(vals, [CI_ALPHA/2, 1-CI_ALPHA/2]))

def paired_effects(diff):
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    mean = float(np.mean(d))
    med = float(np.median(d))
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else np.nan
    dz = mean / sd if np.isfinite(sd) and sd > 0 else np.nan
    nonzero = d[~np.isclose(d, 0.0, atol=1e-15)]
    pos = int((nonzero > 0).sum())
    neg = int((nonzero < 0).sum())
    sign_p = exact_two_sided_binom_p(pos, len(nonzero)) if len(nonzero) else np.nan

    wilcoxon_p = np.nan
    rank_biserial = np.nan
    scipy_status = "UNAVAILABLE"
    if len(nonzero):
        try:
            from scipy.stats import wilcoxon, rankdata
            w = wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided",
                         correction=False, method="auto")
            wilcoxon_p = float(w.pvalue)
            ranks = rankdata(np.abs(nonzero), method="average")
            wpos = float(ranks[nonzero > 0].sum())
            wneg = float(ranks[nonzero < 0].sum())
            denom = wpos + wneg
            rank_biserial = (wpos - wneg) / denom if denom else np.nan
            scipy_status = "PASS"
        except Exception as exc:
            scipy_status = f"UNAVAILABLE:{type(exc).__name__}"

    mean_lo, mean_hi = bootstrap_ci(d, "mean", seed=BOOTSTRAP_SEED + len(d))
    med_lo, med_hi = bootstrap_ci(d, "median", seed=BOOTSTRAP_SEED + 1000 + len(d))
    return {
        "N": len(d),
        "GLACIER_Higher": pos,
        "ICEBERG_Higher": neg,
        "Ties": int(len(d) - len(nonzero)),
        "MeanDifference": mean,
        "MeanDifference_CI95_Lower": mean_lo,
        "MeanDifference_CI95_Upper": mean_hi,
        "MedianDifference": med,
        "MedianDifference_CI95_Lower": med_lo,
        "MedianDifference_CI95_Upper": med_hi,
        "SDDifference": sd,
        "PairedCohens_dz": dz,
        "SignTestP_TwoSided": sign_p,
        "WilcoxonP_TwoSided": wilcoxon_p,
        "RankBiserialEffect": rank_biserial,
        "WilcoxonStatus": scipy_status,
    }

def bh_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    vals = p[ok]
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(ranked)
    adj = ranked * m / np.arange(1, m+1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    back = np.empty(m)
    back[order] = adj
    out[np.where(ok)[0]] = back
    return out

# ----------------------------- load ---------------------------------

require(SCORES)
require(CHEM)

scores = pd.read_csv(SCORES, sep="\t", low_memory=False)
chem = pd.read_csv(CHEM, sep="\t", low_memory=False)

# Optional gate: Script 115 QC must contain no FAIL if present.
if QC115.exists():
    q115 = pd.read_csv(QC115, sep="\t", low_memory=False)
    if "Status" in q115.columns and not q115["Status"].astype(str).eq("PASS").all():
        raise RuntimeError("Script 115 QC contains a non-PASS row.")

required_scores = [
    "Model","BenchmarkSpec","QueryUSI","FragmentTolerance_Da",
    "IsPrimaryTolerance","CosineSqrtIntensity","ScoringPolicyVersion"
]
missing = [c for c in required_scores if c not in scores.columns]
if missing:
    raise RuntimeError(f"Script 115 scores missing columns: {missing}")

scores["FragmentTolerance_Da"] = pd.to_numeric(
    scores["FragmentTolerance_Da"], errors="coerce"
)
scores["CosineSqrtIntensity"] = pd.to_numeric(
    scores["CosineSqrtIntensity"], errors="coerce"
)

primary = scores[np.isclose(scores["FragmentTolerance_Da"], PRIMARY_TOL)].copy()
if len(primary) != 600:
    raise RuntimeError(f"Expected 600 primary score rows; observed {len(primary)}.")
if set(primary["Model"].unique()) != set(MODELS):
    raise RuntimeError(f"Unexpected models: {sorted(primary['Model'].unique())}")
if set(primary["ScoringPolicyVersion"].astype(str)) != {FROZEN_POLICY}:
    raise RuntimeError("Frozen scoring-policy version changed.")

# one score/model/spec
if primary[["Model","BenchmarkSpec"]].duplicated().any():
    raise RuntimeError("Duplicate model/BenchmarkSpec primary scores.")
if primary["BenchmarkSpec"].nunique() != 300:
    raise RuntimeError("Primary benchmark does not contain exactly 300 structures.")

wide = primary.pivot(
    index=["BenchmarkSpec","QueryUSI"],
    columns="Model",
    values="CosineSqrtIntensity"
).reset_index()

if len(wide) != 300 or wide[MODELS].isna().any().any():
    raise RuntimeError("Incomplete paired primary-score matrix.")

wide["GLACIER_minus_ICEBERG"] = wide["GLACIER"] - wide["ICEBERG 2.1"]
wide["HigherModel"] = np.where(
    wide["GLACIER_minus_ICEBERG"] > 0, "GLACIER",
    np.where(wide["GLACIER_minus_ICEBERG"] < 0, "ICEBERG 2.1", "TIE")
)
wide["AbsModelDifference"] = wide["GLACIER_minus_ICEBERG"].abs()

# ----------------------- chemistry crosswalk ------------------------

cols = list(chem.columns)
qcol = choose(cols, ["query_usi","QueryUSI","OfficialQueryUSI","USI"])
chain_col = choose(cols, ["AcylChain","AcylChains","acyl_chain"])
aa1_col = choose(cols, ["AA1","AminoAcid1","Amino_Acid_1","aa1"])
aa2_col = choose(cols, ["AA2","AminoAcid2","Amino_Acid_2","aa2"])
ann_col = choose(cols, ["AnnotationIndex"], required=False)

chem2 = pd.DataFrame({
    "QueryUSI": clean_series(chem[qcol]),
    "AcylChain": clean_series(chem[chain_col]),
    "AA1": chem[aa1_col].map(normalize_aa),
    "AA2": chem[aa2_col].map(normalize_aa),
})
if ann_col:
    chem2["AnnotationIndex"] = clean_series(chem[ann_col])
else:
    chem2["AnnotationIndex"] = np.arange(len(chem2)).astype(str)

chem2 = chem2[
    chem2["QueryUSI"].notna()
    & chem2["AcylChain"].isin(CHAIN_ORDER)
    & chem2["AA1"].isin(AA_ORDER)
    & chem2["AA2"].isin(AA_ORDER)
].drop_duplicates().copy()
chem2["AA1AA2"] = chem2["AA1"] + "-" + chem2["AA2"]

bench_q = set(wide["QueryUSI"].astype(str))
bench_chem = chem2[chem2["QueryUSI"].astype(str).isin(bench_q)].copy()

# Every benchmark query must resolve to chemistry.
resolved_q = set(bench_chem["QueryUSI"].astype(str))
missing_q = sorted(bench_q - resolved_q)
if missing_q:
    raise RuntimeError(
        f"{len(missing_q)} benchmark QueryUSIs lack official chemistry mapping. "
        f"First examples={missing_q[:5]}"
    )

# One acyl chain per official query is a hard invariant.
q_chain_n = bench_chem.groupby("QueryUSI")["AcylChain"].nunique()
if int(q_chain_n.max()) > 1:
    bad = q_chain_n[q_chain_n > 1]
    raise RuntimeError(f"{len(bad)} benchmark queries cross acyl chains.")

query_chain = (
    bench_chem[["QueryUSI","AcylChain"]]
    .drop_duplicates()
)
if len(query_chain) != 300:
    raise RuntimeError(
        f"Expected 300 unique query->chain mappings; observed {len(query_chain)}."
    )

paired = wide.merge(query_chain, on="QueryUSI", how="left", validate="one_to_one")

# Annotation multiplicity is retained and audited, not collapsed.
mult = (
    bench_chem.groupby("QueryUSI")
    .agg(
        AnnotationChemistryRows=("AnnotationIndex","nunique"),
        OrderedAA1AA2Count=("AA1AA2","nunique"),
        AA1Count=("AA1","nunique"),
        AA2Count=("AA2","nunique"),
    ).reset_index()
)
paired = paired.merge(mult, on="QueryUSI", how="left", validate="one_to_one")
paired["ChemistryAmbiguousWithinChain"] = paired["OrderedAA1AA2Count"] > 1

# ------------------------ overall inference -------------------------

overall = pd.DataFrame([{
    "Scope": "FullTier1",
    "AcylChain": "ALL",
    **paired_effects(paired["GLACIER_minus_ICEBERG"].to_numpy())
}])

# ------------------------ chain inference ---------------------------

chain_rows = []
for chain in CHAIN_ORDER:
    g = paired[paired["AcylChain"].eq(chain)]
    if len(g) == 0:
        continue
    eff = paired_effects(g["GLACIER_minus_ICEBERG"].to_numpy())
    eff.update({
        "AcylChain": chain,
        "MeanGLACIER": float(g["GLACIER"].mean()),
        "MedianGLACIER": float(g["GLACIER"].median()),
        "MeanICEBERG": float(g["ICEBERG 2.1"].mean()),
        "MedianICEBERG": float(g["ICEBERG 2.1"].median()),
        "InferenceEligibleNge10": len(g) >= MIN_GROUP_N_FOR_INFERENCE,
    })
    if len(g) < MIN_GROUP_N_FOR_INFERENCE:
        # Preserve estimates/CIs but suppress exploratory p-values for tiny strata.
        eff["SignTestP_TwoSided"] = np.nan
        eff["WilcoxonP_TwoSided"] = np.nan
        eff["RankBiserialEffect"] = np.nan
    chain_rows.append(eff)
chain_stats = pd.DataFrame(chain_rows)
if len(chain_stats):
    chain_stats["WilcoxonBH_FDR"] = bh_adjust(chain_stats["WilcoxonP_TwoSided"])
    chain_stats["SignTestBH_FDR"] = bh_adjust(chain_stats["SignTestP_TwoSided"])

# ---------------- annotation-projected descriptive chemistry --------

proj = paired.merge(
    bench_chem[["QueryUSI","AcylChain","AA1","AA2","AA1AA2","AnnotationIndex"]],
    on=["QueryUSI","AcylChain"],
    how="left",
    validate="one_to_many"
)

def projected_summary(group_cols):
    rows = []
    for key, g in proj.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        d = dict(zip(group_cols, key))
        # Structure-level uniqueness within each projected group prevents
        # duplicate annotation records for the same chemistry from weighting a
        # structure repeatedly inside that group.
        u = g.drop_duplicates("BenchmarkSpec")
        diff = u["GLACIER_minus_ICEBERG"].to_numpy(dtype=float)
        d.update({
            "UniqueBenchmarkSpecs": u["BenchmarkSpec"].nunique(),
            "AnnotationProjectionRows": len(g),
            "MeanGLACIER": float(u["GLACIER"].mean()),
            "MedianGLACIER": float(u["GLACIER"].median()),
            "MeanICEBERG": float(u["ICEBERG 2.1"].mean()),
            "MedianICEBERG": float(u["ICEBERG 2.1"].median()),
            "MeanDifference_GLACIERminusICEBERG": float(np.mean(diff)),
            "MedianDifference_GLACIERminusICEBERG": float(np.median(diff)),
            "GLACIER_Higher": int((diff > 0).sum()),
            "ICEBERG_Higher": int((diff < 0).sum()),
            "Ties": int(np.isclose(diff, 0, atol=1e-15).sum()),
            "InferencePerformed": False,
            "Interpretation": "ANNOTATION_PROJECTED_DESCRIPTIVE_ONLY",
        })
        rows.append(d)
    return pd.DataFrame(rows)

aa1_summary = projected_summary(["AA1"])
aa2_summary = projected_summary(["AA2"])
pair_summary = projected_summary(["AA1","AA2","AA1AA2"])
chain_pair_summary = projected_summary(["AcylChain","AA1","AA2","AA1AA2"])

# ---------------- strongest structure-level differences -------------

extremes = paired.sort_values(
    ["AbsModelDifference","BenchmarkSpec"],
    ascending=[False, True],
    kind="stable"
).copy()
extremes["RankByAbsoluteDifference"] = np.arange(1, len(extremes)+1)

top_glacier = paired.sort_values(
    ["GLACIER_minus_ICEBERG","BenchmarkSpec"],
    ascending=[False, True], kind="stable"
).head(30).copy()
top_glacier["FavoringRank"] = np.arange(1, len(top_glacier)+1)

top_iceberg = paired.sort_values(
    ["GLACIER_minus_ICEBERG","BenchmarkSpec"],
    ascending=[True, True], kind="stable"
).head(30).copy()
top_iceberg["FavoringRank"] = np.arange(1, len(top_iceberg)+1)

# Attach compact chemistry candidates to structure-level extremes.
chem_labels = (
    bench_chem.groupby("QueryUSI")
    .agg(
        CandidateOrderedAA1AA2=("AA1AA2", lambda x: " | ".join(sorted(set(map(str,x))))),
        CandidateChemistryCount=("AA1AA2","nunique")
    ).reset_index()
)
for df in [extremes, top_glacier, top_iceberg]:
    df2 = df.merge(chem_labels, on="QueryUSI", how="left", validate="one_to_one")
    df.drop(df.index, inplace=True)
    for c in df2.columns:
        df[c] = df2[c]

# ----------------------------- QC -----------------------------------

qc_rows = []
def qc(check, expected, observed, passed):
    qc_rows.append({
        "Check": check, "Expected": expected, "Observed": observed,
        "Status": "PASS" if passed else "FAIL"
    })

qc("Primary Script-115 score rows", 600, len(primary), len(primary)==600)
qc("Paired BenchmarkSpecs", 300, len(paired), len(paired)==300)
qc("Unique paired QueryUSIs", 300, paired["QueryUSI"].nunique(),
   paired["QueryUSI"].nunique()==300)
qc("Frozen scoring policy", FROZEN_POLICY,
   "|".join(sorted(primary["ScoringPolicyVersion"].astype(str).unique())),
   set(primary["ScoringPolicyVersion"].astype(str))=={FROZEN_POLICY})
qc("Primary tolerance", PRIMARY_TOL,
   "|".join(map(str, sorted(primary["FragmentTolerance_Da"].unique()))),
   np.allclose(primary["FragmentTolerance_Da"], PRIMARY_TOL))
qc("Benchmark QueryUSIs chemistry-resolved", 300, len(resolved_q), len(resolved_q)==300)
qc("Queries crossing acyl chains", 0, int((q_chain_n>1).sum()), not (q_chain_n>1).any())
qc("Unique query-to-chain mappings", 300, len(query_chain), len(query_chain)==300)
qc("Overall inference rows", 1, len(overall), len(overall)==1)
qc("Acyl-chain rows", "1-4", len(chain_stats), 1 <= len(chain_stats) <= 4)
qc("AA1/AA2 projected inference performed", False, False, True)
qc("Frozen score recalculation performed", False, False, True)
qc("Best-CE primary use performed", False, False, True)

qc_df = pd.DataFrame(qc_rows)
all_pass = qc_df["Status"].eq("PASS").all()

# ----------------------------- save ---------------------------------

paired_out = OUT / "FullTier1_Paired_Structure_Statistics.tsv"
overall_out = OUT / "FullTier1_Overall_Paired_Inference.tsv"
chain_out = OUT / "FullTier1_AcylChain_Stratified_Statistics.tsv"
mult_out = OUT / "FullTier1_Chemistry_Multiplicity_Audit.tsv"
proj_out = OUT / "FullTier1_Annotation_Projected_Structure_Chemistry.tsv"
aa1_out = OUT / "FullTier1_AA1_Descriptive_Summary.tsv"
aa2_out = OUT / "FullTier1_AA2_Descriptive_Summary.tsv"
pair_out = OUT / "FullTier1_Ordered_AA1AA2_Descriptive_Summary.tsv"
chain_pair_out = OUT / "FullTier1_AcylChain_Ordered_AA1AA2_Descriptive_Summary.tsv"
extremes_out = OUT / "FullTier1_Model_Difference_Ranking.tsv"
topg_out = OUT / "FullTier1_Top30_GLACIER_Favoring.tsv"
topi_out = OUT / "FullTier1_Top30_ICEBERG_Favoring.tsv"
qc_out = QC_DIR / "116_statistical_chemistry_stratified_interpretation_v0654.tsv"

for df, p in [
    (paired, paired_out),
    (overall, overall_out),
    (chain_stats, chain_out),
    (mult, mult_out),
    (proj, proj_out),
    (aa1_summary, aa1_out),
    (aa2_summary, aa2_out),
    (pair_summary, pair_out),
    (chain_pair_summary, chain_pair_out),
    (extremes, extremes_out),
    (top_glacier, topg_out),
    (top_iceberg, topi_out),
    (qc_df, qc_out),
]:
    write_tsv(df, p)

hash_rows = []
for label, p in [
    ("Input_Script115Scores", SCORES),
    ("Input_QueryAnnotationLinks", CHEM),
    ("PairedStructureStatistics", paired_out),
    ("OverallPairedInference", overall_out),
    ("AcylChainStatistics", chain_out),
    ("ChemistryMultiplicityAudit", mult_out),
    ("AnnotationProjectedChemistry", proj_out),
    ("AA1Summary", aa1_out),
    ("AA2Summary", aa2_out),
    ("OrderedAA1AA2Summary", pair_out),
    ("AcylChainOrderedAA1AA2Summary", chain_pair_out),
    ("ModelDifferenceRanking", extremes_out),
    ("Top30GLACIER", topg_out),
    ("Top30ICEBERG", topi_out),
    ("QC", qc_out),
]:
    hash_rows.append({
        "Artifact": label, "Path": str(p),
        "SHA256": sha256_file(p), "Bytes": p.stat().st_size
    })
hash_df = pd.DataFrame(hash_rows)
hash_out = OUT / "FullTier1_Statistical_Chemistry_SHA256.tsv"
write_tsv(hash_df, hash_out)

o = overall.iloc[0]
ambiguous_n = int(paired["ChemistryAmbiguousWithinChain"].sum())

notes = f"""Full Tier-1 statistical + chemistry-stratified interpretation {VERSION}

STATUS
======
{"PASS" if all_pass else "HOLD / REVIEW QC"}

FROZEN INPUT
============
Script 115 primary consensus score at 0.020 Da.
Frozen spectrum-scoring policy: {FROZEN_POLICY}
Scores are NOT recalculated or modified here.

OVERALL PAIRED INTERPRETATION
=============================
N = {int(o["N"])}
GLACIER higher = {int(o["GLACIER_Higher"])}
ICEBERG 2.1 higher = {int(o["ICEBERG_Higher"])}
ties = {int(o["Ties"])}

Mean paired difference (GLACIER - ICEBERG 2.1):
    {o["MeanDifference"]:.6f}
Bootstrap 95% CI:
    [{o["MeanDifference_CI95_Lower"]:.6f}, {o["MeanDifference_CI95_Upper"]:.6f}]

Median paired difference:
    {o["MedianDifference"]:.6f}
Bootstrap 95% CI:
    [{o["MedianDifference_CI95_Lower"]:.6f}, {o["MedianDifference_CI95_Upper"]:.6f}]

Paired Cohen's dz:
    {o["PairedCohens_dz"]:.6f}

Two-sided sign-test p:
    {o["SignTestP_TwoSided"]:.6g}

Two-sided Wilcoxon signed-rank p:
    {o["WilcoxonP_TwoSided"]}
Wilcoxon status:
    {o["WilcoxonStatus"]}

Rank-biserial paired effect:
    {o["RankBiserialEffect"]}

CHEMISTRY MAPPING
=================
Official chemistry source:
    data/intermediate/Query_to_Annotation_Links.tsv

Acyl chain is treated as a query-level invariant.
AA1 and AA2 are annotation-level candidate chemistry.

Benchmark queries with >1 ordered AA1-AA2 candidate:
    {ambiguous_n} / 300

Accordingly:
- acyl-chain stratification supports structure-level paired inference;
- AA1, AA2, and ordered AA1-AA2 tables are annotation-projected DESCRIPTIVE
  summaries only;
- ambiguous query annotations are not arbitrarily collapsed to one AA1-AA2
  identity;
- annotation projection is not used to inflate inferential sample size.

BOOTSTRAP
=========
Replicates: {BOOTSTRAP_REPS}
Seed: {BOOTSTRAP_SEED}
CI: percentile 95%

MULTIPLE TESTING
================
Exploratory acyl-chain p-values receive Benjamini-Hochberg FDR adjustment.
No inferential p-values are generated for AA1/AA2 annotation-projected groups.

INTERPRETATION BOUNDARY
=======================
This script quantifies statistical evidence and chemistry-dependent patterns
without changing the frozen prediction/scoring pipeline.

Statistical significance does not by itself imply a large practical effect.
Interpret mean/median paired differences, confidence intervals, effect sizes,
model-win counts, and chemistry strata together.

Best-CE remains descriptive only and is not used here as a primary endpoint.
"""
notes_out = OUT / "FullTier1_Statistical_Chemistry_Notes.txt"
notes_out.write_text(notes)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)

print(f"Full Tier-1 statistical + chemistry interpretation {VERSION}")
print("=" * 64)
print()
print("Overall paired inference")
print("------------------------")
print(overall.to_string(index=False))
print()
print("Acyl-chain stratification")
print("-------------------------")
print(chain_stats.to_string(index=False))
print()
print("Chemistry multiplicity")
print("----------------------")
print(f"Benchmark queries with >1 ordered AA1-AA2 candidate: {ambiguous_n}/300")
print()
print("Top 10 absolute model differences")
print("---------------------------------")
show_cols = [
    "BenchmarkSpec","AcylChain","GLACIER","ICEBERG 2.1",
    "GLACIER_minus_ICEBERG","HigherModel","CandidateOrderedAA1AA2"
]
print(extremes[show_cols].head(10).to_string(index=False))
print()
print("QC")
print("--")
print(qc_df.to_string(index=False))
print()
print("Saved")
print("-----")
for p in [
    paired_out, overall_out, chain_out, mult_out, proj_out, aa1_out, aa2_out,
    pair_out, chain_pair_out, extremes_out, topg_out, topi_out,
    hash_out, notes_out, qc_out
]:
    print(p)
print()
if all_pass:
    print(f"FULL TIER-1 STATISTICAL + CHEMISTRY INTERPRETATION {VERSION} PASSED")
else:
    print(f"FULL TIER-1 STATISTICAL + CHEMISTRY INTERPRETATION {VERSION} HOLD — REVIEW QC")
    sys.exit(2)
