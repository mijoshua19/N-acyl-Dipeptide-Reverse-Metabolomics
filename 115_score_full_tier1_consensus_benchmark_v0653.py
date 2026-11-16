#!/usr/bin/env python3
"""
115_score_full_tier1_consensus_benchmark_v0653.py

Full Tier-1 GLACIER vs ICEBERG 2.1 consensus benchmark
Version: v0.6.5.3

Applies the already-frozen Script-110 / v0.6.4.8 scoring policy to:
  - Script 113c standardized full predictions
  - Script 114 standardized full experimental spectra

PRIMARY POLICY (FROZEN BEFORE FULL RESULTS)
-------------------------------------------
Predicted representation:
  1. CE-specific raw predicted intensities are sqrt-transformed.
  2. Each CE spectrum is independently L2-normalized.
  3. CE 25, 50, 75 each receive equal weight 1/3.
  4. Weighted CE peak lists are pooled.
  5. Adjacent-m/z single-linkage clustering at 0.020 Da.
  6. Cluster m/z = intensity-weighted centroid.
  7. Cluster intensity = sum of weighted CE contributions.
  8. Final consensus vector is L2-normalized.

Experimental representation:
  - measured stepped 25|50|75 eV spectrum;
  - already preprocessed by Script 114;
  - sqrt(raw intensity), then L2 normalization;
  - no fabricated single experimental CE.

Matching:
  - candidate peak pairs within absolute fragment tolerance;
  - deterministic greedy one-to-one matching;
  - descending transformed-intensity product;
  - then smaller absolute mass error;
  - then stable experimental/predicted indices.

Primary score:
  - consensus cosine at 0.020 Da.

Sensitivity:
  - consensus cosine at 0.010 and 0.050 Da.

Diagnostics:
  - CE-specific predicted-vs-experimental scores at 25/50/75 eV;
  - descriptive best CE only;
  - best CE is NEVER used as the primary benchmark score.

This script does not alter the frozen scoring policy after seeing full results.
"""

from __future__ import annotations
from pathlib import Path
import hashlib
import math
import sys

import numpy as np
import pandas as pd

VERSION = "v0.6.5.3"
POLICY_VERSION = "v0.6.4.8"
ROOT = Path.cwd().resolve()

PRED_DIR = ROOT / "data/processed/prediction_v0651_standardized_full_tier1"
PRED_META = PRED_DIR / "MSPRED_FullTier1_Prediction_Record_Metadata.tsv"
PRED_PEAKS = PRED_DIR / "MSPRED_FullTier1_Prediction_Peaks_Long.tsv"

EXP_DIR = ROOT / "data/processed/prediction_v0652_full_tier1_experimental"
EXP_META = EXP_DIR / "FullTier1_Experimental_Spectrum_Metadata.tsv"
EXP_PEAKS = EXP_DIR / "FullTier1_Experimental_Peaks_Long.tsv"

OUT = ROOT / "data/processed/prediction_v0653_full_tier1_consensus_benchmark"
QC_DIR = ROOT / "qc"
OUT.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["GLACIER", "ICEBERG 2.1"]
CES = [25.0, 50.0, 75.0]
PRIMARY_TOL = 0.020
TOLS = [0.010, 0.020, 0.050]
CLUSTER_TOL = 0.020
PRECURSOR_OFFSET = 0.5
EXPECTED_SPECS = 300
EXPECTED_PRED_RECORDS = 1800

def require(p):
    if not p.exists():
        raise FileNotFoundError(f"Required input missing: {p}")

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def write_tsv(df, p):
    df.to_csv(p, sep="\t", index=False)

def as_bool_series(s):
    return s.astype(str).str.lower().isin(["true", "1", "yes"])

for p in [PRED_META, PRED_PEAKS, EXP_META, EXP_PEAKS]:
    require(p)

pm = pd.read_csv(PRED_META, sep="\t", low_memory=False)
pp = pd.read_csv(PRED_PEAKS, sep="\t", low_memory=False)
em = pd.read_csv(EXP_META, sep="\t", low_memory=False)
ep = pd.read_csv(EXP_PEAKS, sep="\t", low_memory=False)

# ---------------------------- input gates ----------------------------

if len(pm) != EXPECTED_PRED_RECORDS:
    raise RuntimeError(f"Expected 1800 prediction metadata rows; observed {len(pm)}.")
if len(em) != EXPECTED_SPECS:
    raise RuntimeError(f"Expected 300 experimental metadata rows; observed {len(em)}.")
if em["BenchmarkSpec"].nunique() != EXPECTED_SPECS:
    raise RuntimeError("Experimental BenchmarkSpec is not unique.")
if set(pm["Model"].unique()) != set(MODELS):
    raise RuntimeError(f"Unexpected prediction models: {sorted(pm['Model'].unique())}")
if "CrosswalkResolved" in pm.columns and not as_bool_series(pm["CrosswalkResolved"]).all():
    raise RuntimeError("Prediction metadata contains unresolved crosswalks.")
if "ExperimentalSingleCEAssigned" in em.columns and as_bool_series(em["ExperimentalSingleCEAssigned"]).any():
    raise RuntimeError("Experimental metadata contains fabricated single-CE assignments.")
if "ScoringPolicyDependency" in em.columns and set(em["ScoringPolicyDependency"].astype(str)) != {POLICY_VERSION}:
    raise RuntimeError("Experimental scoring-policy dependency changed.")

# Numeric coercion and finite-positive gates.
for df, mzcol, icol in [
    (pp, "PeakMz", "PredictedIntensityRaw"),
    (ep, "PeakMz", "ExperimentalIntensityRaw"),
]:
    df[mzcol] = pd.to_numeric(df[mzcol], errors="coerce")
    df[icol] = pd.to_numeric(df[icol], errors="coerce")
    good = np.isfinite(df[mzcol]) & np.isfinite(df[icol]) & (df[mzcol] > 0) & (df[icol] > 0)
    if not good.all():
        raise RuntimeError(f"Non-finite/non-positive peaks found in {mzcol}/{icol}.")

pm["PredictionCollisionEnergy_eV"] = pd.to_numeric(pm["PredictionCollisionEnergy_eV"], errors="coerce")
if sorted(pm["PredictionCollisionEnergy_eV"].dropna().unique().tolist()) != CES:
    raise RuntimeError("Prediction CE set is not exactly 25|50|75 eV.")

# Cross-input identity gate.
pred_specs = set(pm["BenchmarkSpec"].astype(str))
exp_specs = set(em["BenchmarkSpec"].astype(str))
if pred_specs != exp_specs or len(pred_specs) != 300:
    raise RuntimeError(
        f"Prediction/experimental BenchmarkSpec mismatch: "
        f"pred={len(pred_specs)}, exp={len(exp_specs)}, intersection={len(pred_specs & exp_specs)}"
    )

# True QueryUSI must agree.
pmap = pm[["BenchmarkSpec", "QueryUSI"]].drop_duplicates()
if len(pmap) != 300:
    raise RuntimeError("Prediction BenchmarkSpec->QueryUSI mapping is not one-to-one.")
emap = em[["BenchmarkSpec", "QueryUSI"]].drop_duplicates()
merged_ids = pmap.merge(emap, on="BenchmarkSpec", suffixes=("_Pred", "_Exp"), validate="one_to_one")
if not (merged_ids["QueryUSI_Pred"].astype(str) == merged_ids["QueryUSI_Exp"].astype(str)).all():
    raise RuntimeError("Prediction and experimental QueryUSI mappings disagree.")

# -------------------------- scoring helpers --------------------------

def normalize_sqrt(mz, raw_int):
    mz = np.asarray(mz, dtype=float)
    raw = np.asarray(raw_int, dtype=float)
    keep = np.isfinite(mz) & np.isfinite(raw) & (mz > 0) & (raw > 0)
    mz, raw = mz[keep], raw[keep]
    order = np.argsort(mz, kind="stable")
    mz, raw = mz[order], raw[order]
    inten = np.sqrt(raw)
    norm = float(np.linalg.norm(inten))
    if norm <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    return mz, inten / norm

def l2_normalize_existing(mz, intensity):
    mz = np.asarray(mz, dtype=float)
    it = np.asarray(intensity, dtype=float)
    keep = np.isfinite(mz) & np.isfinite(it) & (mz > 0) & (it > 0)
    mz, it = mz[keep], it[keep]
    order = np.argsort(mz, kind="stable")
    mz, it = mz[order], it[order]
    norm = float(np.linalg.norm(it))
    if norm <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    return mz, it / norm

def build_consensus(ce_spectra):
    """
    ce_spectra: dict CE -> (mz, raw predicted intensity)

    Frozen policy:
      sqrt -> CE-wise L2 -> weight 1/3 -> pool -> adjacent single-linkage
      cluster at 0.020 Da -> weighted centroid -> summed intensity -> final L2.
    """
    pooled = []
    for ce in CES:
        mz, norm = normalize_sqrt(*ce_spectra[ce])
        if len(mz) == 0:
            raise RuntimeError(f"CE {ce:g} spectrum has no positive peaks.")
        weighted = norm / 3.0
        for local_idx, (m, i) in enumerate(zip(mz, weighted)):
            pooled.append((float(m), float(i), float(ce), int(local_idx)))

    # Stable deterministic ordering.
    pooled.sort(key=lambda x: (x[0], x[2], x[3]))
    clusters = []
    current = [pooled[0]]
    prev_mz = pooled[0][0]

    # Single-linkage on ADJACENT sorted m/z distances.
    for peak in pooled[1:]:
        if peak[0] - prev_mz <= CLUSTER_TOL + 1e-12:
            current.append(peak)
        else:
            clusters.append(current)
            current = [peak]
        prev_mz = peak[0]
    clusters.append(current)

    out_mz, out_int, cluster_rows = [], [], []
    for ci, cl in enumerate(clusters):
        mzs = np.array([x[0] for x in cl], dtype=float)
        ints = np.array([x[1] for x in cl], dtype=float)
        total = float(ints.sum())
        centroid = float(np.sum(mzs * ints) / total)
        out_mz.append(centroid)
        out_int.append(total)
        cluster_rows.append({
            "ConsensusClusterIndex": ci,
            "ConsensusMz": centroid,
            "PreFinalL2Intensity": total,
            "ContributingPeaks": len(cl),
            "ContributingCEs_eV": "|".join(
                str(int(x)) for x in sorted(set(y[2] for y in cl))
            ),
            "ClusterMinMz": float(mzs.min()),
            "ClusterMaxMz": float(mzs.max()),
        })

    out_mz = np.asarray(out_mz, dtype=float)
    out_int = np.asarray(out_int, dtype=float)
    final_norm = float(np.linalg.norm(out_int))
    if final_norm <= 0:
        raise RuntimeError("Consensus has zero L2 norm.")
    out_int = out_int / final_norm
    for r, val in zip(cluster_rows, out_int):
        r["ConsensusIntensityL2"] = float(val)
    return out_mz, out_int, cluster_rows

def greedy_score(exp_mz, exp_i, pred_mz, pred_i, tol):
    """
    Deterministic one-to-one matching frozen in Script 109/110.
    Both input intensity vectors are already L2 normalized.
    """
    exp_mz = np.asarray(exp_mz, dtype=float)
    exp_i = np.asarray(exp_i, dtype=float)
    pred_mz = np.asarray(pred_mz, dtype=float)
    pred_i = np.asarray(pred_i, dtype=float)

    candidates = []
    # spectra are small enough that deterministic exhaustive candidate creation
    # is transparent and reproducible.
    for ei, emz in enumerate(exp_mz):
        lo = np.searchsorted(pred_mz, emz - tol, side="left")
        hi = np.searchsorted(pred_mz, emz + tol, side="right")
        for pi in range(lo, hi):
            err = abs(float(emz - pred_mz[pi]))
            prod = float(exp_i[ei] * pred_i[pi])
            candidates.append((-prod, err, ei, pi))

    candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    used_e, used_p, matches = set(), set(), []
    for negprod, err, ei, pi in candidates:
        if ei in used_e or pi in used_p:
            continue
        used_e.add(ei); used_p.add(pi)
        prod = -negprod
        matches.append((ei, pi, prod, err))

    cosine = float(sum(x[2] for x in matches))
    exp_frac = float(sum(exp_i[x[0]] ** 2 for x in matches))
    pred_frac = float(sum(pred_i[x[1]] ** 2 for x in matches))
    errors = [x[3] for x in matches]

    return {
        "Cosine": cosine,
        "MatchedPeaks": len(matches),
        "MatchedExperimentalSqrtIntensityFraction": exp_frac,
        "MatchedPredictedSqrtIntensityFraction": pred_frac,
        "MedianAbsMassError_Da": float(np.median(errors)) if errors else np.nan,
        "MaxAbsMassError_Da": float(np.max(errors)) if errors else np.nan,
    }

# Index input tables.
exp_peak_groups = {k: g for k, g in ep.groupby("BenchmarkSpec", sort=False)}
pred_peak_groups = {
    (m, s, float(ce)): g
    for (m, s, ce), g in pp.groupby(
        ["Model", "BenchmarkSpec", "PredictionCollisionEnergy_eV"], sort=False
    )
}

master_info = (
    em.set_index("BenchmarkSpec")[
        ["QueryUSI", "GNPS2LIB_ACCESSION", "BenchmarkPrecursorMz", "Adduct"]
    ].to_dict("index")
)

consensus_rows = []
cluster_rows_all = []
ce_rows = []
primary_rows = []

spec_order = sorted(exp_specs)

print(f"Full Tier-1 consensus benchmark {VERSION}")
print("=" * 50)
print(f"Benchmark structures: {len(spec_order)}")
print(f"Frozen policy: {POLICY_VERSION}")
print()

for si, spec in enumerate(spec_order, start=1):
    info = master_info[spec]
    eg = exp_peak_groups.get(spec)
    if eg is None or len(eg) == 0:
        raise RuntimeError(f"{spec}: experimental peaks missing.")

    # Recompute from RAW intensity to guarantee Script 115 is not trusting a
    # precomputed score-like vector blindly.
    exp_mz, exp_norm = normalize_sqrt(
        eg["PeakMz"].to_numpy(),
        eg["ExperimentalIntensityRaw"].to_numpy()
    )
    if len(exp_mz) == 0:
        raise RuntimeError(f"{spec}: experimental spectrum has zero normalized peaks.")

    precursor = float(info["BenchmarkPrecursorMz"])
    if np.any(exp_mz > precursor + PRECURSOR_OFFSET + 1e-12):
        raise RuntimeError(f"{spec}: experimental peak exceeds frozen precursor ceiling.")

    for model in MODELS:
        ce_spectra = {}
        for ce in CES:
            pg = pred_peak_groups.get((model, spec, ce))
            if pg is None or len(pg) == 0:
                raise RuntimeError(f"{model} {spec} CE={ce:g}: prediction missing.")
            # Defensive ceiling application, matching frozen policy.
            pg2 = pg[
                pd.to_numeric(pg["PeakMz"], errors="coerce")
                <= precursor + PRECURSOR_OFFSET + 1e-12
            ]
            ce_spectra[ce] = (
                pg2["PeakMz"].to_numpy(dtype=float),
                pg2["PredictedIntensityRaw"].to_numpy(dtype=float),
            )

            # CE-specific diagnostic score at primary 0.020 Da.
            pmz, pnorm = normalize_sqrt(*ce_spectra[ce])
            d = greedy_score(exp_mz, exp_norm, pmz, pnorm, PRIMARY_TOL)
            ce_rows.append({
                "Model": model,
                "BenchmarkSpec": spec,
                "QueryUSI": info["QueryUSI"],
                "GNPS2LIB_ACCESSION": info["GNPS2LIB_ACCESSION"],
                "PredictionCE_eV": int(ce),
                "FragmentTolerance_Da": PRIMARY_TOL,
                "CosineSqrtIntensity": d["Cosine"],
                "MatchedPeaks": d["MatchedPeaks"],
                "MatchedExperimentalSqrtIntensityFraction":
                    d["MatchedExperimentalSqrtIntensityFraction"],
                "MatchedPredictedSqrtIntensityFraction":
                    d["MatchedPredictedSqrtIntensityFraction"],
                "MedianAbsMassError_Da": d["MedianAbsMassError_Da"],
                "MaxAbsMassError_Da": d["MaxAbsMassError_Da"],
                "DiagnosticOnly": True,
                "BestCEPrimaryUseAllowed": False,
                "ScoringPolicyVersion": POLICY_VERSION,
            })

        cmz, cint, clusters = build_consensus(ce_spectra)

        for r in clusters:
            r.update({
                "Model": model,
                "BenchmarkSpec": spec,
                "QueryUSI": info["QueryUSI"],
                "ScoringPolicyVersion": POLICY_VERSION,
            })
            cluster_rows_all.append(r)

        consensus_rows.append({
            "Model": model,
            "BenchmarkSpec": spec,
            "QueryUSI": info["QueryUSI"],
            "GNPS2LIB_ACCESSION": info["GNPS2LIB_ACCESSION"],
            "ConsensusPeakCount": len(cmz),
            "ExperimentalPeakCount": len(exp_mz),
            "PredictionCEs_eV": "25|50|75",
            "CEWeighting": "Equal_1over3",
            "ConsensusClusterTolerance_Da": CLUSTER_TOL,
            "ScoringPolicyVersion": POLICY_VERSION,
        })

        for tol in TOLS:
            d = greedy_score(exp_mz, exp_norm, cmz, cint, tol)
            primary_rows.append({
                "Model": model,
                "BenchmarkSpec": spec,
                "QueryUSI": info["QueryUSI"],
                "GNPS2LIB_ACCESSION": info["GNPS2LIB_ACCESSION"],
                "BenchmarkPrecursorMz": precursor,
                "Adduct": info["Adduct"],
                "PredictedRepresentation": "EqualWeight25_50_75Consensus",
                "ExperimentalRepresentation": "MeasuredStepped25_50_75",
                "FragmentTolerance_Da": tol,
                "IsPrimaryTolerance": bool(abs(tol - PRIMARY_TOL) < 1e-12),
                "CosineSqrtIntensity": d["Cosine"],
                "MatchedPeaks": d["MatchedPeaks"],
                "MatchedExperimentalSqrtIntensityFraction":
                    d["MatchedExperimentalSqrtIntensityFraction"],
                "MatchedPredictedSqrtIntensityFraction":
                    d["MatchedPredictedSqrtIntensityFraction"],
                "MedianAbsMassError_Da": d["MedianAbsMassError_Da"],
                "MaxAbsMassError_Da": d["MaxAbsMassError_Da"],
                "ScoringPolicyVersion": POLICY_VERSION,
                "BestCEPrimaryUseAllowed": False,
            })

    if si == 1 or si % 25 == 0 or si == len(spec_order):
        print(f"[{si:03d}/300] scored {spec}", flush=True)

scores = pd.DataFrame(primary_rows)
ce_diag = pd.DataFrame(ce_rows)
consensus = pd.DataFrame(consensus_rows)
clusters = pd.DataFrame(cluster_rows_all)

# ------------------------ descriptive best CE ------------------------

best_ce_rows = []
for (model, spec), g in ce_diag.groupby(["Model", "BenchmarkSpec"], sort=True):
    # Deterministic tie-break: higher cosine, then lower CE.
    z = g.sort_values(
        ["CosineSqrtIntensity", "PredictionCE_eV"],
        ascending=[False, True],
        kind="stable"
    ).iloc[0]
    best_ce_rows.append({
        "Model": model,
        "BenchmarkSpec": spec,
        "QueryUSI": z["QueryUSI"],
        "DescriptiveBestCE_eV": int(z["PredictionCE_eV"]),
        "DescriptiveBestCECosine": float(z["CosineSqrtIntensity"]),
        "PrimaryUseAllowed": False,
        "Interpretation": "DESCRIPTIVE_DIAGNOSTIC_ONLY",
    })
best_ce = pd.DataFrame(best_ce_rows)

# ---------------------------- summaries -----------------------------

summary_rows = []
for (model, tol), g in scores.groupby(["Model", "FragmentTolerance_Da"], sort=True):
    x = g["CosineSqrtIntensity"].to_numpy(dtype=float)
    summary_rows.append({
        "Model": model,
        "FragmentTolerance_Da": tol,
        "IsPrimaryTolerance": bool(abs(float(tol) - PRIMARY_TOL) < 1e-12),
        "N": len(x),
        "MeanCosine": float(np.mean(x)),
        "MedianCosine": float(np.median(x)),
        "SDCosine": float(np.std(x, ddof=1)),
        "Q1Cosine": float(np.quantile(x, 0.25)),
        "Q3Cosine": float(np.quantile(x, 0.75)),
        "MinCosine": float(np.min(x)),
        "MaxCosine": float(np.max(x)),
    })
model_summary = pd.DataFrame(summary_rows)

# Paired model differences at each tolerance.
paired_rows = []
for tol in TOLS:
    z = scores[np.isclose(scores["FragmentTolerance_Da"], tol)].pivot(
        index="BenchmarkSpec", columns="Model", values="CosineSqrtIntensity"
    )
    if set(z.columns) != set(MODELS) or len(z) != 300:
        raise RuntimeError(f"Paired comparison incomplete at tolerance {tol}.")
    diff = z["GLACIER"] - z["ICEBERG 2.1"]
    paired_rows.append({
        "FragmentTolerance_Da": tol,
        "IsPrimaryTolerance": bool(abs(tol - PRIMARY_TOL) < 1e-12),
        "NPaired": len(diff),
        "GLACIER_Higher": int((diff > 0).sum()),
        "ICEBERG_Higher": int((diff < 0).sum()),
        "Ties": int(np.isclose(diff, 0.0, atol=1e-15).sum()),
        "Mean_GLACIER_minus_ICEBERG": float(diff.mean()),
        "Median_GLACIER_minus_ICEBERG": float(diff.median()),
        "Q1_Difference": float(diff.quantile(0.25)),
        "Q3_Difference": float(diff.quantile(0.75)),
        "Min_Difference": float(diff.min()),
        "Max_Difference": float(diff.max()),
    })
paired_summary = pd.DataFrame(paired_rows)

# CE diagnostic summary.
ce_summary = (
    ce_diag.groupby(["Model", "PredictionCE_eV"])["CosineSqrtIntensity"]
    .agg(N="count", MeanCosine="mean", MedianCosine="median",
         MinCosine="min", MaxCosine="max")
    .reset_index()
)
best_ce_counts = (
    best_ce.groupby(["Model", "DescriptiveBestCE_eV"])
    .size().rename("Count").reset_index()
)

# Tolerance stability: model median shifts relative to primary.
stability_rows = []
for model in MODELS:
    m = model_summary[model_summary["Model"].eq(model)].set_index("FragmentTolerance_Da")
    primary_med = float(m.loc[PRIMARY_TOL, "MedianCosine"])
    for tol in TOLS:
        med = float(m.loc[tol, "MedianCosine"])
        stability_rows.append({
            "Model": model,
            "FragmentTolerance_Da": tol,
            "MedianCosine": med,
            "Primary020MedianCosine": primary_med,
            "MedianShiftVs020": med - primary_med,
            "AbsMedianShiftVs020": abs(med - primary_med),
        })
stability = pd.DataFrame(stability_rows)

# ------------------------------- QC --------------------------------

qc_rows = []
def qc(check, expected, observed, passed):
    qc_rows.append({
        "Check": check,
        "Expected": expected,
        "Observed": observed,
        "Status": "PASS" if passed else "FAIL",
    })

qc("Prediction metadata rows", 1800, len(pm), len(pm) == 1800)
qc("Experimental metadata rows", 300, len(em), len(em) == 300)
qc("Shared BenchmarkSpec", 300, len(pred_specs & exp_specs), pred_specs == exp_specs and len(exp_specs) == 300)
qc("QueryUSI identity agreement", 300,
   int((merged_ids["QueryUSI_Pred"].astype(str) == merged_ids["QueryUSI_Exp"].astype(str)).sum()),
   (merged_ids["QueryUSI_Pred"].astype(str) == merged_ids["QueryUSI_Exp"].astype(str)).all())
qc("Consensus spectra", 600, len(consensus), len(consensus) == 600)
qc("Consensus tolerance-score rows", 1800, len(scores), len(scores) == 1800)
qc("Primary 0.020-Da score rows", 600,
   int(scores["IsPrimaryTolerance"].sum()), int(scores["IsPrimaryTolerance"].sum()) == 600)
qc("CE-specific diagnostic rows", 1800, len(ce_diag), len(ce_diag) == 1800)
qc("Descriptive best-CE rows", 600, len(best_ce), len(best_ce) == 600)
qc("Best-CE primary use", False, bool(best_ce["PrimaryUseAllowed"].any()),
   not best_ce["PrimaryUseAllowed"].any())
qc("All consensus cosine finite", True,
   bool(np.isfinite(scores["CosineSqrtIntensity"]).all()),
   np.isfinite(scores["CosineSqrtIntensity"]).all())
qc("All consensus cosine in [0,1]", True,
   bool(((scores["CosineSqrtIntensity"] >= -1e-12) &
         (scores["CosineSqrtIntensity"] <= 1 + 1e-12)).all()),
   ((scores["CosineSqrtIntensity"] >= -1e-12) &
    (scores["CosineSqrtIntensity"] <= 1 + 1e-12)).all())
qc("All CE diagnostic cosine finite", True,
   bool(np.isfinite(ce_diag["CosineSqrtIntensity"]).all()),
   np.isfinite(ce_diag["CosineSqrtIntensity"]).all())
qc("Model/tolerance N", "300 each",
   "|".join(
       f"{r.Model}:{r.FragmentTolerance_Da:.3f}={int(r.N)}"
       for r in model_summary.itertuples()
   ),
   bool((model_summary["N"] == 300).all()))
qc("Frozen scoring policy", POLICY_VERSION,
   "|".join(sorted(scores["ScoringPolicyVersion"].unique())),
   set(scores["ScoringPolicyVersion"]) == {POLICY_VERSION})
qc("Primary representation", "EqualWeight25_50_75Consensus",
   "|".join(sorted(scores["PredictedRepresentation"].unique())),
   set(scores["PredictedRepresentation"]) == {"EqualWeight25_50_75Consensus"})
qc("Experimental representation", "MeasuredStepped25_50_75",
   "|".join(sorted(scores["ExperimentalRepresentation"].unique())),
   set(scores["ExperimentalRepresentation"]) == {"MeasuredStepped25_50_75"})

qc_df = pd.DataFrame(qc_rows)
all_pass = qc_df["Status"].eq("PASS").all()

# ------------------------------ save --------------------------------

scores_out = OUT / "FullTier1_Consensus_Scores.tsv"
consensus_out = OUT / "FullTier1_Consensus_Spectrum_Metadata.tsv"
clusters_out = OUT / "FullTier1_Consensus_Clusters.tsv"
ce_out = OUT / "FullTier1_CE_Specific_Diagnostic_Scores.tsv"
best_ce_out = OUT / "FullTier1_Descriptive_BestCE.tsv"
summary_out = OUT / "FullTier1_Model_Summary.tsv"
paired_out = OUT / "FullTier1_Paired_Model_Differences.tsv"
ce_summary_out = OUT / "FullTier1_CE_Diagnostic_Summary.tsv"
best_ce_counts_out = OUT / "FullTier1_Descriptive_BestCE_Counts.tsv"
stability_out = OUT / "FullTier1_Tolerance_Stability.tsv"
qc_out = QC_DIR / "115_score_full_tier1_consensus_benchmark_v0653.tsv"

for df, p in [
    (scores, scores_out),
    (consensus, consensus_out),
    (clusters, clusters_out),
    (ce_diag, ce_out),
    (best_ce, best_ce_out),
    (model_summary, summary_out),
    (paired_summary, paired_out),
    (ce_summary, ce_summary_out),
    (best_ce_counts, best_ce_counts_out),
    (stability, stability_out),
    (qc_df, qc_out),
]:
    write_tsv(df, p)

hash_rows = []
for label, p in [
    ("Input_PredictionMetadata", PRED_META),
    ("Input_PredictionPeaks", PRED_PEAKS),
    ("Input_ExperimentalMetadata", EXP_META),
    ("Input_ExperimentalPeaks", EXP_PEAKS),
    ("ConsensusScores", scores_out),
    ("ConsensusSpectrumMetadata", consensus_out),
    ("ConsensusClusters", clusters_out),
    ("CESpecificDiagnostics", ce_out),
    ("DescriptiveBestCE", best_ce_out),
    ("ModelSummary", summary_out),
    ("PairedModelDifferences", paired_out),
    ("CEDiagnosticSummary", ce_summary_out),
    ("BestCECounts", best_ce_counts_out),
    ("ToleranceStability", stability_out),
    ("QC", qc_out),
]:
    hash_rows.append({
        "Artifact": label,
        "Path": str(p),
        "SHA256": sha256_file(p),
        "Bytes": p.stat().st_size,
    })
hash_df = pd.DataFrame(hash_rows)
hash_out = OUT / "FullTier1_Benchmark_SHA256.tsv"
write_tsv(hash_df, hash_out)

# Primary descriptive values for notes.
primary_summary = model_summary[model_summary["IsPrimaryTolerance"]].copy()
primary_pair = paired_summary[paired_summary["IsPrimaryTolerance"]].iloc[0]
max_shift = float(stability["AbsMedianShiftVs020"].max())

notes = f"""Full Tier-1 consensus benchmark {VERSION}

STATUS
======
{"PASS" if all_pass else "HOLD / REVIEW QC"}

FROZEN POLICY
=============
Policy version: {POLICY_VERSION}

Predicted primary representation:
    EqualWeight25_50_75Consensus

Prediction CE processing:
    sqrt(raw intensity)
    independent CE-specific L2 normalization
    CE weights = 1/3, 1/3, 1/3
    pool CE-specific peaks
    adjacent-m/z single-linkage clustering at 0.020 Da
    cluster m/z = intensity-weighted centroid
    cluster intensity = summed weighted CE contribution
    final consensus L2 normalization

Experimental representation:
    measured stepped 25|50|75 eV
    sqrt(raw intensity)
    L2 normalization
    no fabricated experimental single CE

Peak matching:
    deterministic greedy one-to-one
    descending transformed-intensity product
    smaller absolute mass error
    stable index tie-break

Primary fragment tolerance:
    0.020 Da

Sensitivity tolerances:
    0.010 Da
    0.050 Da

Best-CE primary use:
    PROHIBITED
    CE-specific and best-CE outputs are descriptive diagnostics only.

PRIMARY FULL-TIER1 RESULTS
==========================
{primary_summary.to_string(index=False)}

Paired GLACIER - ICEBERG 2.1 at 0.020 Da:
    N = {int(primary_pair["NPaired"])}
    GLACIER higher = {int(primary_pair["GLACIER_Higher"])}
    ICEBERG higher = {int(primary_pair["ICEBERG_Higher"])}
    ties = {int(primary_pair["Ties"])}
    mean difference = {primary_pair["Mean_GLACIER_minus_ICEBERG"]:.6f}
    median difference = {primary_pair["Median_GLACIER_minus_ICEBERG"]:.6f}

Tolerance stability:
    maximum absolute model-median shift relative to 0.020 Da = {max_shift:.6f}

INTERPRETATION BOUNDARY
=======================
These are full frozen Tier-1 descriptive benchmark results.

This script intentionally does NOT convert the observed paired model difference
into an inferential superiority claim. Any inferential analysis, confidence
interval, resampling analysis, structural-stratified analysis, or manuscript
claim should be a separately versioned downstream step and must not modify the
frozen v0.6.4.8 scoring policy.

High experimental-vs-predicted cosine is not expected to be equivalent to the
earlier experimental-library-vs-public FASST cosine >=0.90 criterion; those are
different comparison tasks.

PROVENANCE
==========
Prediction metadata SHA256:
    {sha256_file(PRED_META)}
Prediction peaks SHA256:
    {sha256_file(PRED_PEAKS)}
Experimental metadata SHA256:
    {sha256_file(EXP_META)}
Experimental peaks SHA256:
    {sha256_file(EXP_PEAKS)}
"""
notes_out = OUT / "FullTier1_Benchmark_Notes.txt"
notes_out.write_text(notes)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)

print()
print("Primary model summary — 0.020 Da")
print("--------------------------------")
print(primary_summary.to_string(index=False))
print()
print("Paired model difference — GLACIER minus ICEBERG 2.1")
print("----------------------------------------------------")
print(pd.DataFrame([primary_pair]).to_string(index=False))
print()
print("Tolerance summary")
print("-----------------")
print(model_summary.to_string(index=False))
print()
print("CE-specific diagnostic summary")
print("------------------------------")
print(ce_summary.to_string(index=False))
print()
print("Descriptive best-CE counts")
print("--------------------------")
print(best_ce_counts.to_string(index=False))
print()
print("QC")
print("--")
print(qc_df.to_string(index=False))
print()
print("Saved")
print("-----")
for p in [
    scores_out, consensus_out, clusters_out, ce_out, best_ce_out,
    summary_out, paired_out, ce_summary_out, best_ce_counts_out,
    stability_out, hash_out, notes_out, qc_out
]:
    print(p)
print()
if all_pass:
    print(f"FULL TIER-1 CONSENSUS BENCHMARK {VERSION} PASSED")
else:
    print(f"FULL TIER-1 CONSENSUS BENCHMARK {VERSION} HOLD — REVIEW QC")
    sys.exit(2)
