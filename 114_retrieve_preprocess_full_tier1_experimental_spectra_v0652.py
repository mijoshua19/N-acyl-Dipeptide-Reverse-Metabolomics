#!/usr/bin/env python3
"""
114_retrieve_preprocess_full_tier1_experimental_spectra_v0652.py

Full Tier-1 experimental-spectrum retrieval + preprocessing
Version: v0.6.5.2

SCOPE
-----
Retrieves and caches the 300 true experimental QueryUSIs frozen in Script 111,
parses the GNPS2 USI JSON response, applies the experimental preprocessing
policy already established in Script 109, and freezes standardized experimental
spectra for downstream Script 115.

THIS SCRIPT DOES NOT:
- read predicted peak intensities for scoring;
- build the 25/50/75 predicted consensus;
- calculate cosine similarity;
- select a best prediction CE;
- compare/rank GLACIER vs ICEBERG.

FROZEN EXPERIMENTAL PREPROCESSING
---------------------------------
- source: https://metabolomics-usi.gnps2.org/json/?usi1=<USI>
- finite m/z and intensity only
- m/z > 0 and intensity > 0
- remove experimental peaks above benchmark precursor m/z + 0.5 Da
- precursor peak is therefore retained
- no top-N truncation
- preserve raw intensity
- additionally store sqrt(raw intensity) and spectrum-level L2-normalized
  sqrt intensity for downstream frozen scoring
- experimental acquisition remains stepped 25|50|75 eV
- ExperimentalSingleCEAssigned = False
- no fabricated experimental single CE

EXPECTED POPULATION
-------------------
300 BenchmarkSpec
300 unique QueryUSI
one cached raw JSON response per QueryUSI
one standardized experimental spectrum per BenchmarkSpec

RESTART SAFETY
--------------
Existing raw JSON is reused only if:
- it parses as JSON;
- it contains a recoverable peak list;
- its SHA256 is recorded from the actual bytes.

Failed/incomplete retrievals do not authorize Script 115.

OUTPUT
------
data/processed/prediction_v0652_full_tier1_experimental/
    experimental_raw_json/
    FullTier1_Experimental_Spectrum_Metadata.tsv
    FullTier1_Experimental_Peaks_Long.tsv
    FullTier1_Experimental_Retrieval_Manifest.tsv
    FullTier1_Experimental_Precursor_Audit.tsv
    FullTier1_Experimental_SHA256.tsv
    FullTier1_Experimental_Notes.txt
qc/
    114_retrieve_preprocess_full_tier1_experimental_spectra_v0652.tsv
"""

from __future__ import annotations
from pathlib import Path
import hashlib
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
import urllib.error

import numpy as np
import pandas as pd

VERSION = "v0.6.5.2"
ROOT = Path.cwd().resolve()

FREEZE = ROOT / "data/processed/prediction_v0649_full_tier1_execution_freeze"
MASTER = FREEZE / "MASTER/FullTier1_Benchmark_Master.tsv"

STD113 = ROOT / "data/processed/prediction_v0651_standardized_full_tier1"
META113 = STD113 / "MSPRED_FullTier1_Prediction_Record_Metadata.tsv"

OUT = ROOT / "data/processed/prediction_v0652_full_tier1_experimental"
RAW = OUT / "experimental_raw_json"
QC_DIR = ROOT / "qc"
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)

ENDPOINT = "https://metabolomics-usi.gnps2.org/json/"
TIMEOUT_SECONDS = 90
MAX_ATTEMPTS = 4
RETRY_SLEEP_SECONDS = 2.0
PRECURSOR_CEILING_OFFSET_DA = 0.5

EXPECTED_SPECS = 300
EXPECTED_UNIQUE_USI = 300

def require(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Required input missing: {p}")
    return p

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def write_tsv(df: pd.DataFrame, p: Path):
    df.to_csv(p, sep="\t", index=False)

def cache_name(usi: str) -> str:
    return hashlib.sha256(usi.encode("utf-8")).hexdigest()[:24] + ".json"

def finite_float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def recursive_find(obj, keys):
    """Return first value for a case-insensitive key anywhere in nested JSON."""
    wanted = {k.lower() for k in keys}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in wanted:
                return v
        for v in obj.values():
            found = recursive_find(v, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = recursive_find(v, keys)
            if found is not None:
                return found
    return None

def parse_peak_pairs(value):
    """Accept common GNPS USI JSON peak encodings."""
    if value is None:
        return None

    # [[mz,intensity], ...]
    if isinstance(value, list):
        out = []
        for row in value:
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                mz, inten = finite_float(row[0]), finite_float(row[1])
                if mz is not None and inten is not None:
                    out.append((mz, inten))
            elif isinstance(row, dict):
                mz = None
                inten = None
                for k in ["mz", "m/z", "mass", "fragment_mz"]:
                    if k in row:
                        mz = finite_float(row[k]); break
                for k in ["intensity", "i", "abundance", "fragment_intensity"]:
                    if k in row:
                        inten = finite_float(row[k]); break
                if mz is not None and inten is not None:
                    out.append((mz, inten))
        if out:
            return out

    # JSON-encoded string or whitespace/newline-delimited text.
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            nested = json.loads(s)
            parsed = parse_peak_pairs(nested)
            if parsed:
                return parsed
        except Exception:
            pass
        out = []
        for line in s.replace(";", "\n").splitlines():
            toks = line.replace(",", " ").split()
            if len(toks) >= 2:
                mz, inten = finite_float(toks[0]), finite_float(toks[1])
                if mz is not None and inten is not None:
                    out.append((mz, inten))
        if out:
            return out
    return None

def extract_peaks(obj):
    # Preferred keys used by USI resolvers.
    for key in ["peaks", "spectrum", "peaklist", "peak_list"]:
        value = recursive_find(obj, [key])
        parsed = parse_peak_pairs(value)
        if parsed:
            return parsed

    # Some responses expose parallel arrays.
    mzs = recursive_find(obj, ["mzs", "mz_array", "masses"])
    ints = recursive_find(obj, ["intensities", "intensity_array", "abundances"])
    if isinstance(mzs, list) and isinstance(ints, list) and len(mzs) == len(ints):
        out = []
        for m, i in zip(mzs, ints):
            m, i = finite_float(m), finite_float(i)
            if m is not None and i is not None:
                out.append((m, i))
        if out:
            return out

    raise ValueError("No recoverable peak list found in resolver JSON.")

def resolver_precursor(obj):
    v = recursive_find(
        obj,
        ["precursor_mz", "precursormz", "precursorMz", "precursor_mass", "parent_mass"]
    )
    return finite_float(v)

def resolver_charge(obj):
    v = recursive_find(obj, ["charge", "precursor_charge"])
    if v is None:
        return None
    try:
        return str(v)
    except Exception:
        return None

def load_validate_raw(path: Path):
    raw = path.read_bytes()
    obj = json.loads(raw.decode("utf-8"))
    peaks = extract_peaks(obj)
    if not peaks:
        raise ValueError("Cached JSON has no peaks.")
    return raw, obj, peaks

def retrieve_usi(usi: str, cache: Path):
    # Restart-safe cache reuse.
    if cache.exists() and cache.stat().st_size > 0:
        try:
            raw, obj, peaks = load_validate_raw(cache)
            return "CACHE", 0, raw, obj, peaks, ""
        except Exception:
            # Invalid cache is quarantined rather than trusted.
            bad = cache.with_suffix(cache.suffix + ".invalid")
            if bad.exists():
                bad.unlink()
            cache.replace(bad)

    url = ENDPOINT + "?" + urllib.parse.urlencode({"usi1": usi})
    last_error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": f"NAA-Dipeptide-Tier1/{VERSION}"}
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as r:
                raw = r.read()
            obj = json.loads(raw.decode("utf-8"))
            peaks = extract_peaks(obj)
            if not peaks:
                raise ValueError("Resolver returned no peaks.")
            tmp = cache.with_suffix(".json.tmp")
            tmp.write_bytes(raw)
            tmp.replace(cache)
            return "DOWNLOADED", attempt, raw, obj, peaks, ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_SLEEP_SECONDS * attempt)
    return "FAILED", MAX_ATTEMPTS, b"", None, None, last_error

require(MASTER)
master = pd.read_csv(MASTER, sep="\t", low_memory=False)

required = [
    "BenchmarkSpec", "QueryUSI", "GNPS2LIB_ACCESSION", "SMILES",
    "PrecursorMz", "Adduct"
]
missing = [c for c in required if c not in master.columns]
if missing:
    raise RuntimeError(f"MASTER missing required columns: {missing}")

if len(master) != EXPECTED_SPECS:
    raise RuntimeError(f"Expected 300 MASTER rows; observed {len(master)}.")
if master["BenchmarkSpec"].nunique() != EXPECTED_SPECS:
    raise RuntimeError("BenchmarkSpec is not unique in MASTER.")
if master["QueryUSI"].nunique() != EXPECTED_UNIQUE_USI:
    raise RuntimeError(
        f"Expected 300 unique QueryUSIs; observed {master['QueryUSI'].nunique()}."
    )
if master["QueryUSI"].isna().any():
    raise RuntimeError("MASTER contains missing QueryUSI.")

# Optional provenance gate against successful Script 113c output.
if META113.exists():
    m113 = pd.read_csv(META113, sep="\t", low_memory=False)
    if len(m113) != 1800:
        raise RuntimeError(f"Script-113 metadata exists but has {len(m113)} rows, not 1800.")
    if "CrosswalkResolved" in m113.columns:
        vals = m113["CrosswalkResolved"].astype(str).str.lower()
        if not vals.isin(["true", "1"]).all():
            raise RuntimeError("Script-113 metadata contains unresolved crosswalks.")

retrieval_rows = []
metadata_rows = []
peak_rows = []
precursor_rows = []

print(f"Full Tier-1 experimental retrieval {VERSION}")
print("=" * 52)
print(f"Structures / unique QueryUSIs: {len(master)} / {master['QueryUSI'].nunique()}")
print(f"Cache: {RAW}")
print()

for n, row in enumerate(master.itertuples(index=False), start=1):
    spec = str(row.BenchmarkSpec)
    usi = str(row.QueryUSI)
    benchmark_prec = finite_float(row.PrecursorMz)
    if benchmark_prec is None or benchmark_prec <= 0:
        raise RuntimeError(f"{spec}: invalid benchmark precursor m/z {row.PrecursorMz!r}")

    cache = RAW / cache_name(usi)
    action, attempts, raw, obj, raw_peaks, error = retrieve_usi(usi, cache)

    if action == "FAILED":
        retrieval_rows.append({
            "BenchmarkSpec": spec,
            "QueryUSI": usi,
            "Action": action,
            "Attempts": attempts,
            "CacheFile": str(cache),
            "RawJSONBytes": 0,
            "RawJSONSHA256": "",
            "RawPeakCount": 0,
            "Status": "FAIL",
            "Error": error,
        })
        print(f"[{n:03d}/300] FAIL {spec}: {error}", flush=True)
        continue

    raw_sha = sha256_bytes(raw)
    try:
        peaks0 = [(float(m), float(i)) for m, i in raw_peaks]
    except Exception as exc:
        raise RuntimeError(f"{spec}: peak conversion failed: {exc}")

    raw_count = len(peaks0)
    # Frozen preprocessing: finite positive, then precursor+0.5 ceiling.
    arr = np.asarray(peaks0, dtype=float)
    finite_positive = (
        np.isfinite(arr[:, 0]) &
        np.isfinite(arr[:, 1]) &
        (arr[:, 0] > 0) &
        (arr[:, 1] > 0)
    )
    arr_fp = arr[finite_positive]
    ceiling = benchmark_prec + PRECURSOR_CEILING_OFFSET_DA
    arr_keep = arr_fp[arr_fp[:, 0] <= ceiling]

    # Stable ascending m/z ordering; no top-N truncation.
    if len(arr_keep):
        arr_keep = arr_keep[np.argsort(arr_keep[:, 0], kind="stable")]

    raw_int = arr_keep[:, 1] if len(arr_keep) else np.asarray([], dtype=float)
    sqrt_int = np.sqrt(raw_int)
    l2 = float(np.linalg.norm(sqrt_int))
    sqrt_l2 = sqrt_int / l2 if l2 > 0 else np.zeros_like(sqrt_int)

    resolver_prec = resolver_precursor(obj)
    delta = (
        resolver_prec - benchmark_prec
        if resolver_prec is not None and math.isfinite(resolver_prec)
        else None
    )

    retrieval_rows.append({
        "BenchmarkSpec": spec,
        "QueryUSI": usi,
        "Action": action,
        "Attempts": attempts,
        "CacheFile": str(cache),
        "RawJSONBytes": len(raw),
        "RawJSONSHA256": raw_sha,
        "RawPeakCount": raw_count,
        "Status": "PASS",
        "Error": "",
    })

    metadata_rows.append({
        "BenchmarkSpec": spec,
        "QueryUSI": usi,
        "GNPS2LIB_ACCESSION": row.GNPS2LIB_ACCESSION,
        "SMILES": row.SMILES,
        "BenchmarkPrecursorMz": benchmark_prec,
        "Adduct": row.Adduct,
        "ResolverPrecursorMz": resolver_prec,
        "ResolverCharge": resolver_charge(obj),
        "RawPeakCount": raw_count,
        "FinitePositivePeakCountBeforeCeiling": int(len(arr_fp)),
        "StandardizedPeakCount": int(len(arr_keep)),
        "PrecursorCeilingMz": ceiling,
        "SqrtIntensityL2NormBeforeNormalization": l2,
        "ExperimentalAcquisitionCESet_eV": "25|50|75",
        "ExperimentalSingleCEAssigned": False,
        "ExperimentalSpectrumRepresentation": "MeasuredStepped25_50_75",
        "PreprocessingVersion": VERSION,
        "ScoringPolicyDependency": "v0.6.4.8",
        "RawJSONSHA256": raw_sha,
    })

    precursor_rows.append({
        "BenchmarkSpec": spec,
        "QueryUSI": usi,
        "BenchmarkPrecursorMz": benchmark_prec,
        "ResolverPrecursorMz": resolver_prec,
        "ResolverMinusBenchmark_Da": delta,
        "AbsPrecursorDifference_Da": abs(delta) if delta is not None else None,
        "ResolverPrecursorAvailable": resolver_prec is not None,
        # This is an audit, not a destructive gate: resolver metadata may omit
        # precursor m/z even when the spectrum itself is valid.
        "Status": "PASS" if resolver_prec is not None else "NOT_REPORTED",
    })

    for j, ((mz, inten), sq, norm) in enumerate(zip(arr_keep, sqrt_int, sqrt_l2)):
        peak_rows.append({
            "BenchmarkSpec": spec,
            "QueryUSI": usi,
            "ExperimentalPeakIndex": j,
            "PeakMz": float(mz),
            "ExperimentalIntensityRaw": float(inten),
            "ExperimentalIntensitySqrt": float(sq),
            "ExperimentalIntensitySqrtL2": float(norm),
            "BenchmarkPrecursorMz": benchmark_prec,
            "Adduct": row.Adduct,
            "ExperimentalAcquisitionCESet_eV": "25|50|75",
            "ExperimentalSingleCEAssigned": False,
            "PreprocessingVersion": VERSION,
        })

    if n == 1 or n % 25 == 0 or n == len(master):
        print(
            f"[{n:03d}/300] {action:<10} {spec} "
            f"raw={raw_count} standardized={len(arr_keep)}",
            flush=True
        )

retrieval = pd.DataFrame(retrieval_rows)
metadata = pd.DataFrame(metadata_rows)
peaks = pd.DataFrame(peak_rows)
precursor = pd.DataFrame(precursor_rows)

# Deterministic ordering.
if len(metadata):
    metadata = metadata.sort_values(["BenchmarkSpec"]).reset_index(drop=True)
if len(peaks):
    peaks = peaks.sort_values(["BenchmarkSpec", "ExperimentalPeakIndex"]).reset_index(drop=True)
if len(retrieval):
    retrieval = retrieval.sort_values(["BenchmarkSpec"]).reset_index(drop=True)
if len(precursor):
    precursor = precursor.sort_values(["BenchmarkSpec"]).reset_index(drop=True)

# QC
qc_rows = []
def qc(check, expected, observed, passed):
    qc_rows.append({
        "Check": check,
        "Expected": expected,
        "Observed": observed,
        "Status": "PASS" if passed else "FAIL",
    })

successful = retrieval["Status"].eq("PASS").sum() if len(retrieval) else 0
failed = retrieval["Status"].eq("FAIL").sum() if len(retrieval) else 0

qc("Frozen MASTER rows", 300, len(master), len(master) == 300)
qc("Unique frozen BenchmarkSpec", 300, master["BenchmarkSpec"].nunique(),
   master["BenchmarkSpec"].nunique() == 300)
qc("Unique frozen QueryUSI", 300, master["QueryUSI"].nunique(),
   master["QueryUSI"].nunique() == 300)
qc("Retrieval manifest rows", 300, len(retrieval), len(retrieval) == 300)
qc("Successful resolver/cache records", 300, int(successful), successful == 300)
qc("Failed resolver records", 0, int(failed), failed == 0)
qc("Experimental metadata rows", 300, len(metadata), len(metadata) == 300)
qc("Unique standardized BenchmarkSpec",
   300,
   metadata["BenchmarkSpec"].nunique() if len(metadata) else 0,
   len(metadata) == 300 and metadata["BenchmarkSpec"].nunique() == 300)
qc("Unique standardized QueryUSI",
   300,
   metadata["QueryUSI"].nunique() if len(metadata) else 0,
   len(metadata) == 300 and metadata["QueryUSI"].nunique() == 300)
qc("Spectra with >=1 standardized peak",
   300,
   int((metadata["StandardizedPeakCount"] > 0).sum()) if len(metadata) else 0,
   len(metadata) == 300 and (metadata["StandardizedPeakCount"] > 0).all())
qc("Raw JSON SHA256 populated",
   300,
   int(retrieval["RawJSONSHA256"].astype(str).str.len().eq(64).sum()) if len(retrieval) else 0,
   len(retrieval) == 300 and retrieval["RawJSONSHA256"].astype(str).str.len().eq(64).all())
qc("Experimental Single-CE assignments", 0,
   int(metadata["ExperimentalSingleCEAssigned"].astype(bool).sum()) if len(metadata) else 0,
   len(metadata) == 300 and not metadata["ExperimentalSingleCEAssigned"].astype(bool).any())
qc("Experimental acquisition representation",
   "MeasuredStepped25_50_75",
   "|".join(sorted(metadata["ExperimentalSpectrumRepresentation"].unique())) if len(metadata) else "",
   len(metadata) == 300 and
   set(metadata["ExperimentalSpectrumRepresentation"]) == {"MeasuredStepped25_50_75"})
qc("Finite positive standardized peaks",
   f"all {len(peaks)}",
   int((
       np.isfinite(pd.to_numeric(peaks["PeakMz"], errors="coerce")) &
       np.isfinite(pd.to_numeric(peaks["ExperimentalIntensityRaw"], errors="coerce")) &
       (pd.to_numeric(peaks["PeakMz"], errors="coerce") > 0) &
       (pd.to_numeric(peaks["ExperimentalIntensityRaw"], errors="coerce") > 0)
   ).sum()) if len(peaks) else 0,
   len(peaks) > 0 and (
       np.isfinite(pd.to_numeric(peaks["PeakMz"], errors="coerce")) &
       np.isfinite(pd.to_numeric(peaks["ExperimentalIntensityRaw"], errors="coerce")) &
       (pd.to_numeric(peaks["PeakMz"], errors="coerce") > 0) &
       (pd.to_numeric(peaks["ExperimentalIntensityRaw"], errors="coerce") > 0)
   ).all())
qc("All standardized peaks <= precursor+0.5",
   True,
   bool((peaks["PeakMz"] <= peaks["BenchmarkPrecursorMz"] + 0.5 + 1e-12).all()) if len(peaks) else False,
   len(peaks) > 0 and
   (peaks["PeakMz"] <= peaks["BenchmarkPrecursorMz"] + 0.5 + 1e-12).all())
qc("Experimental scoring performed", False, False, True)
qc("Predicted CE consensus constructed", False, False, True)
qc("Best-CE primary selection performed", False, False, True)
qc("Frozen scoring dependency", "v0.6.4.8", "v0.6.4.8", True)

qc_df = pd.DataFrame(qc_rows)
all_pass = qc_df["Status"].eq("PASS").all()

# Write outputs even on HOLD, so failures are diagnosable/restartable.
retrieval_out = OUT / "FullTier1_Experimental_Retrieval_Manifest.tsv"
metadata_out = OUT / "FullTier1_Experimental_Spectrum_Metadata.tsv"
peaks_out = OUT / "FullTier1_Experimental_Peaks_Long.tsv"
precursor_out = OUT / "FullTier1_Experimental_Precursor_Audit.tsv"
qc_out = QC_DIR / "114_retrieve_preprocess_full_tier1_experimental_spectra_v0652.tsv"

write_tsv(retrieval, retrieval_out)
write_tsv(metadata, metadata_out)
write_tsv(peaks, peaks_out)
write_tsv(precursor, precursor_out)
write_tsv(qc_df, qc_out)

# Freeze SHA256 for all cached raw JSON and canonical tables.
hash_rows = []
for p in sorted(RAW.glob("*.json")):
    hash_rows.append({
        "ArtifactType": "RawResolverJSON",
        "Artifact": p.name,
        "Path": str(p),
        "SHA256": sha256_file(p),
        "Bytes": p.stat().st_size,
    })
for label, p in [
    ("RetrievalManifest", retrieval_out),
    ("ExperimentalMetadata", metadata_out),
    ("ExperimentalPeaks", peaks_out),
    ("PrecursorAudit", precursor_out),
    ("QC", qc_out),
]:
    hash_rows.append({
        "ArtifactType": label,
        "Artifact": p.name,
        "Path": str(p),
        "SHA256": sha256_file(p),
        "Bytes": p.stat().st_size,
    })
hash_df = pd.DataFrame(hash_rows)
hash_out = OUT / "FullTier1_Experimental_SHA256.tsv"
write_tsv(hash_df, hash_out)

downloaded = int(retrieval["Action"].eq("DOWNLOADED").sum()) if len(retrieval) else 0
cached = int(retrieval["Action"].eq("CACHE").sum()) if len(retrieval) else 0
resolver_prec_n = int(precursor["ResolverPrecursorAvailable"].sum()) if len(precursor) else 0
max_prec_delta = (
    pd.to_numeric(precursor["AbsPrecursorDifference_Da"], errors="coerce").max()
    if len(precursor) else np.nan
)

notes = f"""Full Tier-1 experimental-spectrum retrieval/preprocessing {VERSION}

STATUS
======
{"PASS / READY FOR SCRIPT 115" if all_pass else "HOLD / REVIEW QC"}

POPULATION
==========
Frozen BenchmarkSpec: {len(master)}
Frozen unique QueryUSI: {master["QueryUSI"].nunique()}
Successful spectra: {successful}
Failed spectra: {failed}
Downloaded this run: {downloaded}
Reused from valid cache: {cached}

EXPERIMENTAL ACQUISITION
========================
Measured study spectrum: stepped 25|50|75 eV
ExperimentalSingleCEAssigned: False
No single experimental CE is fabricated.

FROZEN PREPROCESSING
====================
1. Parse resolver peak list.
2. Retain finite m/z and intensity values.
3. Retain m/z > 0 and intensity > 0.
4. Retain peaks <= frozen benchmark precursor m/z + 0.5 Da.
5. Do not apply top-N truncation.
6. Preserve raw intensity.
7. Store sqrt(raw intensity).
8. Store spectrum-level L2-normalized sqrt intensity for downstream scoring.

Resolver precursor metadata available: {resolver_prec_n}/{len(precursor)}
Maximum reported absolute resolver-vs-benchmark precursor difference (Da):
    {max_prec_delta}

SCORING
=======
Experimental scoring performed: False
Predicted CE consensus constructed: False
Best-CE primary selection performed: False

Frozen downstream scoring dependency:
    v0.6.4.8
    EqualWeight25_50_75Consensus
    primary fragment tolerance 0.020 Da
    tolerance sensitivity 0.010 / 0.020 / 0.050 Da
    best-CE primary use prohibited

NEXT GATE
=========
Only if this script is PASS, Script 115 may combine:
- Script 113c standardized GLACIER / ICEBERG predictions;
- this Script 114 standardized experimental spectra;
and apply the already frozen v0.6.4.8 consensus-scoring policy.

Script 115 must not alter the scoring policy after seeing the full results.
"""
notes_out = OUT / "FullTier1_Experimental_Notes.txt"
notes_out.write_text(notes)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)

print()
print("Retrieval summary")
print("-----------------")
print(f"Downloaded: {downloaded}")
print(f"Valid cache reused: {cached}")
print(f"Successful: {successful}/300")
print(f"Failed: {failed}")
print(f"Standardized peak rows: {len(peaks)}")
print(f"Resolver precursor reported: {resolver_prec_n}/300")
print()
print("QC")
print("--")
print(qc_df.to_string(index=False))
print()
print("Saved")
print("-----")
for p in [
    retrieval_out, metadata_out, peaks_out, precursor_out,
    hash_out, notes_out, qc_out
]:
    print(p)
print()
if all_pass:
    print(f"FULL TIER-1 EXPERIMENTAL RETRIEVAL/PREPROCESSING {VERSION} PASSED — READY FOR SCRIPT 115")
else:
    print(f"FULL TIER-1 EXPERIMENTAL RETRIEVAL/PREPROCESSING {VERSION} HOLD — REVIEW QC")
    sys.exit(2)
