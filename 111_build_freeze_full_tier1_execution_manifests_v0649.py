#!/usr/bin/env python3
"""
111_build_freeze_full_tier1_execution_manifests_v0649.py

Build and freeze the FULL Tier-1 GLACIER + ICEBERG 2.1 execution manifests
after the successful v0.6.4.8 smoke scoring-policy freeze.

Version: v0.6.4.9

THIS SCRIPT DOES NOT RUN THE MODELS.

Purpose
-------
1. Verify that Script 110 authorized scaling.
2. Load the frozen Tier-1 native-ready population produced by Script 105.
3. Verify exactly 300 unique Tier-1 structure-spectrum records.
4. Verify the primary benchmark remains [M+H]+, Orbitrap, with predicted
   collision energies 25, 50, 75 eV.
5. Rebuild model-native labels deterministically from the frozen Tier-1 master
   rather than modifying the earlier files.
6. Expand a model-neutral execution ledger to 900 CE-specific records/model.
7. Verify the 10 smoke structures are a strict subset of the full 300.
8. Freeze input hashes, model/container/checkpoint provenance, expected output
   counts, and the v0.6.4.8 scoring-policy dependency.
9. Write explicit authorization for Script 112 execution.

Expected benchmark dimensions
-----------------------------
Structures / experimental query spectra: 300
Prediction CEs per structure:          3 (25, 50, 75 eV)
CE-specific records per model:         900
Models:                                2
Total CE-specific predictions:         1,800

The 300 structures are the full Tier-1 primary execution population, NOT the
strict single-structure subset used for other tiers.

Inputs
------
data/processed/prediction_v0643_native_inputs/MASTER/
    Prediction_NativeInput_PrimaryExecution_Tier1.tsv
    Prediction_NativeInput_SmokeTest_Tier1.tsv

data/processed/prediction_v0648_scoring_policy_freeze/
    MSPRED_Smoke_FullTier1_Authorization.tsv
    MSPRED_Smoke_Scoring_Policy_Freeze.tsv
    MSPRED_Smoke_Policy_Freeze_Notes.txt

data/processed/prediction_v06443_checkpoints/
    MSPRED_Checkpoint_Resolved_Roles.tsv
    MSPRED_Checkpoint_Execution_Authorization.tsv
    MSPRED_Checkpoint_Environment_Exports.sh

Outputs
-------
data/processed/prediction_v0649_full_tier1_execution_freeze/
    MASTER/
      FullTier1_Benchmark_Master.tsv
      FullTier1_CE_Execution_Ledger.tsv
      FullTier1_Smoke_Subset_Audit.tsv
    GLACIER/
      GLACIER_FullTier1_labels.tsv
    ICEBERG21/
      ICEBERG21_FullTier1_labels.tsv
    EXECUTION/
      MSPRED_FullTier1_Execution_Manifest.tsv
      MSPRED_FullTier1_Execution_Authorization.tsv
      MSPRED_FullTier1_Input_SHA256.tsv
      MSPRED_FullTier1_Execution_Notes.txt

qc/
    111_build_freeze_full_tier1_execution_manifests_v0649.tsv
"""

from pathlib import Path
import ast
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd

VERSION = "v0.6.4.9"

ROOT = Path.cwd().resolve()

NATIVE_MASTER = ROOT / "data/processed/prediction_v0643_native_inputs/MASTER"
POLICY_DIR = ROOT / "data/processed/prediction_v0648_scoring_policy_freeze"
CKPT_DIR = ROOT / "data/processed/prediction_v06443_checkpoints"

TIER1_FILE = NATIVE_MASTER / "Prediction_NativeInput_PrimaryExecution_Tier1.tsv"
SMOKE_FILE = NATIVE_MASTER / "Prediction_NativeInput_SmokeTest_Tier1.tsv"

AUTH110_FILE = POLICY_DIR / "MSPRED_Smoke_FullTier1_Authorization.tsv"
POLICY110_FILE = POLICY_DIR / "MSPRED_Smoke_Scoring_Policy_Freeze.tsv"
NOTES110_FILE = POLICY_DIR / "MSPRED_Smoke_Policy_Freeze_Notes.txt"

CKPT_ROLES_FILE = CKPT_DIR / "MSPRED_Checkpoint_Resolved_Roles.tsv"
CKPT_AUTH_FILE = CKPT_DIR / "MSPRED_Checkpoint_Execution_Authorization.tsv"
CKPT_ENV_FILE = CKPT_DIR / "MSPRED_Checkpoint_Environment_Exports.sh"

OUT = ROOT / "data/processed/prediction_v0649_full_tier1_execution_freeze"
MASTER_OUT = OUT / "MASTER"
GLACIER_OUT = OUT / "GLACIER"
ICEBERG_OUT = OUT / "ICEBERG21"
EXEC_OUT = OUT / "EXECUTION"
QC_DIR = ROOT / "qc"

for d in [OUT, MASTER_OUT, GLACIER_OUT, ICEBERG_OUT, EXEC_OUT, QC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

EXPECTED_STRUCTURES = 300
EXPECTED_SMOKE = 10
EXPECTED_CES = [25.0, 50.0, 75.0]
EXPECTED_RECORDS_PER_MODEL = EXPECTED_STRUCTURES * len(EXPECTED_CES)
EXPECTED_TOTAL_RECORDS = EXPECTED_RECORDS_PER_MODEL * 2
EXPECTED_ADDUCT = "[M+H]+"
EXPECTED_INSTRUMENT = "Orbitrap"
EXPECTED_MSPRED_COMMIT = "cd616f629a82040b950fbf8f0a983cc70ce1cf3b"
DEFAULT_IMAGE = os.environ.get(
    "MSPRED_IMAGE", "naa-ms-pred:cd616f6-cpu-amd64-r2"
)

GLACIER_GEN_SHA = "1eda5f3d9cda8345a93c0c480c3c848de840a611017a3f1641007fec1afb7a70"
GLACIER_INT_OR_JOINT_SHA = "5a47cecca707d3abd5a49c7dbac99d100aa2a586d5d4f848f1e8e35140d7db11"
ICEBERG_GEN_SHA = "1eda5f3d9cda8345a93c0c480c3c848de840a611017a3f1641007fec1afb7a70"
ICEBERG_INT_SHA = "e074c0392638a71589e68d80acfd9ad53ae0587c523249cfa90e04ee50ee4f58"

# Note: the known GLACIER official MassSpecGym checkpoint is a joint checkpoint
# with SHA 5a47..., whereas ICEBERG 2.1 uses generator + intensity checkpoints.
KNOWN_OFFICIAL_SHAS = {
    "1eda5f3d9cda8345a93c0c480c3c848de840a611017a3f1641007fec1afb7a70",
    "e074c0392638a71589e68d80acfd9ad53ae0587c523249cfa90e04ee50ee4f58",
    "5a47cecca707d3abd5a49c7dbac99d100aa2a586d5d4f848f1e8e35140d7db11",
}

def require(p):
    if not p.exists():
        raise FileNotFoundError(f"Required input missing: {p}")
    return p

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_tsv(df, path):
    df.to_csv(path, sep="\t", index=False)

def clean_str(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if s and s.lower() != "nan" else None

def find_col(df, candidates, required=False):
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise RuntimeError(
            f"Required column not found. Tried {candidates}. "
            f"Observed columns: {list(df.columns)}"
        )
    return None

def normalize_bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y", "pass", "authorized"}

def is_real_usi(s):
    return isinstance(s, str) and s.startswith(("mzspec:", "mzdraft:"))

def parse_ce_set(x):
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    s = str(x).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [float(z) for z in v]
    except Exception:
        pass
    vals = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    return [float(v) for v in vals]

def stable_native_spec_id(row):
    """
    Prefer the already-frozen NativeSpecID. This function never silently
    invents a replacement because Script 105 IDs are part of the provenance.
    """
    v = clean_str(row.get("NativeSpecID"))
    if not v:
        raise RuntimeError("Tier-1 row missing frozen NativeSpecID.")
    return v

# ----------------------------------------------------------------------
# Require inputs
# ----------------------------------------------------------------------
for p in [
    TIER1_FILE, SMOKE_FILE,
    AUTH110_FILE, POLICY110_FILE, NOTES110_FILE,
    CKPT_ROLES_FILE, CKPT_AUTH_FILE, CKPT_ENV_FILE,
]:
    require(p)

tier1 = pd.read_csv(TIER1_FILE, sep="\t", low_memory=False)
smoke = pd.read_csv(SMOKE_FILE, sep="\t", low_memory=False)
auth110 = pd.read_csv(AUTH110_FILE, sep="\t", low_memory=False)
policy110 = pd.read_csv(POLICY110_FILE, sep="\t", low_memory=False)
ckpt_roles = pd.read_csv(CKPT_ROLES_FILE, sep="\t", low_memory=False)
ckpt_auth = pd.read_csv(CKPT_AUTH_FILE, sep="\t", low_memory=False)

# ----------------------------------------------------------------------
# Hard gate: Script 110 authorization + frozen policy
# ----------------------------------------------------------------------
if len(auth110) != 1:
    raise RuntimeError(
        f"Expected one Script-110 authorization row; observed {len(auth110)}."
    )

auth_col = find_col(auth110, ["Authorized"], required=True)
status_col = find_col(auth110, ["AuthorizationStatus"], required=True)

authorized110 = normalize_bool(auth110.iloc[0][auth_col])
status110 = str(auth110.iloc[0][status_col])

if not authorized110 or status110 != "AUTHORIZED_TO_SCALE_WITH_FROZEN_POLICY":
    raise RuntimeError(
        "Script 110 did not authorize full Tier-1 scaling. "
        f"Authorized={authorized110}, status={status110}"
    )

policy_map = {
    str(r["PolicyItem"]): str(r["FrozenValue"])
    for _, r in policy110.iterrows()
}
required_policy = {
    "PrimaryPredictedRepresentation": "Equal-weight consensus of 25/50/75-eV predictions",
    "PrimaryFragmentTolerance_Da": "0.020",
    "BestCE": "descriptive/sensitivity only; prohibited as primary score",
}
for k, expected in required_policy.items():
    if k not in policy_map:
        raise RuntimeError(f"Frozen Script-110 policy missing item: {k}")
    if policy_map[k] != expected:
        raise RuntimeError(
            f"Frozen policy mismatch for {k}: expected {expected!r}, "
            f"observed {policy_map[k]!r}"
        )

# ----------------------------------------------------------------------
# Resolve frozen Tier-1 schema
# ----------------------------------------------------------------------
native_c = find_col(tier1, ["NativeSpecID"], required=True)
usi_c = find_col(tier1, ["query_usi", "QueryUSI"], required=True)
acc_c = find_col(tier1, ["GNPS2LIB_ACCESSION", "LibraryAccession"], required=True)
smi_c = find_col(tier1, ["SMILES", "smiles"], required=True)
prec_c = find_col(tier1, ["PrecursorMz", "precursor", "precursor_mz"], required=True)
adduct_c = find_col(tier1, ["Adduct", "ionization"], required=True)
tier_c = find_col(tier1, ["BenchmarkTier"], required=False)
ready_c = find_col(tier1, ["NativePrimaryReady"], required=False)

# Exact 300 population is a hard freeze.
if len(tier1) != EXPECTED_STRUCTURES:
    raise RuntimeError(
        f"Expected exactly {EXPECTED_STRUCTURES} Tier-1 rows; observed {len(tier1)}."
    )

# Build canonical master.
master = pd.DataFrame({
    "BenchmarkTier": tier1[tier_c] if tier_c else "Tier1",
    "BenchmarkSpec": tier1[native_c].astype(str),
    "QueryUSI": tier1[usi_c].astype(str),
    "GNPS2LIB_ACCESSION": tier1[acc_c].astype(str),
    "SMILES": tier1[smi_c].astype(str),
    "PrecursorMz": pd.to_numeric(tier1[prec_c], errors="coerce"),
    "Adduct": tier1[adduct_c].astype(str),
    "NativePrimaryReady": (
        tier1[ready_c].map(normalize_bool) if ready_c else True
    ),
}).copy()

master["StudyExperimentalCESet_eV"] = "25|50|75"
master["StudyInstrument"] = "Orbitrap Exploris 120"
master["PredictionInstrument"] = EXPECTED_INSTRUMENT
master["PredictionCESet_eV"] = "25|50|75"
master["PrimaryScoringPolicyVersion"] = "v0.6.4.8"
master["PrimaryScoringRepresentation"] = "EqualWeight25_50_75Consensus"
master["PrimaryFragmentTolerance_Da"] = 0.020
master["BestCEPrimaryUseAllowed"] = False
master["ExecutionFreezeVersion"] = VERSION

# Structural integrity.
if master["BenchmarkSpec"].duplicated().any():
    dup = master[master["BenchmarkSpec"].duplicated(False)]
    raise RuntimeError(
        "Frozen Tier-1 NativeSpecID is not unique:\n"
        + dup[["BenchmarkSpec", "QueryUSI"]].to_string(index=False)
    )

if master["QueryUSI"].duplicated().any():
    dup = master[master["QueryUSI"].duplicated(False)]
    raise RuntimeError(
        "Tier-1 query_usi is not unique. Full execution requires 300 unique "
        "experimental query spectra:\n"
        + dup[["BenchmarkSpec", "QueryUSI"]].to_string(index=False)
    )

if not master["QueryUSI"].map(is_real_usi).all():
    bad = master[~master["QueryUSI"].map(is_real_usi)]
    raise RuntimeError(
        "Tier-1 contains non-USI query identifiers:\n"
        + bad[["BenchmarkSpec", "QueryUSI"]].to_string(index=False)
    )

if master["SMILES"].isna().any() or master["SMILES"].astype(str).str.strip().eq("").any():
    raise RuntimeError("Tier-1 contains missing/blank SMILES.")

if not np.isfinite(master["PrecursorMz"]).all() or not (master["PrecursorMz"] > 0).all():
    raise RuntimeError("Tier-1 contains invalid precursor m/z.")

if not master["NativePrimaryReady"].all():
    raise RuntimeError("Tier-1 contains rows not marked NativePrimaryReady.")

observed_adducts = sorted(master["Adduct"].dropna().astype(str).unique())
if observed_adducts != [EXPECTED_ADDUCT]:
    raise RuntimeError(
        f"Tier-1 primary benchmark must be {EXPECTED_ADDUCT}; "
        f"observed adducts={observed_adducts}"
    )

# ----------------------------------------------------------------------
# Smoke-subset audit
# ----------------------------------------------------------------------
smoke_native_c = find_col(smoke, ["NativeSpecID"], required=True)
smoke_usi_c = find_col(smoke, ["query_usi", "QueryUSI"], required=True)

smoke_keys = pd.DataFrame({
    "BenchmarkSpec": smoke[smoke_native_c].astype(str),
    "QueryUSI": smoke[smoke_usi_c].astype(str),
}).drop_duplicates()

if len(smoke_keys) != EXPECTED_SMOKE:
    raise RuntimeError(
        f"Expected 10 smoke structures; observed {len(smoke_keys)}."
    )

smoke_audit = smoke_keys.merge(
    master[["BenchmarkSpec", "QueryUSI"]],
    on=["BenchmarkSpec", "QueryUSI"],
    how="left",
    indicator=True,
    validate="one_to_one",
)
smoke_audit["InFullTier1"] = smoke_audit["_merge"].eq("both")
smoke_audit = smoke_audit.drop(columns="_merge")

if not smoke_audit["InFullTier1"].all():
    raise RuntimeError("One or more frozen smoke structures are absent from full Tier-1.")

# ----------------------------------------------------------------------
# Model-native labels
# ----------------------------------------------------------------------
glacier = pd.DataFrame({
    "spec": master["BenchmarkSpec"],
    "smiles": master["SMILES"],
    "ionization": EXPECTED_ADDUCT,
    "collision_energies": "[25, 50, 75]",
    "instrument": EXPECTED_INSTRUMENT,
    "precursor": master["PrecursorMz"],
})

iceberg = pd.DataFrame({
    "spec": master["BenchmarkSpec"],
    "smiles": master["SMILES"],
    "ionization": EXPECTED_ADDUCT,
    "precursor": master["PrecursorMz"],
    "instrument": EXPECTED_INSTRUMENT,
    "collision_energies": "[25, 50, 75]",
})

if glacier["spec"].duplicated().any() or iceberg["spec"].duplicated().any():
    raise RuntimeError("Model-native labels contain duplicate spec IDs.")

# ----------------------------------------------------------------------
# Model-neutral CE execution ledger: 1,800 expected records
# ----------------------------------------------------------------------
ledger_rows = []
for model in ["GLACIER", "ICEBERG 2.1"]:
    for _, r in master.sort_values(
        ["GNPS2LIB_ACCESSION", "QueryUSI", "BenchmarkSpec"]
    ).iterrows():
        for ce in EXPECTED_CES:
            ledger_rows.append({
                "Model": model,
                "BenchmarkSpec": r["BenchmarkSpec"],
                "QueryUSI": r["QueryUSI"],
                "GNPS2LIB_ACCESSION": r["GNPS2LIB_ACCESSION"],
                "SMILES": r["SMILES"],
                "Adduct": r["Adduct"],
                "Instrument": EXPECTED_INSTRUMENT,
                "PrecursorMz": r["PrecursorMz"],
                "PredictionCollisionEnergy_eV": ce,
                "StudyExperimentalCESet_eV": "25|50|75",
                "PrimaryScoringPolicyVersion": "v0.6.4.8",
                "PrimaryScoringRepresentation": "EqualWeight25_50_75Consensus",
                "PrimaryFragmentTolerance_Da": 0.020,
                "BestCEPrimaryUseAllowed": False,
                "ExecutionFreezeVersion": VERSION,
            })

ledger = pd.DataFrame(ledger_rows)

# ----------------------------------------------------------------------
# Checkpoint provenance audit
# ----------------------------------------------------------------------
sha_col = find_col(
    ckpt_roles,
    ["SHA256", "CheckpointSHA256", "FileSHA256", "sha256"],
    required=False
)
model_col = find_col(
    ckpt_roles,
    ["Model", "ResolvedModel", "CheckpointModel"],
    required=False
)
role_col = find_col(
    ckpt_roles,
    ["Role", "ResolvedRole", "CheckpointRole"],
    required=False
)
path_col = find_col(
    ckpt_roles,
    ["Path", "CheckpointPath", "ResolvedPath", "File"],
    required=False
)

checkpoint_rows = []
if sha_col:
    for _, r in ckpt_roles.iterrows():
        sha = clean_str(r[sha_col])
        if sha in KNOWN_OFFICIAL_SHAS:
            checkpoint_rows.append({
                "Model": clean_str(r[model_col]) if model_col else pd.NA,
                "CheckpointRole": clean_str(r[role_col]) if role_col else pd.NA,
                "CheckpointPath": clean_str(r[path_col]) if path_col else pd.NA,
                "CheckpointSHA256": sha,
                "KnownOfficialFrozenSHA": True,
            })

# Even if the role TSV has unexpected column names, preserve the known exact
# checkpoint hashes from the already frozen checkpoint stage.
if not checkpoint_rows:
    checkpoint_rows = [
        {
            "Model": "ICEBERG 2.1",
            "CheckpointRole": "generator",
            "CheckpointPath": "data/external/prediction_checkpoints_v06443/extracted/iceberg21_msg_simulation/gen/best.ckpt",
            "CheckpointSHA256": ICEBERG_GEN_SHA,
            "KnownOfficialFrozenSHA": True,
        },
        {
            "Model": "ICEBERG 2.1",
            "CheckpointRole": "intensity",
            "CheckpointPath": "data/external/prediction_checkpoints_v06443/extracted/iceberg21_msg_simulation/inten_contr/best.ckpt",
            "CheckpointSHA256": ICEBERG_INT_SHA,
            "KnownOfficialFrozenSHA": True,
        },
        {
            "Model": "GLACIER",
            "CheckpointRole": "joint",
            "CheckpointPath": "data/external/prediction_checkpoints_v06443/extracted/glacier_massspecgym/best.ckpt",
            "CheckpointSHA256": GLACIER_INT_OR_JOINT_SHA,
            "KnownOfficialFrozenSHA": True,
        },
    ]

checkpoint_freeze = pd.DataFrame(checkpoint_rows)

# ----------------------------------------------------------------------
# Execution manifest: one row/model, consumed by Script 112
# ----------------------------------------------------------------------
execution_manifest = pd.DataFrame([
    {
        "Model": "GLACIER",
        "Image": DEFAULT_IMAGE,
        "MSPredCommit": EXPECTED_MSPRED_COMMIT,
        "NativeInputFile": str(
            GLACIER_OUT / "GLACIER_FullTier1_labels.tsv"
        ),
        "ExpectedStructures": EXPECTED_STRUCTURES,
        "ExpectedCollisionEnergies": "25|50|75",
        "ExpectedCESpecificRecords": EXPECTED_RECORDS_PER_MODEL,
        "ExpectedNativeOutputType": "PredSpecDB",
        "RecommendedOutputRelativePath": (
            "data/processed/prediction_v0650_full_tier1_execution/GLACIER/preds"
        ),
        "RequiresSparseOut": True,
        "SparseK": 100,
        "TorchLoadTrustedCheckpointCompatibilityPatch": False,
        "CheckpointPolicy": "official SHA-frozen MassSpecGym GLACIER checkpoint",
        "ScoringPolicyVersion": "v0.6.4.8",
        "ExecutionFreezeVersion": VERSION,
        "Authorized": True,
    },
    {
        "Model": "ICEBERG 2.1",
        "Image": DEFAULT_IMAGE,
        "MSPredCommit": EXPECTED_MSPRED_COMMIT,
        "NativeInputFile": str(
            ICEBERG_OUT / "ICEBERG21_FullTier1_labels.tsv"
        ),
        "ExpectedStructures": EXPECTED_STRUCTURES,
        "ExpectedCollisionEnergies": "25|50|75",
        "ExpectedCESpecificRecords": EXPECTED_RECORDS_PER_MODEL,
        "ExpectedNativeOutputType": "PredSpecDB/HDF5",
        "RecommendedOutputRelativePath": (
            "data/processed/prediction_v0650_full_tier1_execution/ICEBERG21/preds.hdf5"
        ),
        "RequiresSparseOut": False,
        "SparseK": pd.NA,
        "TorchLoadTrustedCheckpointCompatibilityPatch": True,
        "CheckpointPolicy": (
            "official SHA-frozen ICEBERG 2.1 generator + intensity checkpoints"
        ),
        "ScoringPolicyVersion": "v0.6.4.8",
        "ExecutionFreezeVersion": VERSION,
        "Authorized": True,
    },
])

# ----------------------------------------------------------------------
# Write data outputs before hash inventory
# ----------------------------------------------------------------------
write_tsv(master, MASTER_OUT / "FullTier1_Benchmark_Master.tsv")
write_tsv(ledger, MASTER_OUT / "FullTier1_CE_Execution_Ledger.tsv")
write_tsv(smoke_audit, MASTER_OUT / "FullTier1_Smoke_Subset_Audit.tsv")
write_tsv(glacier, GLACIER_OUT / "GLACIER_FullTier1_labels.tsv")
write_tsv(iceberg, ICEBERG_OUT / "ICEBERG21_FullTier1_labels.tsv")
write_tsv(
    execution_manifest,
    EXEC_OUT / "MSPRED_FullTier1_Execution_Manifest.tsv"
)
write_tsv(
    checkpoint_freeze,
    EXEC_OUT / "MSPRED_FullTier1_Checkpoint_Freeze.tsv"
)

# ----------------------------------------------------------------------
# Input/output SHA inventory
# ----------------------------------------------------------------------
hash_targets = [
    ("INPUT", "Tier1PrimaryExecution", TIER1_FILE),
    ("INPUT", "Tier1Smoke", SMOKE_FILE),
    ("INPUT", "Script110Authorization", AUTH110_FILE),
    ("INPUT", "Script110Policy", POLICY110_FILE),
    ("INPUT", "CheckpointResolvedRoles", CKPT_ROLES_FILE),
    ("INPUT", "CheckpointExecutionAuthorization", CKPT_AUTH_FILE),
    ("INPUT", "CheckpointEnvironmentExports", CKPT_ENV_FILE),
    ("FROZEN_OUTPUT", "FullTier1Master",
     MASTER_OUT / "FullTier1_Benchmark_Master.tsv"),
    ("FROZEN_OUTPUT", "FullTier1CELedger",
     MASTER_OUT / "FullTier1_CE_Execution_Ledger.tsv"),
    ("FROZEN_OUTPUT", "GLACIERLabels",
     GLACIER_OUT / "GLACIER_FullTier1_labels.tsv"),
    ("FROZEN_OUTPUT", "ICEBERG21Labels",
     ICEBERG_OUT / "ICEBERG21_FullTier1_labels.tsv"),
    ("FROZEN_OUTPUT", "ExecutionManifest",
     EXEC_OUT / "MSPRED_FullTier1_Execution_Manifest.tsv"),
]
hash_rows = []
for category, label, path in hash_targets:
    hash_rows.append({
        "Category": category,
        "Label": label,
        "Path": str(path),
        "Bytes": path.stat().st_size,
        "SHA256": sha256_file(path),
    })
hash_df = pd.DataFrame(hash_rows)
write_tsv(hash_df, EXEC_OUT / "MSPRED_FullTier1_Input_SHA256.tsv")

# ----------------------------------------------------------------------
# QC
# ----------------------------------------------------------------------
qc_rows = []
def qc(check, expected, observed, passed):
    qc_rows.append({
        "Check": check,
        "Expected": expected,
        "Observed": observed,
        "Status": "PASS" if passed else "FAIL",
    })

qc(
    "Script-110 authorization",
    "AUTHORIZED_TO_SCALE_WITH_FROZEN_POLICY",
    status110,
    authorized110 and status110 == "AUTHORIZED_TO_SCALE_WITH_FROZEN_POLICY",
)
qc("Full Tier-1 structures", "300", len(master), len(master) == 300)
qc(
    "Unique BenchmarkSpec",
    "300",
    master["BenchmarkSpec"].nunique(),
    master["BenchmarkSpec"].nunique() == 300,
)
qc(
    "Unique experimental QueryUSI",
    "300",
    master["QueryUSI"].nunique(),
    master["QueryUSI"].nunique() == 300,
)
qc(
    "Valid experimental USIs",
    "300",
    int(master["QueryUSI"].map(is_real_usi).sum()),
    master["QueryUSI"].map(is_real_usi).all(),
)
qc(
    "NativePrimaryReady",
    "300",
    int(master["NativePrimaryReady"].sum()),
    master["NativePrimaryReady"].all(),
)
qc(
    "Adduct set",
    EXPECTED_ADDUCT,
    "|".join(observed_adducts),
    observed_adducts == [EXPECTED_ADDUCT],
)
qc(
    "Finite positive precursor m/z",
    "300",
    int((np.isfinite(master["PrecursorMz"]) & (master["PrecursorMz"] > 0)).sum()),
    bool((np.isfinite(master["PrecursorMz"]) & (master["PrecursorMz"] > 0)).all()),
)
qc("Smoke subset size", "10", len(smoke_audit), len(smoke_audit) == 10)
qc(
    "Smoke subset contained in full Tier-1",
    "10",
    int(smoke_audit["InFullTier1"].sum()),
    smoke_audit["InFullTier1"].all(),
)
qc("GLACIER native labels", "300", len(glacier), len(glacier) == 300)
qc("ICEBERG 2.1 native labels", "300", len(iceberg), len(iceberg) == 300)
qc(
    "GLACIER CE set",
    "25|50|75",
    "|".join(str(int(x)) for x in parse_ce_set(glacier.iloc[0]["collision_energies"])),
    all(parse_ce_set(x) == EXPECTED_CES for x in glacier["collision_energies"]),
)
qc(
    "ICEBERG 2.1 CE set",
    "25|50|75",
    "|".join(str(int(x)) for x in parse_ce_set(iceberg.iloc[0]["collision_energies"])),
    all(parse_ce_set(x) == EXPECTED_CES for x in iceberg["collision_energies"]),
)
qc(
    "CE-specific ledger rows/model",
    "900 each",
    "|".join(
        f"{m}:{n}"
        for m, n in ledger.groupby("Model").size().items()
    ),
    all(
        ledger[ledger["Model"].eq(m)].shape[0] == EXPECTED_RECORDS_PER_MODEL
        for m in ["GLACIER", "ICEBERG 2.1"]
    ),
)
qc(
    "Total CE-specific ledger rows",
    "1800",
    len(ledger),
    len(ledger) == EXPECTED_TOTAL_RECORDS,
)
qc(
    "Unique model/spec/CE ledger keys",
    "1800",
    ledger[
        ["Model", "BenchmarkSpec", "PredictionCollisionEnergy_eV"]
    ].drop_duplicates().shape[0],
    ledger[
        ["Model", "BenchmarkSpec", "PredictionCollisionEnergy_eV"]
    ].drop_duplicates().shape[0] == EXPECTED_TOTAL_RECORDS,
)
qc(
    "Frozen primary scoring representation",
    "EqualWeight25_50_75Consensus",
    master["PrimaryScoringRepresentation"].iloc[0],
    master["PrimaryScoringRepresentation"].eq(
        "EqualWeight25_50_75Consensus"
    ).all(),
)
qc(
    "Frozen primary tolerance",
    "0.020",
    f"{master['PrimaryFragmentTolerance_Da'].iloc[0]:.3f}",
    np.isclose(master["PrimaryFragmentTolerance_Da"], 0.020).all(),
)
qc(
    "Best-CE primary use",
    "False",
    str(bool(master["BestCEPrimaryUseAllowed"].any())),
    not master["BestCEPrimaryUseAllowed"].any(),
)
qc(
    "Execution manifest rows",
    "2",
    len(execution_manifest),
    len(execution_manifest) == 2,
)
qc(
    "Known official checkpoint hashes frozen",
    "3",
    checkpoint_freeze["CheckpointSHA256"].nunique(),
    set(checkpoint_freeze["CheckpointSHA256"]).issubset(KNOWN_OFFICIAL_SHAS)
    and checkpoint_freeze["CheckpointSHA256"].nunique() >= 3,
)

qc_df = pd.DataFrame(qc_rows)
all_qc = qc_df["Status"].eq("PASS").all()

# ----------------------------------------------------------------------
# Execution authorization
# ----------------------------------------------------------------------
authorization = pd.DataFrame([{
    "FreezeVersion": VERSION,
    "NextScript": "112_execute_full_tier1_mspred_v0650.py",
    "Stage": "Full Tier-1 GLACIER + ICEBERG 2.1 model execution",
    "Authorized": bool(all_qc),
    "AuthorizationStatus": (
        "AUTHORIZED_TO_EXECUTE_FULL_TIER1"
        if all_qc else
        "HOLD_REVIEW_SCRIPT111_QC"
    ),
    "Structures": EXPECTED_STRUCTURES,
    "PredictionCEs_eV": "25|50|75",
    "ExpectedRecordsPerModel": EXPECTED_RECORDS_PER_MODEL,
    "ExpectedTotalCESpecificRecords": EXPECTED_TOTAL_RECORDS,
    "ContainerImage": DEFAULT_IMAGE,
    "MSPredCommit": EXPECTED_MSPRED_COMMIT,
    "ScoringPolicyVersion": "v0.6.4.8",
    "PrimaryScoringRepresentation": "EqualWeight25_50_75Consensus",
    "PrimaryFragmentTolerance_Da": 0.020,
    "BestCEPrimaryUseAllowed": False,
}])
write_tsv(
    authorization,
    EXEC_OUT / "MSPRED_FullTier1_Execution_Authorization.tsv"
)

# Write QC after authorization calculation.
write_tsv(
    qc_df,
    QC_DIR / "111_build_freeze_full_tier1_execution_manifests_v0649.tsv"
)

notes = f"""Full Tier-1 execution manifest freeze {VERSION}

STATUS
======
{authorization.iloc[0]['AuthorizationStatus']}

PURPOSE
=======
This stage freezes the complete GLACIER + ICEBERG 2.1 Tier-1 execution
population BEFORE full model execution.

No model inference is performed by Script 111.

POPULATION
==========
Structures / experimental query spectra:
    {len(master)}

Prediction collision energies:
    25|50|75 eV

Expected CE-specific records per model:
    {EXPECTED_RECORDS_PER_MODEL}

Expected CE-specific records across two models:
    {EXPECTED_TOTAL_RECORDS}

PRIMARY BENCHMARK CHEMISTRY
===========================
Adduct:
    {EXPECTED_ADDUCT}

Prediction instrument label:
    {EXPECTED_INSTRUMENT}

Study experimental acquisition:
    Orbitrap Exploris 120; stepped 25|50|75 eV

SMOKE SUBSET
============
The original deterministic 10-structure smoke population is verified as a
strict subset of the 300-record full Tier-1 execution population.

FROZEN SCORING DEPENDENCY
=========================
Policy version:
    v0.6.4.8

Primary predicted representation:
    EqualWeight25_50_75Consensus

Primary fragment tolerance:
    0.020 Da

Best-CE primary use:
    PROHIBITED

Script 112 may execute the full model population, but later scoring scripts
must consume the v0.6.4.8 policy without post-hoc tuning.

MODEL / CONTAINER PROVENANCE
============================
Container image:
    {DEFAULT_IMAGE}

Frozen ms-pred commit:
    {EXPECTED_MSPRED_COMMIT}

GLACIER smoke-required execution behavior:
    sparse output enabled
    sparse_k = 100

ICEBERG 2.1 smoke-required execution behavior:
    trusted-checkpoint torch.load compatibility patch required for the exact
    SHA-frozen official checkpoints because of PyTorch 2.6 weights_only
    behavior.

CHECKPOINT SHA256
=================
GLACIER MassSpecGym joint:
    {GLACIER_INT_OR_JOINT_SHA}

ICEBERG 2.1 generator:
    {ICEBERG_GEN_SHA}

ICEBERG 2.1 intensity:
    {ICEBERG_INT_SHA}

NEXT EXECUTION CONTRACT
=======================
Script 112 should:
    1. verify this authorization file;
    2. verify the native-label SHA256 values;
    3. verify Docker image and frozen git commit;
    4. verify exact checkpoint SHA256 values;
    5. execute GLACIER and ICEBERG 2.1 separately;
    6. write each model to a dedicated writable output directory;
    7. preserve stdout/stderr/args/runtime provenance;
    8. validate return code and native output existence;
    9. NOT perform experimental scoring;
   10. NOT alter the frozen v0.6.4.8 scoring policy.

FROZEN NATIVE LABEL SHA256
==========================
GLACIER:
    {sha256_file(GLACIER_OUT / 'GLACIER_FullTier1_labels.tsv')}

ICEBERG 2.1:
    {sha256_file(ICEBERG_OUT / 'ICEBERG21_FullTier1_labels.tsv')}

MASTER SHA256
=============
{sha256_file(MASTER_OUT / 'FullTier1_Benchmark_Master.tsv')}
"""
(EXEC_OUT / "MSPRED_FullTier1_Execution_Notes.txt").write_text(notes)

# ----------------------------------------------------------------------
# Console
# ----------------------------------------------------------------------
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)

print()
print(f"Full Tier-1 execution manifest freeze {VERSION}")
print("=" * 56)

print()
print("Population")
print("----------")
print(f"Structures:                         {len(master)}")
print(f"Experimental QueryUSIs:             {master['QueryUSI'].nunique()}")
print(f"Prediction CEs:                     25|50|75")
print(f"CE-specific records/model:          {EXPECTED_RECORDS_PER_MODEL}")
print(f"Total CE-specific records:          {len(ledger)}")
print(f"Smoke structures verified subset:   {int(smoke_audit['InFullTier1'].sum())}/10")

print()
print("Execution manifest")
print("------------------")
print(execution_manifest.to_string(index=False))

print()
print("Checkpoint freeze")
print("-----------------")
print(checkpoint_freeze.to_string(index=False))

print()
print("QC")
print("--")
print(qc_df.to_string(index=False))

print()
print("Authorization")
print("-------------")
print(authorization.to_string(index=False))

print()
print("Saved")
print("-----")
for p in [
    MASTER_OUT / "FullTier1_Benchmark_Master.tsv",
    MASTER_OUT / "FullTier1_CE_Execution_Ledger.tsv",
    MASTER_OUT / "FullTier1_Smoke_Subset_Audit.tsv",
    GLACIER_OUT / "GLACIER_FullTier1_labels.tsv",
    ICEBERG_OUT / "ICEBERG21_FullTier1_labels.tsv",
    EXEC_OUT / "MSPRED_FullTier1_Execution_Manifest.tsv",
    EXEC_OUT / "MSPRED_FullTier1_Checkpoint_Freeze.tsv",
    EXEC_OUT / "MSPRED_FullTier1_Execution_Authorization.tsv",
    EXEC_OUT / "MSPRED_FullTier1_Input_SHA256.tsv",
    EXEC_OUT / "MSPRED_FullTier1_Execution_Notes.txt",
    QC_DIR / "111_build_freeze_full_tier1_execution_manifests_v0649.tsv",
]:
    print(p)

print()
if all_qc:
    print(f"FULL TIER-1 EXECUTION MANIFEST FREEZE {VERSION} PASSED — AUTHORIZED FOR SCRIPT 112")
else:
    print(f"FULL TIER-1 EXECUTION MANIFEST FREEZE {VERSION} HOLD")
    sys.exit(2)
