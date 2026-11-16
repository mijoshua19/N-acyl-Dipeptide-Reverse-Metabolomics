#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

from src.common import read_config, setup_logging

log = setup_logging("01_validate_inputs")
cfg = read_config()

required = {
    "annotation_file": [
        "FILENAME","COMPOUND_NAME","COMPOUND_NAME_ORIGINAL","EXTRACTSCAN",
        "SMILES","INCHI","INCHIAUX","ADDUCT","IONMODE","MSLEVEL"
    ],
    "fasst_file": [
        "query_usi","USI","Cosine","Dataset"
    ]
}

log.info("Starting v0.5.0 input validation")

for key, cols in required.items():
    p = Path(cfg[key])
    if not p.exists():
        raise FileNotFoundError(f"{key} missing: {p}")
    df = pd.read_csv(p, sep="\t", nrows=5)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{p}: missing required columns {missing}")
    log.info("%s OK: %s", key, p)

mgf = Path(cfg["dashboard_mgf_file"])
if not mgf.exists():
    raise FileNotFoundError(f"dashboard_mgf_file missing: {mgf}")
log.info("dashboard_mgf_file OK: %s", mgf)

panredu = Path(cfg["panredu_file"])
if panredu.exists():
    log.info("panredu_file OK: %s", panredu)
else:
    log.warning("Pan-ReDU file not present yet: %s", panredu)

print("VALIDATION PASSED")
