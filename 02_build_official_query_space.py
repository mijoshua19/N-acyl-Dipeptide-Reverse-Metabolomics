#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

from src.common import parse_mgf_metadata, read_config, setup_logging, write_json

log = setup_logging("02_build_official_query_space")
cfg = read_config()

mgf = parse_mgf_metadata(cfg["dashboard_mgf_file"])

required = ["TITLE","SPECTRUMID","USI","FILENAME","SCANS","NAME","PEPMASS","ADDUCT"]
missing = [c for c in required if c not in mgf.columns]
if missing:
    raise ValueError(f"Dashboard MGF missing metadata fields: {missing}")

q = mgf[required + [c for c in [
    "SMILES","INCHIAUX","EXACTMASS","IONMODE","FORMULA","SPLASH"
] if c in mgf.columns]].copy()

q = q.rename(columns={
    "TITLE":"GNPS2LIB_ACCESSION",
    "SPECTRUMID":"SPECTRUMID",
    "USI":"query_usi",
    "FILENAME":"SourceFile",
    "SCANS":"SourceScan",
    "NAME":"MGF_Name",
    "PEPMASS":"PrecursorMz",
    "ADDUCT":"MGF_Adduct"
})

q["SourceScan"] = pd.to_numeric(q["SourceScan"], errors="coerce").astype("Int64")
q["PrecursorMz"] = pd.to_numeric(q["PrecursorMz"], errors="coerce")
q["QueryIndex"] = range(1, len(q)+1)

if q["GNPS2LIB_ACCESSION"].duplicated().any():
    raise ValueError("Duplicate official GNPS2LIB accession in dashboard MGF.")
if q["query_usi"].duplicated().any():
    raise ValueError("Duplicate official query USI in dashboard MGF.")
if q[["SourceFile","SourceScan"]].duplicated().any():
    d = q[q[["SourceFile","SourceScan"]].duplicated(keep=False)]
    raise ValueError(f"Duplicate source file+scan in official query space:\n{d}")

expected = cfg["expected_query_spectra"]
if len(q) != expected:
    raise ValueError(f"Expected {expected:,} official spectra, found {len(q):,}")

out = Path("data/intermediate/Official_Query_Space.tsv")
out.parent.mkdir(parents=True, exist_ok=True)
q.to_csv(out, sep="\t", index=False)

qc = {
    "official_query_rows": len(q),
    "unique_accessions": q["GNPS2LIB_ACCESSION"].nunique(),
    "unique_usis": q["query_usi"].nunique(),
    "unique_source_file_scan": q[["SourceFile","SourceScan"]].drop_duplicates().shape[0],
}
write_json(qc, "qc/02_official_query_space.json")

log.info("Official query rows: %s", f"{len(q):,}")
log.info("Unique accessions: %s", f"{q.GNPS2LIB_ACCESSION.nunique():,}")
log.info("Unique USIs: %s", f"{q.query_usi.nunique():,}")
log.info("Output: %s", out)
print("OFFICIAL QUERY SPACE PASSED")
