#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

from src.common import extract_public_filename_from_usi, read_config, setup_logging, write_json

log = setup_logging("07_integrate_panredu")
cfg = read_config()
p = Path(cfg["panredu_file"])

if not p.exists():
    log.warning("Pan-ReDU metadata not found; skipping integration.")
    print("PANREDU INTEGRATION SKIPPED")
    raise SystemExit(0)

f = pd.read_csv("data/processed/FASST_Spectral_Evidence.tsv", sep="\t")
r = pd.read_csv(p, sep="\t")

if "ATTRIBUTE_DatasetAccession" not in r.columns or "filename" not in r.columns:
    raise ValueError("Pan-ReDU file must contain ATTRIBUTE_DatasetAccession and filename.")

f["MatchedFilename"] = f["MatchedPublicUSI"].map(extract_public_filename_from_usi)
r["MatchedFilename"] = r["filename"].astype(str).map(lambda x: Path(x).name)

joined = f.merge(
    r,
    left_on=["Dataset","MatchedFilename"],
    right_on=["ATTRIBUTE_DatasetAccession","MatchedFilename"],
    how="left",
    suffixes=("","_PanReDU"),
    validate="many_to_many"
)

# Store both all joined rows and a coverage summary.
joined.to_csv("data/processed/FASST_Spectral_Evidence_PanReDU.tsv",
              sep="\t", index=False)

meta_hit = joined.groupby("EvidenceKey", dropna=False).apply(
    lambda g: g["ATTRIBUTE_DatasetAccession"].notna().any(),
    include_groups=False
)

qc = {
    "spectral_evidence_rows": len(f),
    "unique_evidence_keys": f["EvidenceKey"].nunique(),
    "evidence_keys_with_panredu_metadata": int(meta_hit.sum()),
    "evidence_keys_without_panredu_metadata": int((~meta_hit).sum()),
    "metadata_coverage_pct": float(100 * meta_hit.mean())
}
write_json(qc, "qc/07_panredu_integration.json")

log.info("Evidence keys with Pan-ReDU metadata: %s / %s (%.2f%%)",
         f"{qc['evidence_keys_with_panredu_metadata']:,}",
         f"{qc['unique_evidence_keys']:,}",
         qc["metadata_coverage_pct"])
print("PANREDU INTEGRATION PASSED")
