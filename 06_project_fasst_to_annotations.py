#!/usr/bin/env python3
"""
Ambiguity-aware projection.

This script DOES NOT treat each projected structural annotation as an independent
spectral observation. The unique evidence key remains:
  query_usi + MatchedPublicUSI

The projection is provided for chemistry-level summarization and plotting only.
"""
from pathlib import Path
import pandas as pd

from src.common import setup_logging, write_json

log = setup_logging("06_project_fasst_to_annotations")

f = pd.read_csv("data/intermediate/FASST_Official_Matches.tsv", sep="\t")
links = pd.read_csv("data/intermediate/Query_to_Annotation_Links.tsv", sep="\t")

# Keep an evidence-level table first.
evidence = f.rename(columns={"USI":"MatchedPublicUSI"}).copy()
evidence["EvidenceKey"] = (
    evidence["query_usi"].astype(str) + "||" +
    evidence["MatchedPublicUSI"].astype(str)
)

# Structural projection is intentionally one-to-many.
p = evidence.merge(
    links[[
        "AnnotationIndex","GNPS2LIB_ACCESSION","query_usi",
        "COMPOUND_NAME","COMPOUND_NAME_ORIGINAL","SMILES","INCHIAUX",
        "AcylName","AcylChain","AA1","AA2","DIPEPTIDE",
        "SourceFile","SourceScan"
    ]],
    on="query_usi",
    how="left",
    validate="many_to_many"
)

if p["AnnotationIndex"].isna().any():
    raise ValueError("At least one FASST query could not be projected to annotations.")

evidence.to_csv("data/processed/FASST_Spectral_Evidence.tsv",
                sep="\t", index=False)
p.to_csv("data/processed/FASST_to_Annotation_Projection.tsv",
         sep="\t", index=False)

# Critical anti-inflation QC:
# Projected rows may be larger, but unique evidence keys must not change.
qc = {
    "spectral_evidence_rows": len(evidence),
    "unique_evidence_keys": evidence["EvidenceKey"].nunique(),
    "projected_annotation_rows": len(p),
    "unique_evidence_keys_after_projection": p["EvidenceKey"].nunique(),
    "projection_multiplier": len(p) / len(evidence) if len(evidence) else None
}
if qc["unique_evidence_keys"] != qc["unique_evidence_keys_after_projection"]:
    raise ValueError("Evidence-key count changed during annotation projection.")
write_json(qc, "qc/06_ambiguity_aware_projection.json")

log.info("Spectral evidence rows: %s", f"{len(evidence):,}")
log.info("Projected annotation rows: %s", f"{len(p):,}")
log.info("Projection multiplier: %.3f", qc["projection_multiplier"])
print("AMBIGUITY-AWARE PROJECTION PASSED")
