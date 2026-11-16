#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

from src.common import parse_compound_name, read_config, setup_logging, write_json

log = setup_logging("03_build_annotation_space")
cfg = read_config()

a = pd.read_csv(cfg["annotation_file"], sep="\t")

if len(a) != cfg["expected_annotation_rows"]:
    raise ValueError(
        f"Expected {cfg['expected_annotation_rows']:,} annotation rows, found {len(a):,}"
    )

parsed = pd.DataFrame([parse_compound_name(x) for x in a["COMPOUND_NAME_ORIGINAL"]])
a = pd.concat([a.reset_index(drop=True), parsed], axis=1)
a.insert(0, "AnnotationIndex", range(1, len(a)+1))
a["SourceScan"] = pd.to_numeric(a["EXTRACTSCAN"], errors="coerce").astype("Int64")
a["SourceFile"] = a["FILENAME"].astype(str)

failed = a[a["ParseStatus"] != "parsed"]
if len(failed):
    failed.to_csv("qc/03_annotation_parse_exceptions.tsv", sep="\t", index=False)
    raise ValueError(f"{len(failed)} annotation rows failed compound parsing.")

if set(a["AcylChain"].dropna()) != set(cfg["expected_acyl_chains"]):
    raise ValueError(
        f"Unexpected acyl-chain set: {sorted(a['AcylChain'].dropna().unique())}"
    )

out = Path("data/intermediate/Library_Annotation_Space.tsv")
out.parent.mkdir(parents=True, exist_ok=True)
a.to_csv(out, sep="\t", index=False)

qc = {
    "annotation_rows": len(a),
    "unique_source_file_scan": a[["SourceFile","SourceScan"]].drop_duplicates().shape[0],
    "unique_smiles": a["SMILES"].nunique(dropna=True),
    "unique_inchikeys": a["INCHIAUX"].nunique(dropna=True),
    "unique_acyl_chains": a["AcylChain"].nunique(),
    "unique_aa1": a["AA1"].nunique(),
    "unique_aa2": a["AA2"].nunique(),
    "unique_dipeptides": a["DIPEPTIDE"].nunique(),
    "unique_complete_chemistries":
        a[["AcylChain","AA1","AA2"]].drop_duplicates().shape[0],
    "acyl_chain_annotation_counts":
        a["AcylChain"].value_counts().sort_index().to_dict()
}
write_json(qc, "qc/03_annotation_space.json")

log.info("Annotation rows: %s", f"{len(a):,}")
log.info("Unique source spectra: %s", f"{qc['unique_source_file_scan']:,}")
log.info("AA1/AA2: %d/%d", qc["unique_aa1"], qc["unique_aa2"])
log.info("Complete chemistries: %s", f"{qc['unique_complete_chemistries']:,}")
print("ANNOTATION SPACE PASSED")
