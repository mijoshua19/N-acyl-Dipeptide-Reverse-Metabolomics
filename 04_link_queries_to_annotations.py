#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

from src.common import read_config, setup_logging, write_json

log = setup_logging("04_link_queries_to_annotations")
cfg = read_config()

q = pd.read_csv("data/intermediate/Official_Query_Space.tsv", sep="\t")
a = pd.read_csv("data/intermediate/Library_Annotation_Space.tsv", sep="\t")

q["SourceScan"] = pd.to_numeric(q["SourceScan"], errors="coerce").astype("Int64")
a["SourceScan"] = pd.to_numeric(a["SourceScan"], errors="coerce").astype("Int64")

links = a.merge(
    q[[
        "QueryIndex","GNPS2LIB_ACCESSION","query_usi",
        "SourceFile","SourceScan","MGF_Name","PrecursorMz","MGF_Adduct"
    ]],
    on=["SourceFile","SourceScan"],
    how="left",
    validate="many_to_one"
)

if links["query_usi"].isna().any():
    missing = links[links["query_usi"].isna()]
    missing.to_csv("qc/04_unlinked_annotations.tsv", sep="\t", index=False)
    raise ValueError(f"{len(missing)} of 5,225 annotations could not be linked.")

# Every official query should be represented at least once.
missing_queries = q.loc[~q["query_usi"].isin(links["query_usi"]), :]
if len(missing_queries):
    missing_queries.to_csv("qc/04_queries_without_annotation.tsv", sep="\t", index=False)
    raise ValueError(f"{len(missing_queries)} official queries have no annotation link.")

out = Path("data/intermediate/Query_to_Annotation_Links.tsv")
links.to_csv(out, sep="\t", index=False)

group = links.groupby(
    ["GNPS2LIB_ACCESSION","query_usi"], dropna=False
).agg(
    AnnotationCount=("AnnotationIndex","size"),
    DistinctStructures=("SMILES","nunique"),
    DistinctInChIKeys=("INCHIAUX","nunique"),
    SourceFile=("SourceFile","first"),
    SourceScan=("SourceScan","first"),
    AllAnnotations=("COMPOUND_NAME_ORIGINAL",
                    lambda x: " | ".join(sorted(set(map(str, x)))))
).reset_index()

group["AmbiguityClass"] = group["AnnotationCount"].apply(
    lambda n: "single_annotation" if n == 1 else "shared_spectral_evidence"
)

group.to_csv("data/processed/Query_Annotation_Ambiguity_Summary.tsv",
             sep="\t", index=False)

dist = group["AnnotationCount"].value_counts().sort_index().to_dict()

qc = {
    "annotation_link_rows": len(links),
    "linked_annotation_rows": links["query_usi"].notna().sum(),
    "official_queries_linked": links["query_usi"].nunique(),
    "single_annotation_queries": int((group["AnnotationCount"] == 1).sum()),
    "shared_evidence_queries": int((group["AnnotationCount"] > 1).sum()),
    "annotation_count_per_query_distribution": {str(k): int(v) for k,v in dist.items()},
    "max_annotations_per_query": int(group["AnnotationCount"].max())
}
write_json(qc, "qc/04_query_annotation_links.json")

log.info("Linked annotation rows: %s", f"{len(links):,}")
log.info("Linked official queries: %s", f"{links.query_usi.nunique():,}")
log.info("Single-annotation queries: %s", f"{qc['single_annotation_queries']:,}")
log.info("Shared-evidence queries: %s", f"{qc['shared_evidence_queries']:,}")
log.info("Group-size distribution: %s", qc["annotation_count_per_query_distribution"])
print("QUERY-ANNOTATION LINKAGE PASSED")
