#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

from src.common import read_config, setup_logging, write_json

log = setup_logging("05_import_official_fasst")
cfg = read_config()

f = pd.read_csv(cfg["fasst_file"], sep="\t")
q = pd.read_csv("data/intermediate/Official_Query_Space.tsv", sep="\t")

f["Cosine"] = pd.to_numeric(f["Cosine"], errors="coerce")

unknown = sorted(set(f["query_usi"].dropna()) - set(q["query_usi"].dropna()))
if unknown:
    pd.DataFrame({"UnknownQueryUSI": unknown}).to_csv(
        "qc/05_unknown_fasst_queries.tsv", sep="\t", index=False
    )
    raise ValueError(
        f"FASST output contains {len(unknown)} query USIs not present "
        "in Official_Query_Space.tsv"
    )

# NOTE: absence from FASST is allowed; it means no public match at the workflow threshold.
qstatus = q[["QueryIndex","GNPS2LIB_ACCESSION","query_usi"]].copy()
hit_stats = f.groupby("query_usi").agg(
    FASSTRows=("USI","size"),
    UniquePublicUSIs=("USI","nunique"),
    MaxCosine=("Cosine","max"),
    MedianCosine=("Cosine","median"),
    UniqueDatasets=("Dataset","nunique")
).reset_index()
qstatus = qstatus.merge(hit_stats, on="query_usi", how="left")
for c in ["FASSTRows","UniquePublicUSIs","UniqueDatasets"]:
    qstatus[c] = qstatus[c].fillna(0).astype(int)
qstatus["HasPublicMatch"] = qstatus["FASSTRows"] > 0

out = Path("data/intermediate/FASST_Official_Matches.tsv")
f.to_csv(out, sep="\t", index=False)
qstatus.to_csv("data/processed/Official_Query_FASST_Summary.tsv",
               sep="\t", index=False)

threshold_stats = {}
for t in cfg["cosine_thresholds"]:
    x = f[f["Cosine"] >= t]
    threshold_stats[str(t)] = {
        "fasst_rows": len(x),
        "unique_query_usis": x["query_usi"].nunique(),
        "unique_public_usis": x["USI"].nunique(),
        "unique_datasets": x["Dataset"].nunique()
    }

qc = {
    "fasst_rows": len(f),
    "unique_queries_with_hits": f["query_usi"].nunique(),
    "official_queries_without_hits":
        int((~qstatus["HasPublicMatch"]).sum()),
    "unique_public_usis": f["USI"].nunique(),
    "unique_datasets": f["Dataset"].nunique(),
    "min_cosine": float(f["Cosine"].min()),
    "max_cosine": float(f["Cosine"].max()),
    "thresholds": threshold_stats
}
write_json(qc, "qc/05_official_fasst.json")

log.info("FASST rows: %s", f"{len(f):,}")
log.info("Queries with hits: %s / %s",
         f"{f.query_usi.nunique():,}", f"{len(q):,}")
log.info("Unique public USIs: %s", f"{f.USI.nunique():,}")
log.info("Datasets: %s", f"{f.Dataset.nunique():,}")
for t, d in threshold_stats.items():
    log.info("Cosine >= %s: %s rows | %s queries | %s public USIs | %s datasets",
             t, f"{d['fasst_rows']:,}", f"{d['unique_query_usis']:,}",
             f"{d['unique_public_usis']:,}", f"{d['unique_datasets']:,}")
print("OFFICIAL FASST IMPORT PASSED")
