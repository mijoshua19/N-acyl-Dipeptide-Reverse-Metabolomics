#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import pandas as pd

def log(x): print(x, flush=True)

def norm_filename(x):
    if pd.isna(x): return None
    s = str(x).strip().replace("\\","/")
    return Path(s).name if s else None

def extract_public_filename(usi):
    if pd.isna(usi): return None
    p = str(usi).split(":")
    return norm_filename(p[2]) if len(p) >= 3 else None

def choose(cols, candidates, label):
    for c in candidates:
        if c in cols: return c
    raise ValueError(f"Could not find {label} column. Tried {candidates}. First columns: {cols[:30]}")

ap = argparse.ArgumentParser()
ap.add_argument("--panredu", required=True)
ap.add_argument("--evidence", default="data/processed/FASST_Spectral_Evidence.tsv")
ap.add_argument("--outdir", default="data/processed/panredu_v051")
ap.add_argument("--chunksize", type=int, default=200000)
args = ap.parse_args()

panredu = Path(os.path.expanduser(args.panredu)).resolve()
evidence = Path(args.evidence)
outdir = Path(args.outdir)
qcdir = Path("qc")
outdir.mkdir(parents=True, exist_ok=True)
qcdir.mkdir(exist_ok=True)

if not panredu.exists(): raise FileNotFoundError(panredu)
if not evidence.exists(): raise FileNotFoundError(evidence)

ev = pd.read_csv(evidence, sep="\t", low_memory=False)
for c in ["query_usi","MatchedPublicUSI","Dataset","EvidenceKey"]:
    if c not in ev.columns: raise ValueError(f"Evidence table missing {c}")

ev["MatchedFilename"] = ev["MatchedPublicUSI"].map(extract_public_filename)
ev["Dataset"] = ev["Dataset"].astype(str).str.strip()

keys_df = ev[["Dataset","MatchedFilename"]].dropna().drop_duplicates()
needed_keys = set(map(tuple, keys_df.itertuples(index=False, name=None)))
needed_datasets = set(keys_df["Dataset"])

keys_df.to_csv(outdir/"PanReDU_Needed_Dataset_File_Keys.tsv", sep="\t", index=False)

cols = list(pd.read_csv(panredu, sep=",", nrows=0).columns)
dataset_col = choose(cols, ["ATTRIBUTE_DatasetAccession","DatasetAccession","dataset_accession","Dataset"], "dataset accession")
filename_col = choose(cols, ["filename","Filename","FILE_NAME","FileName"], "filename")

log("\nPan-ReDU large-file integration")
log("===============================")
log(f"Pan-ReDU: {panredu}")
log(f"Evidence rows: {len(ev):,}")
log(f"Unique evidence keys: {ev['EvidenceKey'].nunique():,}")
log(f"Needed Dataset+filename keys: {len(needed_keys):,}")
log(f"Dataset column: {dataset_col}")
log(f"Filename column: {filename_col}")

retained = []
total = dataset_candidates = exact = 0

reader = pd.read_csv(
    panredu, sep=",", chunksize=args.chunksize, low_memory=False,
    dtype={dataset_col:"string", filename_col:"string"}
)

for i, chunk in enumerate(reader, 1):
    total += len(chunk)
    ds = chunk[dataset_col].astype("string").str.strip()
    c = chunk.loc[ds.isin(needed_datasets)].copy()
    dataset_candidates += len(c)
    if len(c):
        c["_PanReDU_Dataset"] = c[dataset_col].astype("string").str.strip()
        c["_PanReDU_Filename"] = c[filename_col].map(norm_filename)
        mask = [(d,f) in needed_keys for d,f in zip(c["_PanReDU_Dataset"], c["_PanReDU_Filename"])]
        c = c.loc[mask]
        exact += len(c)
        if len(c): retained.append(c)
    if i % 5 == 0:
        log(f"chunks={i:,} scanned={total:,} dataset-candidates={dataset_candidates:,} retained={exact:,}")

meta = pd.concat(retained, ignore_index=True) if retained else pd.DataFrame(columns=cols+["_PanReDU_Dataset","_PanReDU_Filename"])
before = len(meta)
meta = meta.drop_duplicates()
meta.to_csv(outdir/"PanReDU_Relevant_Metadata.tsv", sep="\t", index=False)

key_mult = (meta.groupby(["_PanReDU_Dataset","_PanReDU_Filename"], dropna=False)
            .size().reset_index(name="MetadataRowsPerKey")
            .sort_values("MetadataRowsPerKey", ascending=False)) if len(meta) else pd.DataFrame(
                columns=["_PanReDU_Dataset","_PanReDU_Filename","MetadataRowsPerKey"])
key_mult.to_csv(outdir/"PanReDU_Key_Multiplicity.tsv", sep="\t", index=False)

multi = key_mult[key_mult["MetadataRowsPerKey"] > 1]
if len(multi):
    multi.to_csv(qcdir/"07b_panredu_keys_with_multiple_metadata_rows.tsv", sep="\t", index=False)

joined = ev.merge(
    meta,
    left_on=["Dataset","MatchedFilename"],
    right_on=["_PanReDU_Dataset","_PanReDU_Filename"],
    how="left", validate="many_to_many", suffixes=("","_PanReDU")
) if len(meta) else ev.copy()

joined.to_csv(outdir/"FASST_Spectral_Evidence_PanReDU.tsv", sep="\t", index=False)

if "_PanReDU_Dataset" in joined.columns:
    cov = joined.groupby("EvidenceKey")["_PanReDU_Dataset"].apply(lambda x: x.notna().any())
else:
    cov = pd.Series(False, index=ev["EvidenceKey"].unique())

summary = ev.groupby("EvidenceKey", as_index=False).agg(
    query_usi=("query_usi","first"),
    MatchedPublicUSI=("MatchedPublicUSI","first"),
    Dataset=("Dataset","first"),
    MatchedFilename=("MatchedFilename","first")
)
summary["HasPanReDUMetadata"] = summary["EvidenceKey"].map(cov).fillna(False)
summary.to_csv(outdir/"PanReDU_Evidence_Coverage.tsv", sep="\t", index=False)

dataset_cov = summary.groupby("Dataset", as_index=False).agg(
    EvidenceKeys=("EvidenceKey","nunique"),
    EvidenceKeysWithPanReDU=("HasPanReDUMetadata","sum")
)
dataset_cov["CoveragePct"] = 100 * dataset_cov["EvidenceKeysWithPanReDU"] / dataset_cov["EvidenceKeys"]
dataset_cov.sort_values(["EvidenceKeys","CoveragePct"], ascending=[False,False]).to_csv(
    outdir/"PanReDU_Dataset_Coverage.tsv", sep="\t", index=False)

unique_ev = ev["EvidenceKey"].nunique()
with_meta = int(summary["HasPanReDUMetadata"].sum())
qc = {
    "panredu_file": str(panredu),
    "panredu_rows_scanned": int(total),
    "panredu_rows_in_needed_datasets": int(dataset_candidates),
    "panredu_rows_exactly_retained": int(exact),
    "panredu_exact_duplicate_rows_removed": int(before-len(meta)),
    "panredu_unique_relevant_dataset_filename_keys":
        int(meta[["_PanReDU_Dataset","_PanReDU_Filename"]].drop_duplicates().shape[0]) if len(meta) else 0,
    "panredu_keys_with_multiple_metadata_rows": int(len(multi)),
    "spectral_evidence_rows": int(len(ev)),
    "unique_evidence_keys": int(unique_ev),
    "evidence_keys_with_panredu_metadata": with_meta,
    "evidence_keys_without_panredu_metadata": int(unique_ev-with_meta),
    "metadata_coverage_pct": float(100*with_meta/unique_ev if unique_ev else 0),
    "joined_rows": int(len(joined)),
    "joined_unique_evidence_keys": int(joined["EvidenceKey"].nunique()),
    "evidence_key_integrity_preserved": bool(joined["EvidenceKey"].nunique() == unique_ev)
}
with open(qcdir/"07b_panredu_large_integration.json","w") as f:
    json.dump(qc,f,indent=2)

log("\nPan-ReDU integration QC")
log("=======================")
for k,v in qc.items(): log(f"{k}: {v}")
if not qc["evidence_key_integrity_preserved"]:
    raise RuntimeError("Evidence key count changed during metadata integration.")
log("\nPAN-REDU LARGE-FILE INTEGRATION PASSED")
