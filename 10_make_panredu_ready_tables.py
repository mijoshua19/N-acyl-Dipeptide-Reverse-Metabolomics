#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

base = Path("data/processed/panredu_v051")
df = pd.read_csv(base/"FASST_Spectral_Evidence_PanReDU.tsv", sep="\t", low_memory=False)

core = ["EvidenceKey","query_usi","MatchedPublicUSI","Dataset","MatchedFilename","Cosine"]
context = [
    "NCBITaxonomy","NCBISuperkingdom","NCBIKingdom","NCBIPhylum","NCBIClass",
    "NCBIOrder","NCBIFamily","NCBIGenus","NCBISpecies","UBERONBodyPartName",
    "SampleType","SampleTypeSub1","HealthStatus","DOIDCommonName",
    "DOIDOntologyIndex","ComorbidityListDOIDIndex","ATTRIBUTE_DatasetAccession","filename"
]
keep = [c for c in core+context if c in df.columns]
compact = df[keep].drop_duplicates()
compact.to_csv(base/"PanReDU_AnalysisReady_Evidence.tsv", sep="\t", index=False)

rows=[]
for col in [c for c in context if c in compact.columns]:
    per = compact.groupby("EvidenceKey")[col].apply(
        lambda x: ((x.notna()) & (x.astype(str).str.strip()!="")).any()
    )
    rows.append({
        "Field":col,
        "EvidenceKeysWithValue":int(per.sum()),
        "TotalEvidenceKeys":int(len(per)),
        "CoveragePct":float(100*per.mean())
    })

pd.DataFrame(rows).sort_values("CoveragePct", ascending=False).to_csv(
    base/"PanReDU_Metadata_Field_Completeness.tsv", sep="\t", index=False)
print("Saved compact Pan-ReDU analysis-ready tables.")
