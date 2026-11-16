# ReverseMetabolomics-NAA-Dipeptides v0.5.0
## Official GNPS2 USI Integration and Ambiguity-Aware Rebuild

This release rebuilds the analysis from the authoritative current GNPS2 library:

**2,087 official spectral queries → official FASST output → 5,225 structural annotation records → Pan-ReDU metadata**

The central rule is:

> **A FASST match is spectral evidence at the query-USI/public-USI level.  
> Alternative structural annotations sharing one query USI are not counted as independent spectral observations.**

## Raw files

Place these files in `data/raw/` with exactly these names:

```text
GNPS2_comprehensive_24column_deposition_v1_5225.tsv
N_ACYL_DIPEPTIDES_dashboard.mgf
FASST_output_official_2087.tsv
PanReDU_FASST_exact_files.tsv
```

`PanReDU_FASST_exact_files.tsv` is optional during the first QC pass. Script 07
will skip gracefully if it is not present.

The dashboard MGF is the authoritative source for the current:

- GNPS2 accession
- query USI
- source mzML filename
- source scan
- representative library annotation

## Run

From the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 run_pipeline_v050.py
```

Then run tests:

```bash
python3 -m pytest -q
```

## Pipeline

### 01 — Validate inputs
Checks file existence and required columns.

### 02 — Official query space
Parses the dashboard MGF and requires:

```text
2,087 query spectra
2,087 unique accessions
2,087 unique USIs
2,087 unique source file + scan combinations
```

Output:

```text
data/intermediate/Official_Query_Space.tsv
```

### 03 — Library annotation space
Parses all 5,225 QC-passed annotations and preserves the four acyl chains:

```text
C4:0
C8:0
C16:0
C20:0
```

Output:

```text
data/intermediate/Library_Annotation_Space.tsv
```

### 04 — Query-to-annotation links
Links every structural annotation to the official GNPS2 query using:

```text
Source mzML filename + scan
```

This creates the explicit one-to-many relationship:

```text
one official spectral query
        ↓
one or more structural annotations
```

Outputs:

```text
data/intermediate/Query_to_Annotation_Links.tsv
data/processed/Query_Annotation_Ambiguity_Summary.tsv
```

The expected current shared-USI distribution is checked:

```text
1 annotation : 356 queries
2 annotations: 1,111 queries
3 annotations: 29 queries
4 annotations: 494 queries
6 annotations: 96 queries
8 annotations: 1 query
```

### 05 — Official FASST import
Validates every FASST `query_usi` against the official query space.

Queries absent from the FASST output are retained as legitimate **no-public-match**
queries rather than being treated as missing data.

Outputs:

```text
data/intermediate/FASST_Official_Matches.tsv
data/processed/Official_Query_FASST_Summary.tsv
```

The script reports the full search plus the preplanned thresholds:

```text
>= 0.70
>= 0.85
>= 0.90
```

### 06 — Ambiguity-aware structural projection

Creates two separate tables:

```text
FASST_Spectral_Evidence.tsv
```

This is the **counting layer**.

and

```text
FASST_to_Annotation_Projection.tsv
```

This is the **chemical interpretation / plotting layer**.

The projection may have many more rows because one query spectrum can have multiple
candidate structural annotations. The script verifies that the number of unique
spectral evidence keys does not change during projection.

### 07 — Pan-ReDU integration
Joins public matched spectra to Pan-ReDU using:

```text
Dataset accession + public filename
```

Output:

```text
data/processed/FASST_Spectral_Evidence_PanReDU.tsv
```

### 08 — Master QC
Requires the complete architecture to pass before any figure is rebuilt.

Output:

```text
qc/v0.5.0_master_qc.tsv
```

## Files to retire

Do not use the old accession mapping as the authoritative query source.
The current dashboard MGF and its official USIs now define the spectral-query space.

Previous human and bacterial analyses remain useful as exploratory work, but their
final manuscript versions should be regenerated from v0.5.0.

## Next release

After v0.5.0 passes QC, rebuild the manuscript figures from the new processed
tables rather than modifying the older v0.3/v0.4 outputs in place.
