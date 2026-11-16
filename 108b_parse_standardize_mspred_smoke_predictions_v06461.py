#!/usr/bin/env python3
from pathlib import Path
import ast, hashlib, json, math, os, re, shlex, subprocess, sys
import numpy as np
import pandas as pd

VERSION = "v0.6.4.6.1"
EXPECTED_COMMIT = "cd616f629a82040b950fbf8f0a983cc70ce1cf3b"
IMAGE = os.environ.get("MSPRED_IMAGE", "naa-ms-pred:cd616f6-cpu-amd64-r2").strip()

ROOT = Path.cwd().resolve()
SMOKE = ROOT / "data/processed/prediction_v06452_smoke_compat"
NATIVE = ROOT / "data/processed/prediction_v0643_native_inputs"
OUT = ROOT / "data/processed/prediction_v06461_standardized_smoke"
QC_DIR = ROOT / "qc"
OUT.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "GLACIER": SMOKE / "GLACIER/preds",
    "ICEBERG 2.1": SMOKE / "ICEBERG21/preds.hdf5",
}
MODEL_LABELS = {
    "GLACIER": NATIVE / "GLACIER/GLACIER_Tier1_smoke_labels.tsv",
    "ICEBERG 2.1": NATIVE / "ICEBERG21/ICEBERG21_Tier1_smoke_labels.tsv",
}
EXEC = SMOKE / "MSPRED_Tier1_Smoke_Execution.tsv"
SMOKE_QC = SMOKE / "MSPRED_Tier1_Smoke_QC.tsv"

EXPECTED_STRUCTURES = 10
EXPECTED_RECORDS = 30

for p in [*MODEL_FILES.values(), *MODEL_LABELS.values(), EXEC, SMOKE_QC]:
    if not p.exists():
        sys.exit(f"Missing required prerequisite: {p}")

def run(cmd, timeout=None):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, check=False, timeout=timeout)

def sha256_file(path, chunk=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def project_path(path):
    return "/project/" + str(Path(path).resolve().relative_to(ROOT)).replace(os.sep, "/")

def find_col(df, names):
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None

def parse_ce(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    m = re.search(r"(-?\d+(?:\.\d+)?)", str(x))
    return float(m.group(1)) if m else np.nan

def parse_ce_list(x):
    if isinstance(x, str):
        try:
            vals = ast.literal_eval(x)
        except Exception:
            vals = []
    elif isinstance(x, (list, tuple, np.ndarray)):
        vals = list(x)
    else:
        vals = []
    out = []
    for v in vals:
        ce = parse_ce(v)
        if np.isfinite(ce):
            out.append(float(ce))
    return out

def normalize_pred_name(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).strip()
    if s.startswith("pred_"):
        s = s[5:]
    s = re.sub(r"_collision\s+[-+]?\d+(?:\.\d+)?$", "", s, flags=re.I)
    return s

def canonical_key(spec, ce):
    return (normalize_pred_name(spec), round(float(ce), 6))

def build_crosswalk(model, path):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    spec_c = find_col(df, ["spec", "BenchmarkSpec", "query_usi", "QueryUSI"])
    smi_c = find_col(df, ["smiles", "SMILES"])
    adduct_c = find_col(df, ["ionization", "Adduct", "Precursor_type"])
    inst_c = find_col(df, ["instrument", "Instrument", "Instrument_type"])
    prec_c = find_col(df, ["precursor", "PrecursorMz", "precursor_mz"])
    query_c = find_col(df, ["QueryUSI", "query_usi", "USI", "usi"])
    ce_c = find_col(df, ["collision_energies", "CollisionEnergies"])
    if spec_c is None or ce_c is None:
        raise RuntimeError(f"{model}: missing spec/collision_energies columns.")
    rows = []
    for _, r in df.iterrows():
        spec = str(r[spec_c])
        for ce in parse_ce_list(r[ce_c]):
            rows.append({
                "Model": model,
                "BenchmarkSpec": spec,
                "SpecNorm": normalize_pred_name(spec),
                "QueryUSI": str(r[query_c]) if query_c and pd.notna(r[query_c]) else spec,
                "SMILES": r[smi_c] if smi_c else pd.NA,
                "Adduct": r[adduct_c] if adduct_c else pd.NA,
                "Instrument": r[inst_c] if inst_c else pd.NA,
                "PrecursorMz": float(r[prec_c]) if prec_c and pd.notna(r[prec_c]) else np.nan,
                "CollisionEnergy_eV": float(ce),
            })
    out = pd.DataFrame(rows)
    if len(out) != EXPECTED_RECORDS:
        raise RuntimeError(f"{model}: expected 30 spec-CE rows, observed {len(out)}.")
    return out

exec_df = pd.read_csv(EXEC, sep="\t", low_memory=False)
if not (pd.to_numeric(exec_df["ReturnCode"], errors="coerce") == 0).all():
    raise RuntimeError("Requires successful v0.6.4.5.2 smoke execution.")
smoke_qc = pd.read_csv(SMOKE_QC, sep="\t", low_memory=False)
if "Status" in smoke_qc.columns and not smoke_qc["Status"].eq("PASS").all():
    raise RuntimeError("v0.6.4.5.2 smoke QC contains non-PASS rows.")

r = run(["docker","run","--rm","--platform","linux/amd64",IMAGE,
         "bash","-c","cd /opt/ms-pred && git rev-parse HEAD"], timeout=120)
if r.returncode != 0 or r.stdout.strip() != EXPECTED_COMMIT:
    raise RuntimeError("Frozen Docker image/commit verification failed.")

EXPORTER = r'''
import json, inspect, sys
from pathlib import Path
import numpy as np
import ms_pred.common as common

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
audit_path = Path(sys.argv[3])

audit = {
    "input_path": str(input_path),
    "PredSpecDB_class": str(common.PredSpecDB),
    "methods": sorted([x for x in dir(common.PredSpecDB) if not x.startswith("__")]),
}
try:
    audit["init_signature"] = str(inspect.signature(common.PredSpecDB))
except Exception as e:
    audit["init_signature"] = f"UNAVAILABLE: {e}"
try:
    audit["read_signature"] = str(inspect.signature(common.PredSpecDB.read))
except Exception as e:
    audit["read_signature"] = f"UNAVAILABLE: {e}"

db = common.PredSpecDB(input_path)
audit["instance_attrs"] = sorted([x for x in dir(db) if not x.startswith("__")])

def py(v):
    if v is None: return None
    if isinstance(v, bytes): return v.decode("utf-8", errors="replace")
    if isinstance(v, np.bytes_): return bytes(v).decode("utf-8", errors="replace")
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, np.ndarray): return v.tolist()
    return v

def attr(ms, *names):
    for n in names:
        if hasattr(ms, n):
            v = getattr(ms, n)
            if v is not None:
                return py(v)
    return None

def record_to_json(name_obj, ms, read_key):
    masses = attr(ms, "masses", "mz", "mzs")
    intens = attr(ms, "intens", "intensity", "intensities")
    if masses is None and hasattr(ms, "spec") and getattr(ms, "spec") is not None:
        arr = np.asarray(getattr(ms, "spec"))
        if arr.ndim == 2 and arr.shape[1] >= 2:
            masses, intens = arr[:,0].tolist(), arr[:,1].tolist()
    info = {}
    if hasattr(ms, "info") and isinstance(ms.info, dict):
        for k,v in ms.info.items():
            try: info[str(k)] = py(v)
            except Exception: info[str(k)] = str(v)
    return {
        "record_name": py(name_obj),
        "read_key": repr(read_key),
        "root_canonical_smiles": attr(ms, "root_canonical_smiles", "smiles", "root_smiles"),
        "adduct": attr(ms, "adduct", "ionization"),
        "collision_energy": attr(ms, "collision_energy", "ce"),
        "remark": attr(ms, "remark"),
        "masses": masses,
        "intens": intens,
        "frags": attr(ms, "frags", "int_frags"),
        "info": info,
    }

raw_names = None
name_source = None
for meth in ["get_all_names", "keys", "get_names"]:
    if hasattr(db, meth) and callable(getattr(db, meth)):
        try:
            vals = list(getattr(db, meth)())
            if vals:
                raw_names, name_source = vals, meth
                break
        except Exception as e:
            audit[f"{meth}_error"] = repr(e)

if raw_names is None:
    for an in ["name_to_dict", "name_to_file", "name_map", "names"]:
        if hasattr(db, an):
            try:
                obj = getattr(db, an)
                vals = list(obj.keys()) if isinstance(obj, dict) else list(obj)
                if vals:
                    raw_names, name_source = vals, an
                    break
            except Exception as e:
                audit[f"{an}_error"] = repr(e)

audit["name_source"] = name_source
audit["raw_name_count"] = 0 if raw_names is None else len(raw_names)
audit["raw_name_examples"] = [] if raw_names is None else [repr(x) for x in raw_names[:10]]

if not raw_names:
    audit_path.write_text(json.dumps(audit, indent=2, default=str))
    raise RuntimeError("Could not enumerate PredSpecDB records.")

records, failures = [], []
for item in raw_names:
    attempts = []
    if isinstance(item, (tuple,list)):
        attempts.append(("splat", tuple(item)))
    attempts.append(("single", (item,)))
    loaded = None
    used = None
    errors = []
    for label,args in attempts:
        try:
            loaded = db.read(*args)
            used = (label,args)
            break
        except Exception as e:
            errors.append({"attempt":label,"args":repr(args),"error":repr(e)})
    if loaded is None:
        failures.append({"item":repr(item),"errors":errors})
        continue
    records.append(record_to_json(item, loaded, used))

audit["successful_records"] = len(records)
audit["read_failure_count"] = len(failures)
audit["read_failures"] = failures[:20]
audit_path.write_text(json.dumps(audit, indent=2, default=str))

with output_path.open("w") as fh:
    for rec in records:
        fh.write(json.dumps(rec, default=str) + "\n")
if not records:
    raise RuntimeError("PredSpecDB opened but no records could be read.")
'''

exporter_host = OUT / "_predspecdb_native_exporter.py"
exporter_host.write_text(EXPORTER)

interface_rows = []
native_records = {}

for model, native_path in MODEL_FILES.items():
    slug = "GLACIER" if model == "GLACIER" else "ICEBERG21"
    jsonl = OUT / f"MSPRED_PredSpecDB_Exporter_{slug}.jsonl"
    audit_json = OUT / f"MSPRED_PredSpecDB_Interface_{slug}.json"
    cmd = [
        "docker","run","--rm","--platform","linux/amd64",
        "--mount",f"type=bind,src={ROOT},dst=/project,readonly",
        "--mount",f"type=bind,src={OUT},dst=/export_out",
        IMAGE,"/opt/ms-pred/.venv/bin/python",
        project_path(exporter_host), project_path(native_path),
        f"/export_out/{jsonl.name}", f"/export_out/{audit_json.name}",
    ]
    print(f"\nPredSpecDB-native export: {model}\n" + "-"*40)
    r = run(cmd, timeout=600)
    (OUT / f"{slug}_export_stdout.log").write_text(r.stdout or "")
    (OUT / f"{slug}_export_stderr.log").write_text(r.stderr or "")
    if r.returncode != 0:
        raise RuntimeError(f"{model} PredSpecDB export failed. Inspect {audit_json}.")
    audit = json.loads(audit_json.read_text())
    interface_rows.append({
        "Model": model,
        "NativeOutputFile": str(native_path),
        "NativeOutputSHA256": sha256_file(native_path),
        "NameEnumerationSource": audit.get("name_source"),
        "EnumeratedNames": audit.get("raw_name_count"),
        "SuccessfullyReadRecords": audit.get("successful_records"),
        "ReadFailures": audit.get("read_failure_count"),
        "PredSpecDBReadSignature": audit.get("read_signature"),
        "PredSpecDBInitSignature": audit.get("init_signature"),
    })
    rows = [json.loads(x) for x in jsonl.read_text().splitlines() if x.strip()]
    native_records[model] = rows

interface_audit = pd.DataFrame(interface_rows)
interface_audit.to_csv(OUT/"MSPRED_PredSpecDB_Interface_Audit.tsv", sep="\t", index=False)

crosswalks = {m: build_crosswalk(m,p) for m,p in MODEL_LABELS.items()}
record_rows, peak_rows, crosswalk_rows = [], [], []

for model, recs in native_records.items():
    cw = crosswalks[model]
    cw_index = {canonical_key(r.BenchmarkSpec,r.CollisionEnergy_eV): r
                for r in cw.itertuples(index=False)}
    for idx, rec in enumerate(recs, start=1):
        native_name = normalize_pred_name(rec.get("record_name"))
        ce = parse_ce(rec.get("collision_energy"))
        resolved = cw_index.get(canonical_key(native_name,ce)) if native_name and np.isfinite(ce) else None
        method = "native_name+collision_energy"
        if resolved is None and np.isfinite(ce):
            raw_name = str(rec.get("record_name",""))
            matches = [r for r in cw.itertuples(index=False)
                       if normalize_pred_name(r.BenchmarkSpec) in raw_name
                       and abs(float(r.CollisionEnergy_eV)-ce)<1e-6]
            if len(matches)==1:
                resolved = matches[0]
                method = "benchmark_spec_substring+collision_energy"
        status = "RESOLVED" if resolved is not None else "UNRESOLVED"
        if resolved is not None:
            bs,qu = resolved.BenchmarkSpec,resolved.QueryUSI
            smi = rec.get("root_canonical_smiles") or resolved.SMILES
            adduct = rec.get("adduct") or resolved.Adduct
            inst,prec,ce_final = resolved.Instrument,resolved.PrecursorMz,float(resolved.CollisionEnergy_eV)
        else:
            bs=qu=pd.NA; smi=rec.get("root_canonical_smiles"); adduct=rec.get("adduct")
            inst=pd.NA; prec=np.nan; ce_final=ce

        masses = np.asarray(rec.get("masses") or [], dtype=float).reshape(-1)
        intens = np.asarray(rec.get("intens") or [], dtype=float).reshape(-1)
        if masses.size != intens.size:
            raise RuntimeError(f"{model} record {idx}: masses/intens mismatch.")
        keep = np.isfinite(masses)&np.isfinite(intens)&(masses>0)&(intens>0)
        masses,intens = masses[keep],intens[keep]
        if len(intens):
            max_norm=intens/intens.max(); sum_norm=intens/intens.sum()
            order=np.argsort(-intens,kind="stable")
            ranks=np.empty(len(intens),dtype=int); ranks[order]=np.arange(1,len(intens)+1)
        else:
            max_norm=sum_norm=np.array([],dtype=float); ranks=np.array([],dtype=int)

        rec_id=f"{'GLACIER' if model=='GLACIER' else 'ICEBERG21'}_{idx:03d}"
        record_rows.append({
            "Model":model,"PredictionRecordID":rec_id,"BenchmarkSpec":bs,"QueryUSI":qu,
            "SMILES":smi,"Adduct":adduct,"Instrument":inst,"PrecursorMz":prec,
            "CollisionEnergy_eV":ce_final,"NativeRecordName":rec.get("record_name"),
            "NativeCollisionEnergy":ce,"NativeRemark":rec.get("remark"),
            "NativeReadKey":rec.get("read_key"),"NativeOutputFile":str(MODEL_FILES[model]),
            "NativeOutputSHA256":sha256_file(MODEL_FILES[model]),
            "NativePeakCount":len(np.asarray(rec.get("masses") or []).reshape(-1)),
            "RetainedPositiveFinitePeaks":len(masses),
            "CrosswalkStatus":status,"CrosswalkMethod":method if resolved is not None else "unresolved",
        })
        crosswalk_rows.append({
            "Model":model,"PredictionRecordID":rec_id,"NativeRecordName":rec.get("record_name"),
            "NativeCollisionEnergy":ce,"BenchmarkSpec":bs,"CollisionEnergy_eV":ce_final,
            "CrosswalkStatus":status,"CrosswalkMethod":method if resolved is not None else "unresolved",
        })
        for j in range(len(masses)):
            peak_rows.append({
                "Model":model,"PredictionRecordID":rec_id,"BenchmarkSpec":bs,"QueryUSI":qu,
                "SMILES":smi,"Adduct":adduct,"Instrument":inst,"PrecursorMz":prec,
                "CollisionEnergy_eV":ce_final,"PeakMz":float(masses[j]),
                "PredictedIntensityRaw":float(intens[j]),
                "PredictedIntensityMaxNorm":float(max_norm[j]),
                "PredictedIntensitySumNorm":float(sum_norm[j]),
                "PeakRank":int(ranks[j]),"NativeRecordName":rec.get("record_name"),
                "NativeOutputFile":str(MODEL_FILES[model]),
            })

record_meta=pd.DataFrame(record_rows)
peaks=pd.DataFrame(peak_rows)
crosswalk_audit=pd.DataFrame(crosswalk_rows)

record_meta.to_csv(OUT/"MSPRED_Smoke_Prediction_Record_Metadata.tsv",sep="\t",index=False)
peaks.to_csv(OUT/"MSPRED_Smoke_Prediction_Peaks_Long.tsv",sep="\t",index=False)
crosswalk_audit.to_csv(OUT/"MSPRED_Smoke_Prediction_Crosswalk_Audit.tsv",sep="\t",index=False)

spectrum_summary=(peaks.groupby(
    ["Model","PredictionRecordID","BenchmarkSpec","QueryUSI","CollisionEnergy_eV"],
    dropna=False,as_index=False).agg(
        PredictedPeaks=("PeakMz","size"),MinPeakMz=("PeakMz","min"),
        MaxPeakMz=("PeakMz","max"),MaxRawIntensity=("PredictedIntensityRaw","max"),
        SumRawIntensity=("PredictedIntensityRaw","sum")))
spectrum_summary.to_csv(OUT/"MSPRED_Smoke_Prediction_Spectrum_Summary.tsv",sep="\t",index=False)

coverage_rows=[]; qc_rows=[]
def add_qc(model,check,expected,observed):
    status="PASS" if ((expected==">0" and float(observed)>0) or
                      (expected!=">0" and str(expected)==str(observed))) else "FAIL"
    qc_rows.append({"Model":model,"Check":check,"Expected":expected,"Observed":observed,"Status":status})

for model in MODEL_FILES:
    rec=record_meta[record_meta["Model"].eq(model)].copy()
    pk=peaks[peaks["Model"].eq(model)].copy()
    nrec=rec["PredictionRecordID"].nunique()
    nspec=rec["BenchmarkSpec"].dropna().nunique()
    unresolved=int(rec["CrosswalkStatus"].ne("RESOLVED").sum())
    dup=int(rec.dropna(subset=["BenchmarkSpec","CollisionEnergy_eV"]).duplicated(["BenchmarkSpec","CollisionEnergy_eV"]).sum())
    zero=int((rec["RetainedPositiveFinitePeaks"]==0).sum())
    ces=sorted(pd.to_numeric(rec["CollisionEnergy_eV"],errors="coerce").dropna().unique().tolist())
    cestr="|".join(f"{x:g}" for x in ces)
    coverage_rows.append({"Model":model,"NativePredSpecDBRecords":nrec,"UniqueBenchmarkSpecs":nspec,
                          "CollisionEnergySet_eV":cestr,"CrosswalkUnresolvedRecords":unresolved,
                          "DuplicateSpecCEPairs":dup,"ZeroPositivePeakRecords":zero,
                          "TotalStandardizedPeaks":len(pk),
                          "MedianPeaksPerSpectrum":float(rec["RetainedPositiveFinitePeaks"].median()) if len(rec) else np.nan,
                          "MaxPeaksPerSpectrum":int(rec["RetainedPositiveFinitePeaks"].max()) if len(rec) else 0})
    add_qc(model,"Native PredSpecDB records",30,nrec)
    add_qc(model,"Unique smoke structures",10,nspec)
    add_qc(model,"CE set","25|50|75",cestr)
    add_qc(model,"Unresolved crosswalk records",0,unresolved)
    add_qc(model,"Duplicate spec-CE pairs",0,dup)
    add_qc(model,"Zero-positive-peak spectra",0,zero)
    add_qc(model,"Standardized peaks",">0",len(pk))

coverage=pd.DataFrame(coverage_rows); qc=pd.DataFrame(qc_rows)
coverage.to_csv(OUT/"MSPRED_Smoke_Prediction_Coverage.tsv",sep="\t",index=False)
qc.to_csv(QC_DIR/"108b_parse_standardize_mspred_smoke_predictions_v06461.tsv",sep="\t",index=False)

overall=bool(len(qc) and qc["Status"].eq("PASS").all())
notes=f'''ms-pred PredSpecDB-native smoke standardization {VERSION}

v0.6.4.6.1 uses ms_pred.common.PredSpecDB inside the exact frozen Docker
environment instead of inferring ms-pred's internal raw HDF5 layout.

Image: {IMAGE}
Commit: {EXPECTED_COMMIT}

Expected universe:
10 structures/model x 3 CE values (25|50|75 eV) = 30 records/model.

No CE merging or experimental similarity scoring is performed.
Raw positive finite predicted m/z and intensity values are preserved.
Only max-normalized intensity, sum-normalized intensity, and peak rank are derived.

OVERALL STATUS
==============
{"PASS" if overall else "FAIL"}
'''
(OUT/"MSPRED_Smoke_Standardization_Notes.txt").write_text(notes)

try:
    exporter_host.unlink()
except Exception:
    pass

pd.set_option("display.max_columns",None); pd.set_option("display.width",240)
print(f"\nms-pred PredSpecDB-native smoke standardization {VERSION}\n"+"="*61)
print("\nPredSpecDB interface audit\n--------------------------")
print(interface_audit.to_string(index=False))
print("\nPrediction coverage\n-------------------")
print(coverage.to_string(index=False))
print("\nQC\n--")
print(qc.to_string(index=False))
print("\nSaved\n-----")
for p in [OUT/"MSPRED_PredSpecDB_Interface_Audit.tsv",
          OUT/"MSPRED_Smoke_Prediction_Record_Metadata.tsv",
          OUT/"MSPRED_Smoke_Prediction_Peaks_Long.tsv",
          OUT/"MSPRED_Smoke_Prediction_Spectrum_Summary.tsv",
          OUT/"MSPRED_Smoke_Prediction_Coverage.tsv",
          OUT/"MSPRED_Smoke_Prediction_Crosswalk_Audit.tsv",
          OUT/"MSPRED_Smoke_Standardization_Notes.txt",
          QC_DIR/"108b_parse_standardize_mspred_smoke_predictions_v06461.tsv"]:
    print(p)
print()
if overall:
    print(f"MS-PRED PREDSPECDB SMOKE STANDARDIZATION {VERSION} PASSED")
else:
    print(f"MS-PRED PREDSPECDB SMOKE STANDARDIZATION {VERSION} FAILED")
    sys.exit(2)
