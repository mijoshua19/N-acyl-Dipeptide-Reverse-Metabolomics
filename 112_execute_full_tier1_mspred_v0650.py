#!/usr/bin/env python3
"""112_execute_full_tier1_mspred_v0650.py — restart-safe full Tier-1 execution."""
from pathlib import Path
import datetime as dt, hashlib, json, os, shlex, shutil, subprocess, sys, time, traceback
import pandas as pd

VERSION="v0.6.5.0"
ROOT=Path.cwd().resolve()
FREEZE=ROOT/"data/processed/prediction_v0649_full_tier1_execution_freeze"
EX=FREEZE/"EXECUTION"
AUTH=EX/"MSPRED_FullTier1_Execution_Authorization.tsv"
MANIFEST=EX/"MSPRED_FullTier1_Execution_Manifest.tsv"
HASHES=EX/"MSPRED_FullTier1_Input_SHA256.tsv"
GLABEL=FREEZE/"GLACIER/GLACIER_FullTier1_labels.tsv"
ILABEL=FREEZE/"ICEBERG21/ICEBERG21_FullTier1_labels.tsv"
CKROOT=ROOT/"data/external/prediction_checkpoints_v06443/extracted"
GCK=CKROOT/"glacier_massspecgym/best.ckpt"
IGEN=CKROOT/"iceberg21_msg_simulation/gen/best.ckpt"
IINT=CKROOT/"iceberg21_msg_simulation/inten_contr/best.ckpt"
OUT=ROOT/"data/processed/prediction_v0650_full_tier1_execution"
GOUT=OUT/"GLACIER"; IOUT=OUT/"ICEBERG21"; PROV=OUT/"PROVENANCE"; QC=ROOT/"qc"
for d in [OUT,GOUT,IOUT,PROV,QC]: d.mkdir(parents=True,exist_ok=True)

IMAGE=os.environ.get("MSPRED_IMAGE","naa-ms-pred:cd616f6-cpu-amd64-r2")
COMMIT="cd616f629a82040b950fbf8f0a983cc70ce1cf3b"
G_SHA="5a47cecca707d3abd5a49c7dbac99d100aa2a586d5d4f848f1e8e35140d7db11"
IG_SHA="1eda5f3d9cda8345a93c0c480c3c848de840a611017a3f1641007fec1afb7a70"
II_SHA="e074c0392638a71589e68d80acfd9ad53ae0587c523249cfa90e04ee50ee4f58"
GMOD=os.environ.get("GLACIER_MODULE","ms_pred.glacier.run_joint_inference")
IMOD=os.environ.get("ICEBERG_MODULE","ms_pred.iceberg.run_joint_inference")

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def now(): return dt.datetime.now().astimezone().isoformat(timespec="seconds")
def run(c,timeout=None): return subprocess.run(c,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
def base(out): return ["docker","run","--rm","--platform","linux/amd64","-v",f"{ROOT}:/project:ro","-v",f"{out}:/output:rw","-w","/project",IMAGE]
def dump(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
def native(model): return (GOUT/"preds") if model=="GLACIER" else (IOUT/"preds.hdf5")
def sent(model): return (GOUT if model=="GLACIER" else IOUT)/".execution_complete.json"
def nonempty(p):
    if not p.exists(): return False
    if p.is_file(): return p.stat().st_size>0
    return any(x.is_file() and x.stat().st_size>0 for x in p.rglob("*"))
def hashout(p):
    if p.is_file(): return sha(p)
    h=hashlib.sha256()
    for x in sorted(y for y in p.rglob("*") if y.is_file()):
        h.update(str(x.relative_to(p)).encode()+b"\0"+sha(x).encode()+b"\0")
    return h.hexdigest()
def bytesout(p):
    if not p.exists(): return 0
    return p.stat().st_size if p.is_file() else sum(x.stat().st_size for x in p.rglob("*") if x.is_file())
def clean(model):
    d=GOUT if model=="GLACIER" else IOUT
    stamp=dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    for n in ["stdout.log","stderr.log"]:
        p=d/n
        if p.exists(): p.rename(d/f"{p.stem}.previous_{stamp}{p.suffix}")
    for n in ["preds","preds.hdf5","args.yaml","joint_pred.log",".execution_failed.json"]:
        p=d/n
        if p.exists(): shutil.rmtree(p) if p.is_dir() else p.unlink()
    if sent(model).exists(): sent(model).unlink()

# Gates
for p in [AUTH,MANIFEST,HASHES,GLABEL,ILABEL,GCK,IGEN,IINT]:
    if not p.exists(): raise FileNotFoundError(p)
a=pd.read_csv(AUTH,sep="\t").iloc[0]
if str(a["AuthorizationStatus"])!="AUTHORIZED_TO_EXECUTE_FULL_TIER1": raise RuntimeError("Script 111 authorization missing.")
if int(a["Structures"])!=300 or str(a["PredictionCEs_eV"])!="25|50|75" or int(a["ExpectedRecordsPerModel"])!=900: raise RuntimeError("Frozen execution dimensions changed.")
if str(a["ScoringPolicyVersion"])!="v0.6.4.8" or str(a["PrimaryScoringRepresentation"])!="EqualWeight25_50_75Consensus" or abs(float(a["PrimaryFragmentTolerance_Da"])-.02)>1e-12: raise RuntimeError("Frozen scoring dependency changed.")

hf=pd.read_csv(HASHES,sep="\t"); hm=dict(zip(hf.Label.astype(str),hf.SHA256.astype(str)))
if sha(GLABEL)!=hm["GLACIERLabels"] or sha(ILABEL)!=hm["ICEBERG21Labels"]: raise RuntimeError("Frozen native-label SHA mismatch.")
for p,e in [(GCK,G_SHA),(IGEN,IG_SHA),(IINT,II_SHA)]:
    if sha(p)!=e: raise RuntimeError(f"Checkpoint SHA mismatch: {p}")

dv=run(["docker","--version"])
if dv.returncode: raise RuntimeError(dv.stderr)
di=run(["docker","image","inspect",IMAGE,"--format","{{.Id}}"])
if di.returncode: raise RuntimeError(di.stderr)
image_id=di.stdout.strip()

# CLI preflight: require modules to exist. Help flags are preserved for diagnosis.
pre=[]
for model,mod,out in [("GLACIER",GMOD,GOUT),("ICEBERG 2.1",IMOD,IOUT)]:
    r=run(base(out)+["python","-m",mod,"--help"],180)
    (PROV/f"{model.replace(' ','_').replace('.','')}_help_stdout.txt").write_text(r.stdout)
    (PROV/f"{model.replace(' ','_').replace('.','')}_help_stderr.txt").write_text(r.stderr)
    pre.append({"Model":model,"Module":mod,"ReturnCode":r.returncode,"Status":"PASS" if r.returncode==0 else "FAIL"})
pre=pd.DataFrame(pre); pre.to_csv(PROV/"MSPRED_FullTier1_CLI_Preflight.tsv",sep="\t",index=False)
if not pre.Status.eq("PASS").all():
    print(pre.to_string(index=False))
    raise RuntimeError("CLI preflight failed; no full execution started. Inspect PROVENANCE help logs.")

def gcmd():
    lab=f"/project/{GLABEL.relative_to(ROOT)}"; ck=f"/project/{GCK.relative_to(ROOT)}"
    return base(GOUT)+["python","-m",GMOD,"--dataset-labels",lab,"--checkpoint",ck,"--save-dir","/output","--sparse-out","--sparse-k","100"]

def iceberg_wrapper():
    lab=f"/project/{ILABEL.relative_to(ROOT)}"; gen=f"/project/{IGEN.relative_to(ROOT)}"; inten=f"/project/{IINT.relative_to(ROOT)}"
    # repr() is applied while constructing this host-side string.
    return """
import runpy,sys,torch
_orig=torch.load
def trusted(*args,**kwargs):
    if kwargs.get("weights_only",None) is None:
        kwargs["weights_only"]=False
    return _orig(*args,**kwargs)
torch.load=trusted
sys.argv=[%r,"--dataset-labels",%r,"--gen-checkpoint",%r,"--inten-checkpoint",%r,"--save-dir","/output"]
runpy.run_module(%r,run_name="__main__")
""" % (IMOD,lab,gen,inten,IMOD)

def icmd(): return base(IOUT)+["python","-c",iceberg_wrapper()]

def valid_done(model,inputsha):
    s=sent(model); n=native(model)
    if not s.exists() or not nonempty(n): return False
    try: j=json.loads(s.read_text())
    except: return False
    return j.get("Status")=="PASS" and j.get("InputLabelSHA256")==inputsha and j.get("NativeOutputSHA256")==hashout(n)

def execute(model):
    inp,out,cmd=(GLABEL,GOUT,gcmd()) if model=="GLACIER" else (ILABEL,IOUT,icmd())
    ish=sha(inp); n=native(model)
    if valid_done(model,ish):
        j=json.loads(sent(model).read_text())
        return {"Model":model,"Action":"SKIPPED_ALREADY_COMPLETE","RuntimeSeconds":j.get("RuntimeSeconds"),"ReturnCode":0,"NativeOutputBytes":bytesout(n),"NativeOutputSHA256":hashout(n),"Status":"PASS"}
    clean(model)
    display=cmd if model=="GLACIER" else base(out)+["python","-c","<trusted-checkpoint-wrapper>"]
    dump(out/"execution_command.json",{"Model":model,"Version":VERSION,"Command":display,"InputLabel":str(inp),"InputLabelSHA256":ish,"ScoringPerformed":False})
    st=now(); t=time.perf_counter()
    print(f"\nExecuting {model}\n{'-'*(10+len(model))}\nInput: {inp}\nOutput: {out}\nCPU linux/amd64 emulation may take substantial time.",flush=True)
    with open(out/"stdout.log","w") as so,open(out/"stderr.log","w") as se:
        r=subprocess.run(cmd,cwd=ROOT,text=True,stdout=so,stderr=se)
    runtime=time.perf_counter()-t; ok=r.returncode==0 and nonempty(n)
    j={"Model":model,"ExecutionVersion":VERSION,"Status":"PASS" if ok else "FAIL","StartTime":st,"EndTime":now(),"RuntimeSeconds":runtime,"ReturnCode":r.returncode,"InputLabelSHA256":ish,"NativeOutput":str(n),"NativeOutputBytes":bytesout(n),"NativeOutputSHA256":hashout(n) if nonempty(n) else None,"Image":IMAGE,"ImageID":image_id,"MSPredCommit":COMMIT,"ScoringPerformed":False}
    dump(sent(model) if ok else out/".execution_failed.json",j)
    return {"Model":model,"Action":"EXECUTED","RuntimeSeconds":runtime,"ReturnCode":r.returncode,"NativeOutputBytes":bytesout(n),"NativeOutputSHA256":j["NativeOutputSHA256"],"Status":j["Status"]}

rows=[]
for model in ["GLACIER","ICEBERG 2.1"]:
    try: rows.append(execute(model))
    except KeyboardInterrupt: raise
    except Exception as e:
        (PROV/f"{model.replace(' ','_').replace('.','')}_exception.txt").write_text(traceback.format_exc())
        rows.append({"Model":model,"Action":"EXCEPTION","RuntimeSeconds":"","ReturnCode":"","NativeOutputBytes":bytesout(native(model)),"NativeOutputSHA256":"","Status":"FAIL","Exception":repr(e)})
summary=pd.DataFrame(rows); summary.to_csv(OUT/"MSPRED_FullTier1_Execution_Summary.tsv",sep="\t",index=False)

inv=[]
for model,d in [("GLACIER",GOUT),("ICEBERG 2.1",IOUT)]:
    for p in sorted(x for x in d.rglob("*") if x.is_file()):
        inv.append({"Model":model,"RelativePath":str(p.relative_to(ROOT)),"Bytes":p.stat().st_size,"SHA256":sha(p)})
pd.DataFrame(inv).to_csv(OUT/"MSPRED_FullTier1_Output_Inventory.tsv",sep="\t",index=False)

q=[]
def add(c,e,o,p): q.append({"Check":c,"Expected":e,"Observed":o,"Status":"PASS" if p else "FAIL"})
add("Script-111 authorization","AUTHORIZED_TO_EXECUTE_FULL_TIER1",str(a["AuthorizationStatus"]),str(a["AuthorizationStatus"])=="AUTHORIZED_TO_EXECUTE_FULL_TIER1")
add("GLACIER label SHA","frozen",sha(GLABEL),sha(GLABEL)==hm["GLACIERLabels"])
add("ICEBERG label SHA","frozen",sha(ILABEL),sha(ILABEL)==hm["ICEBERG21Labels"])
add("CLI preflight","2 PASS",int(pre.Status.eq("PASS").sum()),pre.Status.eq("PASS").all())
for model in ["GLACIER","ICEBERG 2.1"]:
    rr=summary[summary.Model.eq(model)].iloc[0]; n=native(model)
    add(f"{model} return code",0,rr["ReturnCode"],str(rr["ReturnCode"]) in {"0","0.0"})
    add(f"{model} native output nonempty",True,nonempty(n),nonempty(n))
    add(f"{model} completion sentinel","PASS",json.loads(sent(model).read_text()).get("Status") if sent(model).exists() else "missing",sent(model).exists() and json.loads(sent(model).read_text()).get("Status")=="PASS")
add("Experimental scoring performed",False,False,True)
add("Frozen scoring dependency","v0.6.4.8",str(a["ScoringPolicyVersion"]),str(a["ScoringPolicyVersion"])=="v0.6.4.8")
q=pd.DataFrame(q); q.to_csv(QC/"112_execute_full_tier1_mspred_v0650.tsv",sep="\t",index=False)
passed=q.Status.eq("PASS").all()

dump(PROV/"MSPRED_FullTier1_Container_Provenance.json",{"Version":VERSION,"Timestamp":now(),"DockerVersion":dv.stdout.strip(),"Image":IMAGE,"ImageID":image_id,"Platform":"linux/amd64","FrozenMSPredCommit":COMMIT,"GLACIERLabelSHA256":sha(GLABEL),"ICEBERG21LabelSHA256":sha(ILABEL),"GLACIERCheckpointSHA256":sha(GCK),"ICEBERGGeneratorCheckpointSHA256":sha(IGEN),"ICEBERGIntensityCheckpointSHA256":sha(IINT),"ScoringPolicyDependency":"v0.6.4.8","ScoringPerformed":False})
(OUT/"MSPRED_FullTier1_Execution_Notes.txt").write_text(f"""Full Tier-1 ms-pred execution {VERSION}
STATUS: {'PASS' if passed else 'HOLD / PARTIAL'}
Scope: model execution only; no experimental scoring.
Population: 300 structures; prediction CE 25|50|75 eV; 900 expected CE-specific records/model.
Frozen scoring dependency: v0.6.4.8, EqualWeight25_50_75Consensus, 0.020 Da, best-CE primary use prohibited.
Restart safety: completed models are skipped only after sentinel + input SHA + native-output SHA validation.
Next gate: Script 113 parses/standardizes native predictions and verifies 900 CE-specific records/model.
""")

pd.set_option("display.max_columns",None); pd.set_option("display.width",240)
print(f"\nFull Tier-1 ms-pred execution {VERSION}\n{'='*44}\n\nExecution summary\n-----------------\n{summary.to_string(index=False)}\n\nQC\n--\n{q.to_string(index=False)}")
print("\nSaved\n-----")
for p in [OUT/"MSPRED_FullTier1_Execution_Summary.tsv",OUT/"MSPRED_FullTier1_Output_Inventory.tsv",OUT/"MSPRED_FullTier1_Execution_Notes.txt",PROV/"MSPRED_FullTier1_Container_Provenance.json",PROV/"MSPRED_FullTier1_CLI_Preflight.tsv",QC/"112_execute_full_tier1_mspred_v0650.tsv"]: print(p)
if passed: print(f"\nFULL TIER-1 MS-PRED EXECUTION {VERSION} PASSED — READY FOR SCRIPT 113")
else:
    print(f"\nFULL TIER-1 MS-PRED EXECUTION {VERSION} HOLD / PARTIAL — REVIEW LOGS AND RERUN SAFELY")
    sys.exit(2)
