"""Build an observation-only client archive and verify it on an isolated 1.0.29 center."""
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.request import urlopen

import condition_chain_audit as A
import condition_chain_run as R
from deploy.research_library import canonical
from deploy.research_ingest_bundle import verified_objects

FOLDER = A.BASE/A.FOLDERS[4]
TITLE = "Qwen3-4B · 测量判定的门控响应（范围内观察）"
ORIGIN = "http://127.0.0.1:56345"


def package():
    plan=A.setup_version(5)
    report=R.load(FOLDER/"analysis.json")
    audit=R.load(FOLDER/"audit.json")
    candidate=next(c for c in report["candidates"] if c["task"]=="assessment")
    assert len(candidate["tests"])==4 and all(t["mean"]>0 for t in candidate["tests"])
    assert not candidate["pass_response"] and not any(t["passed"] for t in candidate["tests"])
    assert audit["verified_rows"]==2784
    for doc in report["screens"]:
        row=doc["totals"]["assessment"]
        assert row["correct"]==row["n"] and row["gates_pass"]==4
    for split in ("words","frames"):
        section=next(s for s in audit["sections"] if s["split"]==split)
        task=next(s for s in section["summary"] if s["task"]=="assessment")
        assert task["families"]==task["exact_distinct_first_prefix_families"]==96

    snapshot=audit["models"]["q4"]["canonical_snapshot"]
    objects={}

    def put(body):
        raw=canonical(dict(kind="real",run_id=R.RUN,model_snapshot=snapshot,**body))
        h=hashlib.sha256(raw).hexdigest()
        objects[h]=raw
        return h

    examples={}
    for split in ("words","frames"):
        rows=R.load(FOLDER/f"materials-{split}-candidate.json")
        run=R.load(FOLDER/f"q4-{split}-candidate.json")
        measured=A.native(split,"candidate")
        selected=[]
        for row,out in zip(rows,run["results"],strict=True):
            if row["task"]!="assessment":
                continue
            values=measured[row["id"]]
            selected.append(dict(family=row["family"],factors=row["factors"],id=row["id"],source=row["pieces"][0],
                                 domain=row["domain_cluster"],choices=row["options"],answer=row["answer"],
                                 option_logits=out["option_logits"],readout_token=out["sites"][0],
                                 G=float(values["G"][0]),V=float(values["V"][0]),A=float(values["A"][0])))
        examples[split]=put(dict(schema="condition_chain_observations.v1",split=split,data=selected,
                                 material_sha256=plan["materials"][f"materials-{split}-candidate.json"]["sha256"]))
    cards=[]
    for split in ("words","frames"):
        for item in R.load(FOLDER/f"materials-{split}-candidate.json"):
            if item["task"]=="assessment" and item["factors"]==[0,1] and item["index"]==0:
                cards.append(dict(split=split,id=item["id"],note="有 testing 关键词，但实际采用猜测"))
    report_hash=put(dict(schema="condition_chain_report.v1",title=TITLE,model="Qwen3-4B BF16",title_scope="研究归档，观察级",
                        public_scope="测量判定候选。人工构造材料，两组各96个题干家族。",
                        interpretation="行为四格均通过，B31/U2619的G在四个对照上正向。预注册预测误差比门槛为0.90，实际为0.912和0.917，候选未过门。",
                        mechanism_promoted=False,independent_node_reproduction=False,intervention="未准入",
                        thresholds=plan["thresholds"],positive_checks=candidate["tests"],
                        behavior={s["split"]:s["totals"] for s in report["screens"]},
                        observed_native_diagnostics=[n for n in audit["native_descriptive"] if n["task"]=="assessment"],
                        counterexamples=cards,readout="题干 Checkpoint 的最后一个前缀稳定 token，答案、问题与选项在之后",
                        examples_sha256=examples,plan_sha256=audit["plan_sha256"],audit_sha256=R.file_sha(FOLDER/"audit.json")))
    graph_hash=put(dict(schema="research_graph.v1",feature_cards=[
        dict(id=R.RUN+":B31:U2619",layer=31,boundary="mlp.gate_proj.output",run_id=R.RUN,
             label="B31/U2619 · 测量判定响应（关联观察，预测门失败）",index=dict(space="mlp",unit=2619,field="G"),
             method="冻结候选与配对题干",scope={"formal_promotion":False,"causal":False},
             evidence={"level":"observation","artifact_sha256":report_hash},counterexamples=cards)],
        graph_edges=[]))
    binding_hash=put(dict(schema="research_space_binding.v1",report_sha256=report_hash,graph_sha256=graph_hash,traces=[],
                          context=dict(study_type="candidate_response_observation",
                                       model_dimensions=dict(n_layers=36,d_ff=9728,d_model=2560,n_heads=32))))
    work_hash=put(dict(schema="research_workpackage.v1",title=TITLE,stages={
        "scope":{"status":"observation","formal_mechanism_count":0,"independent_recomputation":False},
        "prediction_gate":{"status":"failed","threshold":.9,"observed_ratios":[candidate["tests"][i]["mse_ratio"] for i in (0,2)]},
        "registration_validated":{"status":"done","report_sha256":report_hash,"graph_sha256":graph_hash}}))
    study=dict(run_id=R.RUN,model_snapshot=snapshot,workpackage_sha256=work_hash,
               report_sha256=report_hash,graph_sha256=graph_hash,binding_sha256=binding_hash)
    target=R.check_write(FOLDER/"observation-client-bundle.tar.gz")
    assert not target.exists()
    manifest=dict(schema="ai2050_research_upload.v1",studies=[study],objects=[dict(sha256=h) for h in objects])
    with tarfile.open(target,"w:gz") as f:
        for filename,raw in [("MANIFEST.json",canonical(manifest)),*((f"objects/{h}.json",v) for h,v in objects.items())]:
            item=tarfile.TarInfo(filename)
            item.size=len(raw)
            f.addfile(item,io.BytesIO(raw))
    with tarfile.open(target,"r:gz") as f:
        assert dict(verified_objects(f,manifest))==objects
    R.write("publication.json",dict(study=study,bundle_sha256=R.file_sha(target),objects=len(objects),
                                    scope="observation-only research archive, no formal promotion or causal graph edge",
                                    remote_receipt=None))


def sandbox():
    A.setup_version(5)
    publication=R.load(FOLDER/"publication.json")
    archive=R.ROOT/"release/ai2050-center-1.0.29.tar.gz"
    assert (archive.with_suffix(".gz.sha256")).read_text().split()[0]==R.file_sha(archive)
    path=Path(tempfile.mkdtemp(prefix="center-bundle-",dir=R.TMP))
    with tarfile.open(archive,"r:gz") as zipped:
        zipped.extractall(path,filter="data")
    project=path/"ai2050-center-1.0.29"
    env={**os.environ,"AI2050_DIST_DIR":str(path/"data"),"AI2050_NATIVE_DIR":str(path/"native"),
         "AI2050_RESEARCH_DIR":str(path/"research"),"AI2050_SNAPSHOT_DISABLE":"1","AI2050_LOCAL_TOOLS":"0",
         "AI2050_DIST_ADMIN_TOKEN":secrets.token_hex(24),"AI2050_PUBLIC_URL":ORIGIN}
    env.pop("PYTHONPATH",None)
    with (FOLDER/"center-sandbox.log").open("xb") as log:
        service=subprocess.Popen([sys.executable,"-B","-u","deploy/center_app.py","--host","127.0.0.1","--port","56345"],cwd=project,
                                 env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        try:
            for _ in range(60):
                try:
                    health=json.load(urlopen(ORIGIN+"/api/health",timeout=1))
                    break
                except OSError:
                    assert service.poll() is None
                    time.sleep(.25)
            else:
                raise RuntimeError("Isolated center never reached health check")
            assert health["version"]=="1.0.29"
            verified=subprocess.run([sys.executable,"-B","deploy/research_ingest_bundle.py",str(FOLDER/"observation-client-bundle.tar.gz"),
                                     "--center",ORIGIN],cwd=R.ROOT,env={**env,"AI2050_RESEARCH_ADMIN_TOKEN":env["AI2050_DIST_ADMIN_TOKEN"]},
                                    capture_output=True,text=True,timeout=120)
            verified.check_returncode()
            index=json.load(urlopen(ORIGIN+"/api/traces/research-space",timeout=10))
            entry=next(s for s in index["studies"] if s["run_id"]==publication["study"]["run_id"])
            assert entry["report_sha256"]==publication["study"]["report_sha256"]
            doc=json.load(urlopen(ORIGIN+"/api/research-library/artifacts/"+entry["report_sha256"],timeout=10))
            assert hashlib.sha256(canonical(doc)).hexdigest()==entry["report_sha256"]
            assert doc["mechanism_promoted"] is False
            R.write("center-sandbox-proof.json",dict(server_version=health["version"],index_verified=True,report_hash_verified=True,
                                                    study=entry,stdout=verified.stdout,
                                                    scope="Local API used by client; no production or rendered client assertion"))
        finally:
            service.terminate()
            service.wait(timeout=15)


def inprocess():
    """Exercise HTTP routes with the packaged server and forbid any outbound write."""
    A.setup_version(5)
    extracted=sorted(R.TMP.glob("center-bundle-*"))
    assert len(extracted)==1
    stage=extracted[0]
    packaged=stage/"ai2050-center-1.0.29"
    env={**os.environ,"AI2050_DIST_DIR":str(stage/"inprocess-data"),"AI2050_NATIVE_DIR":str(stage/"native"),
         "AI2050_RESEARCH_DIR":str(stage/"research"),"AI2050_SNAPSHOT_DISABLE":"1","AI2050_LOCAL_TOOLS":"0",
         "AI2050_DIST_ADMIN_TOKEN":secrets.token_hex(24),"CHAIN_BUNDLE_PATH":str(FOLDER/"observation-client-bundle.tar.gz")}
    env.pop("PYTHONPATH",None)
    code="""
import os,sys,json,tarfile,hashlib,hmac
from pathlib import Path
from types import SimpleNamespace
from fastapi import HTTPException
from deploy.research_ingest_bundle import verified_objects
import deploy.research_library as L
import server.research_space_service as space
canonical=L.canonical
root=Path(os.environ['AI2050_DIST_DIR'])
library=L.Library(root/'research_library')
L.library=lambda:library
space.library=lambda:library
def admin(token):
    if not hmac.compare_digest(token,os.environ['AI2050_DIST_ADMIN_TOKEN']):
        raise HTTPException(403,'invalid sandbox token')
def deny(fn, token):
    raise HTTPException(401,'sandbox has no signed-in user')
L.services=lambda:SimpleNamespace(_auth_admin=admin,_auth_call=deny,_accounts=SimpleNamespace(require_admin=None))
print('ROUTES_READY',flush=True)
with tarfile.open(os.environ['CHAIN_BUNDLE_PATH'],'r:gz') as tar:
    manifest=json.load(tar.extractfile('MANIFEST.json'))
    objects=list(verified_objects(tar,manifest))
try:
    L.ingest({},x_admin_token='',authorization='')
    raise RuntimeError('anonymous write unexpectedly permitted')
except HTTPException as exc:
    assert exc.status_code in (401,403)
for sha,raw in objects:
    resp=L.ingest(json.loads(raw),x_admin_token=os.environ['AI2050_DIST_ADMIN_TOKEN'])
    assert resp['sha256']==sha,resp
    assert hashlib.sha256(canonical(L.artifact(sha))).hexdigest()==sha
index=space._research_space()
want=manifest['studies'][0]
seen=next(s for s in index['studies'] if s['run_id']==want['run_id'])
assert seen['report_sha256']==want['report_sha256'] and seen['binding_sha256']==want['binding_sha256']
report=L.artifact(seen['report_sha256'])
assert report['mechanism_promoted'] is False
print('PROOF='+json.dumps({'release_routes':'1.0.29','study':seen,'verified_objects':len(objects),'anonymous_test_write_denied':True}))
"""
    check=subprocess.run([sys.executable,"-B","-X","utf8","-c",code],cwd=packaged,env=env,capture_output=True,text=True,timeout=40)
    R.check_write(FOLDER/"center-sandbox-inprocess.log").write_text(check.stdout+check.stderr,encoding="utf-8")
    check.check_returncode()
    proof=json.loads(next(line[6:] for line in check.stdout.splitlines() if line.startswith("PROOF=")))
    R.write("center-sandbox-proof.json",dict(**proof,index_verified=True,report_hash_verified=True,
                                            scope="Direct execution of packaged 1.0.29 archive handlers and client index builder with isolated SQLite and test authentication. No HTTP stack or rendered-client verification."))


if __name__=="__main__":
    {"package":package,"sandbox":sandbox,"inprocess":inprocess}[sys.argv[1]]()
