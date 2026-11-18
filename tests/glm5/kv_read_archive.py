"""Pack and verify experiment artifacts at their measured evidence bounds."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from datetime import datetime
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import kv_read_run as R
from deploy.research_library import canonical
from deploy.research_ingest_bundle import verified_objects

NAME="kv-read-scoped-client.tar.gz"
URL="https://server.ai2050.com"
TITLE="Qwen3-4B · 自生成状态的 Key/Value 与读取时刻"


def package():
    doc=R.read("audit.json")
    confirmation=R.read("confirmation.json")
    suites=[x for x in doc["confirmation"] if x["status"]=="complete"]
    accepted=[(s["suite"],f) for s in suites for f,m in s["findings"].items() if m["broad"]["passed"] or m["sparse"]["passed"]
              or m["complete_rescue"] or any(i["intervention_grade"] for i in m["individual"])]
    if not accepted:
        R.write("publication.json",dict(status="no positive admitted intervention",remote=None))
        return
    model=doc["snapshots"]["q4"]
    docs={}

    def put(body):
        raw=canonical(dict(kind="real",run_id=R.RUN,model_snapshot=model,**body))
        h=hashlib.sha256(raw).hexdigest()
        docs[h]=raw
        return h

    evidence={}
    for row in suites:
        suite=row["suite"]
        raw=next(e for e in confirmation["suites"] if e["suite"]==suite)
        evidence[suite]=put(dict(schema="kv_read_cases.v1",suite=suite,
                                 materials=R.read(f"materials-{suite}.json"),
                                 natural=R.read(f"q4-{suite}-behavior.json")["rows"],
                                 intervention_batches=[R.read(e["name"]) for e in raw["shards"]],
                                 original_native_field_hash=row["structural"]["sampled_native"]["sha"]))
    report=put(dict(schema="kv_read_report.v1",title=TITLE,admitted_suites=list(evidence),accepted=accepted,
                    selection=doc["selection"],behavior=doc["behavior"],reference_behavior=doc["reference"],
                    discovery=doc["discovery"],confirmation=doc["confirmation"],case_object_refs=evidence,
                    audit_sha=R.sha(R.OUT/"audit.json"),plan_sha=R.sha(R.OUT/"plan.json"),
                    evidence_grade="scoped intervention",independent_gpu_recomputation=False,
                    scope="Local Qwen3-4B BF16. K-only and V-only at the generated state slot. Timing groups and individual query heads as listed.",
                    caveats=["Fixed-prefix contrasts retain natural non-answer format; free continuation scored separately.",
                             "Shared-KV group floor(query_head/4), patched at individual query output before o_proj.",
                             "A failed positive Key contrast does not establish Key is unnecessary.",
                             "Successful removal and recovery have separate scope; code intervals can contain unequal numbers of reads."]))
    cards=[]
    edges=[]

    def add(suite,field,suffix,layer,head,label,stat,point=None,more=None):
        h=evidence[suite]
        outgoing=f"{R.RUN}:{suite}:{field}:{suffix}"
        target=f"{R.RUN}:{suite}:{field}:{suffix}:code"
        ev=dict(level="intervention",artifact_sha256=h)
        proof=[stat.get("least_effect")] if stat.get("least_effect") else []
        scope=dict(suite=suite,modality=field,measurement="alternate vs correct code margin",**(more or {}))
        ix=dict(space="attention",field=f"generated_state_{field}")
        if head is not None:
            ix.update(head=head,kv_group=head//4)
        else:
            ix["layer_anchor_only"]=True
        cards.append(dict(id=outgoing,layer=layer,boundary="attention.query_output_before_o_proj",index=ix,
                           method=("native SDPA selected-head neutralization and controlled partial recovery" if suffix=="remove-and-restore"
                                   else "native SDPA selected-head reread with donor K or V"),label=label,run_id=R.RUN,
                           scope=scope,evidence=ev,counterexamples=proof))
        cards.append(dict(id=target,layer=35,boundary="lm_head.before_answer_token",index=dict(space="residual",field="final_readout"),
                           method="next-token code logit contrast at frozen natural non-answer prefix",
                           label=f"{suite} · P/Q 答码读出 · {suffix}",run_id=R.RUN,scope=scope,evidence=ev,counterexamples=proof))
        edges.append(dict(id=outgoing+":edge",source=outgoing,target=target,relation="causal score shift under scoped native read edit",
                          run_id=R.RUN,evidence=ev,scope=scope))

    for suite in suites:
        sid=suite["suite"]
        for f,record in suite["findings"].items():
            if record["broad"]["passed"]:
                add(sid,f,"all",0,None,f"{sid} · {'Key' if f=='keys' else 'Value'} 全层状态槽",
                    record["broad"],more={"joint_36_layers":True})
            if record["sparse"]["passed"]:
                add(sid,f,"selected",doc["selection"]["choices"][f]["points"][0][0],None,
                    f"{sid} · {f} 冻结三查询头联合",record["sparse"],
                    more={"points":doc["selection"]["choices"][f]["points"],"timing_contrast_passed":record["local_and_timed"],
                          "complete_controlled_rescue":record["complete_rescue"]})
            if record["complete_rescue"]:
                add(sid,f,"remove-and-restore",doc["selection"]["choices"][f]["points"][0][0],None,
                    f"{sid} · {f} 冻结三头移除与匹配恢复",record["rescue_specific"],
                    more={"points":doc["selection"]["choices"][f]["points"],"only_supported_under_tested_lesions":True,
                          "removal_vs_dose_matched":record["removal"]["mean"],
                          "restore_vs_global_zero":record["rescue"]["mean"]})
            for x in record["individual"]:
                if not x["intervention_grade"]:
                    continue
                layer,head,window=x["point"]
                add(sid,f,f"L{layer}Q{head}",layer,head,f"{sid} · {f} 层{layer} 查询头{head}",
                    x["specific"],more={"edited_at":window,"timing_contrast_passed":x["timing_grade"]})
    graph=put(dict(schema="research_graph.v1",feature_cards=cards,graph_edges=edges))
    binding=put(dict(schema="research_space_binding.v1",report_sha256=report,graph_sha256=graph,traces=[],
                     context=dict(study_type="native_K_V_reread",
                                  model_dimensions=dict(n_layers=36,d_ff=9728,d_model=2560,n_heads=32))))
    work=put(dict(schema="research_workpackage.v1",title=TITLE,stages={
        "discovery":{"status":"done","scanner_arms":226},
        "heldout_intervention":{"status":"done","material_suites":list(evidence),"accepted_scope_edges":len(edges)},
        "independent_recomputation":{"status":"pending"},
        "registration_validated":{"status":"done","report_sha256":report,"graph_sha256":graph}}))
    s=dict(run_id=R.RUN,model_snapshot=model,report_sha256=report,graph_sha256=graph,binding_sha256=binding,workpackage_sha256=work)
    manifest=dict(schema="ai2050_research_upload.v1",studies=[s],objects=[dict(sha256=h) for h in docs])
    target=R.Base.Previous.check_write(R.OUT/NAME)
    assert not target.exists()
    with tarfile.open(target,"w:gz") as out:
        for name,record in [("MANIFEST.json",canonical(manifest)),*((f"objects/{h}.json",v) for h,v in docs.items())]:
            m=tarfile.TarInfo(name)
            m.size=len(record)
            out.addfile(m,io.BytesIO(record))
    with tarfile.open(target,"r:gz") as f:
        assert dict(verified_objects(f,manifest))==docs
    R.write("publication.json",dict(status="eligible scoped intervention",study=s,sha=R.sha(target),
                                    objects=len(docs),cards=len(cards),edges=len(edges),remote=None))


def isolated():
    release=R.Base.ROOT/"release/ai2050-center-1.0.29.tar.gz"
    assert release.with_suffix(".gz.sha256").read_text().split()[0]==R.sha(release)
    root=R.Base.Previous.check_write(R.TEMP/"isolated-center")
    root.mkdir()
    with tarfile.open(release,"r:gz") as f:
        f.extractall(root,filter="data")
    cwd=root/"ai2050-center-1.0.29"
    env={**os.environ,"KV_READ_UPLOAD_PACKAGE":str(R.OUT/NAME),"KV_READ_ISOLATED_LIBRARY":str(root/"db")}
    env.pop("PYTHONPATH",None)
    cmd="""
import json,hashlib,os,tarfile
from pathlib import Path
from deploy.research_library import Library,canonical
from deploy.research_ingest_bundle import verified_objects
from server import research_space_service as s
l=Library(Path(os.environ['KV_READ_ISOLATED_LIBRARY']))
s.library=lambda:l
with tarfile.open(os.environ['KV_READ_UPLOAD_PACKAGE'],'r:gz') as f:
    manifest=json.load(f.extractfile('MANIFEST.json'))
    docs=list(verified_objects(f,manifest))
for sha,raw in docs:
    assert l.ingest(json.loads(raw),'registered',kind='registered')['sha256']==sha
    assert hashlib.sha256(canonical(l.artifact(sha))).hexdigest()==sha
indices=s._research_space()
target=manifest['studies'][0]
got=next(x for x in indices['studies'] if x['run_id']==target['run_id'])
assert got['report_sha256']==target['report_sha256'] and got['binding_sha256']==target['binding_sha256']
assert {x['body']['evidence']['level'] for x in indices['objects']}=={'intervention'}
print(json.dumps(dict(objects=len(docs),report=got,client_index=True,scope='packaged release code with isolated SQLite')))
"""
    p=subprocess.run([sys.executable,"-B","-X","utf8","-c",cmd],cwd=cwd,env=env,capture_output=True,text=True,timeout=45)
    R.Base.Previous.check_write(R.OUT/"isolated.log").write_text(p.stdout+p.stderr,encoding="utf-8")
    p.check_returncode()
    R.write("isolated-proof.json",json.loads(p.stdout.strip().splitlines()[-1]))


def published():
    release=json.load(urlopen(URL+"/api/health",timeout=15))
    index=json.load(urlopen(URL+"/api/traces/research-space",timeout=15))
    info=R.read("publication.json")
    want=info["study"]
    entry=next((x for x in index["studies"] if x["run_id"]==want["run_id"] and x["model_snapshot"]==want["model_snapshot"]),None)
    if entry is None:
        R.write("production.json",dict(version=release["version"],verified=False,reason="study not indexed yet"))
        return
    assert entry["report_sha256"]==want["report_sha256"] and entry["binding_sha256"]==want["binding_sha256"]
    verified=[]
    with tarfile.open(R.OUT/NAME,"r:gz") as f:
        manifest=json.load(f.extractfile("MANIFEST.json"))
        for item in manifest["objects"]:
            sha=item["sha256"]
            material=json.load(urlopen(URL+"/api/research-library/artifacts/"+sha,timeout=15))
            assert hashlib.sha256(canonical(material)).hexdigest()==sha
            verified.append(sha)
    receipt=dict(center=URL,version=release["version"],verified=True,sha256=verified,study=entry,
                 time=datetime.now(ZoneInfo("America/Chicago")).isoformat(timespec="minutes"))
    R.write("production.json",receipt)
    R.write("publication.json",dict(info,remote=receipt))
    print("Verified server objects",len(verified),"and the public client index",flush=True)


if __name__=="__main__":
    R.setup()
    {"package":package,"isolated":isolated,"production":published}[sys.argv[1]]()
