"""Content-addressed archive, isolated release check, and authenticated production readback."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from datetime import datetime
from zoneinfo import ZoneInfo

import own_state_explicit as E
R=E.R
from deploy.research_library import canonical
from deploy.research_ingest_bundle import verified_objects

BUNDLE="own-state-client-v2.tar.gz"
ORIGIN="https://server.ai2050.com"


def package():
    audit=R.read("audit.json")
    studies=audit["causal"]["tasks"]
    if not any(t["native_gate"] or t["token_gate"] for t in studies):
        R.write("publication.json",dict(status="not eligible",reason="no local causal gate accepted",remote=None))
        return
    graph_checks=R.read("causal-run.json")
    title="Qwen3-4B · 自生成中间状态对最终答码的作用"
    model=audit["v2"]["models"]["q4"]["snapshot"]
    objects={}

    def put(doc):
        raw=canonical(dict(kind="real",model_snapshot=model,run_id=R.RUN,**doc))
        key=hashlib.sha256(raw).hexdigest()
        objects[key]=raw
        return key

    rows={t["task"]:[] for t in studies}
    materials={r["id"]:r for r in R.read("materials-confirm.json")}
    for shard in graph_checks["batches"]:
        observed=R.read(shard["name"])
        for r in observed["rows"]:
            row=materials[r["id"]]
            rows[r["task"]].append(dict(**r,prompt=row["text"],expected=[row["state"],row["code"]],
                                        alternate=[row["alt"],row["alt_code"]],native_array_sha=observed["slot_sha"]))
    evidence={key:put(dict(schema="own_state_cases.v1",task=key,rows=value)) for key,value in rows.items()}
    report=put(dict(schema="own_state_report.v1",title=title,protocol_sha=R.digest(R.OUT/"causal-plan.json"),
                    audit_sha=R.digest(R.OUT/"audit.json"),source_studies=evidence,
                    behavioral_reference=audit["v2"]["runs"],
                    outcomes=studies,original_values_omitted=True,independent_node_reproduction=False,
                    evidence_level="intervention in the frozen scope",official_evidence_registry="pending separate review",
                    scope="One Qwen3-4B BF16 checkpoint, two English two-field tasks, prompted report then code. V-only transfer spans all 36 attention layers at one generated token.",
                    limits=["Not a single-head localization", "Does not establish necessity, unique internal representation or generalization outside this prompt protocol",
                            "Qwen3-14B and GLM4 small behavior-only references; no cross-model intervention"],
                    verification="Exact same-generation unedited replay, BF16 raw target and dose reconstruction, no independent GPU rerun."))
    cards=[]
    edges=[]
    for study in studies:
        if not study["native_gate"]:
            continue
        task=study["task"]
        source=f"{R.RUN}:{task}:generated-token-v"
        target=f"{R.RUN}:{task}:final-code"
        reference={"level":"intervention","artifact_sha256":evidence[task]}
        counter=[dict(case=x["id"],artifact_sha256=evidence[task]) for x in study["failures"]["value_patch"][:5]]
        cards.append(dict(id=source,layer=0,boundary="attention.cache.values.generated_state",run_id=R.RUN,
                           label=f"{task} · 自生成词的 Value 缓存（36层联合）",
                           index=dict(space="attention",field="kv_value",blocks=list(range(36)),layer_anchor_only=True),
                           method="V only from native same-prompt alternate-token branch; fixed original K and original next generated token",
                           evidence=reference,counterexamples=counter))
        cards.append(dict(id=target,layer=35,boundary="unembedding.free_generated_code",run_id=R.RUN,
                           label=f"{task} · 自由生成的最终答码",index=dict(space="residual",field="final_readout"),
                           method="Strict two-line parsing and EOS",evidence=reference,counterexamples=counter))
        edges.append(dict(id=source+"->"+target,source=source,target=target,
                          relation="V transplant caused mapped alternative final code in admitted scope",run_id=R.RUN,
                          scope={"views":study["n"],"families":study["families"],"exact_cache_replay":True,
                                 "licensed_scope":"Original fictional text","independent_replica":False},
                          evidence=reference))
    graph=put(dict(schema="research_graph.v1",feature_cards=cards,graph_edges=edges))
    binding=put(dict(schema="research_space_binding.v1",report_sha256=report,graph_sha256=graph,traces=[],
                     context=dict(study_type="self_generated_state_kv_intervention",
                                  model_dimensions=dict(n_layers=36,d_ff=9728,d_model=2560,n_heads=32))))
    work=put(dict(schema="research_workpackage.v1",title=title,stages={
        "behavior":{"status":"done","qwen3_4b_confirm_views":audit["causal"]["views"]},
        "same_generation_intervention":{"status":"done","accepted_scoped_native_edges":len(edges)},
        "independent_recomputation":{"status":"pending"},
        "registration_validated":{"status":"done","report_sha256":report,"graph_sha256":graph}}))
    study=dict(run_id=R.RUN,model_snapshot=model,workpackage_sha256=work,report_sha256=report,graph_sha256=graph,binding_sha256=binding)
    manifest=dict(schema="ai2050_research_upload.v1",studies=[study],objects=[dict(sha256=k) for k in objects])
    target=R.Previous.check_write(R.OUT/BUNDLE)
    assert not target.exists()
    with tarfile.open(target,"w:gz") as stream:
        for name,raw in [("MANIFEST.json",canonical(manifest)),*((f"objects/{k}.json",v) for k,v in objects.items())]:
            member=tarfile.TarInfo(name)
            member.size=len(raw)
            stream.addfile(member,io.BytesIO(raw))
    with tarfile.open(target,"r:gz") as stream:
        assert dict(verified_objects(stream,manifest))==objects
    R.write("publication.json",dict(status="eligible scoped archive",study=study,objects=len(objects),edges=len(edges),
                                   bundle_sha=R.digest(target),remote=None,
                                   scope="The intervention grade describes this one admitted experiment. Independent node/official promotion are separate."))


def isolated():
    packed=R.ROOT/"release/ai2050-center-1.0.29.tar.gz"
    assert packed.with_suffix(".gz.sha256").read_text().split()[0]==R.digest(packed)
    root=R.Previous.check_write(R.TEMP/"isolated-release")
    root.mkdir()
    with tarfile.open(packed,"r:gz") as archive:
        archive.extractall(root,filter="data")
    release=root/"ai2050-center-1.0.29"
    env={**os.environ,"OWN_STATE_BUNDLE":str(R.OUT/BUNDLE),"OWN_STATE_LIBRARY":str(root/"test-library")}
    env.pop("PYTHONPATH",None)
    code="""
import hashlib,json,os,tarfile
from pathlib import Path
from deploy.research_ingest_bundle import verified_objects
from deploy.research_library import Library,canonical
from server import research_space_service as space
l=Library(Path(os.environ['OWN_STATE_LIBRARY']))
space.library=lambda:l
with tarfile.open(os.environ['OWN_STATE_BUNDLE'],'r:gz') as f:
    manifest=json.load(f.extractfile('MANIFEST.json'))
    docs=list(verified_objects(f,manifest))
for sha,raw in docs:
    assert l.ingest(json.loads(raw),'registered',kind='registered')['sha256']==sha
    assert hashlib.sha256(canonical(l.artifact(sha))).hexdigest()==sha
index=space._research_space()
s=manifest['studies'][0]
entry=next(x for x in index['studies'] if x['run_id']==s['run_id'])
assert entry['report_sha256']==s['report_sha256'] and entry['binding_sha256']==s['binding_sha256']
assert l.artifact(s['report_sha256'])['independent_node_reproduction'] is False
assert {x['body']['evidence']['level'] for x in index['objects']}=={'intervention'}
print(json.dumps(dict(release='1.0.29',verified_objects=len(docs),observed=entry,scope='isolated release code and client index only')))
"""
    verification=subprocess.run([sys.executable,"-B","-X","utf8","-c",code],cwd=release,env=env,
                                capture_output=True,text=True,timeout=50)
    R.Previous.check_write(R.OUT/"isolated-client.log").write_text(verification.stdout+verification.stderr,encoding="utf-8")
    verification.check_returncode()
    R.write("isolated-client-proof.json",json.loads(verification.stdout.splitlines()[-1]))


def remote(do_upload):
    from urllib.request import urlopen
    health=json.load(urlopen(ORIGIN+"/api/health",timeout=12))
    if do_upload:
        if not (os.getenv("AI2050_RESEARCH_ADMIN_TOKEN") or os.getenv("AI2050_RESEARCH_SESSION")):
            raise RuntimeError("No existing authorized research administrator session found")
        subprocess.run([sys.executable,"-B",str(R.ROOT/"deploy/research_ingest_bundle.py"),str(R.OUT/BUNDLE),
                        "--center",ORIGIN],check=True)
    index=json.load(urlopen(ORIGIN+"/api/traces/research-space",timeout=15))
    assert index["schema"]=="research_space_index.v1"
    publication=R.read("publication.json")
    target=publication["study"]
    found=next((s for s in index["studies"] if s["run_id"]==target["run_id"] and
                s["model_snapshot"]==target["model_snapshot"]),None)
    evidence_verified=False
    if found:
        assert any(o["body"]["run_id"]==target["run_id"] and o["body"]["evidence"]["level"]=="intervention" for o in index["objects"])
        assert target["report_sha256"]==found["report_sha256"] and target["binding_sha256"]==found["binding_sha256"]
        report=json.load(urlopen(ORIGIN+"/api/research-library/artifacts/"+found["report_sha256"],timeout=15))
        assert hashlib.sha256(canonical(report)).hexdigest()==found["report_sha256"]
        evidence_verified=True
    if do_upload:
        assert evidence_verified
    receipt=dict(center=ORIGIN,version=health["version"],client_index_verified=evidence_verified,study=found,
                 checked_at=datetime.now(ZoneInfo("America/Chicago")).isoformat(timespec="minutes"))
    R.write("production-readback.json",receipt)
    if evidence_verified:
        publication["remote"]=receipt
        R.write("publication.json",publication)
    print("Production",health["version"],"study indexed",evidence_verified,flush=True)


if __name__=="__main__":
    E.select()
    R.configure()
    {"package":package,"isolated":isolated,"remote-check":lambda:remote(False),"upload":lambda:remote(True)}[sys.argv[1]]()
