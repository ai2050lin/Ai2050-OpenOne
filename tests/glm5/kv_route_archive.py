"""Bundle only admitted native results and verify the client registration."""

from hashlib import sha256
import io
import sys
import tarfile

import kv_read_archive as Verify
from kv_route_contrast import configure
import kv_route_run as R
from deploy.research_library import canonical
from deploy.research_ingest_bundle import verified_objects

NAME = "kv-route-scoped-client.tar.gz"
TITLE = "Qwen3-4B · 自生成状态 Value 的单头与三头定点读取"
FIELDS = {"content": "内容特异性", "source": "状态槽对照 Note 槽", "clock": "答码时刻对照早期时刻"}


def package():
    audit = R.read("audit.json")
    contrast = R.read("matched-contrast.json")
    transfer = R.OUT.with_name("kv_route_20261115_v3")
    old = R.OUT.with_name("kv_route_20261115_v1")
    load = R.Base.Previous.load
    base = load(old / "audit.json")
    transfer_audit = load(transfer/"audit.json")
    transfer_contrast = load(transfer/"matched-contrast.json")
    model = audit["snapshot"]
    assert contrast["audit_sha"] == R.sha(R.OUT/"audit.json")
    assert model==transfer_audit["snapshot"] and transfer_contrast["audit_sha"]==R.sha(transfer/"audit.json")
    docs = {}

    def put(**body):
        raw = canonical(dict(kind="real",run_id=R.RUN,model_snapshot=model,**body))
        h = sha256(raw).hexdigest()
        docs[h] = raw
        return h

    cases = {}
    cards, edges = [], []

    def add(sid, suffix, group, label, stat, evidence_run, **extra):
        start = f"{R.RUN}:{sid}:{suffix}"
        end = start + ":code"
        evidence = dict(level="intervention",artifact_sha256=cases[sid])
        example = [stat["worst"]] if "worst" in stat else []
        details = dict(suite=sid,experiment_run=evidence_run,heads=group,comparison=suffix,
                       alternate_minus_correct=stat["mean"],**extra)
        single = len(group) == 1
        layer, head = group[0] if group else (0,None)
        ix = (dict(space="attention",field="generated_state_Value",head=head,kv_group=head//4)
              if single else dict(space="attention",field="generated_state_Value",layer_anchor_only=True))
        cards.extend([
            dict(id=start,layer=layer,boundary="attention.query_output_before_o_proj",index=ix,
                 method="native SDPA controlled Value read edit",label=f"{sid} · {label}",
                 run_id=R.RUN,scope=details,evidence=evidence,counterexamples=example),
            dict(id=end,layer=35,boundary="lm_head.before_answer_token",
                 index=dict(space="residual",field="final_readout"),method="native P/Q logit margin",
                 label=f"{sid} · P/Q 答码读出 · {label}",run_id=R.RUN,
                 scope=details,evidence=evidence,counterexamples=example),
        ])
        edges.append(dict(id=start+":edge",source=start,target=end,relation="scoped causal effect on next-code score",
                          run_id=R.RUN,scope=details,evidence=evidence))

    for diff, folder, verified in [(x,R.OUT,audit) for x in contrast["suites"]]+[
                                  (x,transfer,transfer_audit) for x in transfer_contrast["suites"]]:
        sid = diff["suite"]
        confirmation = load(folder/"confirmation.json")
        info = next(a for a in verified["holdout"] if a["suite"] == sid)
        raw = next(a for a in confirmation["ledger"] if a["suite"] == sid)
        assert info["status"] == raw["status"] == "complete"
        cases[sid] = put(schema="kv_route_cases.v1",suite=sid,experiment_run=verified["run"],
                         materials=load(folder/f"materials-{sid}.json"),natural=load(folder/f"behavior-{sid}.json")["rows"],
                         arms=confirmation["arms"],intervention_batches=[load(folder/a["name"]) for a in raw["shards"]],
                         original_native_field_hash=R.sha(folder/f"native-{sid}.npz"),
                         plan_sha=R.sha(folder/"plan.json"),thresholds=load(folder/"plan.json")["contrasts"])
        for tag, d in diff["groups"].items():
            group = R.HEADS[:1] if tag=="L21H19" else R.HEADS
            for f, result in d["tests"].items():
                if result["passed"]:
                    add(sid,f"{tag}-{f}",group,f"{tag} {FIELDS[f]}",result,verified["run"],
                        one_read_per_query_head=True,matched_raw_value_norm=True)
        for subset, stats in info["subsets"].items():
            if stats["controlled_pair"]:
                add(sid,"remove-restore-"+subset,tuple(R.HEADS[int(i)] for i in subset),
                    f"组 {subset} 移除与恢复",stats["remove"],verified["run"],restore=stats["restore"],
                    restoration_from_all_three_zero=True)
        if info["measures"]["whole_content"]["passed"]:
            add(sid,"whole-content",(), "全查询头内容对照",info["measures"]["whole_content"],verified["run"],
                edited_query_heads=1152,matched_rotation=True)

    if not edges:
        R.write("publication.json",dict(status="no admitted causal edge",remote=None))
        return
    retro = audit["retrospective"]
    report = put(schema="kv_route_report.v1",title=TITLE,admitted_suites=list(cases),case_object_refs=cases,
                 audit_sha=R.sha(R.OUT/"audit.json"),plan_sha=R.sha(R.OUT/"plan.json"),
                 matched_sha=R.sha(R.OUT/"matched-contrast.json"),retrospective={k:v for k,v in retro.items() if k!="rows"},
                 behavior=audit["behavior"],failed_first_version=base["behavior"],reference_behavior=base["references"],
                 source_time=contrast["suites"]+transfer_contrast["suites"],
                 holdout=audit["holdout"]+transfer_audit["holdout"],selection=audit["selection"],
                 transfer_run=dict(run_id=transfer_audit["run"],plan_sha=transfer_contrast["plan_sha"],
                                   audit_sha=transfer_contrast["audit_sha"],behavior=transfer_audit["behavior"]),
                 evidence_grade="scoped intervention",independent_gpu_recomputation=False,
                 caveats=["固定自然非答案前缀；自由答码只登记可保留原生成边界的批次。",
                          "剂量匹配在 BF16 Value 改变量空间完成；自然注意力权重和实际残差写入可不同。",
                          "全头、其余头和三头的直接比较使用不同读取次数；不据此求独立路径占比。",
                          "匹配循环旋转控制幅度；它不建立零语义。移除与恢复只说明本实验定义的干预。"])
    graph = put(schema="research_graph.v1",feature_cards=cards,graph_edges=edges)
    binding = put(schema="research_space_binding.v1",report_sha256=report,graph_sha256=graph,traces=[],
                  context=dict(study_type="value_source_time_factorial",
                               model_dimensions=dict(n_layers=36,d_ff=9728,d_model=2560,n_heads=32)))
    work = put(schema="research_workpackage.v1",title=TITLE,stages={
        "discovery":dict(status="done",families=16),
        "heldout_intervention":dict(status="done",suites=list(cases),scoped_edges=len(edges)),
        "independent_recomputation":dict(status="pending"),
        "registration_validated":dict(status="done",report_sha256=report,graph_sha256=graph)})
    study = dict(run_id=R.RUN,model_snapshot=model,report_sha256=report,graph_sha256=graph,
                 binding_sha256=binding,workpackage_sha256=work)
    manifest = dict(schema="ai2050_research_upload.v1",studies=[study],objects=[dict(sha256=h) for h in docs])
    target = R.Base.Previous.check_write(R.OUT/NAME)
    with tarfile.open(target,"x:gz") as f:
        for name, data in [("MANIFEST.json",canonical(manifest)),*((f"objects/{h}.json",v) for h,v in docs.items())]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            f.addfile(member,io.BytesIO(data))
    with tarfile.open(target,"r:gz") as f:
        assert dict(verified_objects(f,manifest)) == docs
    R.write("publication.json",dict(status="eligible scoped intervention",study=study,sha=R.sha(target),
                                    objects=len(docs),cards=len(cards),edges=len(edges),remote=None))


if __name__ == "__main__":
    configure()
    Verify.R,Verify.NAME = R,NAME
    {"package":package,"isolated":Verify.isolated,"production":Verify.published}[sys.argv[1]]()
