"""Append a delivery revision; never overwrite the original scientific report."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_function_materials_v4 as P
from native_mlp_function_publish import register_binding
from shared.dist_protocol.mechanism_trace import TraceStore,file_sha,digest
from deploy.research_library import Library

def main():
    lib=Library(P.ROOT/'distributed_data/research_library');r=P.read(P.OUT/'client-registration.json');old=lib.artifact(r['report_sha256']);prov=old['provenance'];snap=old['model_snapshot']
    def archive(name,doc,run=P.RUN):
        d={'schema':'research_native_evidence.v1','kind':'real','run_id':run,'model_snapshot':snap,'provenance':prov,**doc};p=P.PUBLIC/(name+'-'+digest(d)[:12]+'.research.json');P.save(p,d)
        return lib.ingest(d,str(p.relative_to(P.ROOT)),kind='local_report')['sha256']
    with lib.db() as db:rows=db.execute("SELECT sha FROM artifacts WHERE kind='space_binding' ORDER BY created DESC,sha DESC").fetchall()
    allb=[];seen=set()
    for row in rows:
        b=lib.artifact(row['sha']);key=(b['run_id'],b['model_snapshot'])
        if key not in seen:allb.append(b);seen.add(key)
    sources=[P.ROOT/x for x in ['server/research_space_service.py','server/trace_catalog.py','server/mechanism_trace_service.py','deploy/research_library.py','deploy/datapack.py',
        'frontend/src/components/v8/ResearchSpace.jsx','frontend/src/components/v8/NativeFunctionReport.jsx','frontend/src/components/v8/MechanismTraceExplorer.jsx','frontend/src/components/v8/MechanismScene.jsx','frontend/src/components/v8/ResearchInterventionReport.jsx',
        'tests/glm5/test_native_mlp_function.py','tests/glm5/native_mlp_function_report.test.jsx','tests/glm5/native_mlp_function_space.test.jsx','tests/glm5/native_mlp_function_closeout.py']]
    src=archive('delivery-source-bundle',{'schema':'research_source_bundle.v1','files':[{'path':str(p.relative_to(P.ROOT)),'sha256':file_sha(p),'text':p.read_text(encoding='utf-8')} for p in sources]})
    plan={'schema':'research_stage_plan.v1','status':'proposed_not_executed_not_frozen','question':'Separate world fact, question relation, abstract decision and lexical answer, then explain a small heldout-predictive subgraph.',
        'motivation':'Six coordinates show answer-aligned conditional responses; no individual switch or fixed property has been established.',
        'stages':[
            {'stage':'controlled observation','design':'cross fact × question polarity × answer coding (Yes/No, independently permuted arbitrary codes, another language), names/keys/frames/length balanced; new source-disjoint backgrounds','entry_gate':'discovery-only behavioral calibration; per-cell sample size, cluster unit, effect size and FWER/power fixed before confirm/lockbox'},
            {'stage':'three extraction engines','design':'same frozen inputs compare native G/V/A, a locally trained sparse dictionary/transcoder and supervised token/position baselines; fit only discovery; retain all-coordinate views','boundary':'dictionary directions are not native neurons; no one-coordinate-one-concept assumption; resource plan before all-token collection'},
            {'stage':'compute path and intervention','design':'arithmetic head→G/V→A→residual/logit sources; separate question-end, assistant query and historical cue positions; predeclared local ablation/control/rescue tests','boundary':'do not infer necessity from large local terms or a null last-query edit; retain negative controls, finite intervals and capability NI'},
            {'stage':'graph-only heldout prediction','design':'features/edges learned on discovery predict unseen frame/entity/code activations and next-token distribution; report heldout KL and loss recovery against lexical, position and n-gram baselines','boundary':'no target-layer state, future token, answer label or full original forward may be a predictor input'},
            {'stage':'scale and independent replication','design':'content-addressed job input/output, source-split shards, cross-node canary/audit, then sequential local Qwen3-14B/GLM4 comparisons','boundary':'full model snapshot and precision separate; native coordinates not equated across models'}],
        'client_contract':'Graph objects organize external language patterns and internal features; datasets are versioned evidence filters. Bind run+full snapshot+report+graph+case index; preserve counterexamples and superseded revisions. No total deciphering percentage.'}
    ps=archive('next-stage-plan',plan)
    factorial=P.read(P.OUT/'factorial-results.json');diagnostic=archive('factorial-explained',{**factorial,'definition':{'s':'fact level mapped to -1,+1','q':'question level mapped to -1,+1','b_s':'mean(s*x) over the four cells per background','b_q':'mean(q*x)','b_sq':'mean(s*q*x)'},'interpretation':'Interaction RMS dominates fact RMS in these selected units at question-end. Standard four-cell decomposition, not a newly discovered algebra or proof of causal multiplication.'})
    evidence={}
    for name in ['software-tests-final.log','client-tests-final.log','trace-tests-final.log','protocol-tests-loopback.log','frontend-build-final.log']:
        p=P.OUT/name;text=p.read_text(encoding='utf-8',errors='replace');evidence[name]={'sha256':file_sha(p),'tail':'\n'.join(text.splitlines()[-10:])}
    qa=archive('verification',{'checks':evidence,'note':'Initial sandbox temp/socket failures were rerun with an isolated project temp/research root and loopback permissions; the passing run is retained alongside the initial logs.'})
    visibility=archive('visibility-source-mapping-v3',{**P.read(P.OUT/'visibility-revisions.v3.json'),'supersedes_wrapper_sha256':old['artifact_refs']['visibility'],
        'wrapper_erratum':'Use this untouched original→v3 mapping and the case-index mapping. The earlier publication wrapper recursively remapped old_trace_id strings as well as new IDs. Tensor files, original source mapping and case-index mapping were unaffected.'})
    figure=archive('figure-index',{'files':[{'path':str((P.OUT/('research-results.'+e)).relative_to(P.ROOT)),'sha256':file_sha(P.OUT/('research-results.'+e))} for e in ['png','pdf']],'data':'Direct plot of original measurements; loss intervals nonzero at zero discordance; capability intervals individually95%, not simultaneous95% across operations.'})
    report={**old,'supersedes_report_sha256':r['report_sha256'],'delivery_revision':2,'next_stage':plan,'factorial_diagnostic':factorial,'artifact_refs':{**old['artifact_refs'],'visibility':visibility,'delivery_sources':src,'next_stage_plan':ps,'factorial_explained':diagnostic,'verification':qa,'figures':figure},
        'limitations':old['limitations']+['原生自然极点提名曾观察旧语料discovery；其中一些背景在新受控任务的留出中再次使用。新实体/句式/任务标签与当前划分独立，但不能声称整个研究链未见过这些原文；下一阶段需来源级新背景。',
            '两组能力非劣效区间各自95%，不是两组同时95%保证。主要剂量四对比区间使用98.75%各自保证、联合至少95%。']}
    rr=archive('report-delivery-v2',report);store=TraceStore()
    for b in allb:
        if b['run_id'] in [P.RUN,P.RUN+':place-polarity',P.RUN+':capability-audit',P.RUN+':operation-audit']:
            register_binding(archive,store,b['run_id'],rr,r['graph_sha256'],b['traces'],{**b['context'],'next_stage_plan_sha256':ps},'原生MLP：答案方向反例与真实3D闭环' if b['run_id']==P.RUN else b['run_id'].split(':')[-1]+' · 真实对照复采',native=r['native'] if b['run_id']==P.RUN else None)
    result={**r,'original_report_sha256':r['report_sha256'],'report_sha256':rr,'next_stage_plan_sha256':ps,'verification_sha256':qa};P.save(P.OUT/'client-registration-final.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='native'},ensure_ascii=False))
if __name__=='__main__':main()
