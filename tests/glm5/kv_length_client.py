"""Register verified Phase 2952 evidence using the existing research-library schema."""

from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile

import kv_length_run as C
from deploy.research_ingest_bundle import verified_objects
from deploy.research_library import Library, canonical


SOURCE=C.OUT
CATALOG=C.ROOT/'research/kv_length/v1'
OUTPUT=C.ROOT/'tests/glm5/result/kv_length_client_20261117_v1'
TITLE='Qwen3-14B · 实发 State 的距离、寻址与写入'
LABELS={'distance-preference':'中性段延长后的实发词答码优势','address-state':'H18 寻址阻断 · 等幅对照',
        'value-state':'H18 Value 写入阻断 · 等幅对照','fact-preference':'事实头阻断 · 等幅对照',
        'book-preference':'码本头阻断 · 等幅对照','joint-margin':'三路联合阻断','h18-rescue':'恢复原生 H18',
        'parallel-rescue':'恢复两条原生并行读取'}


def load(name):
    return json.loads((SOURCE/name).read_text(encoding='utf-8'))


def documents():
    ledger=load('manifest.json')
    for name in ('audit.json','plan.json','freeze.json','common-shape.json','native-recheck.json','confirm-natural.json','report.md','rounding-exclusions.json'):
        assert C.A.digest(SOURCE/name)==ledger['files'][name],name
    audit,plan,shape=load('audit.json'),load('plan.json'),load('common-shape.json')['result']
    model=plan['snapshot']['snapshot']; common=dict(kind='real',run_id=C.RUN,model_snapshot=model)
    payload={}; hashed={}
    def put(file,doc):
        raw=canonical(dict(common,**doc)); h=sha256(raw).hexdigest()
        payload[file]=raw; hashed[h]=raw
        return h

    cases=[]
    for tag,r in audit['natural']['records'].items():
        observed=r.get('observed') or r['short']; row=r['row']
        cases.append(dict(kind='greedy',tag=tag,id=row['id'],fact=row['words'][row['fact']],state=row['values']['State'],
                          factual_code=row['code_f'],emitted_state_code=row['code_s'],output=observed['rendered'],
                          paired_output=r.get('long',{}).get('rendered'),input=row['prompt']))
    names=load('freeze.json')['arm_names']
    for tag,rows in audit['reads']['examples'].items():
        for r in rows[:1]:
            row=r['row']; letters=row['token']['letters']; a=('sham','address','value','key','joint','restore-h18','restore-parallel','restore-all')
            scored={n:dict(code='PQ'[letters.index(r['top'][names.index(n)])] if r['top'][names.index(n)] in letters else str(r['top'][names.index(n)]),
                           logits=r['pq'][names.index(n)]) for n in a}
            cases.append(dict(kind='intervention',tag=tag,id=row['id'],fact=row['words'][row['fact']],state=row['values']['State'],
                              factual_code=row['code_f'],emitted_state_code=row['code_s'],input=row['prompt'],arms=scored))
    links={k:ledger['files'][k] for k in ('plan.json','freeze.json','confirm-natural.json','audit.json','native-recheck.json',
                                          'common-shape.json','rounding-exclusions.json','report.md')}
    ev=put('00-cases.research.json',dict(schema='kv_length_cases.v1',title='真实生成、改码与反例',cases=cases,source_file_sha256=links))
    stats=audit['primary']; assert all(stats[k]['passed'] for k in ('address-state','value-state','joint-margin','h18-rescue','parallel-rescue'))
    assert not any(stats[k]['passed'] for k in ('distance-preference','fact-preference','book-preference'))
    n=audit['natural']; reads=audit['reads']; exact=load('native-recheck.json')['lexical']['matched_conflict']
    report=put('01-report.research.json',dict(schema='kv_length_report.v1',title=TITLE,phase=2952,case_sha256=ev,
        result=dict(complete=n['complete'],exact_pairs=n['exact_pairs'],intervention_families=reads['n'],gap=n['exact_gap'],
                    counts=exact,transitions=n['length_transitions'],attention=shape['pairs']['alpha'],same_k=shape['pairs']['key']['exact'],
                    same_v=shape['pairs']['value']['exact'],shape_top_matches=shape['exact_top'],shape_top_total=shape['n']*2,
                    actual_joint_flips=reads['arms']['joint']['flip'],all_restore_flips=reads['arms']['restore-all']['flip']),
        tests=[dict(id=k,label=LABELS[k],**v) for k,v in stats.items()],dose=reads['dose'],
        highlights=[r for r in cases if r['tag'] in ('fact_to_state','state_to_fact','address_and_value_give_different_code')],
        limitations=['在同一提示和实发词下比较这一段指定中性文本的插入，内容与距离同时变化。',
                    '用原生末层读出决定分叉的首个 token，随后逐 token 核验真实贪心续写。',
                    '两个并行头的单独等幅主检验未过，一组码本剂量偏差超过 3%。原生部分恢复量不能加总。',
                    '这是该快照和受控 P/Q 任务的范围干预。独立物理节点复算待完成。'],source_file_sha256=links))
    cards=[]; edges=[]
    scope=dict(dataset='Phase 2952 同提示中性段',panel='confirm',runs_only_this_snapshot=True,read_boundary='Code: 后、答码前')
    examples=[r['id'] for r in cases if r['tag'] in ('address_and_value_give_different_code','h18_restore_lowers_original_margin')]
    def card(suffix,layer,index,label,bdy,kind):
        evidence=dict(level=kind,artifact_sha256=report)
        id_=C.RUN+':'+suffix
        cards.append(dict(id=id_,layer=layer,index=index,label=label,boundary=bdy,method='原生 BF16 同提示受控前向及留出干预',
                          run_id=C.RUN,evidence=evidence,scope=scope,counterexamples=examples))
        return id_
    target=card('answer',39,dict(space='residual',field='next_token_logit'),'P/Q · 最终答码','lm_head.before_answer', 'heldout')
    for suffix,label in (('address','L25H18 · State 的 Q/K 寻址'),('write','L25H18 · State 的 V/O 写入')):
        source=card(suffix,25,dict(space='attention',head=18,field=suffix),label,'attention.query_output_before_o_proj','intervention')
        edges.append(dict(id=source+':code',source=source,target=target,relation='范围干预改变实发词答码优势',run_id=C.RUN,
                          evidence=dict(level='intervention',artifact_sha256=report),scope=scope))
    graph=put('02-graph.research.json',dict(schema='research_graph.v1',feature_cards=cards,graph_edges=edges))
    binding=put('03-binding.research.json',dict(schema='research_space_binding.v1',report_sha256=report,graph_sha256=graph,traces=[],
                context=dict(study_type='kv_length',model_dimensions=dict(n_layers=40,d_model=5120,d_ff=17408,n_heads=40))))
    work=put('04-workpackage.research.json',dict(schema='research_workpackage.v1',title=TITLE,stages={
        'heldout_behavior':dict(status='done',family_count=n['complete']),
        'native_intervention':dict(status='done',family_count=reads['n'],evidence='scoped'),
        'independent_recomputation':dict(status='pending'),
        'registration_validated':dict(status='done',report_sha256=report,graph_sha256=graph)}))
    study=dict(run_id=C.RUN,model_snapshot=model,report_sha256=report,graph_sha256=graph,binding_sha256=binding,workpackage_sha256=work)
    return payload,hashed,study


def build():
    C.P.check_write(CATALOG,10*2**20).mkdir(parents=True,exist_ok=True)
    C.P.check_write(OUTPUT,10*2**20).mkdir(parents=True,exist_ok=True)
    files,hashed,study=documents()
    for name,raw in files.items():
        dest=C.P.check_write(CATALOG/name)
        if dest.exists():assert dest.read_bytes()==raw
        else:dest.write_bytes(raw)
    package=C.P.check_write(OUTPUT/'phase-2952-research-upload.tar.gz')
    manifest=dict(schema='ai2050_research_upload.v1',studies=[study],objects=[dict(sha256=h) for h in hashed])
    with tarfile.open(package,'w:gz') as packed:
        for name,data in [('MANIFEST.json',canonical(manifest)),*((f'objects/{h}.json',raw) for h,raw in hashed.items())]:
            item=tarfile.TarInfo(name);item.size=len(data);packed.addfile(item,io.BytesIO(data))
    with tarfile.open(package,'r:gz') as packed:
        assert dict(verified_objects(packed,manifest))==hashed
    lib=Library(C.P.check_write(OUTPUT/'verified-library'))
    from server import research_space_service as s
    first=lib.scan(CATALOG); second=lib.scan(CATALOG)
    assert first=={'new_revisions':len(files),'errors':[]} or first=={'new_revisions':0,'errors':[]}
    assert second=={'new_revisions':0,'errors':[]}
    s.library=lambda:lib
    index=s._research_space(run_id=C.RUN)
    found=next(x for x in index['studies'] if x['run_id']==C.RUN)
    for key in ('run_id','model_snapshot','report_sha256','binding_sha256','workpackage_sha256'):
        assert found[key]==study[key],key
    assert lib.artifact(study['binding_sha256'])['graph_sha256']==study['graph_sha256']
    assert len(index['objects'])==5 and all(x['body']['run_id']==C.RUN for x in index['objects'])
    for h,data in hashed.items():assert canonical(lib.artifact(h))==data
    receipt=dict(scope='repository catalog auto-index and center bundle verified locally; formal deployment unverified',
                  bundle_sha256=C.A.digest(package),study=found,graph_objects=len(index['objects']),object_count=len(hashed),manifest=manifest)
    C.P.check_write(OUTPUT/'client-readback.json').write_bytes(canonical(receipt))
    print(receipt,flush=True)


if __name__=='__main__':build()
