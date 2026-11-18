"""Build the public Phase 2955 report and observation card from the frozen audit."""

import io
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from deploy.research_ingest_bundle import verified_objects
from deploy.storage_policy import check_write, staging_file
from shared.dist_protocol.mainline_state import canonical
from hashlib import sha256


SOURCE=ROOT/'tests/glm5/result/kv_receipt_20261117_v1'
TITLE='Qwen3-14B · 固定实发 State 的等长插入与真实改码'


def digest(data):
    return sha256(data).hexdigest()


def records():
    audit_file=SOURCE/'audit.json'
    audit_raw=audit_file.read_bytes()
    audit=json.loads(audit_raw)
    manifest=json.loads((SOURCE/'manifest.json').read_bytes())
    required=('audit.json','plan.json','frozen.json','words-intervention-binding.json','frames-intervention-binding.json')
    for name in required:
        assert digest((SOURCE/name).read_bytes())==manifest['files'][name],name
    common=dict(kind='real',run_id=audit['run'],model_snapshot=audit['snapshot'])
    objects={}

    def add(schema,**fields):
        raw=canonical(dict(schema=schema,**common,**fields))
        sha=digest(raw)
        objects[sha]=raw
        return sha

    studies=[]
    for split,label in [('words','新词'),('frames','新词与新句式')]:
        p=audit['panels'][split]; e=p['interventions']
        studies.append(dict(split=split,label=label,families=p['admitted_families'],planned=p['planned_families'],
                            real_flips=sum(q['flip'] for q in p['flip_by_route'].values()),
                            inserted=sum(q['n'] for q in p['flip_by_route'].values()),
                            receipt_wins=sum(q['n'] for route,q in p['flip_by_route'].items() if route.endswith('→Receipt')),
                            peak_ratio=p['max_receipt_over_state'],flip_signature=p['flip_signature'],
                            read_pass=p['prediction']['read_gain']['passed'],
                            flip_pass=p['prediction']['flip_gain']['passed'],
                            dose=e['doses'],dose_pass={k:v['matched_dose'] for k,v in e['effects'].items() if 'matched_dose' in v},
                            eligible_flips=e['observed_flips'],state_reverted=e['arms']['state-value']['reversion'],
                            receipt_address_reverted=e['arms']['address']['reversion'],receipt_value_reverted=e['arms']['value']['reversion'],
                            joint_reverted=e['arms']['joint']['reversion'],
                            receipt_restore=e['restore_original_after_joint_reversion'],
                            state_restore=e['restore_state_only_after_joint_reversion'],
                            cases=e['cases'],equal_text=p['equal_text_cases'],excluded_sham=e['excluded_sham_ids'],
                            failed_cells=[x for x in p['trigger_cells'] if x['good']/x['total']<.85]))
    report=add('kv_receipt_report.v1',phase=2955,title=TITLE,panels=studies,
               limitations=['三段英文文本，固定输出格式，插入 7、14、21 token。反角色 State 由同模型 donor 实发，再用回原提示。',
                            'State 的原生 K/V 在同家族十个视图逐位一致。插入同时改变可见文本与其后位置。',
                            '冻结的改码及寻址预测均未过留出主判。State 的阻断与恢复是同路径的范围证据，独立节点复算待完成。'],
               source_file_sha256={name:manifest['files'][name] for name in required})
    examples=[c['id'] for p in studies for c in p['cases']]
    card=dict(id=audit['run']+':h18-state-v',layer=25,index=dict(space='attention',head=18,field='value'),
              label='L25H18 · 实发 State 的读出',boundary='attention.query_output_before_o_proj',
              description='同一提示下的固定 State。Receipt 未超过 State。特定反角色样本保留单查询阻断及原生恢复。',
              method='原生 BF16 固定历史单查询测量',run_id=audit['run'],
              evidence=dict(level='observation',artifact_sha256=report),
              scope=dict(phase=2955,read_boundary='Code: 后、答码前',model_snapshot=audit['snapshot']),
              counterexamples=examples)
    graph=add('research_graph.v1',feature_cards=[card],graph_edges=[])
    binding=add('research_space_binding.v1',report_sha256=report,graph_sha256=graph,traces=[],
                context=dict(study_type='kv_receipt',model_dimensions=dict(n_layers=40,d_model=5120,d_ff=17408,n_heads=40)))
    work=add('research_workpackage.v1',title=TITLE,stages=dict(
        registration_validated=dict(status='done',report_sha256=report,graph_sha256=graph),
        heldout=dict(status='done',scope='两个预注册面板及新词反角色行为失败'),
        prediction=dict(status='not_supported'),native_intervention=dict(status='done',evidence='scoped'),
        independent_recomputation=dict(status='pending')))
    study=dict(run_id=audit['run'],model_snapshot=audit['snapshot'],report_sha256=report,graph_sha256=graph,
               binding_sha256=binding,workpackage_sha256=work)
    return objects,study


def build(output):
    objects,study=records()
    manifest=dict(schema='ai2050_research_upload.v1',studies=[study],objects=[{'sha256':k} for k in objects])
    members={'MANIFEST.json':canonical(manifest)} | {f'objects/{k}.json':v for k,v in objects.items()}
    target=check_write(output,sum(map(len,members.values()))+2**16,operation='Phase 2955 展示数据')
    with staging_file(target) as partial:
        with tarfile.open(partial,'w:gz') as archive:
            for name,raw in members.items():
                info=tarfile.TarInfo(name)
                info.size=len(raw); info.mode=0o644
                archive.addfile(info,io.BytesIO(raw))
        with tarfile.open(partial,'r:gz') as archive:
            assert dict(verified_objects(archive,manifest))==objects
    print(json.dumps(study,ensure_ascii=False),flush=True)


if __name__=='__main__':
    build(Path(sys.argv[1]))
