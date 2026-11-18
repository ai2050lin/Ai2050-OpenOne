"""Verify and publish the frozen authorization study to local 3D and the upload protocol."""
import io
from pathlib import Path
import tarfile

import numpy as np

import mlp_b31_authority_design as D
from mlp_b31_authority_report import capture
from deploy.research_library import Library, canonical
from tools import provenance


def main():
    plan=D.checked()
    out=D.OUT
    report=D.READ(out/'report.json')
    assert report['plan_sha256']==D.HASH(out/'prereg.json')
    assert report['selection_sha256']==D.HASH(out/'selection.json')
    for obj in report['raw_manifest']:
        p=out/obj['path']
        assert p.stat().st_size==obj['bytes'] and D.HASH(p)==obj['sha256']
    gate_path=out/'holdout-behavior-gate.json'
    gate=D.READ(gate_path)
    assert gate['passed'] and gate['plan_sha256']==report['plan_sha256'] and gate['selection_sha256']==report['selection_sha256']
    assert gate['materials_sha256']==plan['inputs']['holdout']['sha256']
    assert gate['source_sha256']==D.HASH(Path(__file__).with_name('mlp_b31_authority_confirm.py'))
    rows=[r for path in sorted(out.glob('holdout-[0-9]*.json')) for r in D.READ(path)['rows']]
    assert [(r['id'],r['base_scores']) for r in rows]==[(r['id'],r['scores']) for r in gate['rows']]
    assert not provenance.check_provenance(report)
    samples,records,delta,_=capture(plan,'holdout')
    card=D.READ(out/'selection.json')['B32']
    natural=float(np.mean(delta[32][:,card['units']]*card['orientation']/card['scale']))
    margins=[]
    for sample,record in zip(samples,records):
        i=sample['recipient_index']
        gold=sample['correct'][i]
        scores=record['base_scores'][i]
        margins.append(scores[gold]-scores[1-gold])
    report['margin_diagnostic']={'scope':'事后描述，未用于改变判据或选择剂量',
          'baseline_median_ab_margin':float(np.median(margins)),
          'b32_natural_normalized_shift':natural,
          'norm_by_dose':[{'dose':dose,'median':float(np.median([r['doses'][i]['norm'] for r in records]))}
                          for i,dose in enumerate(D.DOSES)]}
    report['behavior_gate_before_edits']={k:v for k,v in gate.items() if k!='rows'}
    report['raw_manifest'].append({'path':gate_path.name,'bytes':gate_path.stat().st_size,'sha256':D.HASH(gate_path)})
    audit=provenance.write_result('mlp_b31_b32_authority',{'run_id':D.RUN,'plan_sha256':report['plan_sha256'],
          'selection_sha256':report['selection_sha256'],'report_sha256':D.HASH(out/'report.json'),
          'model_snapshot':D.SNAPSHOT,'behavior_gate_sha256':D.HASH(gate_path),'restored_pairs':report['restored_pairs'],
          'scope':'single CUDA BF16 node, no independent hardware replication'},
          model_path=D.MODEL,seed=D.SEED,tool='mlp_b31_authority_publish')
    assert not provenance.check_provenance(D.READ(audit))
    revision=D.RUN.rsplit('-',1)[-1]
    public=D.ROOT/f'research/mlp_tail/b31_b32_{revision}'
    roots=[D.ROOT/'distributed_data/research_library',
           D.ROOT/'tests/glm5/result/mlp_tail_acceptance/center/research_library']
    libraries=[Library(path) for path in roots]
    objects={}

    def put(name,body):
        doc={**body,'kind':'real','run_id':D.RUN,'model_snapshot':D.SNAPSHOT,'provenance':plan['provenance']}
        hashes={lib.ingest(doc,f'{D.RUN}/{name}',kind='local_report')['sha256'] for lib in libraries}
        assert len(hashes)==1
        sha=hashes.pop()
        D.SAVE(public/f'{name}.research.json',doc)
        objects[sha]=canonical(doc)
        return sha

    audit_ref=put('provenance-audit',{'schema':'mlp_b31_provenance_audit.v1','audit':D.READ(audit)})
    plan_ref=put('preregistration',plan)
    kw_ref=put('keyword-audit',report['keyword_audit'])
    selection_ref=put('frozen-selection',{'schema':'mlp_b31_authority_selection.v1','selection':D.READ(out/'selection.json')})
    if plan.get('prior_negative'):
        previous=out.with_name('mlp_b31_b32_authority_v1')
        earlier_gate=previous/'holdout-behavior-gate.json'
        assert D.HASH(earlier_gate)==plan['prior_negative']['gate_sha256']
        assert D.HASH(previous/'prereg.json')==plan['prior_negative']['prereg_sha256']
        earlier=D.READ(earlier_gate)
        material_path=previous/'holdout-materials.json'
        assert D.HASH(material_path)==earlier['materials_sha256']
        old_material={row['id']:row for row in D.READ(material_path)['items']}
        seen=set()
        examples=[]
        for row in earlier['rows']:
            frame=row['frame']
            if not row['both_correct'] and frame not in seen:
                seen.add(frame)
                material=old_material[row['id']]
                examples.append({**row,'prompts':material['prompts'],'correct':material['correct']})
        report['prior_behavior_examples']=examples
        report['prior_behavior_failure_sha256']=put('v1-behavior-negative',{
            'schema':'mlp_b31_authority_behavior_negative.v1',
            'prior_run_id':'MLP-B31-B32-AUTHORITY-v1',
            'measurements':earlier,'examples':examples,'material_sha256':D.HASH(material_path),
            'conclusion':'行为门未通过，保留反例；未执行该版本的留出干预。'})
    report['audit_artifact_sha256']=audit_ref
    report['keyword_audit_artifact_sha256']=kw_ref
    report['selection_artifact_sha256']=selection_ref
    report_ref=put('report',report)
    cards=[]
    edges=[]
    for block,tests in report['semantic'].items():
        b=int(block[1:])
        for test in tests['units']:
            unit=test['unit']
            cards.append({'id':f'{D.RUN}:{block}:U{unit}','run_id':D.RUN,'layer':b,'boundary':'mlp.down_proj.input',
                          'label':f'{block}/U{unit} · 授权角色方向 {test["hits"]}/{test["n"]}',
                          'index':{'space':'mlp','unit':unit},'method':'发现集冻结极性，未见长文本语义角色差值',
                          'evidence':{'level':'heldout' if test['passed'] and report['baseline_gate'] else 'negative','artifact_sha256':report_ref},
                          'scope':{'test':test,'claim':'英文受控授权角色绑定','unit_api':f'/api/research/mlp-tail/units/{b}/{unit}'},
                          'counterexamples':[{'artifact_sha256':report_ref,'field':f'semantic.{block}'}]})
        group=tests['group']
        passed=group['passed'] and report['baseline_gate']
        group_id=f'{D.RUN}:{block}:group'
        cards.append({'id':group_id,'run_id':D.RUN,'layer':b,'boundary':'mlp.down_proj.input',
                      'label':f'{block} · 授权角色 32 单元组','index':{'space':'mlp','field':'residual_write'},
                      'method':'留出角色差值与同层随机单元组对照',
                      'evidence':{'level':'heldout' if passed else 'negative','artifact_sha256':report_ref},
                      'scope':{'test':group,'units':[u['unit'] for u in tests['units']]},
                      'counterexamples':[{'artifact_sha256':report_ref,'field':f'semantic.{block}.failed_group_ids'}]})
        if passed:
            edges.append({'id':f'{group_id}:language','run_id':D.RUN,'source':'language:approval_role_binding',
                          'target':group_id,'relation':'未见长文本的授权角色响应','evidence':{'level':'heldout','artifact_sha256':report_ref}})
    edges.append({'id':f'{D.RUN}:B31-B32','run_id':D.RUN,'source':f'{D.RUN}:B31:group','target':f'{D.RUN}:B32:group',
                  'relation':'B31 写入与 B32 定点恢复实测',
                  'scope':{'mediation_passed':report['mediation_passed'],'restored_pairs':report['restored_pairs'],
                           'single_node':True,'primary':'A/B token comparison'},
                  'evidence':{'level':'intervention' if report['mediation_passed'] else 'negative','artifact_sha256':report_ref}})
    graph_ref=put('graph',{'schema':'research_graph.v1','feature_cards':cards,'graph_edges':edges,
                          'note':'因果操作使用固定单元组，保留未通过的路径。单元方向为留出观测。'})
    bind_ref=put('binding',{'schema':'research_space_binding.v1','report_sha256':report_ref,'graph_sha256':graph_ref,'traces':[],
                           'context':{'study_type':'mlp_b31_authority','native_bank':True,
                                      'model_dimensions':{'n_layers':36,'d_ff':9728,'d_model':2560,'n_heads':32}}})
    work_ref=put('workpackage',{'schema':'research_workpackage.v1','title':'Qwen3-4B · B31→B32 授权角色、留出与跨层恢复',
                  'stages':{'keyword_audit':{'status':'done','artifact_sha256':kw_ref,'n':77824},
                            'preregistration':{'status':'done','artifact_sha256':plan_ref},
                            'heldout':{'status':'done','families':report['holdout_pairs'],'baseline_gate':report['baseline_gate']},
                            'causal':{'status':'done','switch_gate':report['controlled_behavior_passed'],'restoration_gate':report['mediation_passed']},
                            'registration_validated':{'status':'done','report_sha256':report_ref,'graph_sha256':graph_ref}}})
    registry={'run_id':D.RUN,'model_snapshot':D.SNAPSHOT,'report_sha256':report_ref,'graph_sha256':graph_ref,
              'binding_sha256':bind_ref,'workpackage_sha256':work_ref}
    bundle=D.ROOT/f'release/ai2050-research-b31-b32-authority-{revision}.tar.gz'
    with tarfile.open(bundle.with_suffix('.tmp'),'w:gz',compresslevel=5) as tar:
        def member(name,data):
            info=tarfile.TarInfo(name)
            info.size=len(data)
            tar.addfile(info,io.BytesIO(data))
        member('MANIFEST.json',canonical({'schema':'ai2050_research_upload.v1','objects':[{'sha256':s} for s in objects],'studies':[registry]}))
        for sha,data in objects.items():
            member(f'objects/{sha}.json',data)
    bundle.with_suffix('.tmp').replace(bundle)
    registry.update(provenance=plan['provenance'],audit_path=audit.relative_to(D.ROOT).as_posix(),
                    bundle=bundle.relative_to(D.ROOT).as_posix(),bundle_sha256=D.HASH(bundle),
                    cards=len(cards),edges=len(edges),objects=len(objects))
    D.SAVE(public/'registration.json',registry)
    D.SAVE(out/'registration.json',registry)
    print(registry,flush=True)


if __name__=='__main__':
    main()
