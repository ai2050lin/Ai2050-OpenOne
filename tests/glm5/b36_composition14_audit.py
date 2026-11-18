"""End-to-end identity, native shapes, split and intervention audit."""
import argparse,time
from pathlib import Path
import numpy as np
import b36_composition14 as B
import b36_composition14_collect as C
import b36_composition14_tail as T


def audit():
    B.configure();C.admit();T.admit();start=time.monotonic();checks=[];native_bytes=0
    model=B.read(B.OUT/'model-identity.json')
    for i,f in enumerate(model['files']):
        p=B.MODEL/f['path'];assert p.stat().st_size==f['bytes'] and B.file_sha(p)==f['sha256']
        B.log('audit_model',i+1,len(model['files']),start)
    checks.append(dict(name='model_and_tokenizer',files=len(model['files']),passed=True))
    allrows=B.read(B.OUT/'materials.json')['rows'];byid={r['id']:r for r in allrows};families={s:{r['family'] for r in allrows if r['split']==s} for s in ['pilot','discovery','confirm','format','order']}
    assert not families['pilot']&families['discovery'];assert not (families['pilot']|families['discovery'])&families['confirm']
    assert families['format']<=families['confirm'] and families['order']<=families['confirm']
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(B.MODEL,local_files_only=True)
    for r in allrows:
        assert tok.encode(r['prompt'],add_special_tokens=False)==r['tokens'];assert r['answer_pos']==len(r['tokens'])-1
        for c,token in zip(r['format'],r['choice_ids']):assert tok.encode(r['prompt']+' '+c,add_special_tokens=False)==r['tokens']+[token]
    checks.append(dict(name='families_tokens_choices',passed=True,rows=len(allrows),unique_confirm_families=len(families['confirm'])))
    for split in ['pilot','discovery','confirm','format']:
        meta=B.read(B.OUT/split/'collection.json');origin=Path(meta.get('native_origin',B.OUT/split));ids=[]
        assert meta['row_ids']==[allrows[i]['id'] for i in meta['row_indices']]
        for chunk in meta['chunks']:
            folder=origin/chunk['name'];cp=folder/'chunk.json';assert B.file_sha(cp)==chunk['sha256'];m=B.read(cp);ids+=m['row_ids'];n=len(m['row_ids'])
            if m.get('numeric_reference'):
                assert all(c['bitwise'] and c['relative_L2']==0 for c in m['numeric_reference'])
            for f in m['files']:
                p=folder/f['path'];assert p.stat().st_size==f['bytes'] and B.file_sha(p)==f['sha256'];native_bytes+=f['bytes']
            with np.load(folder/'states.npz') as d:
                assert d['hidden'].shape==(n,41,2,B.DIM) and d['hidden'].dtype==np.uint16
                assert d['B36_group_trajectory'].shape==(n,B.LENGTH,32,3)
                assert np.isfinite(B.unbits(d['hidden'])).all()
            for block in [36,37,38,39]:
                with np.load(folder/f'B{block}.npz') as d:
                    for f in 'GVA':
                        assert d[f].shape==(n,B.WIDTH if block in [36,39] else len(B.tracked())) and d[f].dtype==np.uint16
                        assert d[f+'_content'].shape==(n,len(B.tracked()))
        assert ids==meta['row_ids']
        checks.append(dict(name='native_'+split,passed=True,rows=len(ids),read_only_origin=str(origin)))
        B.log('audit_native',len(checks)-2,4,start,split=split)
    for split in ['confirm','format']:
        p=B.OUT/f'tail-{split}'/'summary.json'
        if not p.exists():continue
        s=B.read(p);admission=B.read(B.OUT/(split+'-admissibility.json'));assert s['row_ids']==admission['eligible_row_ids']
        for name,sha in s['arrays'].items():assert B.file_sha(p.parent/name)==sha
        for c in s['checks']:
            assert c['restore_bitwise'] and c['prefix_bitwise'] and c['baseline_bitwise']
            assert c['smooth_attention_max_relative_L2']<=.02
        assert len(s['effects'])==len(s['row_ids'])*len(s['arms'])
        collection=B.read(B.OUT/split/'collection.json');scores=dict(zip(collection['row_ids'],collection['scores']))
        for e in s['effects']:
            if e['arm'] in ['baseline','restore']:assert e['scores']==scores[e['id']],(split,e['id'],e['arm'])
        checks.append(dict(name='intervention_'+split,passed=True,rows=len(s['row_ids']),arms=len(s['arms']),separate_independent_N=False))
    output=B.read(B.OUT/'confirm-output-report.json');collection=B.read(B.OUT/'confirm/collection.json');scores=dict(zip(collection['row_ids'],collection['scores']))
    for r in output['rows']:assert r['teacher_scores']==scores[r['id']],r['id']
    checks.append(dict(name='conditional_output_teacher_scores_exact',passed=True,rows=len(output['rows'])))
    diagnostic=B.read(B.OUT/'readout-diagnostic.json');plan=B.read(B.OUT/'readout-diagnostic-plan.json')
    assert diagnostic['diagonal_max_abs_error']==0 and diagnostic['decomposition_identity_passed']
    assert diagnostic['plan_sha256']==B.file_sha(B.OUT/'readout-diagnostic-plan.json')
    assert diagnostic['source_sha256']==plan['source_sha256']==B.file_sha(B.ROOT/'tests/glm5/b36_composition14_readout_diagnostic.py')
    for r in diagnostic['rows']:
        assert abs(r['code_component']+r['state_component']-r['total'])<1e-10
    checks.append(dict(name='posthoc_readout_exact_diagonal_and_identity',passed=True,pairs=diagnostic['pairs'],new_independent_N=0))
    summary=B.read(B.OUT/'research-summary.json');assert not summary['formal_promotion']
    B.save('audit.json',dict(kind='local_technical_audit_not_independent_replication',passed=True,checks=checks,native_bytes_counting_referenced_pilot_once=native_bytes,
        source_sha256=B.file_sha(__file__),seconds=time.monotonic()-start,
        boundaries=['Technical audit is not scientific acceptance or independent node replication.','Failed pilot strata stay behavior-only.','No training-time, cross-model or free-generation claim.']))


def manifest():
    B.configure();T.admit();files={}
    required=['report.html','report-build.json','audit.json','browser-check.json','feature-cards.json','mechanism-graph.json','delivery-source.zip','delivery-source.json','cleanup.json','memo-append.json','research-summary.json','path-output-report.json','readout-diagnostic.json']
    assert all((B.OUT/n).exists() for n in required)
    assert B.read(B.OUT/'audit.json')['passed'] and B.read(B.OUT/'browser-check.json')['passed']
    for p in sorted(B.OUT.rglob('*')):
        if p.is_file() and p.name!='manifest.json' and not p.name.endswith('_b36_composition14_manifest.json.json'):
            files[p.relative_to(B.OUT).as_posix()]=dict(sha256=B.file_sha(p),bytes=p.stat().st_size)
    B.save('manifest.json',dict(kind='delivery_manifest',files=files,external_reused_pilot_manifest='pilot-reuse.json',source_sha256=B.file_sha(__file__),
        exclusions=['manifest.json','The manifest own provenance helper, created after enumerating files; reference in provenance_helper.'],
        scientific_joint_composition_confirmed=False,local_work_completed=True,public_deployed=False,formal_promotion=False))
    print('Manifest entries',len(files),flush=True)
    if B.TEMP.exists() and not any(B.TEMP.iterdir()):B.TEMP.rmdir()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['audit','manifest']);a=p.parse_args();globals()[a.stage]()
