"""Verify local publication, resource records and lossless numeric presentation."""
import base64,hashlib,json,shutil,tarfile
from datetime import datetime
import numpy as np
import state_formation_core as C
from deploy.research_library import Library,canonical

def main():
    p=C.checked();r=C.read(C.OUT/'client-report.json');reg=C.read(C.OUT/'registration.json')
    accept=C.read(C.OUT/'client-acceptance.json');audit=C.read(C.OUT/'audit.json')
    assert accept['passed'] and not accept['errors'] and not accept['gpuRequests'] and audit['passed']
    assert 'Ran 6 tests' in (C.OUT/'unit-tests.log').read_text(encoding='utf-8-sig')
    assert 'OK' in (C.OUT/'unit-tests.log').read_text(encoding='utf-8-sig')
    assert '56 passed' in (C.OUT/'frontend-tests.log').read_text(encoding='utf-8-sig')
    assert 'built in' in (C.OUT/'build.log').read_text(encoding='utf-8-sig')
    assert not any(s in (C.OUT/'component-lint.log').read_text(encoding='utf-8-sig') for s in ['error ','Oops!'])
    libs=[Library(path) for path in reg['local_libraries']]
    est=np.load(C.OUT/'estimator.npz');arr=np.load(C.OUT/'analysis-arrays.npz');coords=[];values=0
    assert C.sha(reg['bundle'])==reg['bundle_sha256']
    with tarfile.open(reg['bundle'],'r:gz') as tar:
        manifest=json.load(tar.extractfile('MANIFEST.json'));assert len(manifest['objects'])==reg['objects']
        for obj in manifest['objects']:
            digest=obj['sha256'];raw=tar.extractfile(f'objects/{digest}.json').read();assert hashlib.sha256(raw).hexdigest()==digest
            doc=json.loads(raw);assert not C.provenance.check_provenance(doc)
            for lib in libs:assert canonical(lib.artifact(digest))==raw
            if doc.get('schema')=='state_formation_coordinates.v1':
                index=doc['index'];kind=doc['field'];actual=np.frombuffer(base64.b64decode(doc['array']['data']),'<f4').reshape(doc['array']['shape'])
                if kind=='state':expected=np.stack([est['beta'][index,[3,6,7]],arr['evaluation_beta'][index,[3,6,7]]])
                else:expected=arr['module_coefficients'][0 if kind=='attention' else 1,index,[3,6,7]][None]
                np.testing.assert_array_equal(actual,expected);coords.append((kind,index));values+=actual.size
    assert len(coords)==146 and set(coords)=={('state',i) for i in range(74)}|{(k,i) for k in ['attention','mlp'] for i in range(36)}
    # Control is bounded to this run; no old evidence or native fields are removed.
    temp=C.ROOT/'tests/glm5_temp';archived=[]
    for name in ['state-formation-storage.log','state-formation-storage-finalize.log','state-formation-storage.json']:
        source=temp/name;target=C.OUT/name
        if source.exists():
            shutil.copyfile(source,target);assert C.sha(source)==C.sha(target);source.unlink();archived.append(name)
        else:assert target.exists()
    cleanup=C.read(C.OUT/'cleanup.json');assert cleanup['removed']
    migrated=C.read(C.OUT/'storage-audit.json');old_root=C.ROOT/'tests/glm5/result/mlp_puzzle_v1'
    assert old_root.resolve()==C.OUT.parent/'mlp_puzzle_v1'
    for item in migrated['manifest']:
        old_file=old_root/item['path'];assert old_file.stat().st_size==item['bytes'] and C.sha(old_file)==item['sha256']
    monitor=[json.loads(line) for line in (C.OUT/'gpu-monitor.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    active=[row for row in monitor if row['gpu_process_memory']]
    shared=max(sum(v['SharedUsage'] for v in row['gpu_process_memory']) for row in active)
    dedicated=max(sum(v['DedicatedUsage'] for v in row['gpu_process_memory']) for row in active)
    progress=[json.loads(line) for line in (C.OUT/'progress.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    stages={}
    for row in progress:
        key=row['stage']
        if key in ['confirm','diagnostic_causal','capability','discovery']:
            values_for_stage=stages.setdefault(key,[]);values_for_stage.append(row)
    times={stage:{'last_done':rows[-1]['done'],'total':rows[-1]['total'],'seconds':rows[-1]['seconds'],
        'largest_log_gap_seconds':max((datetime.fromisoformat(b['time'])-datetime.fromisoformat(a['time'])).total_seconds() for a,b in zip(rows,rows[1:]))} for stage,rows in stages.items()}
    for stage,v in times.items():assert v['last_done']==v['total'] and v['largest_log_gap_seconds']<300
    retained=sum(f.stat().st_size for f in C.OUT.rglob('*') if f.is_file());assert retained<32*2**30
    sources=[f'tests/glm5/{v}' for v in ['state_formation_core.py','state_formation_design.py','state_formation_math.py','state_formation_runner.py','state_formation_analyze.py',
        'state_formation_publish.py','state_formation_diagnostic.py','state_formation_failure_analysis.py','state_formation_delivery.py','state_formation_storage.ps1','state_formation_monitor.ps1','state_formation_campaign.ps1','test_state_formation.py','verify_state_formation.cjs']]
    sources+=['frontend/src/components/v8/StateFormationReport.jsx','frontend/src/components/v8/ResearchInterventionReport.jsx','frontend/src/components/v8/hiddenstate_components.css']
    C.save('delivery-audit.json',{'schema':'state_formation_delivery.v1','passed':True,'registration':reg,'all_objects_verified_in_both_local_libraries':reg['objects'],
        'coordinate_artifacts':len(coords),'coordinate_values_per_library':values,'scientific_unit_tests':6,'frontend_tests':56,'browser_checks':len(accept['checks']),
        'build':True,'component_lint':True,'visual_review':'separately inspected desktop and mobile screenshots',
        'source_hashes':{name:C.sha(C.ROOT/name) for name in sources},
        'executed_entrypoints':['state_formation_storage.ps1; then -FinalizeVerifiedCopy with verified-copy recovery',
            '.venv/Scripts/python.exe -B -m unittest tests/glm5/test_state_formation.py',
            '.venv/Scripts/python.exe -B -u tests/glm5/state_formation_design.py',
            'state_formation_runner.py --stage pilot/discovery/calibrate (sequential); state_formation_math.py between discovery and calibrate',
            'state_formation_campaign.ps1 (confirm completed; original causal stopped at behavior gate)',
            'state_formation_diagnostic.py freeze; state_formation_diagnostic.py run',
            'state_formation_runner.py --stage capability; state_formation_analyze.py',
            'state_formation_failure_analysis.py behavior/effects',
            'state_formation_publish.py audit/publish',
            'frontend: node node_modules/vitest/vitest.mjs run',
            'frontend: node node_modules/eslint/bin/eslint.js src/components/v8/StateFormationReport.jsx',
            'frontend: node node_modules/vite/bin/vite.js build --outDir ../tests/glm5_temp/state-formation-build',
            'node tests/glm5/verify_state_formation.cjs',
            'state_formation_delivery.py; state_formation_memo.py'],
        'artifact_hashes':{name:C.sha(C.OUT/name) for name in ['plan.json','selection.json','effects-frozen.json','effects.npz','analysis-arrays.npz','report.json','audit.json','audit-recovery.json','client-report.json','client-acceptance.json','gpu-monitor.jsonl','cleanup.json']},
        'stage_durations':times,'wddm_peak_shared_bytes':shared,'wddm_peak_dedicated_bytes':dedicated,'wddm_samples':len(monitor),
        'resource_scope':'sampled WDDM allocated process memory, not measured GPU-to-host transfer volume; does not prove zero CPU spill',
        'result_retained_bytes':retained,'result_budget_bytes':32*2**30,'C_free_bytes':shutil.disk_usage(C.OUT).free,'D_free_bytes':shutil.disk_usage(C.ROOT).free,
        'cleanup':cleanup,'archived_storage_logs':archived,'remote_published':False,'formal_promotions':0,
        'original_confirmation_failed':not r['behavior']['passed'],'intervention_scope':'separately frozen post-failure diagnostic; no confirmatory use claims',
        'migrated_old_files_reverified':len(migrated['manifest']),'migrated_old_bytes_preserved':migrated['bytes'],
        'unverified':['independent node replication','new language instruction/query constructions','other languages/colors/tasks/style/grammar','single neuron or QK/VO mechanisms','fair SAE/transcoder comparison','installed desktop and offline package','remote deployment']})
    print(json.dumps({'verified_objects':reg['objects'],'coordinate_artifacts':len(coords),'values':values,'retained_GiB':retained/2**30,'peak_shared_MiB':shared/2**20}),flush=True)

if __name__=='__main__':main()
