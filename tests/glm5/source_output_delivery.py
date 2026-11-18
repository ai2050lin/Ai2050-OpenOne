"""Verify the client bundle against full native coordinates and record resources."""
import base64,hashlib,json,re,shutil,tarfile
from datetime import datetime
import numpy as np
import source_output_core as S
from source_output_core import C
from source_output_execute import resources
from source_output_runner import check_analysis
from deploy.research_library import Library,canonical

def main():
    p=C.checked();check_analysis();r=C.read(C.OUT/'report.json');reg=C.read(C.OUT/'registration.json');audit=C.read(C.OUT/'audit.json');browser=C.read(C.OUT/'client-acceptance.json')
    assert audit['passed'] and browser['passed'] and not browser['errors'] and not browser['gpuRequests']
    assert 'Ran 6 tests' in (C.OUT/'unit-tests-final.log').read_text(encoding='utf-8-sig')
    frontend_tests=int(re.search(r'Tests\s+(\d+) passed',(C.OUT/'frontend-tests-final.log').read_text(encoding='utf-8-sig')).group(1));assert frontend_tests>=60
    assert 'built in' in (C.OUT/'build.log').read_text(encoding='utf-8-sig')
    lint=(C.OUT/'component-lint-final.log').read_text(encoding='utf-8-sig');assert ' error ' not in lint and 'Parsing error' not in lint
    libs=[Library(path) for path in reg['local_libraries']];coords=0;values=0;cal=np.load(C.OUT/'calibration-coordinates.npz');conf=np.load(C.OUT/'confirm-coordinates.npz')
    assert C.sha(reg['bundle'])==reg['bundle_sha256']
    with tarfile.open(reg['bundle'],'r:gz') as tar:
        manifest=json.load(tar.extractfile('MANIFEST.json'));assert len(manifest['objects'])==reg['objects']
        for item in manifest['objects']:
            digest=item['sha256'];raw=tar.extractfile(f'objects/{digest}.json').read();assert hashlib.sha256(raw).hexdigest()==digest;doc=json.loads(raw)
            assert not C.provenance.check_provenance(doc)
            for lib in libs:assert canonical(lib.artifact(digest))==raw
            if doc.get('schema')=='source_output_coordinates.v1':
                key=doc['branch'].replace(':','_')+'__';field=doc['field'];index=doc['index']
                actual=np.frombuffer(base64.b64decode(doc['array']['data']),'<f4').reshape(doc['array']['shape'])
                if field=='hidden':expected=np.stack([np.stack([z[key+'hidden_'+factor][index] for factor in ['source','output']]) for z in [cal,conf]])
                elif field=='units':expected=np.stack([np.stack([z[key+'unit_'+factor] for factor in ['source','output']]) for z in [cal,conf]])
                else:expected=np.stack([z[key+'head_route'][None] for z in [cal,conf]])
                np.testing.assert_array_equal(actual,expected);coords+=1;values+=actual.size
    assert coords==reg['coordinate_artifacts']==39*len(r['active_branches'])
    monitor=[json.loads(s) for s in (C.OUT/'gpu-monitor.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    active=[v for v in monitor if v['gpu_process_memory']];assert active,'A live WDDM sample is required'
    shared=max(sum(x['SharedUsage'] for x in v['gpu_process_memory']) for v in active);dedicated=max(sum(x['DedicatedUsage'] for x in v['gpu_process_memory']) for v in active)
    progress=[json.loads(s) for s in (C.OUT/'progress.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()];durations={}
    for stage in ['behavior_calibration','native_calibration','behavior_confirm','native_confirm','causal']:
        rows=[x for x in progress if x['stage']==stage]
        if not rows:continue
        assert rows[-1]['done']==rows[-1]['total'],stage
        gap=max([(datetime.fromisoformat(b['time'])-datetime.fromisoformat(a['time'])).total_seconds() for a,b in zip(rows,rows[1:])]+[0]);assert gap<300,(stage,gap)
        durations[stage]={'seconds':rows[-1]['seconds'],'max_log_gap_seconds':gap,'done':rows[-1]['done']}
    resource=resources();retained=sum(x.stat().st_size for x in C.OUT.rglob('*') if x.is_file());assert retained<resource['budget_GiB']*2**30
    C.save('delivery-audit.json',{'schema':'source_output_delivery.v1','passed':True,'registration':reg,'engineering_tests':6,'frontend_tests':frontend_tests,'browser_checks':len(browser['checks']),
        'native_inputs':audit['native_inputs'],'causal_inputs':audit['causal_inputs'],'full_coordinate_artifacts':coords,'coordinate_values_per_library':values,
        'objects_verified_in_two_libraries':reg['objects'],'all_whole_branch_failures_retained':True,'old_candidates_unchanged':True,'formal_promotions':0,'remote_published':False,
        'wddm_peak_shared_bytes':shared,'wddm_peak_dedicated_bytes':dedicated,'wddm_samples':len(monitor),'resource_scope':'WDDM allocated bytes, not actual transfer traffic',
        'durations':durations,'retained_bytes':retained,'C_free_bytes':shutil.disk_usage(C.OUT).free,'D_free_bytes':shutil.disk_usage(C.ROOT).free,
        'storage_cleanup':C.read(C.OUT/'storage-cleanup-summary.json'),'execution_resume':C.read(C.OUT/'execution-resume.json'),
        'active_resource_policy':resource,'stop_registration_recovery':C.read(C.OUT/'stop-registration-recovery.json'),
        'resource_execution_deviation':C.read(C.OUT/'resource-deviation.json'),
        'final_cleanup':C.read(C.OUT/'storage-cleanup-final.json'),
        'visual_review':'Desktop and mobile screenshots independently inspected after real browser acceptance.',
        'source_hashes':{str(x.relative_to(C.ROOT)):C.sha(x) for x in [*sorted((C.ROOT/'tests/glm5').glob('source_output*.py')),
            C.ROOT/'tests/glm5/test_source_output.py',C.ROOT/'tests/glm5/verify_source_output.cjs',C.ROOT/'frontend/src/components/v8/SourceOutputReport.jsx',C.ROOT/'frontend/src/components/v8/ResearchInterventionReport.jsx',C.ROOT/'frontend/src/components/v8/record_selection.css']},
        'artifact_hashes':{name:C.sha(C.OUT/name) for name in ['plan.json','analysis-plan.json','resource-amendment.json','resource-amendment-behavior-only.json','stop-registration-recovery.json','secondary-power-freeze.json','predictor-freeze.json','observations.json','effects.json','failure-diagnostics.json','report.json','audit.json','registration.json','client-acceptance.json']},
        'unverified':['independent node or model replication','complete source-to-answer causal mediation','general natural generation or broad capability preservation','SAE/transcoder comparison','remote deployment and installed desktop package']})
    print(json.dumps({'coordinate_values':values,'objects':reg['objects'],'retained_GiB':retained/2**30}),flush=True)

if __name__=='__main__':main()
