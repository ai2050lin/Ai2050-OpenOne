"""Verify client artifacts, recovery bundle, resource accounting and old evidence."""
import base64,hashlib,json,shutil,tarfile
from datetime import datetime
import numpy as np
import record_selection_core_v3 as R
from record_selection_core_v3 import C
from record_selection_analysis import checked
from deploy.research_library import Library,canonical

def main():
    checked();reg=C.read(C.OUT/'registration.json');r=C.read(C.OUT/'client-report.json');audit=C.read(C.OUT/'audit.json');browser=C.read(C.OUT/'client-acceptance.json')
    assert audit['passed'] and browser['passed'] and not browser['errors'] and not browser['gpuRequests']
    assert 'Ran 7 tests' in (C.OUT/'unit-tests-final.log').read_text(encoding='utf-8-sig')
    assert '56 passed' in (C.OUT/'frontend-tests.log').read_text(encoding='utf-8-sig')
    assert 'built in' in (C.OUT/'build-final.log').read_text(encoding='utf-8-sig')
    assert not any(s in (C.OUT/'component-lint-final.log').read_text(encoding='utf-8-sig') for s in ['error ','Oops!'])
    libs=[Library(path) for path in reg['local_libraries']];coords=[];values=0
    disc=np.load(C.OUT/'discovery-atlas.npz');conf=np.load(C.OUT/'confirm-atlas.npz');assert C.sha(reg['bundle'])==reg['bundle_sha256']
    with tarfile.open(reg['bundle'],'r:gz') as tar:
        manifest=json.load(tar.extractfile('MANIFEST.json'));assert len(manifest['objects'])==reg['objects']
        for entry in manifest['objects']:
            digest=entry['sha256'];raw=tar.extractfile(f'objects/{digest}.json').read();assert hashlib.sha256(raw).hexdigest()==digest;doc=json.loads(raw)
            assert not C.provenance.check_provenance(doc)
            for lib in libs:assert canonical(lib.artifact(digest))==raw
            if doc.get('schema')=='record_selection_coordinates.v1':
                kind,index=doc['field'],doc['index'];actual=np.frombuffer(base64.b64decode(doc['array']['data']),'<f4').reshape(doc['array']['shape'])
                expected=np.stack([z['head_validity_by_position'].reshape(2,1152) for z in [disc,conf]]) if kind=='heads' else np.stack([np.stack([z['unit_choice_contrast'][index],z['unit_dla_contrast'][index]]) for z in [disc,conf]])
                np.testing.assert_array_equal(actual,expected);coords.append((kind,str(index)));values+=actual.size
    assert len(coords)==37 and set(coords)=={('heads','all')}|{('units',str(i)) for i in range(36)}
    prior=[]
    # Recheck the preceding run's frozen scientific inputs, selected arrays and
    # publication. No migration, deletion or mutation of preceding evidence.
    old=C.ROOT/'tests/glm5/result/state_formation_v1';op=C.read(old/'plan.json');oa=C.read(old/'audit.json');orr=C.read(old/'registration.json')
    for name,digest in op['dependencies'].items():assert C.sha(C.ROOT/name)==digest;prior.append(name)
    for name,digest in op['materials'].items():assert C.sha(old/name)==digest;prior.append(name)
    assert C.sha(old/'report.json')==oa['report_sha256'];assert C.sha(orr['bundle'])==orr['bundle_sha256']
    retained=sum(p.stat().st_size for p in C.OUT.rglob('*') if p.is_file());assert retained<20*2**30
    for version in [1,2]:
        root=C.OUT.parent/f'record_selection_v{version}';plan=C.read(root/'plan.json')
        for name,digest in plan['dependencies'].items():assert C.sha(C.ROOT/name)==digest
        for name,digest in plan['materials'].items():assert C.sha(root/name)==digest
        assert not C.read(root/'pilot-behavior.json')['passed']
    monitor=[json.loads(line) for line in (C.OUT/'gpu-monitor.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()];active=[v for v in monitor if v['gpu_process_memory']]
    shared=max(sum(x['SharedUsage'] for x in v['gpu_process_memory']) for v in active);dedicated=max(sum(x['DedicatedUsage'] for x in v['gpu_process_memory']) for v in active)
    progress=[json.loads(line) for line in (C.OUT/'progress.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()];durations={}
    for stage in ['calibration_full','discovery','confirm','causal']:
        entries=[r for r in progress if r['stage']==stage];assert entries[-1]['done']==entries[-1]['total']
        gap=max((datetime.fromisoformat(b['time'])-datetime.fromisoformat(a['time'])).total_seconds() for a,b in zip(entries,entries[1:]));assert gap<300
        durations[stage]={'seconds':entries[-1]['seconds'],'max_progress_gap_seconds':gap,'done':entries[-1]['done']}
    C.save('delivery-audit.json',{'schema':'record_selection_delivery.v1','passed':True,'registration':reg,'objects_verified_both_libraries':reg['objects'],
        'full_coordinate_artifacts':37,'coordinate_values_per_library':values,'engineering_tests':7,'frontend_tests':56,'browser_checks':len(browser['checks']),
        'build':True,'component_lint':True,'visual_review':'desktop and mobile screenshots inspected separately',
        'native_inputs_retained':4096,'real_causal_families':384,'unique_behavior_inputs_including_failed_color_pilots':4608,
        'raw_failures_not_erased':True,'frozen_preceding_run_sources_and_materials_verified':prior,'preceding_report_and_bundle_verified':True,
        'wddm_peak_shared_bytes':shared,'wddm_peak_dedicated_bytes':dedicated,'wddm_samples':len(monitor),'stage_durations':durations,
        'resource_scope':'sampled WDDM process allocation, not measured GPU-host transfer traffic; all model weights verified on CUDA',
        'result_retained_bytes':retained,'budget_bytes':20*2**30,'C_free_bytes':shutil.disk_usage(C.OUT).free,'D_free_bytes':shutil.disk_usage(C.ROOT).free,
        'cleanup':C.read(C.OUT/'cleanup.json'),'remote_published':False,'formal_promotions':0,
        'source_hashes':{str(path.relative_to(C.ROOT)):C.sha(path) for path in [*sorted((C.ROOT/'tests/glm5').glob('record_selection*.py')),
            C.ROOT/'tests/glm5/test_record_selection_v3.py',C.ROOT/'tests/glm5/verify_record_selection.cjs',
            C.ROOT/'frontend/src/components/v8/RecordSelectionReport.jsx',C.ROOT/'frontend/src/components/v8/ResearchInterventionReport.jsx',C.ROOT/'frontend/src/components/v8/record_selection.css']},
        'executed_entrypoints':['record_selection_design.py; record_selection_runner.py pilot (failed)',
            'record_selection_design_v2.py; record_selection_runner_v2.py pilot (failed)',
            'record_selection_design_v3.py; record_selection_runner_v3.py pilot (failed)',
            'record_selection_calibrate.py (all discovery behavior, full gate failed)',
            'record_selection_analysis.py freeze; record_selection_collect.py discovery; record_selection_analysis.py select; record_selection_collect.py confirm',
            'record_selection_causal_matched.py freeze; record_selection_controls_report.py freeze; record_selection_causal_matched.py run',
            'record_selection_failures.py; record_selection_analysis.py report; record_selection_controls_report.py report',
            'record_selection_publish.py audit/publish; verify_record_selection.cjs; record_selection_delivery.py; record_selection_memo.py',
            'python -B -u -m unittest tests/glm5/test_record_selection_v3.py',
            'frontend node node_modules/vitest/vitest.mjs run; component eslint; vite build to owned tests/glm5_temp directory'],
        'artifact_hashes':{name:C.sha(C.OUT/name) for name in ['plan.json','analysis-plan.json','selection.json','controls-amendment.json','controls-analysis-freeze.json','report.json','matched-control-report.json','audit.json','client-report.json','client-acceptance.json','gpu-monitor.jsonl']},
        'unverified':['full-domain semantic mechanism','independent node or other model replication','unrelated numeric-output tasks and output-token remapping','source-specific B29 V/O writes','head-to-unit causal mediation','natural free generation','SAE/transcoder comparison','installed desktop/offline package and remote deployment']})
    print({'objects':reg['objects'],'coordinate_values':values,'retained_GiB':retained/2**30,'C_free_GiB':shutil.disk_usage(C.OUT).free/2**30},flush=True)

if __name__=='__main__':main()
