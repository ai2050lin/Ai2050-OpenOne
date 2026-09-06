"""Completed-data UI integration preflight, not final publication or browser QA."""
import json, subprocess, sys, zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import RESULT, read, save
import server.native_source_parameter_query as source

OUT = RESULT/'phase2684_source_campaign_delivery'
SCALAR = RESULT/'phase2682_resolved_scalar_paths'


def shapes(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                assert version in ((1, 0), (2, 0)), version
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dtype = reader(stream)
                result[name.removesuffix('.npy')] = (shape, str(dtype))
    return result


def main():
    assert read(SCALAR/'analysis/final.json')['all_checks_passed'] is True
    metadata = read(OUT/'material/staged_published_source_cases.json')
    # Staged metadata only redirects the file catalog; actual measured values
    # and production query code are unchanged. Never marks live delivery true.
    source.metadata = lambda: metadata
    records = source.numeric_records()
    controls = read(SCALAR/'protocol/frozen.json')['controls']
    checks = {}; examples = []
    fresh = [r for r in metadata if r['dataset'] == 'fresh']
    checks['64_matched_actual_numeric_prefixes'] = len(fresh) == 64 and all(
        r['case_index'] in records and records[r['case_index']]['case_id'] == r['case_id']
        and records[r['case_index']]['observed_ids'] == r['natural']['generated_ids'] for r in fresh)
    for r in fresh[:4]+fresh[-4:]:
        for control in (controls[0], controls[1], controls[-1]):
            obj = source.query('fresh', r['case_index'], control['layer'], control['unit'],
                               control['coordinate'], 0, len(r['token_strings'])-1, 31, 127, 1, 0)
            got = obj['numeric_scalar_validation']
            expected = [c for c in records[r['case_index']]['conditions'] if
                        (c['layer'], c['unit'], c['coordinate']) ==
                        (control['layer'], control['unit'], control['coordinate'])]
            assert got is not None and got['effects'] == expected and len(expected) >= 4
            assert got['noop_exact'] and got['all12_matrices_restored']
            assert obj['values']['embedding_coordinate'] == obj['values']['hidden_coordinate']
            key = {'gate':'actual_Wgate_jk', 'up':'actual_Wup_jk', 'down':'actual_Wdown_kj'}[control['kind']]
            assert obj['values'][key] == control['original_weight']
            examples.append({'case_id':r['case_id'], 'control':control['key']+'/'+control['control'],
                             'coordinate':control['coordinate'], 'actual_weight':obj['values'][key],
                             'actual_conditions':len(expected), 'FP64_readout_available':got['baseline_logprobs64'] is not None})
    checks['24_actual_numeric_source_queries'] = len(examples) == 24
    all_shapes = {}; full_mlp = 0
    for r in metadata:
        header = shapes(RESULT/r['native_path'])
        if 'full__x' not in header:
            continue
        assert header['full__h'][0][-1] == 2560
        for key, (shape, dtype) in header.items():
            if not key.startswith('full__') or key == 'full__h': continue
            assert len(shape) == 3 and shape[0] == 4 and shape[1] == len(r['token_strings'])
            assert shape[-1] in (2560, 9728) and dtype == 'uint16'
            all_shapes.setdefault(key, []).append(shape)
        full_mlp += 1
    checks['32_real_fulltoken_MLP_descriptor_shapes'] = full_mlp == 32 and bool(all_shapes)
    checks['no_torch_or_model_import'] = 'torch' not in sys.modules
    node = 'C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
    build = subprocess.run([node, 'node_modules/vite/bin/vite.js', 'build'], cwd=ROOT/'frontend',
                           capture_output=True, text=True, encoding='utf-8')
    checks['current_frontend_build'] = build.returncode == 0
    report = {'all_checks_passed':all(checks.values()), 'checks':checks, 'preview_only':True,
              'real_HTTP':False, 'real_browser':False, 'actual_numeric_examples':examples,
              'fulltoken_shapes':all_shapes, 'build':{'returncode':build.returncode, 'stdout':build.stdout, 'stderr':build.stderr},
              'boundary':'Real completed native/scalar records, read-only production query, staged metadata and real frontend build. Not final catalog, live HTTP, React interaction test, or browser QA. Delayed-response behavior still requires actual browser verification.'}
    save(OUT/'analysis/numeric_display_preflight.json', report)
    print(checks, flush=True)
    assert report['all_checks_passed']


if __name__ == '__main__':
    main()
