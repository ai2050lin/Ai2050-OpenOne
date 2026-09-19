"""Complete-model native atlas/parameter API and authored-app browser audit.

Run only after the requested original model collections genuinely complete.
The old partial-state qualification remains separate and unchanged.
"""
import argparse
import hashlib
import re
import sys
import time
from itertools import permutations
from urllib.request import urlopen
from urllib.parse import urlencode
from phase2748_rdc_client_test import ROOT, API, get, equal
from server import rdc_question_service as service
from rdc_question_common import OUT, read, sha, stamp, immutable
from rdc_relation_native_parameters import parameter, decode
import numpy as np


def original_parameter_path(model, sample, block, unit, output):
    response = get('/parameter-path', model=model, question=sample['tokens']['question_id'],
                   block=block, unit=unit, output=output)
    model_name = service.MODEL_FOLDERS[model]; prefix = f'model.layers.{block}.mlp.'
    config = read(ROOT/'models/hf'/model_name/'config.json'); middle = config['intermediate_size']
    if model == 'glm4':
        joined = parameter(ROOT, prefix+'gate_up_proj.weight', model_name)
        assert joined.shape == (2*middle, config['hidden_size'])
        gate, up = decode(joined[unit]).astype(float), decode(joined[middle+unit]).astype(float)
        assert response['native_parameter_addresses'][0]['gate_row'] == unit
        assert response['native_parameter_addresses'][0]['up_row'] == middle+unit
    else:
        gate = decode(parameter(ROOT, prefix+'gate_proj.weight', model_name)[unit]).astype(float)
        up = decode(parameter(ROOT, prefix+'up_proj.weight', model_name)[unit]).astype(float)
    down = parameter(ROOT, prefix+'down_proj.weight', model_name)
    column, outrow = decode(down[:, unit]).astype(float), decode(down[output]).astype(float)
    equal(response['factors'], np.stack([gate, up, column]))
    inputs, writes = [], []
    for sibling, record in zip(sample['siblings'], response['records']):
        assert sibling['material']['question_id'] == record['question_id']
        ref = sibling['record']['field']; assert sha(ROOT/ref['path']) == ref['sha256']
        with np.load(ROOT/ref['path'], allow_pickle=False) as z:
            x = decode(z[f'block{block}_MLP_input_BF16']).astype(float)
            a = decode(z[f'block{block}_product_BF16']).astype(float)
            native = float(decode(z[f'block{block}_MLP_write_BF16'])[output])
        inputs.extend([gate*x, up*x]); writes.append(outrow*a)
        assert record['gate_FP64_sum'] == float(inputs[-2].sum())
        assert record['up_FP64_sum'] == float(inputs[-1].sum())
        assert record['all_unit_write_FP64_sum'] == float(writes[-1].sum())
        assert record['native_MLP_write_BF16_at_output'] == native
        assert record['selected_unit_write_term_FP64'] == float(writes[-1][unit])
        assert record['write_FP64_minus_native_BF16'] == float(writes[-1].sum()-native)
    equal(response['input_terms'], np.stack(inputs)); equal(response['write_terms'], np.stack(writes))
    return {'model': model, 'question_id': sample['tokens']['question_id'], 'block': block,
            'unit': unit, 'output': output, 'input_scalars': 8*config['hidden_size'],
            'write_scalars': 4*middle, 'source_SHA256': [r['source_field_sha256'] for r in response['records']]}


def geometry(model):
    result = read(OUT/'relation_geometry'/model/'result.json'); assert result['all_passed']
    ref = result['field']; assert sha(ROOT/ref['path']) == ref['sha256']
    with np.load(ROOT/ref['path'], allow_pickle=False) as z: arrays = {k: z[k].copy() for k in z.files}
    config = read(ROOT/'models/hf'/service.MODEL_FOLDERS[model]/'config.json')
    for split in ['train', 'validation', 'diagnostic']:
        for cohort in ['drop', 'quoref']:
            for metric in ['similarity', 'permutation_mean', 'excess']:
                response = get('/relation-geometry', model=model, split=split, cohort=cohort, metric=metric)
                assert response['labels'] == result['labels']
                counts = arrays[f'{split}__{cohort}__valid_contexts']; values = arrays[f'{split}__{cohort}__{metric}_mean']
                assert response['valid_context_counts'] == counts.tolist()
                for i, row in enumerate(response['values']):
                    for j, v in enumerate(row): assert (v is None) if counts[i, j] == 0 else (v == values[i, j])
            n = next(i for i, r in enumerate(result['identities']) if r['split'] == split and r['cohort'] == cohort)
            ident = result['identities'][n]
            group = read(OUT/'native'/model/'nonconfirmation/groups'/(ident['group_id']+'.json'))
            qq = {q['question_id']: q for q in group['questions']}; samples = []
            for qid in ident['question_ids']:
                reference = qq[qid]['field']; assert sha(ROOT/reference['path']) == reference['sha256']
                with np.load(ROOT/reference['path'], allow_pickle=False) as z:
                    samples.append({k: decode(z[k]).astype(float) for k in ['hidden_BF16', 'postnorm_BF16']})
            for label in ['H12', 'H24', 'H32', 'H'+str(config['num_hidden_layers'])]:
                a = np.stack([s['hidden_BF16'][int(label[1:])] for s in samples]); a -= a.mean(0)
                b = np.stack([s['postnorm_BF16'] for s in samples]); b -= b.mean(0)
                ka, kb = a@a.T/a.shape[1], b@b.T/b.shape[1]
                i, j = result['labels'].index(label), result['labels'].index('postnorm')
                assert np.array_equal(ka, arrays['full_coordinate_Gram_by_context'][n, i])
                assert np.array_equal(kb, arrays['full_coordinate_Gram_by_context'][n, j])
                denom = np.linalg.norm(ka)*np.linalg.norm(kb)
                baseline = np.mean([np.sum(ka*kb[np.ix_(p, p)])/denom for p in permutations(range(4))])
                assert abs(np.sum(ka*kb)/denom-arrays['similarity'][n, i, j]) < 1e-13
                assert abs(baseline-arrays['permutation_mean'][n, i, j]) < 1e-13
    for figure in read(OUT/'relation_geometry'/model/'display.json')['figures']:
        with urlopen(API+'/relation-figure?'+urlencode({'model': model, 'name': figure['name']}), timeout=60) as r:
            assert hashlib.sha256(r.read()).hexdigest() == figure['sha256']
    return {'model': model, 'spaces': len(result['labels']), 'full_API_matrices': 18,
            'independent_raw_field_pairs_and_all24permutation_means': 24, 'reviewed_figure_bytes': 2}


def main(models, confirmation=False, api_only=False):
    start = time.monotonic(); scope = 'confirmation' if confirmation else 'nonconfirmation'
    summary = get('/overview'); assert not confirmation or summary['confirmation_open']
    folder = OUT/'client'/('multimodel_'+scope+'_'+str(time.time_ns()))
    selected = {}; checks = []; parameter_checks = []; full_examples = {}; geometry_checks = []
    for model in models:
        original = read(OUT/'native'/model/scope/'result.json'); assert original['all_passed']
        config = read(ROOT/'models/hf'/service.MODEL_FOLDERS[model]/'config.json')
        split = 'confirmation' if confirmation else 'diagnostic'
        rows = get('/samples', model=model, split=split)
        assert len(rows) == (256 if confirmation else 384) and all(r['captured'] for r in rows)
        qid = rows[0]['question_id']; selected[model] = qid
        sample = get('/sample', model=model, question=qid); q = sample['record']
        assert len(sample['siblings']) == 4 and sample['tokens']['question_id'] == qid
        expected = np.concatenate([service.array(q['field'], 'hidden_BF16'), service.array(q['field'], 'postnorm_BF16')[None]])
        equal(get('/field', model=model, question=qid), expected)
        blocks = sorted(int(re.fullmatch(r'block(\d+)_MLP_input_BF16', name).group(1))
                        for name in q['field']['arrays'] if re.fullmatch(r'block(\d+)_MLP_input_BF16', name))
        declared = read(OUT/'effective_experiment_contract.json')['capture']['selected_MLP_blocks'][model]
        assert blocks == sorted(declared)
        for block in blocks:
            equal(get('/field', model=model, question=qid, mode='MLP', block=block),
                  np.stack([service.array(q['field'], f'block{block}_{k}_BF16') for k in ['gate', 'up', 'product']]))
            for unit, output in [(0, 0), (config['intermediate_size']-1, config['hidden_size']-1)]:
                parameter_checks.append(original_parameter_path(model, sample, block, unit, output))
        equal(get('/field', model=model, question=qid, mode='attention'), service.array(q['field'], 'block12_attention_BF16'))
        source = service.array(sample['context_field'], 'source_H12_BF16')
        for begin in [0, len(source)-1]:
            equal(get('/field', model=model, question=qid, mode='source_H12', start=begin), source[begin:begin+64])
        for mode, name in [('history_postnorm', 'postnorm_BF16'), ('history_H12', 'H12_last_BF16'), ('history_read', 'native_source_read_BF16')]:
            equal(get('/field', model=model, question=qid, mode=mode), service.array(q['history']['field'], name))
        equal(get('/field', model=model, question=qid, mode='teacher_postnorm'), service.array(q['teacher']['field'], 'postnorm_BF16'))
        full_ids = set(read(OUT/'effective_experiment_contract.json')['capture']['full_history_context_ids'])
        full_id = next(r['question_id'] for r in rows if r['group_id'] in full_ids)
        full_q = get('/sample', model=model, question=full_id)['record']; h = full_q['history']
        assert h['full_H_all_layers_every_generated_step']
        for step in [0, len(h['generated_ids'])-1]:
            equal(get('/field', model=model, question=full_id, mode='history_all_hidden', step=step),
                  service.array(h['field'], 'all_hidden_BF16')[step])
        full_examples[model] = {'question_id': full_id, 'steps': len(h['generated_ids'])}
        if not confirmation:
            for cohort in ['drop', 'quoref']:
                for channel in ['hidden', 'MLP']:
                    for statistic in ['within_RMS', 'train_standardized_RMS', 'mean']:
                        response = get('/atlas', model=model, cohort=cohort, channel=channel, statistic=statistic)
                        equal(response, np.asarray(service.atlas(model=model, cohort=cohort, channel=channel,
                                                                 statistic=statistic, block=12)['values']))
            assert len(summary['models'][model]['figures']) == 2
            for figure in summary['models'][model]['figures']:
                with urlopen(API+'/figure?'+urlencode({'model': model, 'name': figure['name']}), timeout=60) as r:
                    assert hashlib.sha256(r.read()).hexdigest() == figure['sha256']
            geometry_checks.append(geometry(model))
        checks.append({'model': model, 'scope': scope, 'full_sample_count': len(rows), 'all_sample_groups_captured': True,
                       'selected_MLP_blocks': blocks, 'all_native_API_axes_and_histories_exact': True})
        print('NATURAL_MULTIMODEL_API_PASS', model, scope, len(blocks), flush=True)
    images = []; errors = []; displayed_history_cells = 0
    if not api_only:
        sys.path.insert(0, str(ROOT/'tests/glm5_temp/rdc_update_browser_packages'))
        from playwright.sync_api import sync_playwright, expect
        folder.mkdir(parents=True)
        with sync_playwright() as p:
            browser = p.chromium.launch(channel='msedge', headless=True, chromium_sandbox=True)
            context = browser.new_context(viewport={'width': 1440, 'height': 1080}); page = context.new_page()
            page.on('pageerror', lambda e: errors.append(str(e)))
            try:
                page.goto('http://127.0.0.1:5173/rdc-construction', wait_until='networkidle', timeout=60000)
                page.add_style_tag(content='html,body,main{scroll-behavior:auto !important}')
                study = page.locator('#construction-natural-questions'); field = page.locator('#natural-question-fields')
                parameter_panel = page.locator('#natural-parameter-path')
                for model in models:
                    study.get_by_label('自然研究模型', exact=True).select_option(model)
                    if not confirmation:
                        from phase2748_rdc_source_coupling_client_test import cells
                        h = summary['models'][model]['native_history_prediction']
                        expected = []
                        for row in h['summaries']:
                            s = row['summary']['equal_cohort']
                            p = next(x for x in h['paired_comparisons'] if x['bin'] == row['bin'])['primary_minus_target_shuffle_control']['equal_cohort']
                            expected.append([row['bin'], s['prediction_MSE'], s['training_mean_MSE'], s['native_norm_squared_mean'], p['right_mean'], s['query_standardized_RMS']])
                        displayed_history_cells += cells(page, 'natural-history-comparison', expected)
                    field.get_by_label('自然材料划分', exact=True).select_option('confirmation' if confirmation else 'diagnostic')
                    expect(field.get_by_label('自然问题', exact=True).locator('option')).to_have_count(256 if confirmation else 384, timeout=60000)
                    field.get_by_label('自然问题', exact=True).select_option(selected[model])
                    field.get_by_label('自然场类型', exact=True).select_option('first_hidden')
                    field.get_by_role('button', name='读取自然问题与完整坐标', exact=True).click()
                    config = read(ROOT/'models/hf'/service.MODEL_FOLDERS[model]/'config.json')
                    expect(field.locator('canvas')).to_have_attribute('width', str(config['hidden_size']), timeout=60000)
                    expect(field.locator('canvas')).to_have_attribute('height', str(config['num_hidden_layers']+2))
                    field.locator('canvas').evaluate('(e)=>e.scrollIntoView({block:"center",behavior:"instant"})')
                    path = folder/(model+'_first_all_H.png'); page.screenshot(path=str(path)); images.append(path)
                    parameter_panel.get_by_role('button', name='读取自然问题的完整参数路径', exact=True).click()
                    expect(parameter_panel.locator('canvas')).to_have_count(2, timeout=60000)
                    expect(parameter_panel.locator('canvas').nth(0)).to_have_attribute('width', str(config['hidden_size']))
                    expect(parameter_panel.locator('canvas').nth(1)).to_have_attribute('width', str(config['intermediate_size']))
                    parameter_panel.locator('h4').first.evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
                    path = folder/(model+'_full_parameter_path.png'); page.screenshot(path=str(path)); images.append(path)
                page.set_viewport_size({'width': 390, 'height': 844})
                parameter_panel.locator('h4').first.evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
                path = folder/'multimodel_mobile.png'; page.screenshot(path=str(path)); images.append(path)
                assert study.evaluate('(e)=>e.scrollWidth<=e.clientWidth+1') and not errors, errors
            finally: context.close(); browser.close()
    value = {'timestamp': stamp(), 'all_passed': True, 'scope': scope, 'models': models, 'checks': checks,
             'parameter_settings': parameter_checks, 'relation_geometry_checks': geometry_checks,
             'full_history_examples': full_examples, 'selected': selected, 'browser_executed': not api_only,
             'displayed_native_history_cells_compared': displayed_history_cells,
             'screenshots': [str(p.relative_to(ROOT).as_posix()) for p in images], 'visual_QA_pending': bool(images),
             'source_sha256': sha(__file__), 'seconds': time.monotonic()-start,
             'limits': 'Actual complete requested models only. Native full-coordinate arrays, complete registered MLP boundary indices and independent scalar contractions; not semantic-unit identification or new GPU arithmetic qualification.'}
    immutable(folder/'result.json', value)
    print('NATURAL_MULTIMODEL_CLIENT_PASS', str(folder), len(parameter_checks), round(value['seconds'], 2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--models', nargs='+', choices=['qwen4', 'qwen14', 'glm4'], default=['qwen4', 'qwen14', 'glm4'])
    parser.add_argument('--confirmation', action='store_true'); parser.add_argument('--api-only', action='store_true')
    args = parser.parse_args(); main(args.models, args.confirmation, args.api_only)
