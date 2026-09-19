"""Every displayed completed-analysis cell against its actual source receipt.

No synthetic scientific outputs are inserted. Missing analyses block this test.
"""
import argparse
import sys
import time
import urllib.error
from phase2748_rdc_client_test import ROOT, get
from rdc_question_common import OUT, read, sha, stamp, immutable


def rows_for(kind, data, cohort, split):
    if kind == 'readout':
        columns = ['native_to_predicted_KL', 'native_top1_agreement',
                   'predicted_first_teacher_NLL', 'native_first_teacher_NLL']
        return [[r['variant']]+[r['summary'][cohort][k] for k in columns]
                for r in data['summaries'] if r['split'] == split and r['kind'] == 'selected']
    if kind == 'prospective':
        columns = ['complete_correct_and_stopped', 'native_complete_correct_and_stopped',
                   'strict_JSON', 'natural_EOS', 'cap_censored', 'complete_sequence_matches_native']
        return [[r['variant']]+[r['summaries'][cohort]['means'][k] for k in columns] for r in data['rules']]
    if kind == 'learning':
        columns = ['teacher_mean_NLL', 'native_teacher_mean_NLL', 'teacher_later_mean_NLL',
                   'complete_correct_and_stopped', 'native_complete_correct_and_stopped', 'within_state_change_MSE']
        return [[r['run']]+[next(x for x in r['summaries'] if x['split'] == split)['summary'][cohort][k]
                           for k in columns] for r in data['runs']]
    assert kind == 'parameter'
    return [[r['run'], r['FP32_update_L2'], r['BF16_update_L2'], str(r['FP32_changed_words']),
             str(r['BF16_changed_words']), r['actual_FP32_to_actual_BF16_rounding_L2']] for r in data['summaries']]


def check_table(page, table_index, raw):
    # Same display precision, but an independently explicit mapping of each
    # source metric to each table column. Never writes to the authored page.
    expected = page.evaluate("rows=>rows.map(row=>row.map(v=>typeof v==='string'?v:v==null?'未定义':Number(v).toPrecision(6)))", raw)
    page.wait_for_function("""({index,expected})=>{
      const table=document.querySelectorAll('#natural-complete-analyses table')[index];
      if(!table)return false;
      const actual=Array.from(table.querySelectorAll('tbody tr'),r=>Array.from(r.querySelectorAll('td'),c=>c.textContent.trim()));
      return JSON.stringify(actual)===JSON.stringify(expected);
    }""", arg={'index': table_index, 'expected': expected}, timeout=60000)
    return sum(len(r) for r in raw)


def main(models, confirmation=False, requested_kinds=None):
    start = time.monotonic(); overview = get('/overview')
    assert not confirmation or overview['confirmation_open']
    requested_kinds = requested_kinds or ['readout', 'prospective', 'learning', 'parameter']
    responses = {}; checks = []; images = []; errors = []
    scopes = ['nonconfirmation', 'confirmation'] if confirmation else ['nonconfirmation']
    for model in models:
        for scope in scopes:
            kinds = ['readout', 'prospective']+(['learning'] if model == 'qwen4' else [])
            if model == 'qwen4' and scope == 'nonconfirmation': kinds.append('parameter')
            kinds = [kind for kind in kinds if kind in requested_kinds]
            for kind in kinds:
                path = {'readout': OUT/'readout_analysis'/model/scope/'result.json',
                        'prospective': OUT/'prospective_analysis'/model/('confirmation' if scope == 'confirmation' else 'diagnostic')/'result.json',
                        'learning': OUT/'learning_analysis'/scope/'result.json',
                        'parameter': OUT/'parameter_formation/result.json'}[kind]
                source = read(path); assert source['all_passed']
                response = get('/analysis', kind=kind, model=model, scope=scope)
                assert response['complete'] and response['result'] == source
                assert response['receipt']['sha256'] == sha(path)
                responses[(model, kind, scope)] = response
    assert responses, 'No registered model/kind/scope combination requested'
    folder = OUT/'client'/('numeric_analysis_'+str(time.time_ns())); folder.mkdir(parents=True)
    sys.path.insert(0, str(ROOT/'tests/glm5_temp/rdc_update_browser_packages'))
    from playwright.sync_api import sync_playwright, expect
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True, chromium_sandbox=True)
        context = browser.new_context(viewport={'width': 1440, 'height': 1080}); page = context.new_page()
        page.on('pageerror', lambda e: errors.append(str(e)))
        try:
            page.goto('http://127.0.0.1:5173/rdc-construction', wait_until='networkidle', timeout=60000)
            page.add_style_tag(content='html,body,main{scroll-behavior:auto !important}')
            study = page.locator('#construction-natural-questions'); panel = page.locator('#natural-complete-analyses')
            for (model, kind, scope), response in responses.items():
                study.get_by_label('自然研究模型', exact=True).select_option(model)
                panel.get_by_label('完整证据类型', exact=True).select_option(kind)
                panel.get_by_label('完整证据范围', exact=True).select_option(scope)
                expect(panel.locator('pre')).to_contain_text(response['receipt']['sha256'], timeout=60000)
                data = response['result']; cohorts = ['equal_cohort', 'drop', 'quoref'] if kind != 'parameter' else ['not_applicable']
                splits = ['diagnostic', 'validation'] if scope == 'nonconfirmation' and kind in ['readout', 'learning'] else [scope]
                for cohort in cohorts:
                    if kind != 'parameter': panel.get_by_label('完整证据语料', exact=True).select_option(cohort)
                    for split in splits:
                        if scope == 'nonconfirmation' and kind in ['readout', 'learning']:
                            panel.get_by_label('完整证据划分', exact=True).select_option(split)
                        raw = rows_for(kind, data, cohort, split)
                        assert len(raw) == {'readout': 9, 'prospective': 2, 'learning': 6, 'parameter': 6}[kind]
                        cells = check_table(page, 0, raw)
                        checks.append({'model': model, 'kind': kind, 'scope': scope, 'cohort': cohort,
                                       'split': split, 'actual_source_cell_count': cells, 'source_sha256': response['receipt']['sha256']})
                if kind == 'parameter':
                    for index, precision in enumerate(['FP32', 'BF16'], 1):
                        raw = [[str(i+1)+' · '+data['runs'][i]]+r for i, r in enumerate(data[precision+'_direction_cosines'])]
                        cells = check_table(page, index, raw)
                        checks.append({'model': model, 'kind': kind, 'precision': precision, 'scope': scope,
                                       'actual_source_cell_count': cells, 'source_sha256': response['receipt']['sha256']})
                else:
                    panel.get_by_label('完整证据语料', exact=True).select_option('equal_cohort')
                    if scope == 'nonconfirmation' and kind in ['readout', 'learning']:
                        panel.get_by_label('完整证据划分', exact=True).select_option('diagnostic')
                    check_table(page, 0, rows_for(kind, data, 'equal_cohort', 'diagnostic' if scope == 'nonconfirmation' else scope))
                if model == 'qwen4' or kind == 'readout':
                    panel.locator('h3').evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
                    path = folder/(model+'_'+kind+'_'+scope+'.png'); page.screenshot(path=str(path)); images.append(path)
                if kind == 'parameter':
                    for precision in ['FP32', 'BF16']:
                        toggle = panel.get_by_text(precision+' · 六运行全部参数更新方向余弦', exact=True)
                        toggle.click(); toggle.evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
                        path = folder/('all_parameter_'+precision+'_cosines.png'); page.screenshot(path=str(path)); images.append(path)
                print('NATURAL_NUMERIC_ANALYSIS_UI', model, kind, scope, flush=True)
            # Local training is absent from the two untrained original models.
            study.get_by_label('自然研究模型', exact=True).select_option('qwen14')
            panel.get_by_label('完整证据类型', exact=True).select_option('learning')
            expect(panel.get_by_role('status')).to_contain_text('当前模型没有这项训练证据')
            expect(panel.locator('table')).to_have_count(0)
            study.get_by_label('自然研究模型', exact=True).select_option('qwen4')
            panel.get_by_label('完整证据类型', exact=True).select_option('parameter')
            panel.get_by_label('完整证据范围', exact=True).select_option('confirmation')
            expect(panel.get_by_role('status')).to_contain_text('参数终点不按确认材料重复计数')
            expect(panel.locator('table')).to_have_count(0)
            mobile_key = next((k for k in responses if k[0] == 'qwen4' and k[2] == 'nonconfirmation'), None)
            if mobile_key:
                mobile_kind = mobile_key[1]; mobile = responses[mobile_key]
                panel.get_by_label('完整证据范围', exact=True).select_option('nonconfirmation')
                panel.get_by_label('完整证据类型', exact=True).select_option(mobile_kind)
                if mobile_kind in ['readout', 'learning']: panel.get_by_label('完整证据划分', exact=True).select_option('diagnostic')
                if mobile_kind != 'parameter': panel.get_by_label('完整证据语料', exact=True).select_option('equal_cohort')
                expect(panel.locator('pre')).to_contain_text(mobile['receipt']['sha256'], timeout=60000)
                check_table(page, 0, rows_for(mobile_kind, mobile['result'], 'equal_cohort', 'diagnostic'))
                page.set_viewport_size({'width': 390, 'height': 844})
                panel.locator('h3').evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
                path = folder/'numeric_analysis_mobile.png'; page.screenshot(path=str(path)); images.append(path)
                assert study.evaluate('(e)=>e.scrollWidth<=e.clientWidth+1')
            assert not errors, errors
        finally: context.close(); browser.close()
    value = {'timestamp': stamp(), 'all_passed': True, 'models': models, 'confirmation_included': confirmation,
             'requested_kinds': requested_kinds,
             'complete_analysis_receipts': len(responses), 'checked_tables': checks,
             'displayed_cells_compared_to_actual_sources': sum(r['actual_source_cell_count'] for r in checks),
             'all_data_from_real_complete_results': True, 'screenshots': [p.relative_to(ROOT).as_posix() for p in images],
             'visual_QA_pending': True, 'seconds': time.monotonic()-start,
             'source_sha256': sha(__file__), 'UI_sources': {name: sha(ROOT/name) for name in [
                 'server/rdc_question_service.py', 'frontend/src/components/app/RdcNaturalAnalyses.jsx',
                 'frontend/src/components/app/RdcNaturalQuestions.jsx']},
             'limits': 'Exact actual-API receipt equality and displayed metric-to-cell mapping at six significant digits. This checks software presentation, not a new scientific result or replication of GPU generation.'}
    immutable(folder/'result.json', value)
    print('NATURAL_NUMERIC_ANALYSIS_CLIENT_PASS', str(folder), len(responses), value['displayed_cells_compared_to_actual_sources'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--models', nargs='+', choices=['qwen4', 'qwen14', 'glm4'], default=['qwen4', 'qwen14', 'glm4'])
    parser.add_argument('--kinds', nargs='+', choices=['readout', 'prospective', 'learning', 'parameter'])
    parser.add_argument('--confirmation', action='store_true'); args = parser.parse_args(); main(args.models, args.confirmation, args.kinds)
