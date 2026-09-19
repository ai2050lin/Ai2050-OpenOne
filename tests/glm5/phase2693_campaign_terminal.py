"""Read-only campaign verification and strictly gated terminal accounting.

No model is loaded, no field is deleted, and preview evidence cannot complete a
phase. Browser observations must come from actual browser work, not this script.
"""
import argparse
import math
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from phase2620_native_coordinate_contract import *
from phase2693_qkv_campaign_delivery import OUT, FIELD, FRESH, TERMS, SCALAR, LINK, CROSS

sys.path.insert(0, str(ROOT))
FLOOR = 8 * 1024**3
KEYS = ('qwen14', 'glm4', 'ds7', 'ds7_answer')
PHASES = {
    2685: 'phase2685_native_attention_contract',
    2686: 'phase2686_independent_role_contract',
    2687: FIELD.name, 2688: TERMS.name, 2689: SCALAR.name,
    2690: FRESH.name, 2691: CROSS.name, 2692: LINK.name,
}
CODE = [ROOT / p for p in (
    'server/native_atlas_heatmap_query.py', 'server/native_qkv_parameter_query.py',
    'server/research_asset_service.py',
    'frontend/src/components/app/NativeQKVParameterInspector.jsx',
    'frontend/src/components/app/NativeParameterInspector.jsx',
    'tests/glm5/phase2693_qkv_campaign_delivery.py',
)]
LEGACY = (
    RESULT / 'phase2669_symmetric_multitoken_delivery',
    RESULT / 'phase2676_native_mlp_delivery',
    RESULT / 'phase2684_source_campaign_delivery',
)
LEGACY_ASSET = RESULT / 'client_visualization_assets/research_kernel/c42641_output_conditioned_crossmodel_field.json'
LEGACY_ASSET_SHA = '4e15b56f30a89f5f523ddea4b35ab46394f2a9a9015ac565b1672b25142207cb'


def checked_path(path):
    p = Path(path).resolve()
    assert p.is_relative_to(RESULT.resolve()) and p.is_file(), str(p)
    return p


def catalog(preview=False):
    path = OUT / ('material/staged_client_panel_catalog.json' if preview else 'material/client_panel_catalog.json')
    obj = read(path)
    assert obj['preview_only'] is preview and obj['phase'] == 2693
    return path, obj


def binding(preview=False):
    path, _ = catalog(preview)
    return {'catalog_sha256': sha(path), 'preview_only': preview,
            'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in CODE},
            'checker_sha256': sha(Path(__file__))}


def checked_report(name):
    obj = read(OUT / f'analysis/{name}.json')
    assert obj['all_checks_passed'] and obj.get('preview_only') is False, name
    assert obj['catalog_sha256'] == binding()['catalog_sha256'], (name, 'stale catalog')
    for path, digest in obj['source_sha256'].items():
        assert sha(ROOT / path) == digest, (name, path)
    if 'checker_sha256' in obj:
        assert obj['checker_sha256'] == sha(Path(__file__)), (name, 'checker changed')
    test_sources = {'panel_direct_audit': 'phase2693_direct_preview_audit.py',
                    'parameter_direct_audit': 'phase2693_direct_preview_audit.py',
                    'checkpoint_byte_audit': 'phase2693_checkpoint_byte_audit.py'}
    if name in test_sources:
        assert obj['code_sha256'] == sha(ROOT / 'tests/glm5_temp' / test_sources[name])
    return obj


def phase_records(require_terminal=False):
    text = MEMO.read_text(encoding='utf-8-sig')
    ids = [int(v) for v in re.findall(r'^## Phase (\d+):', text, re.M)]
    if require_terminal:
        assert ids[-1] == 2692 and 2693 not in ids
        assert not (OUT / 'analysis/final.json').exists(), 'Do not overwrite an existing final'
    result = {}
    for phase, name in PHASES.items():
        path = RESULT / name / 'analysis/final.json'
        value = read(path)
        assert value['phase'] == phase and value['all_checks_passed'] and ids.count(phase) == 1
        assert value['language_mechanism_closed'] is False
        result[phase] = value
    return result


def numeric_audit():
    folder = LINK / 'numerical_baseline_audit'
    a = read(folder / 'result.json'); protocol = read(folder / 'protocol.json')
    assert a['all_audit_execution_checks_passed'] and a['baseline_examples'] == 32 and a['layer_cases'] == 256
    assert a['new_model_forwards'] == a['intervened_parameters'] == 0
    assert a['protocol_sha256'] == sha(folder / 'protocol.json')
    for key, name in (
        ('native_math_sha256', TESTS / 'phase2692_native_rounding_math.py'),
        ('ideal_math_sha256', TESTS / 'phase2685_native_attention_math.py'),
        ('audit_code_sha256', ROOT / 'tests/glm5_temp/phase2692_rounding_baseline_audit.py'),
    ):
        assert sha(name) == protocol[key], name
    for case in a['cases']:
        assert sha(checked_path(case['numerical_map'])) == case['numerical_map_sha256']
    assert a['fixed_point_checks']['all_checks_passed']
    assert a['fixed_point_checks']['finite_BF16_fixed_points'] == 65280
    return a


def scientific_review():
    finals = phase_records()
    independent = read(SCALAR / 'analysis/independent_condition_audit.json')
    assert independent['all_independent_checks_passed'] and independent['all24_full_matrix_hashes_restored']
    assert independent['actual_conditions'] == 24576 and independent['unique_parameter_dose_sign_edits'] == 192
    numeric = numeric_audit()
    supplement = read(LINK / 'analysis/numerical_review_append.json')
    assert supplement['actual_phase2692_complete']
    assert supplement['numerical_result_sha256'] == sha(LINK / 'numerical_baseline_audit/result.json')
    for entry in read(LINK / 'analysis/linked_manifest.json'):
        assert sha(checked_path(entry['path'])) == entry['sha256']
    assert sha(LINK / 'field/natural_full_vocabulary_readout.npz') == read(LINK / 'analysis/natural_readout.json')['artifact_sha256']
    cross = finals[2691]['summary']['models']
    assert cross['qwen14']['cases'] == 4096 and all(cross[k]['cases'] == 512 for k in KEYS[1:])
    observations = {
        'initial': finals[2687]['summary'], 'fresh': finals[2690]['summary'],
        'native_scalar': finals[2689]['summary'],
        'independent_scalar_audit': {k: independent[k] for k in (
            'actual_conditions', 'unique_parameter_dose_sign_edits', 'full_vocabulary_zero_effects',
            'next_token_argmax_changes', 'next_token_changes_by_function', 'recomputed_groups', 'limits')},
        'crossmodel': finals[2691]['summary'], 'linked': finals[2692]['summary'],
        'baseline_numerical_error': numeric['all_stage_coordinate_metrics'],
        'resource_runtime_amendment': read(CROSS / 'analysis/resource_runtime_amendment.json'),
    }
    report = {**binding(), 'all_checks_passed': True, 'phase_completed': False,
        'phase_final_sha256': {str(n): sha(RESULT / p / 'analysis/final.json') for n, p in PHASES.items()},
        'observations': observations,
        'non_closure_boundaries': [
            'Execution checks, no-ops, known arithmetic and same-prefix equality are not semantic mechanism closure.',
            '24576 finite conditions reuse128 prefixes/192 edits. 3072 natural paths reuse16 truth/v0 prefixes; weights changed but observed outputs did not.',
            'Fixed256 argmax changes are not natural-cache output changes. Same model/checkpoint is not a guarantee of bit-identical execution across shapes or device maps.',
            'Baseline rounding agreement does not validate finite changed-weight predictions. Original Q/K delta error remains a separate observed failure.',
            'Ordinary/low controls differ in selected coordinates and relative doses. Large low-weight response alone does not establish greater semantic importance.',
            'Four-function same-sign counts retain every partial/zero/opposed coordinate; no same-index semantic correspondence is assumed across models.',
            'Natural postnorm readout and fixed256 H36 are separate protocols. First-token contrast is not whole-word or sentence semantics.',
            'Observed endogenous RMS denominators make source allocations conditional accounting, not source ablation.',
        ]}
    save(OUT / 'analysis/scientific_checks.json', report)
    print('2693 actual completed scientific records reviewed; no mechanism closure claimed', flush=True)


def all_references(preview=False):
    _, current = catalog(preview)
    references = set(); old_panels = []
    for folder in LEGACY:
        obj = read(folder / 'material/client_panel_catalog.json')
        old_panels.extend(obj['panels'])
        for panel in obj['panels']:
            if panel.get('storage') == 'native_descriptor':
                references.update(checked_path(RESULT / r['file']) for r in panel['rows'])
            else:
                references.add(checked_path(folder / 'maps/client_panels' / (panel['key'] + '.npz')))
    assert len(old_panels) == 75
    current_refs = {checked_path(RESULT / b['file']) for p in current['panels'] for b in p['blocks']}
    publication = read(OUT / ('analysis/staged_publication.json' if preview else 'analysis/publication.json'))
    assert publication['preview_only'] is preview
    assert current_refs == {checked_path(r['path']) for r in publication['referenced_files']}
    references |= current_refs
    assert LEGACY_ASSET.stat().st_size == 572994561 and sha(LEGACY_ASSET) == LEGACY_ASSET_SHA
    references.add(checked_path(LEGACY_ASSET))
    return references, current_refs


def storage_inventory(preview=False):
    """Exact current raw inventory. Never delete a guessed/unregistered file."""
    if not preview:
        phase_records()
        for name in ('scientific_checks', 'panel_direct_audit', 'parameter_direct_audit', 'live_api_checks', 'browser_checks'):
            checked_report(name)
    references, current_refs = all_references(preview)
    scopes = [FIELD, FRESH]
    scopes += [CROSS / k for k in KEYS if (CROSS / k / 'analysis/completion.json').exists()]
    if not preview:
        assert len(scopes) == 6
    actual = set(); manifests = {}
    for folder in scopes:
        manifest = read(folder / 'analysis/published_manifest.json')
        for row in manifest:
            path = checked_path(row['path'])
            assert path.parent in ((folder / 'field').resolve(), (folder / 'source').resolve())
            assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
            assert path in current_refs
            manifests[path] = row['sha256']
        for sub in ('field', 'source'):
            scope = (folder / sub).resolve()
            if not scope.exists():
                continue
            for path in scope.iterdir():
                assert path.is_file() and path.suffix == '.npz', ('unclassified raw path', path)
                assert path.resolve().parent == scope
                assert re.fullmatch(r'(case|natural)_\d{4}\.npz', path.name), path
                actual.add(checked_path(path))
    assert actual == set(manifests), 'Unexpected/unregistered raw field: inspect, do not delete'
    unshown = actual - current_refs
    assert not unshown, 'Unshown raw requires a separate reviewed exact allowlist'
    before = shutil.disk_usage(RESULT).free
    assert before > FLOOR
    retained = []
    for path in sorted(references):
        retained.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    prefix = 'staged_' if preview else ''
    report = {**binding(preview), 'all_checks_passed': True, 'phase_completed': False,
        'audited_completed_raw_scopes': [str(p) for p in scopes],
        'active_model_scopes_excluded_from_preview': [k for k in KEYS if CROSS / k not in scopes],
        'native_raw_files': len(actual), 'native_raw_bytes': sum(p.stat().st_size for p in actual),
        'all_native_raw_published': True, 'remaining_unpublished_raw_files': 0,
        'deleted_files': 0, 'deleted_bytes': 0, 'deletion_performed': False,
        'retained_published_files': retained, 'all_retained': True,
        'before_free_bytes': before, 'after_free_bytes': shutil.disk_usage(RESULT).free,
        'storage_interpretation': 'Current completed protocols stream unpublished fields into full-coordinate maps. Persisted raw fields are prospectively published. No removable raw file was found; no old campaign cleanup repeated.'}
    save(OUT / f'analysis/{prefix}storage_inventory.json', report)
    print('2693 actual storage inventory', prefix, len(actual), 'raw protected;', len(retained), 'total references; deleted0', flush=True)


def http_json(base, endpoint, **query):
    parsed = urllib.parse.urlparse(base)
    assert parsed.scheme == 'http' and parsed.hostname in ('127.0.0.1', 'localhost')
    url = base.rstrip('/') + endpoint + ('?' + urllib.parse.urlencode(query) if query else '')
    with urllib.request.urlopen(url, timeout=45) as response:
        assert response.status == 200
        return json.load(response)


def live_api(base):
    """Actual HTTP, compared with independently indexed complete native rows."""
    phase_records()
    from server import native_qkv_parameter_query as qkv
    sys.path.insert(0, str(ROOT / 'tests/glm5_temp'))
    from phase2693_direct_preview_audit import expected_row
    _, cat = catalog(); remote = http_json(base, '/native-atlas-panels', compact='true')
    assert len(remote['panels']) == 75 + len(cat['panels'])
    by_key = {r['key']: r for r in remote['panels']}
    pages = []
    for panel in cat['panels']:
        assert by_key[panel['key']]['coordinate_count'] == panel['coordinate_count']
        assert by_key[panel['key']]['row_count'] == panel['row_count']
        starts = {0, panel['row_count'] - 1}
        if len(panel['blocks']) > 1:
            starts.add(max(0, panel['blocks'][0]['row_count'] - 1))
        for start in sorted(starts):
            data = http_json(base, '/native-atlas-rows', panel=panel['key'], start=start, count=2)
            for row in data['rows']:
                assert np.array_equal(row['values'], expected_row(panel, row['row_index']))
                assert len(row['values']) == panel['coordinate_count']
            pages.append({'key': panel['key'], 'start': start, 'count': len(data['rows']),
                          'columns': panel['coordinate_count'], 'last_value': data['rows'][-1]['values'][-1]})
    meta = http_json(base, '/native-qkv-cases'); assert meta == qkv.options()
    queries = []
    for row in meta['cases']:
        for i, control in enumerate(meta['controls']):
            args = {'case': row['case'], 'layer': control['layer'], 'kind': control['kind'],
                'output_row': control['output_row'], 'input_coordinate': control['input_coordinate'],
                'token': row['tokens'] - 1 if i % 2 else 0, 'query_position': i % 2,
                'source_token': row['tokens'] - 1, 'head': 31 if i % 2 else 0,
                'head_coordinate': 127, 'checkpoint': 36, 'unit': 9727, 'output_coordinate': 2559}
            actual = http_json(base, '/native-qkv-parameter', **args)
            assert actual == qkv.query(**args)
            queries.append({'case': row['case'], 'control': i, 'all_response_fields_equal': True})
    for endpoint, query in (
        ('/native-atlas-rows', {'panel': cat['panels'][0]['key'], 'start': -1}),
        ('/native-qkv-parameter', {'input_coordinate': 2560}),
        ('/native-qkv-parameter', {'unit': 9728}),
        ('/native-qkv-parameter', {'kind': 'invalid'}),
    ):
        try:
            http_json(base, endpoint, **query)
        except urllib.error.HTTPError as exc:
            assert exc.code in (400, 404)
        else:
            raise AssertionError('Invalid physical address accepted over HTTP')
    assert len(queries) == 768
    save(OUT / 'analysis/live_api_checks.json', {**binding(), 'all_checks_passed': True,
        'actual_HTTP': True, 'base_url': base, 'full_native_pages': pages, 'parameter_queries': queries,
        'real_browser_verified': False, 'phase_completed': False})
    print('2693 live HTTP passed', len(pages), 'full-column pages and768 parameter requests', flush=True)


def frontend_build():
    phase_records()
    node = Path('C:/Users/Admin/.workbuddy/binaries/node/versions/22.21.0/node.exe')
    assert node.is_file()
    command = [str(node), 'node_modules/vite/bin/vite.js', 'build']
    logpath = OUT / 'runtime/frontend_build.log'; logpath.parent.mkdir(parents=True, exist_ok=True)
    before = binding()
    with logpath.open('w', encoding='utf-8') as log:
        process = subprocess.run(command, cwd=ROOT / 'frontend', stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    assert process.returncode == 0
    assert before == binding(), 'Source changed during build'
    save(OUT / 'analysis/frontend_build.json', {**before, 'all_checks_passed': True,
        'actual_build_executed': True, 'exit_code': process.returncode, 'command': command,
        'log_sha256': sha(logpath), 'phase_completed': False, 'real_browser_verified': False})


def post_storage(base):
    storage = checked_report('storage_inventory')
    for record in storage['retained_published_files']:
        path = checked_path(record['path'])
        assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256']
    sys.path.insert(0, str(ROOT / 'tests/glm5_temp'))
    from phase2693_direct_preview_audit import expected_row
    _, cat = catalog(); observations = []
    for panel in cat['panels']:
        last = panel['row_count'] - 1
        data = http_json(base, '/native-atlas-rows', panel=panel['key'], start=last, count=1)
        assert data['coordinate_count'] == panel['coordinate_count']
        assert len(data['rows']) == 1 and np.array_equal(data['rows'][0]['values'], expected_row(panel, last))
        observations.append({'key': panel['key'], 'last_row': last, 'all_columns_exact': panel['coordinate_count']})
    save(OUT / 'analysis/post_storage_checks.json', {**binding(), 'all_checks_passed': True,
        'actual_HTTP': True, 'all_published_hashes_retained': True, 'all_panel_last_rows': observations,
        'storage_report_sha256': sha(OUT / 'analysis/storage_inventory.json'), 'phase_completed': False})


def review_and_plan():
    science = checked_report('scientific_checks')
    next_plan = {
        'same_goal': True, 'created_after_actual_crossmodel_results': True,
        'scientific_checks_sha256': sha(OUT / 'analysis/scientific_checks.json'),
        'prior_Q14_candidates': science['observations']['crossmodel']['models']['qwen14']['Q14_old_candidates'],
        'objective': '外部语言操作如何改变原生单参数输入项、竞争路由、MLP乘积及自然答案；先提取跨条件可复用与可预测变化的原坐标拼图，再讨论机制闭合。',
        'key_revision': '在真实改动参数的同一数值协议内分离线性GEMM、headnorm、RoPE、softmax误差。基线重算相符与真实有限变化可预测是两个问题；不能持续重复已知记账公式充当语义进展。',
        'stages': {
            '2694': '完成本轮实际证据审查后冻结下一大阶段，分别记录模型、坐标、神经元、标量、执行精度、资源与抽样单位；保留所有旧候选及完整背景。',
            '2695': '八族双语至少4096新条件；覆盖真值、代码映射、实体名、多token填空，反平衡角色、事实顺序、提及顺序、实体与表达；确认集在首次输出前独立冻结。',
            '2696': '原生Qwen4B全坐标基线采集：精确记录执行形状、实际投影、归一化、RoPE、完整P/AV与自然答案；逐步舍入参考与实际内核差别保留。',
            '2697': '至少256分层前缀的真实单参数正负多剂量；绝对与相对剂量独立匹配，记录BF16有效改变量；从真实完整W行重算线性结果，测量真实变化，不用donor或已舍入投影加理想差项冒充原生响应。',
            '2698': '对预定分层条件做同值FP32真实模型数值对照，模型串行；区分舍入吞没、局部公式误差和真实下游耦合，不以FP64分析冒充FP64模型。',
            '2699': '结合外部角色/词汇/形式变化的全坐标输入项、归一化竞争及MLP乘积图，找具体可复用与系统性差异的条件关系；每个候选必须给出新条件方向/幅值预测及失败范围，而非只贴语义名称。',
            '2700': '冻结算法与语言条件预测后，在新填充和新的表达组合扩大确认；包含预定自然多token输出、不按事后翻转筛样本，保留全部低值、零效应和反向。',
            '2701': 'Qwen14B、GLM4、DS7B串行非量化本模型坐标复验；规模与资源在2694冻结，不借用相同坐标编号命名相同语义，DS原生与答案区分账。',
            '2702': '重要完整坐标/实际参数图与预测误差加入客户端，完成真实API/浏览器与引用安全存储终审，按真实结果自动选择同目标下一阶段。',
        },
        'first_principles': '有限参数复用已知运算不等于解释语言机制。关键证据应是同一真实参数/坐标对不同语言条件的变化关系可预测且在新材料保留；模板、token位置、输出接口与数值协议都需分账。有限上下文中的巨大组合空间不是已证明无条件无限能力。',
        'no_future_completion_claimed': True,
    }
    save(OUT / 'analysis/next_campaign.json', next_plan)


def terminal():
    phase_records(require_terminal=True)
    required = ('scientific_checks', 'panel_direct_audit', 'parameter_direct_audit',
                'checkpoint_byte_audit', 'live_api_checks', 'browser_checks',
                'storage_inventory', 'post_storage_checks', 'frontend_build')
    reports = {name: checked_report(name) for name in required}
    browser = reports['browser_checks']
    assert browser['real_browser_verified'] is True and browser['observations']
    assert browser['full_columns_last_coordinate'] and browser['gain_control'] and browser['actual_scalar_selection']
    assert reports['live_api_checks']['actual_HTTP'] and reports['frontend_build']['actual_build_executed']
    assert len(reports['checkpoint_byte_audit']['complete_checkpoint_arrays']) == 48
    assert len(reports['checkpoint_byte_audit']['direct_scalar_points']) == 192
    assert reports['post_storage_checks']['actual_HTTP']
    assert reports['post_storage_checks']['storage_report_sha256'] == sha(OUT / 'analysis/storage_inventory.json')
    storage = reports['storage_inventory']
    assert storage['remaining_unpublished_raw_files'] == 0 and storage['deleted_files'] == storage['deleted_bytes'] == 0
    for r in storage['retained_published_files']:
        assert sha(checked_path(r['path'])) == r['sha256']
    plan = read(OUT / 'analysis/next_campaign.json')
    assert plan['same_goal'] and plan['created_after_actual_crossmodel_results']
    assert plan['scientific_checks_sha256'] == sha(OUT / 'analysis/scientific_checks.json')
    assert shutil.disk_usage(RESULT).free > FLOOR
    save(OUT / 'analysis/terminal_audit.json', {**binding(), 'all_checks_passed': True,
        'required_report_sha256': {k: sha(OUT / f'analysis/{k}.json') for k in required},
        'all_published_hashes_retained': True, 'same_goal_next': True, 'phase_completed': False})


def finalize():
    phase_records(require_terminal=True)  # Check BEFORE finish() can write final.json.
    terminal_report = checked_report('terminal_audit')
    for name, digest in terminal_report['required_report_sha256'].items():
        assert sha(OUT / f'analysis/{name}.json') == digest
    _, published = catalog(); storage = checked_report('storage_inventory')
    checks = {'actual_science_and_crossmodels_complete': True, 'live_HTTP_and_real_browser': True,
              'full_native_columns_and_real_parameters': True, 'published_reference_hashes_retained': True,
              'same_goal_continuation_planned': read(OUT / 'analysis/next_campaign.json')['same_goal']}
    finish(2693, '原生QKV参数与跨层完整坐标图谱：舍入纠错及整轮客户端终审', OUT,
        {'provenance': str(Path(__file__)), 'checks': checks,
         'summary': {'published_types': len(published['panels']), 'logical_rows': sum(p['row_count'] for p in published['panels']),
                     'old_types_preserved': 75, 'actual_parameter_queries': 768,
                     'raw_files_preserved': storage['native_raw_files'], 'deleted_files': 0, 'deleted_bytes': 0,
                     'science': str(OUT / 'analysis/scientific_checks.json'),
                     'next_campaign': str(OUT / 'analysis/next_campaign.json')}},
        '从原生词嵌入、全坐标输入项、真实Q/K/V与MLP参数到来源路由建立可查询账本；语言模式观察、有限单参数效果、基线数值重算与机制解释分别验收。',
        r'z_{t,r}=\sum_k W_{r,k}x_{t,k};\quad P=\operatorname{softmax}(QK^T/\sqrt d+M);\quad E_1=\sum_i|\hat x_i-x_i^{native}|;\quad \delta_{eff}=\operatorname{BF16}(\theta+\delta)-\theta.',
        'C0018192原始与8192新条件的完整坐标背景；C00224576真实单参数条件及独立审计；C003Q14 4096和GLM/DS两协议各512；C00416源例×8层全部MLP/残差账本与64自然完整词表读出；C00532基线例×8层显式舍入核查；C006全列热力图/真实参数直接、HTTP、浏览器与构建验收；C007完整发布引用SHA与当前原场存储核查。',
        '条件纹理可以落到真实物理地址与实际参数输入项，不需要搬运另一个样本的差分。显式舍入纠正了部分数值测量，而跨条件语言响应仍须靠新材料预测来积累规律。',
        '基线重算不是有限参数改动预测；已知运算分账不是语义闭合。全坐标方向门失败不关闭路线。低值与普通标量剂量未完全匹配；自然输出只覆盖预定16前缀且3072条输出均未改变。固定256和自然cache不拼接。当前流式处理没有留下待删未展示原场，删除数量确为零而非重复旧清理。',
        '本轮2685–2693实际全部完成；按绑定实际结果的next_campaign.json自动进入同目标2694–2702。完成检查不意味着破解语言编码或证明无限语言能力。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('science', 'storage', 'http', 'build', 'post_storage', 'plan', 'terminal', 'finalize'))
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--base-url', default='http://127.0.0.1:5001/api/research-assets')
    args = parser.parse_args()
    assert not args.preview or args.action == 'storage'
    if args.action == 'science': scientific_review()
    elif args.action == 'storage': storage_inventory(args.preview)
    elif args.action == 'http': live_api(args.base_url)
    elif args.action == 'build': frontend_build()
    elif args.action == 'post_storage': post_storage(args.base_url)
    elif args.action == 'plan': review_and_plan()
    elif args.action == 'terminal': terminal()
    elif args.action == 'finalize': finalize()
