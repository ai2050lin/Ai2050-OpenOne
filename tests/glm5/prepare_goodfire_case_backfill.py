"""Inventory existing Goodfire evidence. Write a local draft, with no center writes."""

import base64
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np

from export_graph_observation_preview import export as verify_b36

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.test_archive_client import save


RESULT = ROOT / 'tests/glm5/result'
OUT = RESULT / 'goodfire_case_backfill'
CLIENT = ROOT / 'frontend/src/components/v8'
PILOT = ROOT / 'shared/dist_protocol/test_records/pilot.json'

# Compare the stated question and method with specific earlier designs.
# These links express added scope. Only the archive protocol can decide hash equivalence.
DELTAS = {
    'gf-2959': ('gf-2957-v3', '固定句式、标签、通知位置和长度，只成对更改授权，区分保旧与抄提议。'),
    'gf-2961': ('gf-2959', '两个未过门设置各用 128 个独立家族先复测原句式，再在同家族只换授权同义词，沿用原行为门。'),
    'gf-2968': ('gf-2967', '沿用既有采集，逐行对账原生坐标、G/U/A 与原参数，属于复算。'),
    'gf-2970': ('gf-2964', '增加独立家族和 16 种组合，在等预算下和多套随机单元比较留出预测。'),
    'gf-2972': ('gf-2970', '先修正六个基线失配与写入剂量，再逐头做阻断和原生恢复。'),
    'gf-2973': ('gf-2972', '增加多套等长档案与新句式，检验措辞和标点，并冻结启用二阶项的规则。'),
    'gf-2974': ('gf-2973', '固定活动文字与长度，只移档案或请求，定位反转发生的布局。'),
    'gf-2975': ('gf-2974', '分开检验两条移动边，换整套等长标记，在同剂量与同距离下测试路径。'),
    'gf-2976': ('gf-2975', '独立交换两处占位标记，扩大联合阻断到档案邻位和共同后缀。'),
    'gf-2978-v1': ('gf-2976', '增加新的回答说明和独立试跑家族，验证能否准入跨句式确认。'),
    'gf-2978-v2': ('gf-2978-v1', '重新冻结回答说明和词表，再检查后缀窗口及完整续写。两项同时变化。'),
    'gf-2979': ('gf-2976', '将固定路径放到新等长句式，与纯文本预测比较，并核对供体覆盖率。'),
    'gf-2980': ('gf-2978-v2', '沿用既有自由生成，提取首次分歧与事后字符串，属于探索性再分析。'),
    'gf-2981': ('gf-2980', '独立交叉档案样式、档案内容与指定来源，在新家族、新句式预先分别评分对象和数字。'),
    'gf-2990': ('gf-2983', '换独立确认家族，并冻结按记录竞争导数选 Q/K 坐标，与旧排序和同预算对照在同一批上比较。'),
}


def read(path):
    return json.loads(path.read_bytes())


def file_ref(path, expected=None):
    path = path.resolve(strict=True)
    relative = path.relative_to(ROOT).as_posix()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    if expected is not None and sha != expected:
        raise ValueError(f'{relative}: archived SHA-256 differs')
    return {'path': relative, 'sha256': sha}


def study_cards():
    records = []
    for seed in read(PILOT)['records']:
        row = {k: seed[k] for k in ('id', 'phase', 'group', 'title', 'status', 'objective',
                                  'principle', 'result', 'counter', 'gate', 'related')}
        row['sources'] = []
        row['runs'] = [{k: item[k] for k in ('id', 'identity', 'activity')} for item in seed['runs']]
        refs = [seed.get('protocol'), *(item.get('source') for item in seed['runs'])]
        for ref in filter(None, refs):
            path = ROOT / ref['path']
            if not path.is_file():
                row['sources'].append({**ref, 'state': 'source_missing_locally'})
                continue
            row['sources'].append({**file_ref(path, ref['sha256']), 'state': 'sha_verified'})
        row['coverage'] = 'case_link_pending' if seed['runs'] else 'no_independent_run'
        row['reason'] = ('本次尚未逐题核对原场对应关系。' if seed['runs'] else '来源是综述或交接，未登记独立运行。')
        records.append(row)
    for row in records:
        if row['id'] == 'gf-2981':
            path = RESULT / 'content_origin_20261118_v1/counterexamples.json.gz'
            row['sources'].append({**file_ref(path), 'state': 'local_file_hashed', 'role': 'scored_cases_and_counterexamples'})
            candidates = json.loads(gzip.decompress(path.read_bytes()))
            row['case_examples'] = [
                {'case_id': case['id'], 'frame': case['frame'], 'style': case['style'],
                 'observation': '对象抄档案，数字取指定来源'}
                for case in candidates if case['split'] == 'confirm' and case['requested'] == 0
                and case['score']['obj'] == 'archive' and case['score']['num'] == 'revised'
            ]
            row.update(coverage='behavior_only', reason='冻结的内容来源行为门未过，B26/H36 原生干预与恢复没有执行。')
        if row['id'] == 'gf-2961':
            path = RESULT / 'record_authorization_retest_20261117_v1/behavior-audit.json'
            row['sources'].append({**file_ref(path), 'state': 'local_file_hashed', 'role': 'paired_behavior_counterexamples'})
            row.update(coverage='behavior_only', reason='原词和同义词都未通过成对门，未开启留出与干预。')
        if row['id'] == 'gf-2979':
            path = RESULT / 'paradigm_verdict_20261118_v1/counterexamples.json'
            row['sources'].append({**file_ref(path), 'state': 'local_file_hashed', 'role': 'heldout_counterexamples'})
    return records


def head_cases(run, phase, bundled):
    folder = RESULT / run['id']
    provenance = run['provenance']
    receipt_file = folder / ('delivery.json' if phase == 2983 else 'verified.json')
    receipt = read(receipt_file)
    if any(receipt['files'][name] != expected for name, expected in provenance.items()):
        raise ValueError('published Head provenance disagrees with the original experiment receipt')
    sources = [file_ref(receipt_file), *(file_ref(folder / name, expected) for name, expected in provenance.items())]
    collection = read(folder / 'confirm-collection.json')
    evaluation = read(folder / 'evaluation.json')
    if collection['provenance']['model_revision'] != run['snapshot'] or run['readout'] != {
        'layer': 26, 'head': 36, 'position': '最后输入 token',
    }:
        raise ValueError('B26/H36 source snapshot or readout changed')
    material = read(folder / 'confirm-materials.json')
    by_id = {case['id']: (i, case) for i, case in enumerate(material)}
    published = {case['id']: case for family in run['families'] for case in family['cases']}
    cases = []
    seen = set()
    for name, expected in sorted(collection['shards'].items()):
        array_ref = file_ref(folder / name, expected)
        meta_file = folder / Path(name).with_suffix('.json')
        meta_ref = file_ref(meta_file, receipt['files'][meta_file.name])
        meta = read(meta_file)
        if meta['sha'] != expected or receipt['files'][name] != expected:
            raise ValueError(f'{name}: row metadata references different bytes')
        sources.extend((array_ref, meta_ref))
        with np.load(folder / name, allow_pickle=False) as archive:
            vectors, masses = archive['head_actual'], archive['mass_actual']
        count = len(meta['scores'])
        if vectors.dtype != np.float32 or vectors.shape != (count, 128) or masses.shape != (count, 5):
            raise ValueError(f'{name}: unexpected native capture layout')
        if not (np.isfinite(vectors).all() and np.isfinite(masses).all()):
            raise ValueError(f'{name}: non-finite native measurement')
        for i, score in enumerate(meta['scores']):
            case_id = score['id']
            material_row, case = by_id[case_id]
            if case_id in seen or not case['tokens']:
                raise ValueError(f'{name}: ambiguous case membership')
            seen.add(case_id)
            position = len(case['tokens']) - 1
            token_id = case['tokens'][-1]
            if score['top1_correct'] != (score['top1'] == case['gold_id']):
                raise ValueError(f'{case_id}: original scoring metadata disagrees with the frozen answer')
            public = published[case_id]
            if (public['inputTokens'], public['tokenId'], public['gold'], public['correct']) != (
                position + 1, score['top1'], case['target_record'], score['top1_correct'],
            ) or public['mass'] != [round(float(value), 8) for value in masses[i]]:
                raise ValueError(f'{case_id}: published summary does not match the original shard row')
            in_client = case_id in bundled
            if in_client:
                excerpt = bundled[case_id]
                if (position, token_id) != (excerpt['position'], excerpt['tokenId']) or not np.array_equal(
                    vectors[i], np.asarray(excerpt['values'], dtype=np.float32),
                ):
                    raise ValueError(f'{case_id}: bundled replay differs from original')
            cases.append({
                'case_id': case_id, 'family': case['family'], 'cell': case['cell'], 'rule': case['rule'],
                'material_row': material_row, 'position': position, 'input_token_id': token_id,
                'next_token_id': score['top1'], 'gold_token_id': case['gold_id'],
                'output_correct': score['top1_correct'], 'head_top_record': int(np.argmax(masses[i])),
                'target_record': case['target_record'],
                'source_sha256': expected, 'source_row': i, 'array': 'head_actual',
                'vector_sha256': hashlib.sha256(vectors[i].astype('<f4', copy=False).tobytes()).hexdigest(),
                'replay': 'bundled_client' if in_client else 'local_native_no_registered_trace',
            })
    if seen != published.keys() or len(seen) != len(material) or len(by_id) != len(material):
        raise ValueError('B26/H36 sources did not cover original cases exactly once')
    return {'record_id': f'gf-{phase}', 'phase': phase, 'run_id': run['id'], 'snapshot': run['snapshot'],
            'split': 'confirm', 'address': 'B26/H36', 'behavior_gate_passed': collection['passed'],
            'prediction_branches_admitted': evaluation['admitted'],
            'layer': 27, 'block': 26, 'head': 36, 'boundary': 'head_z', 'width': 128,
            'sources': sources, 'cases': cases, 'missing': '这里核对的是末输入 token 的单个 Head。其他头、位置及 MLP 未由该绑定提供。'}


def b36_cases():
    archive = CLIENT / 'data/graphObservationExample.json'
    original = verify_b36()
    for key in ('hiddenShape', 'mlpShape'):
        original[key] = list(original[key])
    if original != read(archive):
        raise ValueError('B36 client excerpt differs from original capture')
    folder = RESULT / 'b36_transition14_20261114_v1'
    sources = [
        file_ref(archive), file_ref(folder / original['source']['file'], original['source']['sha256']),
        file_ref(folder / 'native/confirm-000/record.json', original['source']['recordSha256']),
    ]
    return {'run_id': original['run'], 'snapshot': original['snapshot'], 'phase': original['phase'],
            'case_id': original['caseId'], 'address': 'B36/U2429', 'block': 36, 'layer': 37,
            'positions': [{'position': site['index'], 'input_token_id': site['tokenId'], 'site': site['site']}
                          for site in original['positions']],
            'boundary': 'mlp_activation', 'width': original['mlpShape'][1],
            'sources': sources, 'replay': 'bundled_client',
            'missing': '客户端摘录只回放此题四个位置的 FFN 激活和 HiddenState。'}


def b23_cases():
    path = CLIENT / 'data/nativeObservationB23.json'
    archive = read(path)
    folder = RESULT / 'goodfire_middle_20261113_v2/qwen14'
    manifest_file = folder / 'manifest.json'
    manifest_ref = file_ref(manifest_file, archive['manifestSha256'])
    manifest = read(manifest_file)
    if (manifest['kind'], manifest['state'], manifest['run_id'], manifest['model_snapshot']) != (
        'real', 'complete', archive['run'], archive['snapshot'],
    ):
        raise ValueError('B23 manifest identity differs from the bundled archive')
    files = {item['path']: item['sha256'] for item in manifest['files']}
    array_ref = file_ref(folder / 'B23-A.npy', files['B23-A.npy'])
    docs_ref = file_ref(folder / 'documents.json', files['documents.json'])
    documents = read(folder / 'documents.json')
    raw = np.load(folder / 'B23-A.npy', allow_pickle=False, mmap_mode='r')
    if array_ref['sha256'] != archive['arraySha256'] or raw.dtype != np.dtype('<u2') or raw.shape != (
        len(documents['docs']), len(documents['positions']), archive['width'],
    ):
        raise ValueError('B23 original array identity or layout differs')
    by_doc = {row['doc_id']: row for row in documents['docs']}
    cases = []
    for case in archive['rows']:
        doc = by_doc[case['doc']]
        slot = documents['positions'].index(case['position'])
        if doc['split'] != 'holdout' or doc['tokens'][case['position']] != case['tokenId'] or (
            raw[doc['doc_index'], slot].tobytes() != base64.b64decode(case['bits'], validate=True)
        ):
            raise ValueError(f'{case["doc"]}: original B23 document, position or values differ')
        cases.append({'case_id': case['doc'], 'source_row': doc['doc_index'], 'site_row': slot,
                      'position': case['position'], 'input_token_id': case['tokenId'], 'replay': 'bundled_client'})
    return {'run_id': archive['run'], 'snapshot': archive['snapshot'], 'split': 'holdout', 'address': 'B23/U7708',
            'block': 23, 'layer': 24, 'width': archive['width'], 'boundary': 'mlp_activation', 'cases': cases,
            'sources': [file_ref(path), manifest_ref, array_ref, docs_ref],
            'missing': '这两条绑定是语料片段的激活读出；未绑定 Head、其他层或生成轨迹。'}


def main():
    rows = study_cards()
    cues = read(CLIENT / 'threeCueEvidence.json')['runs']
    module_file = CLIENT / 'data/nativeHeadChannels.js'
    bundle = json.loads(module_file.read_text(encoding='utf-8').split(' = ', 1)[1].strip().removesuffix(';'))
    if (bundle['run'], bundle['snapshot']) != (cues[0]['id'], cues[0]['snapshot']):
        raise ValueError('bundled Head metadata disagrees with published record')
    head = [head_cases(run, phase, bundle['cases'] if run['id'] == bundle['run'] else {})
            for run, phase in zip(cues, (2983, 2990), strict=True)]
    head[0]['sources'].append(file_ref(module_file))
    print(f'[draft] verified {sum(len(run["cases"]) for run in head)} original Head cases', flush=True)
    b36, b23 = b36_cases(), b23_cases()
    keys = [(run['run_id'], run['snapshot'], case['case_id']) for run in head for case in run['cases']]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate identity across prepared Head cases')
    counterexample_counts = {}
    for run in head:
        counterexample_counts[run['run_id']] = dict(Counter(
            str(case['rule']) for case in run['cases']
            if case['output_correct'] and case['head_top_record'] != case['target_record']
        ))
    coverage = {'record_cards': len(rows), 'cards_by_state': dict(Counter(row['coverage'] for row in rows)),
                'head_cases': len(keys), 'head_cases_in_client': len(bundle['cases']),
                'mlp_cases': 1 + len(b23['cases']), 'mlp_positions': len(b36['positions']) + len(b23['cases']),
                'explicit_deltas': len(DELTAS), 'output_correct_head_disagrees': counterexample_counts}
    save(OUT / 'draft.json', {'schema': 'goodfire_case_backfill_draft.v1', 'state': 'local_draft',
                             'center_acceptance': 'required_before_import',
                             'study_cards': rows, 'head': head, 'mlp': [b36, b23], 'coverage': coverage,
                             'comparisons': [{'record_id': key, 'earlier_record_id': pair[0], 'added_verification': pair[1]}
                                             for key, pair in DELTAS.items()]})
    print(json.dumps(coverage, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
