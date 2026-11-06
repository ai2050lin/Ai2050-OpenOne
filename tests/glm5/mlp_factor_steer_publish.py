"""Archive the frozen study, its native values and its full case audit."""
import base64
import io
import re
import tarfile

import mlp_factor_steer_common as C
from deploy.research_library import canonical
from mlp_atlas_publish_final import Archive
from mlp_functional_publish import archive_cases, without


TASKS = {'lookup': '属性查询', 'dispatch': '派遣规则'}
DIMENSIONS = {'n_layers': 36, 'd_ff': 9728, 'd_model': 2560, 'n_heads': 32}


def native_cases(archive, task, result):
    path = C.OUT / f'{task}-native.json.gz'
    assert C.file_sha(path) == result['native_sha256']
    data = C.read_gz(path)
    rows = data['rows']
    size = 8
    assert [r['id'] for r in rows] == [r['id'] for r in result['rows']]
    chunks = [
        archive.put(f'{task}-kv-{i}', {
            'schema': 'factor_steer_native_kv.v1', 'layer': data['layer'],
            'dtype': data['dtype'], 'start': i, 'cases': rows[i:i + size],
        })
        for i in range(0, len(rows), size)
    ]
    return {'n': len(rows), 'chunk_size': size, 'chunks': chunks,
            'dtype': data['dtype'], 'layer': data['layer'], 'native_sha256': result['native_sha256']}


def record_graph(archive, recipe, tasks, refs, case_indexes, counterexamples):
    cards, edges = [], []
    method = '64 背景中拟合方向，投影去除草稿与同答案选择键张成的子空间；在独立发现集冻结写入组。'

    def card(name, block, boundary, index, label, level, ref, scope, examples):
        key = f'{archive.snap}:{archive.run}:{name}'
        cards.append({'id': key, 'layer': block, 'boundary': boundary, 'index': index,
                      'label': label, 'method': method, 'run_id': archive.run,
                      'evidence': {'level': level, 'artifact_sha256': ref},
                      'scope': scope, 'counterexamples': examples})
        return key

    def edge(source, target, relation, level, ref, scope, examples):
        edges.append({'id': C.digest([archive.run, source, target, relation]),
                      'source': source, 'target': target, 'relation': relation,
                      'run_id': archive.run, 'evidence': {'level': level, 'artifact_sha256': ref},
                      'scope': scope, 'counterexamples': examples})

    candidates = []
    pending = [{'reason': '组合筛选不提供单一写入充分性或必要性',
                'case_index': case_indexes['selection'], 'selection_artifact': refs['selection']}]
    for b, h in recipe['heads']:
        candidates.append(card(f'B{b}-H{h}', b, 'attention.o_proj.input',
                               {'space': 'attention', 'head': h}, f'B{b}/H{h} · 冻结写入候选',
                               'observation', refs['selection'], {'joint_recipe': recipe['id']}, pending))
    for b in recipe['mlps']:
        candidates.append(card(f'B{b}-MLP', b, 'mlp.output',
                               {'space': 'mlp', 'field': 'residual_write'}, f'B{b} MLP · 冻结写入候选',
                               'observation', refs['selection'], {'joint_recipe': recipe['id']}, pending))
    for task, v in tasks.items():
        s = v['summary']
        examples = [x for x in counterexamples if x['task'] == task][:12]
        scope = {'task': task, 'selected_case_count': s['n'], 'case_index': case_indexes[task],
                 'first_word_gate_passed': s['passed'], 'eligible': s['eligible'],
                 'vocabulary': ['mining', 'fishing'], 'joint_recipe': recipe['id']}
        cache = card(f'{task}-KV', 35, 'attention.past_key_value',
                     {'space': 'attention', 'field': 'kv_cache'}, f'{TASKS[task]} · B35 缓存采样',
                     'observation', refs[task], {**scope, 'replay_scope': 'all changed KV, all layers',
                     'sampled_native_layer': 35, 'B35_alone_tested': False}, examples)
        result = card(f'{task}-gate', 35, 'logits.first_generated_word',
                      {'space': 'residual', 'field': 'final_readout'},
                      f'{TASKS[task]} · 首词门' + ('通过' if s['passed'] else '未通过'),
                      'intervention' if s['passed'] else 'negative' if s['eligible'] else 'observation',
                      refs[task], scope, examples)
        edge(f'language:{task}_selected_signed_attribute', result, 'frozen_first_word_test',
             'intervention' if s['passed'] else 'negative' if s['eligible'] else 'observation',
             refs[task], scope, examples)
        for candidate in candidates:
            edge(candidate, cache, 'member_of_joint_write_set', 'observation', refs['selection'],
                 {'individual_causality': False, 'joint_recipe': recipe['id']}, pending)
    return archive.put('graph', {'schema': 'research_graph.v1', 'feature_cards': cards,
                               'graph_edges': edges, 'note': '组合成员、缓存观测和行为结论各有证据类型。'})


def publish():
    archive = Archive(C.PUBLIC, C.RUN, C.identity())
    docs = C.read_gz(C.OUT / 'documents.json.gz')['docs']
    inputs = C.read_gz(C.OUT / 'inputs.json.gz')
    factorial = C.read_gz(C.OUT / 'factorial-result.json.gz')
    selection = C.read_gz(C.OUT / 'selection-result.json.gz')
    screen = C.read_gz(C.OUT / 'screen.json.gz')
    frozen = C.read(C.OUT / 'frozen-recipe.json')
    capability = C.read_gz(C.OUT / 'capability-result.json.gz')
    prereg = {'design': C.read(C.OUT / 'prereg.json')}
    for stage in ['fit', 'selection', 'screen', 'lookup', 'dispatch', 'capability']:
        prereg[stage] = C.read(C.OUT / f'{stage}-execution.json')
    cases, native, tasks = {}, {}, {}
    refs = {name: archive.put(name, without(doc, 'rows')) for name, doc in [
        ('factorial', factorial), ('selection', selection), ('screen', screen), ('capability', capability)]}

    def source(row):
        return without(docs[row['b']], 'tokens')

    for name, split, results in [('factorial', 'fit', factorial), ('selection', 'selection', selection)]:
        assert [r['id'] for r in inputs[split]] == [r['id'] for r in results['rows']]
        cases[name] = archive_cases(archive, name, [
            {**r, 'source': source(r), 'outcome': v} for r, v in zip(inputs[split], results['rows'])])
    counterexamples = []
    for task in TASKS:
        pool = [r for r in inputs['heldout'] if r['task'] == task]
        screened = {v['id']: v for v in screen['rows'][task]}
        cases[f'screen_{task}'] = archive_cases(archive, f'screen_{task}', [
            {**r, 'source': source(r), 'outcome': screened.get(r['id'])} for r in pool])
        result = C.read_gz(C.OUT / f'{task}-result.json.gz')
        tasks[task] = without(result, 'rows')
        cases[task] = archive_cases(archive, task, [
            {'input': pool[r['screen_index']], 'source': source(pool[r['screen_index']]), **r} for r in result['rows']])
        for r in result['rows']:
            assert pool[r['screen_index']]['id'] == r['id']
            a, b = r['conditions']['targeted']['word'], r['conditions']['matched']['word']
            counterexamples.append({'task': task, 'index': r['index'], 'expected': r['expected'],
                                    'targeted': a, 'matched': b, 'target_reached': a == r['contrast']})
        native[task] = native_cases(archive, task, result)
        refs[task] = archive.put(f'{task}-evidence', {**tasks[task], 'screen': screen['summary'][task],
                                                    'case_index': cases[task], 'native': native[task]})
    cases['capability'] = archive_cases(archive, 'capability', capability['rows'])
    directions = C.OUT / 'directions.npz'
    refs['directions'] = archive.put('native-directions', {'schema': 'research_native_binary.v1',
        'filename': directions.name, 'sha256': C.file_sha(directions), 'encoding': 'base64',
        'payload': base64.b64encode(directions.read_bytes()).decode('ascii')})
    refs['source'] = archive.sources([
        f'tests/glm5/mlp_factor_steer_{name}.py' for name in
        ['common', 'sources', 'materials', 'fit', 'edit', 'select', 'screen', 'holdout', 'publish']
    ] + ['tests/glm5/mlp_read_route_causal.py', 'tests/glm5/mlp_kv_mediation.py',
         'tests/glm5/mlp_fidelity_v2_tail.py'])
    refs['audits'] = archive.put('one-token-audits', {'audits': inputs['audits']})
    report = archive.put('report', {'schema': 'mlp_factor_steer_report.v1',
        'factorial': without(factorial, 'rows'), 'selection': frozen, 'selection_summaries': selection['summaries'],
        'screen': screen['summary'], 'tasks': tasks, 'capability': without(capability, 'rows'),
        'native': native, 'cases': cases, 'counterexamples': counterexamples, 'prereg': prereg, 'refs': refs,
        'conclusion': '冻结写入的首词效应与能力保持分门报告。缓存回放覆盖全部改变槽，B35 仅为原生展示采样。',
        'limitations': ['两个任务沿用同一答案词汇', '前缀定向写入使用已知反目标答案定向',
                        '逐题能力结果可查，中心仅发布基准题索引和评分，不包含完整题干']})
    graph = record_graph(archive, frozen['recipe'], tasks, refs, cases, counterexamples)
    context = {'study_type': 'mlp_factor_steer', 'native_bank': True, 'model_dimensions': DIMENSIONS}
    archive.bind(report, graph, context, '属性／键／草稿最小对照与冻结写入跨任务检验', {
        'factorial': {'status': 'done', 'exact_single_token_pairs': factorial['one_token_pairs']},
        'lookup': {'status': 'done', 'gate_passed': tasks['lookup']['summary']['passed']},
        'dispatch': {'status': 'done', 'gate_passed': tasks['dispatch']['summary']['passed']},
        'capability': {'status': 'done', 'n': capability['summary']['targeted']['n']}})
    with archive.lib.db() as db:
        candidates = db.execute("SELECT sha FROM artifacts WHERE kind='space_binding' AND model=? ORDER BY created DESC",
                                (archive.snap,)).fetchall()
    binding_sha = next(row['sha'] for row in candidates
                       if archive.lib.artifact(row['sha']).get('report_sha256') == report)
    w = next(w['sha'] for w in archive.lib.workpackages()['workpackages'] if w['body']['run_id'] == C.RUN)
    reg = {'run_id': C.RUN, 'model_snapshot': archive.snap, 'report_sha256': report,
           'graph_sha256': graph, 'binding_sha256': binding_sha, 'workpackage_sha256': w}
    C.save(C.OUT / 'registration-final.json', reg)
    return archive, reg


def hashes(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from hashes(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from hashes(v)
    elif isinstance(obj, str) and re.fullmatch('[0-9a-f]{64}', obj):
        yield obj


def bundle(archive, registration):
    root = archive.lib.root / 'objects'
    objects = {}
    visiting = set()

    def visit(sha):
        path = root / sha[:2] / f'{sha}.json'
        if sha in objects or not path.is_file():
            return
        doc = C.read(path)
        if doc.get('run_id') != C.RUN:
            return
        if sha in visiting:
            raise ValueError(f'cyclic research artifact reference {sha}')
        visiting.add(sha)
        assert doc['kind'] == 'real' and doc['model_snapshot'] == archive.snap
        for child in hashes(doc):
            visit(child)
        visiting.remove(sha)
        assert C.file_sha(path) == sha and path.stat().st_size < 16 * 2**20
        objects[sha] = path

    for key in ('report_sha256', 'graph_sha256', 'binding_sha256', 'workpackage_sha256'):
        visit(registration[key])
    out = C.ROOT / 'release' / 'ai2050-research-factor-steer.tar.gz'
    tmp = out.with_suffix('.tmp')
    manifest = {'schema': 'ai2050_research_upload.v1', 'objects': [{'sha256': sha} for sha in objects],
                'studies': [registration]}
    with tarfile.open(tmp, 'w:gz', compresslevel=5) as tar:
        raw = canonical(manifest)
        info = tarfile.TarInfo('MANIFEST.json')
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
        for sha, path in objects.items():
            tar.add(path, arcname=f'objects/{sha}.json')
    tmp.replace(out)
    sha = C.file_sha(out)
    out.with_suffix('.gz.sha256').write_text(f'{sha}  {out.name}\n', encoding='ascii')
    C.save(C.OUT / 'upload-package.json', {'name': out.name, 'bytes': out.stat().st_size,
                                           'sha256': sha, 'objects': len(objects), **registration})
    print(out, len(objects), sha, flush=True)


if __name__ == '__main__':
    bundle(*publish())
