"""Publish the frozen transfer result, raw cases and native graph references."""
import base64
import io
import tarfile

from attribute_binding_holdout import RUN, SEED, SIZES
import mlp_factor_steer_common as prior
from deploy.research_library import canonical
from mlp_atlas_publish_final import Archive
from mlp_factor_steer_publish import DIMENSIONS


def main():
    out = prior.ROOT/'tests/glm5/result/attribute_prefix_transfer_v1'
    summary = prior.read(out/'summary.json')
    prereg = prior.read(out/'prereg.json')
    assert prior.file_sha(out/'prereg.json') == summary['prereg_sha256']
    assert prior.file_sha(out/'inputs.json.gz') == prereg['inputs_sha256']
    inputs = {r['id']: r for r in prior.read_gz(out/'inputs.json.gz')['rows']}
    rows = prior.read_gz(out/'scored.json.gz')['rows']
    assert len(inputs) == len(rows) and set(inputs) == {r['id'] for r in rows}
    identity = {**prior.identity(), 'provenance': {**prior.identity()['provenance'], 'seed': SEED}}
    archive = Archive(prior.ROOT/'research/attribute_prefix_transfer/v1', RUN, identity)
    shas = []

    def put(name, body):
        sha = archive.put(name, body)
        shas.append(sha)
        return sha

    source = archive.sources(['tests/glm5/attribute_binding_holdout.py', 'tests/glm5/attribute_binding_publish.py',
                              'tests/glm5/mlp_factor_steer_edit.py', 'tests/glm5/mlp_read_route_causal.py'])
    shas.append(source)
    design = put('design', prereg)
    case_index, native = {}, {}
    for split in SIZES:
        cases = [{**r, 'input_tokens': inputs[r['id']]['tokens']} for r in rows if r['split'] == split]
        assert len(cases) == SIZES[split]
        case_index[split] = {'n': len(cases), 'chunk_size': 32, 'chunks': [
            put(f'{split}-cases-{i}', {'schema': 'functional_cases.v1', 'start': i, 'cases': cases[i:i+32]})
            for i in range(0, len(cases), 32)]}
        path = out/f'{split}-native.npz'
        native[split] = put(f'{split}-native', {'schema': 'research_native_binary.v1', 'filename': path.name,
            'sha256': prior.file_sha(path), 'encoding': 'base64', 'payload': base64.b64encode(path.read_bytes()).decode('ascii'),
            'boundary': 'B35 pre-question tail value, exact BF16 bits, row order matches case chunks'})
    predictor = out/'predictor.npz'
    weights = put('predictor', {'schema': 'research_native_binary.v1', 'filename': predictor.name,
        'sha256': prior.file_sha(predictor), 'encoding': 'base64', 'payload': base64.b64encode(predictor.read_bytes()).decode('ascii')})
    title = '属性绑定与问题前写入 · 新句式和未见答案词'
    reports = summary['summary']
    capability_path = out/'capability-retest.json.gz'
    capability = None
    if capability_path.exists():
        retest = prior.read_gz(capability_path)
        assert retest['recipe_sha256'] == prereg['recipe_sha256'] and retest['directions_sha256'] == prereg['directions_sha256']
        capability = {'summary': retest['summary'], 'scope': retest['scope'],
                      'artifact_sha256': put('capability-retest', {**retest, 'kind': 'real'})}
    counterexamples = [r['id'] for r in rows if r['split'] != 'discovery' and
                       (r['conditions']['targeted']['word'] != r['contrast'] or
                        abs(r['prediction']['predicted_gap']-r['prediction']['actual_gap']) >
                        abs(r['prediction']['baseline_gap']-r['prediction']['actual_gap']))]
    report = put('report', {'schema': 'attribute_prefix_transfer_report.v1', 'title': title, 'summary': reports,
        'cases': case_index, 'native': native, 'counterexamples': counterexamples, 'prereg_sha256': design,
        'source_sha256': source, 'predictor_sha256': weights, 'capability': capability,
        'conclusion': '缓存回放和恢复逐位通过。冻结的预测与行为翻转未过门。',
        'limitations': ['发现集原始首词正确率低，预测失败同时反映发现材料与留出任务的差异。',
                        '新答案词与新句式同时变化。结论仅限联合迁移。',
                        '本轮为受控长背景。方向编辑沿用已知目标的定向方案。',
                        '能力复测沿用 Phase 2801 的选项题与固定写入偏移。其结果不外推到未测领域。' if capability else
                        'Phase 2801 的能力监测属于历史证据。当前长背景设置没有新的独立能力保持结论。']})
    cache = f'{RUN}:B35:pre-question-values'
    replay_cache = f'{RUN}:all-edited-cache-slots'
    output = f'{RUN}:B35:answer-logits'
    feature_cards = [
        {'id': cache, 'layer': 35, 'label': 'B35 问题前 V 与留出预测', 'boundary': 'pre_question_cache',
         'index': {'space': 'attention', 'field': 'kv_cache'}, 'method': 'frozen_ridge_over_role_baseline', 'run_id': RUN,
         'evidence': {'level': 'negative', 'artifact_sha256': report}, 'counterexamples': counterexamples},
        {'id': replay_cache, 'layer': 35, 'label': '跨层缓存 · 全部改动槽', 'boundary': 'all_edited_prefix_slots',
         'index': {'space': 'attention', 'field': 'cross_layer_cache'}, 'method': 'native_cache_slot_diff', 'run_id': RUN,
         'evidence': {'level': 'parameter_source', 'artifact_sha256': report}, 'counterexamples': counterexamples},
        {'id': output, 'layer': 35, 'label': '全词表 logits 的缓存回放与恢复', 'boundary': 'first_answer_logits',
         'index': {'space': 'residual', 'field': 'final_readout'}, 'method': 'native_cache_replay_rescue', 'run_id': RUN,
         'evidence': {'level': 'parameter_source', 'artifact_sha256': report}, 'counterexamples': counterexamples},
    ]
    graph = put('graph', {'schema': 'research_graph.v1', 'feature_cards': feature_cards, 'graph_edges': [
        {'id': f'{RUN}:language-binding', 'source': 'language:attribute_binding', 'target': cache,
         'relation': 'heldout_prediction_not_supported', 'run_id': RUN, 'evidence': {'level': 'negative', 'artifact_sha256': report}},
        {'id': f'{RUN}:replay', 'source': replay_cache, 'target': output, 'relation': 'replay_of_all_changed_cache_slots', 'run_id': RUN,
         'evidence': {'level': 'parameter_source', 'artifact_sha256': report}, 'scope': 'All altered cache slots replayed across layers.'},
    ]})
    binding = put('binding', {'schema': 'research_space_binding.v1', 'report_sha256': report, 'graph_sha256': graph,
        'traces': [], 'context': {'study_type': 'attribute_prefix_transfer', 'model_dimensions': DIMENSIONS, 'native_bank': True}})
    package = put('workpackage', {'schema': 'research_workpackage.v1', 'title': title, 'stages': {
        'prediction': {'status': 'done', 'confirm_passed': reports['confirm']['prediction_gate'], 'lockbox_passed': reports['lockbox']['prediction_gate']},
        'intervention': {'status': 'done', 'confirm_passed': reports['confirm']['causal_gate'], 'lockbox_passed': reports['lockbox']['causal_gate']},
        'registration_validated': {'status': 'done', 'report_sha256': report, 'graph_sha256': graph}}})
    registration = {'run_id': RUN, 'model_snapshot': archive.snap, 'report_sha256': report,
                    'graph_sha256': graph, 'binding_sha256': binding, 'workpackage_sha256': package}
    revision = binding[:12]
    bundle = out/f'ai2050-research-attribute-prefix-transfer-{revision}.tar.gz'
    if not bundle.exists():
        temporary = bundle.with_suffix('.tmp')
        with tarfile.open(temporary, 'w:gz', compresslevel=5) as tar:
            manifest = canonical({'schema': 'ai2050_research_upload.v1', 'objects': [{'sha256': s} for s in shas], 'studies': [registration]})
            member = tarfile.TarInfo('MANIFEST.json')
            member.size = len(manifest)
            tar.addfile(member, io.BytesIO(manifest))
            for sha in shas:
                path = archive.lib.root/'objects'/sha[:2]/f'{sha}.json'
                assert prior.file_sha(path) == sha
                tar.add(path, arcname=f'objects/{sha}.json')
        temporary.replace(bundle)
    prior.save(out/f'registration-{revision}.json', {**registration, 'objects': len(shas),
               'bundle': bundle.name, 'bundle_sha256': prior.file_sha(bundle)})
    print(bundle, registration, flush=True)


if __name__ == '__main__':
    main()
