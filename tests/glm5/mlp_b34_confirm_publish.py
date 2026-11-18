"""Verify the complete run and register its cards, evidence and 3D binding."""
import io
import tarfile

import mlp_b34_confirm_design as D
from deploy.research_library import Library, canonical


def main():
    from mlp_b34_confirm_v3 import configure

    first = D.OUT
    previous_run = configure()
    plan = D.checked_plan()
    result = D.READ(D.OUT/'report.json')
    assert result['plan_sha256'] == D.HASH(D.OUT/'prereg.json')
    assert result['selection_sha256'] == D.HASH(D.OUT/'selection.json')
    for row in result['raw_manifest']:
        path = D.OUT/row['path']
        if path.stat().st_size != row['bytes'] or D.HASH(path) != row['sha256']:
            raise ValueError(f'Raw evidence changed: {path}')
    raw = D.READ(D.OUT/'capability.json')
    if raw['plan_sha256'] != result['plan_sha256']:
        raise ValueError('Capability plan mismatch')
    if list(first.glob('holdout-*.npz')):
        raise ValueError('The failed v1 discovery gate should have kept v1 holdout unopened')
    previous = D.READ(first/'selection.json')
    public = D.ROOT/'research/mlp_tail/b34_b35_v3'
    library = Library(D.ROOT/'distributed_data/research_library')
    local = Library(D.ROOT/'tests/glm5/result/mlp_tail_acceptance/center/research_library')
    objects = {}

    def put(name, body):
        doc = {'kind': 'real', 'run_id': D.RUN, 'model_snapshot': D.SNAPSHOT, 'provenance': plan['provenance'], **body}
        result = library.ingest(doc, f'{D.RUN}/{name}', kind='local_report')
        sha = result['sha256']
        assert local.ingest(doc, f'{D.RUN}/{name}', kind='local_report')['sha256'] == sha
        objects[sha] = canonical(doc)
        D.SAVE(public/f'{name}.research.json', doc)
        return sha

    design_sha = put('preregistration', plan)
    selection_sha = put('selection', {'schema': 'mlp_b34_confirm_selection.v1', **D.READ(D.OUT/'selection.json')})
    prior_sha = put('v1-negative', {'schema': 'mlp_b34_discovery_negative.v1', 'failed_run': D.READ(first/'prereg.json')['run_id'],
        'prereg_sha256': D.HASH(first/'prereg.json'), 'selection_sha256': D.HASH(first/'selection.json'),
        'gate': previous['baseline_gate'], 'paired_accuracy': previous['discovery_both_correct'], 'holdout_opened': False,
        'failed_provenance': previous['provenance']})
    repair_sha = put('repair', {'schema': 'mlp_b34_execution_repair.v1', **D.READ(D.OUT/'repair-calibration.json'),
        'aborted_run_id': D.READ(previous_run/'prereg.json')['run_id'], 'repair_contract': plan['repair']})
    historical = D.READ(D.OUT/'historical-diagnosis.json')
    historical_sha = put('historical-diagnosis', historical)
    selectivity = D.READ(D.OUT/'selectivity-diagnosis.json')
    selectivity_sha = put('selectivity-diagnosis', selectivity)
    report_sha = put('report', {**result, 'preregistration': {k:plan[k] for k in ('power','calibration','causal','semantic','inputs','software','custody','repair')},
        'historical_diagnosis': historical, 'selectivity_diagnosis': selectivity, 'artifact_refs': {'design': design_sha, 'selection': selection_sha,
            'v1_failed_gate': prior_sha, 'v2_execution_repair': repair_sha, 'historical_diagnosis': historical_sha,
            'selectivity_diagnosis': selectivity_sha}})
    cards = []
    for block, test in result['semantic'].items():
        b = int(block[1:])
        for row in test['units']:
            unit = row['unit']
            level = 'heldout' if row['passed'] and result['baseline_gate'] else 'negative'
            cards.append({'id': f'{D.RUN}:{block}:U{unit}', 'label': f'{block}/U{unit} · 收件角色方向 {row["hits"]}/{row["n"]}',
                'run_id': D.RUN, 'layer': b, 'boundary': 'mlp.down_proj.input', 'index': {'space': 'mlp', 'unit': unit},
                'method': 'Discovery-oriented paired A score on new frames with answer-code reversal',
                'scope': {'test': row, 'family': 'authored English recipient role binding'},
                'evidence': {'level': level, 'artifact_sha256': report_sha},
                'counterexamples': [{'artifact_sha256': report_sha, 'field': f'semantic.{block}'}]})
        cards.append({'id': f'{D.RUN}:{block}:group', 'label': f'{block} · 32 单元组', 'run_id': D.RUN, 'layer': b,
            'boundary': 'mlp.down_proj.input', 'index': {'space': 'mlp', 'field': 'residual_write'},
            'method': 'Frozen native group role direction and group intervention',
            'scope': {'group_test': test['group'], 'members': [row['unit'] for row in test['units']]},
            'evidence': {'level': 'heldout' if test['group']['passed'] and result['baseline_gate'] else 'negative', 'artifact_sha256': report_sha},
            'counterexamples': [{'artifact_sha256': report_sha, 'field': 'counterexamples'}]})
    edges = [{'id': f'{D.RUN}:34-35', 'run_id': D.RUN, 'source': f'{D.RUN}:B34:group', 'target': f'{D.RUN}:B35:group',
              'relation': '成对角色编辑与B35恢复路径',
              'evidence': {'level': 'intervention' if result['mediation_passed'] else 'negative', 'artifact_sha256': report_sha},
              'scope': {'mediation_passed': result['mediation_passed'], 'native_replay_gate': result['diagnosis']['native_replay_gate']}}]
    graph_sha = put('graph', {'schema': 'research_graph.v1', 'feature_cards': cards, 'graph_edges': edges,
        'note': 'The group relation records its measured outcome and gate. No individual neuron-to-neuron path is inferred.'})
    bind_sha = put('binding', {'schema': 'research_space_binding.v1', 'report_sha256': report_sha, 'graph_sha256': graph_sha,
        'traces': [], 'context': {'study_type': 'mlp_b34_confirm', 'native_bank': True,
            'model_dimensions': {'n_layers': 36, 'd_ff': 9728, 'd_model': 2560, 'n_heads': 32}}})
    work_sha = put('workpackage', {'schema': 'research_workpackage.v1', 'title': 'Qwen3-4B · B34→B35 收件角色留出与递推诊断',
        'stages': {'preregistration': {'status': 'done', 'sha256': design_sha},
            'heldout': {'status': 'done', 'families': result['holdout_pairs'], 'baseline_gate': result['baseline_gate']},
            'causal': {'status': 'done', 'controlled_behavior_passed': result['controlled_behavior_passed'], 'mediation_passed': result['mediation_passed']},
            'registration_validated': {'status': 'done', 'report_sha256': report_sha, 'graph_sha256': graph_sha}}})
    reg = {'run_id': D.RUN, 'model_snapshot': D.SNAPSHOT, 'report_sha256': report_sha, 'graph_sha256': graph_sha,
           'binding_sha256': bind_sha, 'workpackage_sha256': work_sha}
    path = D.ROOT/'release/ai2050-research-b34-b35-confirm-v3.tar.gz'
    with tarfile.open(path.with_suffix('.tmp'), 'w:gz', compresslevel=5) as tar:
        def add(name, data):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        add('MANIFEST.json', canonical({'schema': 'ai2050_research_upload.v1', 'objects': [{'sha256': s} for s in objects], 'studies': [reg]}))
        for sha, data in objects.items():
            add(f'objects/{sha}.json', data)
    path.with_suffix('.tmp').replace(path)
    reg.update(provenance=plan['provenance'], object_count=len(objects), cards=len(cards), edges=len(edges),
               bundle=str(path), bundle_sha256=D.HASH(path))
    D.SAVE(D.OUT/'registration.json', reg)
    print(reg, flush=True)


if __name__ == '__main__':
    main()
