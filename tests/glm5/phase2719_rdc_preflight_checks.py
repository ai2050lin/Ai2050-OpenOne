"""Independent numerical receipts and material alignment/annotation checks before broad collection."""
import re
import hashlib
from collections import Counter
from rdc_joint_common import *
from rdc_relation_native_parameters import parameter, decode
from rdc_relation_common import load_field as old_field
from phase2719_rdc_joint_material import parse, document_annotations


def main():
    checks = []
    def near(name, actual, expected, atol=1e-6):
        assert abs(actual-expected) <= atol, (name, actual, expected)
        checks.append({'name': name, 'actual': actual, 'expected': expected, 'tolerance': atol})
    p = read(PREVIOUS / 'relation_atlas/parser_baseline_result.json')
    for name, expected in [('distance_only', .1658790926578171), ('predicted_POS_only', .09155493333752394), ('distance_predicted_POS', .41235281187367245)]:
        near('parser AP ' + name, p['baselines'][name]['average_precision'], expected, 1e-12)
    r = read(PREVIOUS / 'native/result.json')
    assert r['sources'] == 128 and r['queries'] == 256 and r['oracle_all_bitwise_equal']
    near('native all-source H24 MSE', r['methods']['all_predicted_sources']['H24_MSE'], 1.6405945516612732, 1e-12)
    near('native true-past hybrid H24 MSE', r['methods']['actual_past_hybrid']['H24_MSE'], 1.3865745079741894, 1e-12)
    r = read(PREVIOUS / 'boundary_relations/result.json')
    for rel, expected in [('nmod', .03394713085638077), ('conj', .043478860772403156), ('compound', .012859394007318496)]:
        near('noninitial full matrix cosine ' + rel, next(x for x in r['entries'] if x['relation'] == rel)['all_coordinate_train_test_cosine'], expected, 1e-12)
    r = read(PREVIOUS / 'source_normalization/result.json')
    near('RMS matched-df fresh H36 MSE', r['reports']['source_RMS_mean_matched_df']['evaluation']['fresh']['H36_MSE'], 54.49089431762695, 1e-9)
    for name in ('source_RMS_relation', 'source_RMS_relation_bilinear', 'source_RMS_permuted_relation'):
        near('RMS zero-history mixture ' + name, r['reports'][name]['selection']['mix'], 0, 0)
    r = read(PREVIOUS / 'output_geometry/result.json')
    for name, expected in [('current', 4.802979546831921), ('temporal', 5.374064651336084), ('self_generation', 2.365269477828406)]:
        near('same-FP32 full vocabulary KL ' + name, r['reports'][name]['mean_KL_same_FP32'], expected, 1e-12)
        assert r['reports'][name]['max_attribution_conservation_error'] < 1e-5
    n = next(x for x in r['normalized_history_readout'] if x['route'] == 'source_RMS_mean_matched_df' and x['split'] == 'fresh')
    near('RMS matched-df native-reference KL', n['KL'], 4.96576394001022, 1e-12)
    # Recompute a crucial simple explanation from actual old first-position arrays, not the prose.
    first = [(unbits(old_field(row)['h12'][0]).astype(float), unbits(old_field(row)['h23'][0]).astype(float))
             for row in old_rows() if row['split'] == 'train']
    x, y = (np.stack([pair[j] for pair in first]) for j in (0, 1))
    c = (y.mean(0) - x.mean(0)).astype(np.float32)
    with np.load(PREVIOUS / 'boundary_compilation/constant_update_control.npz') as z:
        assert np.array_equal(c, z['training_common_update'])
    errors = []
    for row in read(PREVIOUS / 'fresh_material.json'):
        z = old_field(row, True)
        delta = (unbits(z['h12'][0]) + c).astype(float) - unbits(z['h23'][0])
        errors.append(np.mean(delta*delta))
    near('independently recomputed all-coordinate first-boundary common-update MSE', float(np.mean(errors)), .01366774781405603, 1e-12)
    for key, index, expected in [('model.layers.23.mlp.gate_proj.weight', (19, 17), .09619140625),
                                 ('model.layers.23.mlp.up_proj.weight', (19, 17), .0224609375),
                                 ('model.layers.23.mlp.down_proj.weight', (23, 19), .00677490234375)]:
        near('original BF16 scalar ' + key, float(decode(parameter(ROOT, key)[index])), expected, 0)
    # GUM document and annotation indices are independently checked against frozen compressed source.
    parsed, annotations = {}, {}
    for name, part in [('gum_train', 'train'), ('gum_dev', 'dev'), ('gum_test', 'test')]:
        raw = gzip.decompress((BASE / 'sources' / (name + '.conllu.gz')).read_bytes()).decode('utf-8')
        rr = parse(raw, 'en', 'gum', part)
        parsed.update({r['source_sentence_id']: r for r in rr})
        annotations.update(document_annotations(rr))
    all_material = rows() + rows(True)
    graphs, same_piece, source_count = 0, 0, Counter()
    for row in all_material:
        n = len(row['prompt_ids'])
        assert len(row['token_offsets']) == n and len(row['tokens']) == n
        assert len(set(row['positions'])) == 6 and all(0 <= p < n for p in row['positions'])
        assert row['positions'] == [0, 1, row['anchors'][0], row['anchors'][0]+1, row['anchors'][1], row['anchors'][1]+1]
        for a, b in row['token_offsets']:
            assert 0 <= a <= b <= len(row['text'])
        if row['treebank'] == 'gum':
            parts = [parsed[sid] for sid in row['component_ids']]
            assert all(p['source_group'] == row['source_group'] for p in parts)
            assert [p['document_sentence_index'] for p in parts] == list(range(parts[0]['document_sentence_index'], parts[0]['document_sentence_index']+len(parts)))
            assert row['text'] == ' '.join(p['text'] for p in parts)
            gold_mentions = {(m['entity_id'], m['start_doc_word'], m['end_doc_word']) for m in annotations[row['source_group']]['mentions']}
            for m in row['entity_mentions']:
                assert (m['entity_id'], m['start_doc_word'], m['end_doc_word']) in gold_mentions
                a, b = m['char_span']
                assert row['text'][a:b] == m['text']
                assert row['token_offsets'][m['end_token']][1] >= b
        for edge in row['retrospective_graph']:
            a, b = edge['dependent_token'], edge['head_token']
            assert a != b and 0 <= a < n and 0 <= b < n
            assert edge['available_after_token'] == max(a, b)
            assert 'not_online_input' in edge['scope']
            graphs += 1
        source_count[row['treebank'] + '/' + row['split']] += 1
    output = {'timestamp': stamp(), 'passed': True, 'source': snapshot(Path(__file__)), 'numeric_checks': checks,
              'independent_old_boundary_recompute_sources': [320, 128], 'material_units': len(all_material), 'graph_edges_checked': graphs,
              'source_counts': dict(source_count), 'all_GUM_window_adjacency_and_entity_span_checks': True,
              'all_token_offsets_and_anchor_positions_checked': True,
              'limits': 'Annotation parsing is checked against retained corpus brackets, not independent human semantic adjudication. Native model outputs for the new fresh material remain unseen.'}
    save(BASE / 'preflight_checks.json', output)
    print('JOINT_PREFLIGHT_CHECKS', len(checks), 'numerical assertions,768 material units,', graphs, 'edges', flush=True)


if __name__ == '__main__':
    main()
