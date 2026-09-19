"""Audited self-generated histories, common native ability and paired changes."""
import argparse
from collections import defaultdict
from rdc_formation_common import *
from rdc_formation_history import summarize
from rdc_formation_readout import checked_arrays
from phase2747_rdc_training_analysis import source_stat


def audit_scoring(records):
    """Replay the unchanged whole-answer grammar; do not revise any score."""
    from phase2744_rdc_query_identifiability import language_score
    controlled = [r for r in records if r['kind'] == 'controlled']
    unresolved = []
    for r in controlled:
        replay = language_score({**r, 'kind': 'controlled_language'}, r['generated_text'],
            r['generated_ids'], set(r['native_stop_ids']), 128)
        assert replay == r['answer_scoring'], ('Frozen scoring replay mismatch', r['sample_id'])
        assert replay['EOS'] == r['EOS'] and replay['censored'] == r['censored']
        if r['EOS'] and replay['conservative_final_answer'] is None:
            unresolved.append({'sample_id': r['sample_id'], 'source_group': r['source_group'],
                'family': r['family'], 'language': r['language'],
                'status': 'Stopped but frozen parser has no answer; not automatically a semantic error.',
                'generated_text_sha256': hashlib.sha256(r['generated_text'].encode()).hexdigest()})
    return {'all_frozen_scoring_objects_recomputed_exactly': True,
        'controlled_expressions': len(controlled), 'EOS_unparsed_count': len(unresolved),
        'EOS_unparsed_records': unresolved,
        'scope': 'Audit of the original grammar, not a new parser or human semantic adjudication. '
                 'Strict format, stopped content and cap censorship remain distinct.'}


def pairs(records):
    groups = defaultdict(list)
    for r in records:
        if r['kind'] == 'controlled': groups[r['pair_id']].append(r)
    assert all(len(v) == 2 and len({r['world'] for r in v}) == 2 for v in groups.values())
    return {key: {'pair_id': key, 'family': rr[0]['family'], 'source_group': rr[0]['source_group'],
        'both_correct_stopped': all(r['answer_scoring']['parsed_and_stopped_correct'] for r in rr),
        'both_first_B8_correct': all(r['first_target']['B8_correct'] for r in rr),
        'both_first_B1_correct': all(r['first_target']['B1_correct'] for r in rr)} for key, rr in groups.items()}


def main(partial=False):
    start = time.monotonic(); folder = OUT/'own_history/analysis'
    finish = folder/('partial_'+str(time.time_ns())+'.json' if partial else 'result.json')
    if finish.exists(): return
    variants = [r['condition']+'_'+str(r['seed']) for r in read(OUT/'training/result.json')['runs']]
    required = [('qwen4','native')]+[('qwen4',v) for v in variants]+[('qwen14','native'),('glm4','native')]
    complete = {}; reports = []; boundaries = []
    for model, variant in required:
        run = OUT/'own_history'/model/variant; endpoint = run/'result.json'
        if not endpoint.exists():
            assert partial, ('Missing complete ownhistory run', model, variant)
            continue
        result = read(endpoint); assert result['all_passed'] and result['trajectories'] == 512
        rr = [read(path) for path in sorted((run/'records').glob('*.json'))]
        assert len(rr) == 512 and len({r['sample_id'] for r in rr}) == 512
        for r in rr:
            a = checked_arrays(r['field']); n = len(r['generated_ids'])
            assert a['generated_ids'].tolist() == r['generated_ids']
            assert a['postnorm_BF16'].shape[0] == n and a['statistics'].shape == (n,2)
            assert ('all_hidden_BF16' in a) == r['full_hidden_collected']
            if 'all_hidden_BF16' in a: assert a['all_hidden_BF16'].shape[0] == n
            assert bool(a['first_target_available']) == ('first_target' in r)
        assert sum(r['full_hidden_collected'] for r in rr) == 44
        complete[(model,variant)] = rr
        reports.append({'model': model, 'variant': variant, 'summary': summarize(rr),
            'scoring_audit': audit_scoring(rr),
            'run_result': {'path': str(endpoint.relative_to(BASE)), 'sha256': sha(endpoint)}})
        print('FORMATION_OWN_ANALYSIS_RUN', model, variant, flush=True)
    if ('qwen4','native') in complete:
        native = {r['sample_id']: r for r in complete[('qwen4','native')]}
        for (model,variant), rr in complete.items():
            if model != 'qwen4' or variant == 'native': continue
            measures = []; changes = []
            for r in rr:
                old = native[r['sample_id']]; a, b = r['generated_ids'], old['generated_ids']
                div = next((i for i in range(max(len(a),len(b))) if i >= len(a) or i >= len(b) or a[i] != b[i]), None)
                assert div == r['first_divergence_from_native']
                nowa, olda = checked_arrays(r['field']), checked_arrays(old['field'])
                same_prefix_steps = min(len(a),len(b),div+1 if div is not None else min(len(a),len(b)))
                current, base = unbits(nowa['postnorm_BF16']).astype(float), unbits(olda['postnorm_BF16']).astype(float)
                mse = ((current[:same_prefix_steps]-base[:same_prefix_steps])**2).mean(-1)
                changes.append({'sample_id': r['sample_id'], 'family': r['family'], 'source_group': r['source_group'],
                    'first_divergence': div, 'both_have_step_on_identical_prefix': same_prefix_steps,
                    'same_prefix_full_coordinate_MSE': mse.tolist(),
                    'after_divergence_scope': 'Different self-generated text histories; no same-position causal equivalence or forced gold alignment asserted.'})
                measures.append({**r, 'EOS_change': int(r['EOS'])-int(old['EOS']),
                    'correct_stopped_change': int(r.get('answer_scoring',{}).get('parsed_and_stopped_correct',False))-int(old.get('answer_scoring',{}).get('parsed_and_stopped_correct',False)),
                    'history_changed': div is not None})
            current_pairs, native_pairs = pairs(rr), pairs(list(native.values()))
            for family in ['all']+sorted({r['family'] for r in rr}):
                selected = [r for r in measures if family == 'all' or r['family'] == family]
                cr = [r for r in selected if r['kind'] == 'controlled']
                pr = [v for v in current_pairs.values() if family == 'all' or v['family'] == family]
                record = {'variant': variant, 'family': family,
                    'EOS_change': source_stat([r['EOS_change'] for r in selected], selected),
                    'any_generated_history_change': source_stat([r['history_changed'] for r in selected], selected)}
                if cr: record['correct_and_stopped_change'] = source_stat([r['correct_stopped_change'] for r in cr], cr)
                if pr: record['both_worlds_correct_stopped_change'] = source_stat([int(v['both_correct_stopped'])-int(native_pairs[v['pair_id']]['both_correct_stopped']) for v in pr], pr)
                boundaries.append(record)
            compressed(folder/(variant+'_same_prefix_records.json.gz'), changes)
    common = None
    if all((model,'native') in complete for model in MODELS):
        native_pairs = {model: pairs(complete[(model,'native')]) for model in MODELS}
        assert all(set(p) == set(native_pairs['qwen4']) for p in native_pairs.values())
        ids = sorted(key for key in native_pairs['qwen4'] if all(pp[key]['both_correct_stopped'] for pp in native_pairs.values()))
        common = {'native_all3_complete_correct_pair_ids': ids, 'pairs': len(ids),
            'scope': 'Outcome-conditioned native ability panel, descriptive only, not a newly unselected prospective test.',
            'trained_Q4_retention': []}
        for variant in variants:
            if ('qwen4',variant) not in complete: continue
            pp = pairs(complete[('qwen4',variant)])
            common['trained_Q4_retention'].append({'variant': variant, 'correct_pairs': sum(pp[key]['both_correct_stopped'] for key in ids), 'eligible_pairs': len(ids)})
    value = {'timestamp': stamp(), 'source': snapshot(__file__), 'all_passed': len(complete) == 9,
        'partial': partial, 'complete_runs': len(complete), 'required_runs': 9, 'reports': reports,
        'paired_Q4_changes': boundaries, 'common_ability': common, 'seconds': time.monotonic()-start,
        'limits': ['No unique gold natural continuation; natural statistics describe behavior, not correctness.',
            'Model-specific tokenizer units and native chat wrappers prevent raw token-count equivalence.',
            'B1 first-prefix and B8 cached greedy own-history shape effects are explicitly separate.',
            'Fixed 96/128caps do not establish arbitrarily long composition. All censored runs remain included.',
            'FullH is collected for44frozenexpressions, not all512; every generatedstep postnorm is complete.']}
    save(finish, value); ledger('phase2747_own_history_analysis', value['seconds'])
    print('FORMATION_OWN_HISTORY_ANALYSIS', len(complete), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--partial',action='store_true'); main(parser.parse_args().partial)
