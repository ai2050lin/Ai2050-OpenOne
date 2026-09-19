"""Separate post-outcome review of the ENTIRE native-language EOS-unparsed set.

Authored only after the main agent read all five original inputs and complete
outputs. This does not change any frozen scorer, primary score, model output,
training choice, or original common-ability panel. Not an independent review.
"""
from collections import defaultdict
from rdc_formation_common import *


REVIEWS = {
    'q2746new_03f3e0cebf0772a37e58': (
        '因此，根据条款丙，诺岚15属于泽类3-15。', False,
        'The terminal assertion is affirmative, but rule1 points from class1 to class0. Membership in class0 does not establish membership in class1; the requested entailment is unsupported.'),
    'q2746new_213e94f775cf85b0ef1b': (
        '因此，可以推出诺岚13属于泽类3-13。', False,
        'The terminal assertion is affirmative despite the reversed first implication; the requested entailment is unsupported.'),
    'q2746new_21ae5bb3c7cfc47ff889': (
        '通过以上步骤，可以推出诺岚14 属于 泽类3-14。', True,
        'The final requested membership follows from the three forward implications. The explanation has awkward punctuation and a misattributed premise; only the terminal conclusion is adjudicated.'),
    'q2746new_54bf74018d16763a94ed': (
        '因此，诺岚03属于泽类3-03。', True,
        'The terminal requested membership follows from the forward chain. Clause references in the generated explanation are not all accurate; no whole-reasoning-chain correctness claim.'),
    'q2746new_aaa08a044f6221ab2c3a': (
        '因此，诺岚13属于泽类3-13。', True,
        'The terminal requested membership follows from the forward chain. Misattributed intermediate clause references are retained; only the terminal conclusion is adjudicated.'),
}


def main():
    start = time.monotonic()
    folder = OUT/'own_history/terminal_review'
    finish = folder/'result.json'
    if finish.exists():
        assert read(finish)['all_passed']
        return
    analysis_path = OUT/'own_history/analysis/result.json'
    primary = read(analysis_path)
    assert primary['all_passed'] and primary['complete_runs'] == 9 and not primary['partial']
    analysis_sha = sha(analysis_path)
    remaining = []
    for report in primary['reports']:
        for row in report['scoring_audit']['EOS_unparsed_records']:
            remaining.append((report['model'], report['variant'], row['sample_id']))
    assert set(remaining) == {('glm4', 'native', sid) for sid in REVIEWS} and len(remaining) == 5
    run = OUT/'own_history/glm4/native'
    run_sha = sha(run/'result.json')
    reviewed = []
    for sid, (quote, correct, reason) in REVIEWS.items():
        path = run/'records'/(sid+'.json')
        row = read(path)
        assert row['EOS'] and not row['censored'] and row['answer_scoring']['conservative_final_answer'] is None
        assert row['generated_text'].strip().endswith(quote)
        assert row['generated_ids'][-1] in row['native_stop_ids']
        assert correct == (row['target'] == '是')
        assert not row['answer_scoring']['strict_answer_only']
        reviewed.append({'model': 'glm4', 'variant': 'native', 'sample_id': sid,
            'source_group': row['source_group'], 'pair_id': row['pair_id'],
            'record_path': path.relative_to(ROOT).as_posix(), 'record_sha256': sha(path),
            'actual_input_sha256': hashlib.sha256(row['actual_input'].encode()).hexdigest(),
            'generated_text_sha256': hashlib.sha256(row['generated_text'].encode()).hexdigest(),
            'exact_terminal_quote': quote, 'terminal_answer': '是', 'target': row['target'],
            'supplemental_terminal_correct_and_stopped': correct,
            'original_frozen_scoring': row['answer_scoring'], 'reason': reason,
            'review_status': 'Main-agent post-outcome unblinded review of full input/output, not independent or prospective.',
            'reasoning_chain_graded': False, 'strict_format_passed': False})
    rows = [read(p) for p in (run/'records').glob('*.json')]
    controlled = [r for r in rows if r['kind'] == 'controlled']
    lookup = {r['sample_id']: r for r in reviewed}
    def supplemental(row):
        if row['sample_id'] in lookup:
            return lookup[row['sample_id']]['supplemental_terminal_correct_and_stopped']
        return row['answer_scoring']['parsed_and_stopped_correct']
    groups = defaultdict(list)
    for row in controlled:
        groups[row['pair_id']].append(row)
    assert len(controlled) == 320 and len(groups) == 160 and all(len(v) == 2 for v in groups.values())
    summary = {'model': 'glm4', 'variant': 'native', 'controlled_expressions': 320, 'pairs': 160,
        'primary_correct_and_stopped': sum(r['answer_scoring']['parsed_and_stopped_correct'] for r in controlled),
        'primary_both_worlds_correct_and_stopped': sum(all(r['answer_scoring']['parsed_and_stopped_correct'] for r in v) for v in groups.values()),
        'reviewed_EOS_unparsed': 5, 'additional_terminal_correct': 3, 'additional_terminal_wrong': 2,
        'supplemental_terminal_correct_and_stopped': sum(supplemental(r) for r in controlled),
        'supplemental_both_worlds_correct_and_stopped': sum(all(supplemental(r) for r in v) for v in groups.values()),
        'primary_common_ability_panel_unchanged': True}
    assert summary['primary_correct_and_stopped'] == 230 and summary['supplemental_terminal_correct_and_stopped'] == 233
    assert sha(analysis_path) == analysis_sha and sha(run/'result.json') == run_sha
    value = {'timestamp': stamp(), 'source': snapshot(__file__), 'all_passed': True,
        'original_analysis_sha256': analysis_sha, 'original_GLM_result_sha256': run_sha,
        'complete_remaining_set': remaining, 'reviews': reviewed, 'summary': summary,
        'seconds': time.monotonic()-start,
        'scope': 'Separate unblinded terminal-conclusion audit of all five remaining stopped-unparsed outputs across all nine runs. '
                 'All three correct conclusions still fail the strict answer-only format; reasoning-chain correctness is not graded. '
                 'Frozen230/320 primary scores and original79/160 pair/common-ability analyses are not rewritten.'}
    immutable(finish, value)
    print('LANGUAGE_TERMINAL_REVIEW', summary, flush=True)


if __name__ == '__main__':
    main()
