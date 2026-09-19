"""Prospective source/whole-answer semantics checks, gated after Phase2747."""
from rdc_construction_common import *
from copy import deepcopy
from rdc_question_material import (article_identity, conservative_complete_answer,
                                   drop_rows, inventory, quoref_rows)


def main():
    assert read(BASE/'phase2747/delivery_manifest.json')['phase2747_complete']
    start = time.monotonic()
    # Synthetic parser fixtures. These are never passed off as source data.
    context = '  Mira met Tomas. She gave him a book.  '
    fixture = {'data': [{'title': 'A title', 'url': 'https://en.wikipedia.org/wiki/A_title',
        'paragraphs': [{'context': context, 'qas': [
            {'id': 'q1', 'question': 'Who met Tomas?',
             'answers': [{'answer_start': 2, 'text': 'Mira'}]},
            {'id': 'q2', 'question': 'Who met each other?',
             'answers': [{'answer_start': 2, 'text': 'Mira'}, {'answer_start': 11, 'text': 'Tomas'}]},
            {'id': 'q3', 'question': 'Who was met?',
             'answers': [{'answer_start': 11, 'text': 'Tomas'}]},
        ]}]}]}
    rows = list(quoref_rows(fixture, 'synthetic'))
    assert rows[0]['passage'] == context
    assert rows[0]['answer_source_spans'][0]['start'] == 2
    assert rows[1]['answer_annotations'] == [{'required_spans': ['Mira', 'Tomas'], 'type': 'multiple_required_spans'}]
    assert article_identity('unused', 'https://en.m.wikipedia.org/wiki/A_title#Section')[0] == rows[0]['article_id']
    counts = inventory(iter(rows))
    assert counts['questions'] == 3 and counts['exact_contexts'] == 1
    assert counts['nonoverlapping_answer_pairs_per_context_histogram'] == {3: 1}
    wrong = {'data': [{'paragraphs': [{'context': context.strip(), 'qas': fixture['data'][0]['paragraphs'][0]['qas']}]}]}
    rejected_offsets = False
    try:
        list(quoref_rows(wrong, 'synthetic'))
    except AssertionError:
        rejected_offsets = True
    assert rejected_offsets
    one_bad = deepcopy(fixture)
    one_bad['data'][0]['paragraphs'][0]['qas'][0]['answers'][0]['answer_start'] += 1
    explicitly_kept = list(quoref_rows(one_bad, 'synthetic', excluded_question_ids={'q1'}))
    assert [r['original_question_id'] for r in explicitly_kept] == ['q2', 'q3']
    assert explicitly_kept == rows[1:]
    annotation = {'spans': [], 'number': '3', 'date': {'day': '', 'month': '', 'year': ''}}
    drop_fixture = {'source_only_not_an_article': {'passage': context, 'qa_pairs': [
        {'query_id': 'd1', 'question': 'How many?', 'answer': annotation,
         'validated_answers': [annotation, {'spans': ['three'], 'number': '', 'date': {}}]},
    ]}}
    dropped = list(drop_rows(drop_fixture, 'synthetic'))
    assert len(dropped[0]['answer_annotations']) == 2
    assert dropped[0]['article_id'] is None
    empty_drop = deepcopy(drop_fixture)
    empty_drop['source_only_not_an_article']['qa_pairs'].append({
        'query_id': 'empty', 'question': 'No annotation?', 'answer': {'spans': [], 'number': '', 'date': {}}})
    assert list(drop_rows(empty_drop, 'synthetic', excluded_question_ids={'empty'})) == dropped
    empty_drop['source_only_not_an_article']['qa_pairs'].append(deepcopy(empty_drop['source_only_not_an_article']['qa_pairs'][-1]))
    assert list(drop_rows(empty_drop, 'synthetic', excluded_question_ids={'empty'})) == dropped
    answer = [{'required_spans': ['Mira', 'Tomas'], 'type': 'multiple_required_spans'}]
    cases = [
        ('["Mira", "Tomas"]', True, True, True),
        ('["Tomas", "Mira"]', True, True, True),
        ('["Mira"]', True, False, True),
        ('["Mira", "Tomas", "Mira"]', True, False, True),
        ('The answer is Mira and Tomas.', True, False, False),
        ('["Mira", "Tomas"]', False, True, True),
    ]
    checks = []
    for text, eos, exact, strict in cases:
        scored = conservative_complete_answer(text, answer, eos)
        assert scored['whole_response_normalized_exact'] is exact
        assert scored['strict_nonempty_json_string_array'] is strict
        assert scored['whole_response_exact_and_stopped'] is (exact and eos)
        checks.append({'synthetic_text': text, 'eos': eos, 'score': scored})
    numeric = [{'required_spans': ['3'], 'type': 'number'}]
    assert conservative_complete_answer('3', numeric, True)['whole_response_exact_and_stopped']
    assert not conservative_complete_answer('3\nActually the answer is 4.', numeric, True)['whole_response_normalized_exact']
    folder = BASE/'phase2748/unit'
    value = {'timestamp': stamp(), 'source': snapshot(__file__),
        'algorithm': snapshot(Path(__file__).with_name('rdc_question_material.py')),
        'all_passed': True, 'source_text_offsets_preserved': True,
        'unverified_article_identity_not_inferred': True, 'multiple_required_spans_not_alternatives': True,
        'wrong_offsets_rejected': rejected_offsets, 'synthetic_whole_response_checks': checks,
        'explicit_exclusion_keeps_original_remaining_rows_unchanged': True,
        'seconds': time.monotonic()-start,
        'scope': 'Synthetic parsing and prospective scoring checks only; no native model test, '
                 'real corpus qualification, official benchmark scoring, or Phase2747 score modification.'}
    path = folder/('material_semantics_'+str(time.time_ns())+'.json')
    save(path, value)
    print('QUESTION_MATERIAL_SYNTHETIC_CHECKS_PASSED', flush=True)


if __name__ == '__main__':
    main()
