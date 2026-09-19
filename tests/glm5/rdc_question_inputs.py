"""Native-template, losslessly aligned inputs for the prospective question study."""
import json
from rdc_question_material import text_id


INSTRUCTION = ('Read the passage and answer the question using the passage. '
               'Return only a nonempty JSON array of strings containing the complete answer. '
               'If several names or spans are required, include all of them as separate strings. '
               'For a number or date, put the answer in one string. Do not add an explanation.')


def native_input(tok, row, stop_ids):
    user = INSTRUCTION + '\n\nPassage:\n' + row['passage'] + '\n\nQuestion:\n' + row['question']
    rendered = tok.apply_chat_template([{'role': 'user', 'content': user}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False)
    assert isinstance(rendered, str) and rendered.count(user) == 1
    base = rendered.index(user)
    passage_start = base + len(INSTRUCTION + '\n\nPassage:\n')
    passage_end = passage_start + len(row['passage'])
    question_start = passage_end + len('\n\nQuestion:\n')
    question_end = question_start + len(row['question'])
    assert rendered[passage_start:passage_end] == row['passage']
    assert rendered[question_start:question_end] == row['question']
    enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc['input_ids'], enc['offset_mapping']
    assert len(ids) == len(offsets) and ids
    assert ids == tok.apply_chat_template([{'role': 'user', 'content': user}], tokenize=True,
                                         return_dict=False, add_generation_prompt=True, enable_thinking=False)
    # A token that straddles the boundary belongs to the question branch. The
    # shared prefix is never lengthened by an LCP extending into known questions.
    split = next(i for i, (a, b) in enumerate(offsets) if b > passage_end)
    assert 0 < split < len(ids)
    assert all(b <= passage_end for a, b in offsets[:split])
    context_positions = [i for i, (a, b) in enumerate(offsets[:split])
                         if b > passage_start and a < passage_end and b > a]
    question_positions = [i for i, (a, b) in enumerate(offsets)
                          if b > question_start and a < question_end and b > a]
    assert context_positions and question_positions and min(question_positions) >= split
    teacher_text = json.dumps(row['answer_annotations'][0]['required_spans'], ensure_ascii=False)
    answer_ids = tok(teacher_text, add_special_tokens=False)['input_ids']
    assert answer_ids and tok.eos_token_id in stop_ids
    assert not set(answer_ids) & set(stop_ids), 'A source answer contains a native stop token'
    return {'question_id': row['question_id'], 'actual_user_content': user, 'actual_input': rendered,
            'actual_input_sha256': text_id(rendered), 'input_ids': ids,
            'offset_mapping': [list(v) for v in offsets], 'context_prefix_length': split,
            'context_prefix_ids': ids[:split], 'question_branch_ids': ids[split:],
            'context_token_positions': context_positions, 'question_token_positions': question_positions,
            'passage_char_span': [passage_start, passage_end], 'question_char_span': [question_start, question_end],
            'teacher_text': teacher_text, 'teacher_answer_ids': answer_ids,
            'teacher_ids_including_EOS': answer_ids + [int(tok.eos_token_id)],
            'teacher_EOS_id': int(tok.eos_token_id), 'native_stop_ids': list(stop_ids),
            'teacher_tokenization': 'Separately tokenized complete JSON answer appended to fixed native prompt IDs, then tokenizer EOS; no joint boundary retokenization.',
            'template_boundary': 'Native chat template, add_generation_prompt=True, enable_thinking=False; actual returned text retained even when the template ignores an option.'}


def qualify_context(tok, rows, stop_ids):
    encoded = [native_input(tok, row, stop_ids) for row in rows]
    assert len(encoded) == 4 and len({r['question_id'] for r in encoded}) == 4
    assert all(r['context_prefix_ids'] == encoded[0]['context_prefix_ids'] for r in encoded)
    return encoded
