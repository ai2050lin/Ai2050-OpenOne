"""Native tokenizer alignment controls only; no language-model execution."""
from transformers import AutoTokenizer
from rdc_construction_common import *
from rdc_question_inputs import qualify_context
from phase2748_rdc_context_inventory import four_questions


def main():
    start = time.monotonic()
    cases = []
    for text in ['  Mira met Tomas. Lila brought apples, and Noah brought tea.  ',
                 'Mira said “café”. Tomas wrote 中文. Lila used an emoji: 🍎. Noah waited.\n']:
        rows = [{'question_id': 'synthetic:'+str(i), 'passage': text,
                 'question': 'Who is person number '+str(i+1)+' in the passage?',
                 'answer_annotations': [{'required_spans': [name]}]}
                for i, name in enumerate(['Mira', 'Tomas', 'Lila', 'Noah'])]
        chosen = four_questions(rows)
        assert chosen == four_questions(list(reversed(rows))) and len(chosen) == 4
        for key in ['qwen4', 'qwen14', 'glm4']:
            folder = ROOT/'models/hf'/MODELS[key]
            tok = AutoTokenizer.from_pretrained(folder, local_files_only=True, use_fast=True, trust_remote_code=True)
            stop = read(folder/'generation_config.json')['eos_token_id']
            stop = stop if isinstance(stop, list) else [stop]
            encoded = qualify_context(tok, rows, stop)
            for record, row in zip(encoded, rows):
                assert record['context_prefix_ids'] + record['question_branch_ids'] == record['input_ids']
                a, b = record['passage_char_span']
                assert record['actual_input'][a:b] == text
                assert record['teacher_ids_including_EOS'][:-1] == record['teacher_answer_ids']
                assert record['teacher_ids_including_EOS'][-1] in stop
                assert record['context_prefix_length'] <= min(record['question_token_positions'])
            cases.append({'model': key, 'synthetic_context': text,
                'question_count': len(encoded), 'prefix_tokens': encoded[0]['context_prefix_length'],
                'all_exact_spans_and_ID_decompositions_passed': True,
                'all_four_shared_prefix_IDs_equal': True, 'teacher_EOS_id': tok.eos_token_id})
    duplicate = [dict(r) for r in rows]
    duplicate[-1]['answer_annotations'] = duplicate[0]['answer_annotations']
    assert four_questions(duplicate) is None
    duplicate = [dict(r) for r in rows]
    duplicate[-1]['question'] = duplicate[0]['question']
    assert four_questions(duplicate) is None
    value = {'timestamp': stamp(), 'source': snapshot(__file__),
        'input_algorithm': snapshot(Path(__file__).with_name('rdc_question_inputs.py')),
        'all_passed': True, 'native_tokenizer_context_checks': cases,
        'four_question_stable_order_and_duplicate_rejection_passed': True,
        'preselection_assertion_repair': 'First material attempt stopped before any accepted sample because the installed apply_chat_template defaults to return_dict=True. Equality check now explicitly requests return_dict=False; rendered text/IDs and sampling rule unchanged. Original selection_rules source snapshot retained.',
        'scope': '24synthetic prompt/answer alignment checks, not real native language-model behavior.',
        'seconds': time.monotonic()-start}
    save(BASE/'phase2748/unit'/('native_inputs_'+str(time.time_ns())+'.json'), value)
    print('NATIVE_INPUT_ALIGNMENT_PASSED', len(cases), sum(r['question_count'] for r in cases), flush=True)


if __name__ == '__main__':
    main()
