"""Six gold-free one-shot program/text readouts followed by native own history."""
import argparse
from rdc_formation_common import *
from rdc_formation_readout import CUDA_FORMATION, checked_arrays
from rdc_formation_history import HiddenObserver

BRANCHES = ['native', 'code_identity', 'mapped_code', 'shuffled_map', 'mapped_digit_bias', 'mapped_letter_bias']


def main(pilot=False):
    import torch
    from rdc_native_tail import cuda_singleton
    from rdc_query_scoring import score, checks
    from rdc_operator_qa import repeated_ngrams
    cuda_singleton(CUDA_FORMATION)
    folder = OUT/'program_own_history'; finish = folder/('pilot.json' if pilot else 'result.json')
    if finish.exists(): return
    if not pilot: assert read(folder/'pilot.json')['all_passed']
    assert read(OUT/'transfer/readout_result.json')['all_passed']
    targets = gzread(OUT/'followup/material.json.gz')['program_own_targets']
    assert len(targets) == 32
    target_ids = [r['sample_id'] for r in targets]
    frozen = gzread(OUT/'transfer/records.json.gz')
    sources = {r['target_sample_id']: r for r in frozen if r['direction'] == 'python_to_en' and r['target_sample_id'] in target_ids}
    probes = read(OLD/'probes/protocol.json')['probes']
    qi = next(i for i, p in enumerate(probes) if p['text'] == '\nAnswer:')
    start = time.monotonic(); model = observer = None
    try:
        model, tok = load('qwen4', folder/('pilot_load' if pilot else 'main_load'))
        def one(text):
            ids = tok.encode(text, add_special_tokens=False); assert len(ids) == 1; return ids[0]
        digits, letters = [one(str(i)) for i in range(1, 9)], [one(c) for c in 'abcdefgh']
        stop = model.generation_config.eos_token_id or tok.eos_token_id
        stop = set(stop if isinstance(stop, list) else [stop]); observer = HiddenObserver(model)
        protocol_path = folder/'protocol.json'
        if not protocol_path.exists():
            immutable(protocol_path, {'timestamp': stamp(), 'source': snapshot(__file__), 'groups': 32, 'branches': BRANCHES,
                'cap': 256, 'query_index': qi, 'query_token_ids': probes[qi]['token_ids'],
                'extra_information': 'Matching original observed Python expression response at known fixed Answer query. Target response/gold never enter mapping or chosen bias set.',
                'digit_ids': digits, 'letter_ids': letters, 'uniform_bias': 8,
                'fields': 'Every generatedstep full native postnorm; full37H boundaries first and final step; initial actual readout postnorm separately saved. Intermediate allH axes not retained.',
                'decoding': 'B1 unchanged original targetprefill and queryKV; exactly one postnorm readout replacement then native self-fed greedy generation. Digit/letter bias only at step0.',
                'qualification': 'All frozen paths including failed mappings retained. Not relabeled a validated repair.',
                'parser': checks(), 'pilot': 'First2frozen mixedholdout groups, all6branches, full256cap. Numerical/control admission, not success-based candidate selection.'})
        selected = targets[:2] if pilot else targets
        records = []; replay_checks = []
        for branch in BRANCHES:
            for row in selected:
                sid = row['sample_id']; record_path = folder/('pilot_records' if pilot else 'records')/branch/(sid+'.json')
                if record_path.exists():
                    r = read(record_path); checked_arrays(r['field']); records.append(r); continue
                tick = time.monotonic(); source = sources[sid]; arrays = checked_arrays(source['field'])
                ci = 0 if branch == 'code_identity' else (4 if branch == 'shuffled_map' else 3)
                replacement = torch.tensor(arrays['predictions'][ci, qi], device='cuda', dtype=torch.bfloat16)
                with torch.inference_mode():
                    observer.active = False
                    pre = model.model(input_ids=torch.tensor([row['prompt_ids']], device='cuda'), use_cache=True)
                    cache = pre.past_key_values; del pre
                    ids = torch.tensor([probes[qi]['token_ids']], device='cuda')
                    tokens = []; postnorm = []; statistics = []; first_hidden = None; cache_audit = None; first_readout = None
                    native_path = folder/('pilot_records' if pilot else 'records')/'native'/(sid+'.json')
                    native = read(native_path) if branch != 'native' else None
                    native_packet = checked_arrays(native['field']) if native is not None else None
                    for step in range(256):
                        observer.active = True
                        value = model.model(input_ids=ids, past_key_values=cache, use_cache=True)
                        cache = value.past_key_values; h = value.last_hidden_state[0, -1]
                        current_hidden = np.stack([observer.state[i][0] for i in range(len(model.model.layers)+1)])
                        if step == 0:
                            first_hidden = current_hidden
                            before = cache_id(cache)
                            used = h if branch == 'native' else replacement
                            logits = model.lm_head(used[None]).float()[0]
                            if branch == 'mapped_digit_bias': logits[digits] += 8
                            elif branch == 'mapped_letter_bias': logits[letters] += 8
                            after = cache_id(cache); assert before == after
                            first_readout = bits(used)
                            if native_packet is not None:
                                assert np.array_equal(bits(h), native_packet['native_postnorm_BF16'][0])
                                assert np.array_equal(first_hidden, native_packet['first_final_all_hidden_BF16'][0])
                                assert before == native['cache_audit']['original_query_cache_identity']
                            cache_audit = {'original_query_cache_identity': before, 'readout_does_not_mutate_cache': True,
                                'same_prefill_and_query_as_native': True,
                                'first_B1_vs_stored_B16_postnorm_MSE': float(np.mean((h.float().cpu().numpy()-unbits(arrays['target_postnorm_BF16'][qi]))**2)),
                                'source_expression_id': source['source_sample_id'], 'source_query_index': qi,
                                'correct_answer_enters_mapping_or_bias': False, 'original_parameters_changed': False}
                        else: logits = model.lm_head(h[None]).float()[0]
                        lp = logits.double().log_softmax(-1); chosen = int(logits.argmax())
                        tokens.append(chosen); postnorm.append(bits(h)); statistics.append([float(-(lp.exp()*lp).sum()), float(lp[chosen])])
                        last = chosen in stop or step == 255
                        del value, h, lp, logits
                        if last: break
                        ids = torch.tensor([[chosen]], device='cuda')
                    observer.active = False; observer.state.clear(); del cache, ids
                generated = tok.decode(tokens, skip_special_tokens=True)
                packet = {'native_postnorm_BF16': np.stack(postnorm), 'first_actual_readout_BF16': first_readout,
                    'first_final_all_hidden_BF16': np.stack([first_hidden, current_hidden]),
                    'statistics': np.array(statistics), 'generated_ids': np.array(tokens, np.int64)}
                if not pilot:
                    earlier = folder/'pilot_records'/branch/(sid+'.json')
                    if earlier.exists():
                        pp = checked_arrays(read(earlier)['field'])
                        assert set(pp) == set(packet) and all(np.array_equal(pp[k], v) for k, v in packet.items())
                        replay_checks.append({'branch': branch, 'sample_id': sid, 'pilot_complete_array_replay_bit_equal': True})
                field = commit_array('program_own_history/'+branch+('/pilot' if pilot else ''), sid, **packet)
                result = {k: row[k] for k in ['sample_id', 'source_group', 'target', 'depth', 'representation', 'operation_sequence']}
                result.update(timestamp=stamp(), branch=branch, field=field, generated_ids=tokens, generated_text=generated,
                    answer_scoring=score(row, generated, tokens, stop, 256), cache_audit=cache_audit,
                    repeated_4gram_fraction=repeated_ngrams(tokens), prompt_ids=row['prompt_ids'], query_token_ids=probes[qi]['token_ids'],
                    first_divergence_from_native=None if native is None else next((i for i in range(max(len(tokens), len(native['generated_ids'])))
                        if i >= len(tokens) or i >= len(native['generated_ids']) or tokens[i] != native['generated_ids'][i]), None),
                    seconds=time.monotonic()-tick)
                save(record_path, result); records.append(result)
                save(folder/'progress.json', {'timestamp': stamp(), 'pilot': pilot, 'completed': len(records), 'total': len(selected)*6,
                    'generated_tokens': sum(len(r['generated_ids']) for r in records)})
                print('FORMATION_PROGRAM_OWN', pilot, branch, len(records), len(selected)*6, len(tokens), round(time.monotonic()-start, 2), flush=True)
                storage_guard()
        summary = []
        for branch in BRANCHES:
            rr = [r for r in records if r['branch'] == branch]
            summary.append({'branch': branch, 'groups': len(rr),
                **{k: sum(r['answer_scoring'][k] for r in rr) for k in ['parsed_and_stopped_correct', 'EOS', 'censored']},
                'mean_tokens': float(np.mean([len(r['generated_ids']) for r in rr])),
                'correct_cluster': clustered([int(r['answer_scoring']['parsed_and_stopped_correct']) for r in rr], [r['source_group'] for r in rr])})
        result = {'timestamp': stamp(), 'source': snapshot(__file__), 'all_passed': True, 'pilot': pilot,
            'trajectories': len(records), 'summary': summary, 'pilot_replay_checks': replay_checks,
            'actual_generated_tokens': sum(len(r['generated_ids']) for r in records),
            'seconds': time.monotonic()-start, 'peak_CUDA_bytes': torch.cuda.max_memory_allocated(),
            'scope': 'Single original postnorm readout replaced once with extra observed source response; later native own history. Allbias tokens fixed independent of gold. Not an autonomous extracted reasoning-state mechanism.'}
        save(finish, result); ledger('phase2747_program_own'+('_pilot' if pilot else ''), result['seconds'])
    except Exception as exc:
        failure(folder, start, exc); raise
    finally:
        if observer is not None: observer.close()
        if model is not None: del model
        del observer
        gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--pilot', action='store_true'); main(parser.parse_args().pilot)
