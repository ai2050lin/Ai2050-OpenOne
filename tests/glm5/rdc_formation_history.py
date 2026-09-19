"""Frozen text identities and native greedy B8 histories, every postnorm kept."""
from collections import Counter, defaultdict
from rdc_formation_common import *


def freeze():
    from transformers import AutoTokenizer
    folder = OUT/'own_history'
    path = folder/'protocol.json'
    if path.exists(): return read(path), gzread(folder/'material.json.gz')
    follow = gzread(OUT/'followup/material.json.gz')['own_history']
    qtok = AutoTokenizer.from_pretrained(ROOT/'models/hf'/MODELS['qwen4'], local_files_only=True)
    complete = []
    for family in sorted({r['family'] for r in follow}):
        rr = [r for r in follow if r['family'] == family]
        if rr[0]['kind'] == 'natural':
            complete += [r['sample_id'] for r in sorted(rr, key=lambda r: rank('ownfull/'+r['sample_id']))[:24]]
        else:
            group = min({r['source_group'] for r in rr}, key=lambda g: rank('ownfull/'+g))
            complete += [r['sample_id'] for r in rr if r['source_group'] == group]
    assert len(complete) == 44
    material = {}
    for key, name in MODELS.items():
        tok = AutoTokenizer.from_pretrained(ROOT/'models/hf'/name, local_files_only=True, trust_remote_code=True)
        rows = []
        for row in follow:
            if row['kind'] == 'natural':
                text = qtok.decode(row['prompt_ids'], skip_special_tokens=False)
                assert '\ufffd' not in text and qtok.encode(text, add_special_tokens=False) == row['prompt_ids']
                ids = tok.encode(text, add_special_tokens=False)
                full_text = qtok.decode(row['source_input_ids'], skip_special_tokens=False)
                full_ids = tok.encode(full_text, add_special_tokens=False)
                aligned = full_ids[:len(ids)] == ids and len(full_ids) > len(ids)
                target_ids = [full_ids[len(ids)]] if aligned else []
                target_scope = 'Authentic next token in model-specific full-source tokenization' if aligned else 'Unscored: model-specific full-source BPE merges across this fixed text-prefix boundary'
                if key == 'qwen4': assert ids == row['ids'] and target_ids == [row['target']]
            else:
                text = tok.apply_chat_template([{'role': 'user', 'content': row['original_text']}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                ids = tok.encode(text, add_special_tokens=False)
                target_ids = tok.encode(row['target'], add_special_tokens=False)
                target_scope = 'First token of declared literal relation answer; full answer and format scored independently'
                if key == 'qwen4': assert ids == row['prompt_ids']
            rows.append({**row, 'model': key, 'actual_input': text, 'model_prompt_ids': ids,
                'model_target_ids': target_ids, 'first_target_scope': target_scope,
                'collect_all_hidden': row['sample_id'] in complete})
        material[key] = rows
    compressed(folder/'material.json.gz', material)
    protocol = {'timestamp': stamp(), 'source': snapshot(__file__), 'material_sha256': sha(folder/'material.json.gz'),
        'models': list(MODELS), 'expressions_each': 512, 'full_hidden_sample_ids': complete,
        'batch': 8, 'padding': 'Left padding; explicit attention masks and cumulative position IDs; inactive rows append masked pad tokens.',
        'shape_control': 'Every expression also gets original B1 first-postnorm and fullV readout. Q4native and all6trained firstB1 states checked against actual training evaluation packets.',
        'larger_model_pilot': 'First8natural and first8controlled, per-model original B1 capture plus fullcap B8 generation. Measured memory/time before full execution.',
        'generation': 'Greedy full native vocabulary, model native EOS IDs, natural96/controlled128 cap; no calibration, teacher token or gold constraint.',
        'fields': 'Every actual token has full postnormBF16. AllH0..Hdepth at each generatedstep for44hash/source-balanced expressions; other fullH axes explicitly not collected.',
        'native_text': 'Identical natural textprefix across models; same controlled user text with each native tokenizer/chatwrapper. Tokens/step counts are not cross-model identical units.',
        'first_natural_target': 'Only score when complete-source model tokenization exactly extends the tokenized fixed prefix; otherwise mark unavailable, never pretend a cross-boundary merge is the authentic next token.',
        'natural_unscored_first_positions': {key: sum(not r['model_target_ids'] for r in rr if r['kind'] == 'natural') for key, rr in material.items()},
        'remaining': 'All3native and6nativeBF16trainedQ4 trajectories; this document freezes implementation details, not a completion claim.'}
    immutable(path, protocol)
    return protocol, material


class HiddenObserver:
    def __init__(self, model):
        self.active = False; self.state = {}; self.handles = []
        def put(index, value):
            if self.active: self.state[index] = bits(value[:, -1])
        self.handles.append(model.model.embed_tokens.register_forward_hook(lambda m, a, o: put(0, o)))
        for i, layer in enumerate(model.model.layers, 1):
            self.handles.append(layer.register_forward_hook(lambda m, a, o, i=i: put(i, o)))
    def close(self):
        for handle in self.handles: handle.remove()


def generate_batch(model, tok, rows, observer):
    import torch
    stop = model.generation_config.eos_token_id or tok.eos_token_id
    stop = set(stop if isinstance(stop, list) else [stop])
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    b1, b1stats = [], []
    with torch.inference_mode():
        observer.active = False
        for row in rows:
            value = model.model(input_ids=torch.tensor([row['model_prompt_ids']], device='cuda'), use_cache=False)
            h = value.last_hidden_state[0, -1]
            lp = model.lm_head(h[None]).float()[0].double().log_softmax(-1)
            b1.append(bits(h)); b1stats.append([int(lp.argmax()), float(-(lp.exp()*lp).sum()),
                float(-lp[row['model_target_ids'][0]]) if row['model_target_ids'] else 0.])
            del value, h, lp
        n, maxlen = len(rows), max(len(r['model_prompt_ids']) for r in rows)
        ids = torch.full((n, maxlen), pad, device='cuda', dtype=torch.long); mask = torch.zeros_like(ids)
        for b, row in enumerate(rows):
            ids[b, -len(row['model_prompt_ids']):] = torch.tensor(row['model_prompt_ids'], device='cuda')
            mask[b, -len(row['model_prompt_ids']):] = 1
        positions = (mask.cumsum(-1)-1).clamp_min(0)
        cache = None; done = [False]*n
        tokens = [[] for _ in rows]; posts = [[] for _ in rows]; fields = [[] for _ in rows]; stats = [[] for _ in rows]
        first_b8_target_nll = [0.]*n
        observer.active = any(r['collect_all_hidden'] for r in rows)
        for step in range(max(r['max_new_tokens'] for r in rows)):
            value = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, past_key_values=cache, use_cache=True)
            cache = value.past_key_values; h = value.last_hidden_state[:, -1]
            lp = model.lm_head(h).float().double().log_softmax(-1); choice = lp.argmax(-1)
            post = bits(h)
            for b, row in enumerate(rows):
                if done[b]: continue
                chosen = int(choice[b]); tokens[b].append(chosen); posts[b].append(post[b])
                stats[b].append([float(-(lp[b].exp()*lp[b]).sum()), float(lp[b, chosen])])
                if step == 0 and row['model_target_ids']:
                    first_b8_target_nll[b] = float(-lp[b, row['model_target_ids'][0]])
                if row['collect_all_hidden']:
                    fields[b].append(np.stack([observer.state[i][b] for i in range(len(model.model.layers)+1)]))
                done[b] = chosen in stop or step+1 == row['max_new_tokens']
            del value, h, lp
            if all(done): break
            active = torch.tensor([not d for d in done], device='cuda', dtype=torch.long)
            mask = torch.cat([mask, active[:, None]], -1)
            positions = (mask.sum(-1)-1).clamp_min(0)[:, None]
            ids = choice[:, None]; ids[active == 0] = pad
        observer.active = False; observer.state.clear()
        del cache, ids, mask
    arrays = []
    for i, row in enumerate(rows):
        packet = {'generated_ids': np.array(tokens[i], np.int64), 'postnorm_BF16': np.stack(posts[i]),
            'statistics': np.array(stats[i]), 'first_B1_postnorm_BF16': b1[i], 'first_B1_statistics': np.array(b1stats[i]),
            'first_B8_target_NLL': np.array(first_b8_target_nll[i]),
            'first_target_available': np.array(bool(row['model_target_ids']))}
        if fields[i]: packet['all_hidden_BF16'] = np.stack(fields[i])
        arrays.append(packet)
    return arrays, stop


def record(row, packet, tok, stop, variant):
    from phase2744_rdc_query_identifiability import language_score
    from rdc_operator_qa import repeated_ngrams
    ids = packet['generated_ids'].tolist()
    generated = tok.decode(ids, skip_special_tokens=True)
    result = {k: row[k] for k in ['sample_id', 'source_group', 'cohort', 'family', 'language', 'kind', 'split', 'model']}
    result.update(variant=variant, model_prompt_ids=row['model_prompt_ids'], actual_input=row['actual_input'],
        generated_ids=ids, generated_text=generated, native_stop_ids=sorted(stop),
        EOS=ids[-1] in stop, censored=ids[-1] not in stop and len(ids) == row['max_new_tokens'],
        first_visible_content_step=next((i for i in range(len(ids)) if tok.decode(ids[:i+1], skip_special_tokens=True).strip()), None),
        first_token_text=tok.decode(ids[:1], skip_special_tokens=False), repeated_4gram_fraction=repeated_ngrams(ids),
        first_shape={'B1_argmax': int(packet['first_B1_statistics'][0]), 'B8_argmax': ids[0],
            'B1_B8_postnorm_full_coordinate_MSE': float(np.mean((unbits(packet['first_B1_postnorm_BF16']).astype(float)-unbits(packet['postnorm_BF16'][0]))**2))},
        full_hidden_collected=row['collect_all_hidden'], first_target_scope=row['first_target_scope'])
    if row['model_target_ids']:
        result['first_target'] = {'id': row['model_target_ids'][0], 'B1_NLL': float(packet['first_B1_statistics'][2]),
            'B8_NLL': float(packet['first_B8_target_NLL']), 'B1_correct': int(packet['first_B1_statistics'][0]) == row['model_target_ids'][0],
            'B8_correct': ids[0] == row['model_target_ids'][0]}
    if row['kind'] == 'controlled':
        result.update(pair_id=row['pair_id'], world=row['world'], target=row['target'],
            answer_scoring=language_score({**row, 'kind': 'controlled_language'}, generated, ids, stop, row['max_new_tokens']))
    return result


def summarize(records):
    result = []
    for family in ['all']+sorted({r['family'] for r in records}):
        rr = [r for r in records if family == 'all' or r['family'] == family]
        value = {'family': family, 'expressions': len(rr), 'source_groups': len({r['source_group'] for r in rr}),
            'EOS': sum(r['EOS'] for r in rr), 'censored': sum(r['censored'] for r in rr),
            'tokens': sum(len(r['generated_ids']) for r in rr),
            'mean_tokens': float(np.mean([len(r['generated_ids']) for r in rr])),
            'mean_repeated_4gram_fraction': float(np.mean([r['repeated_4gram_fraction'] for r in rr])),
            'first_B1_B8_argmax_mismatches': sum(r['first_shape']['B1_argmax'] != r['first_shape']['B8_argmax'] for r in rr)}
        controlled = [r for r in rr if 'answer_scoring' in r]
        if controlled:
            pairs = defaultdict(list)
            for r in controlled: pairs[r['pair_id']].append(r)
            value.update(controlled_expressions=len(controlled), pairs=len(pairs),
                correct_and_stopped=sum(r['answer_scoring']['parsed_and_stopped_correct'] for r in controlled),
                both_worlds_correct_and_stopped=sum(len(v) == 2 and all(r['answer_scoring']['parsed_and_stopped_correct'] for r in v) for v in pairs.values()),
                strict_answer_only=sum(r['answer_scoring']['strict_answer_only'] for r in controlled),
                parsed_wrong=sum(r['answer_scoring']['conservative_final_answer'] is not None and not r['answer_scoring']['conservative_final_correct'] for r in controlled))
        result.append(value)
    return result


if __name__ == '__main__':
    p, data = freeze(); print('FORMATION_OWN_HISTORY_MATERIAL', {k: len(v) for k, v in data.items()}, p['natural_unscored_first_positions'], flush=True)
