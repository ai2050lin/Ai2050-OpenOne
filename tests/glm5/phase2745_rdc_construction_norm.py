"""Direction-versus-radius identification with natural and own-history outcomes."""
import argparse
from collections import defaultdict
from phase2745_rdc_construction_contract import freeze
from rdc_construction_direction import Direction, variants
from rdc_construction_common import *
from phase2744_rdc_query_identifiability import TEMPERATURES, MIXTURES, language_score, language_checks

OUT = BASE / 'norm_controls'


def freeze_norm():
    path = OUT / 'protocol.json'
    if path.exists():
        return read(path)
    protocol, material = freeze()
    source = snapshot(__file__)
    value = {'timestamp': stamp(), 'source': source, 'phase': 2745, 'variants': variants(),
        'natural_positions': len(material['natural']), 'controlled_expressions': 320,
        'reuse_originals': 'Five complete native/originally-trained condition results from2744 are immutable baselines, not newly collected replicas. New radii/direction controls are executed on the same material.',
        'actual_matching': protocol['training_direction_control'],
        'natural_scoring': {'temperatures': TEMPERATURES, 'prior_mixtures': MIXTURES,
            'prior': 'Same add-one full-vocabulary target counts from original576trainingpositions.',
            'selection': 'Equal-document validation mean only; prospective_natural positions remain held out from new calibrator selection, but their old outputs were exposed in2744.'},
        'relation_scoring': 'B1 original prompt full-vocabulary/conditional answer scores; B8 exact old row groups,128token cap, native own histories, explicit yes/no parser unchanged. Calibration is not used to generate.',
        'fields': 'Every one of9728 gate/up/activation units at blocks16/35 for all320controlled expressions, allH16/H17/H36/postnorm coordinates, every natural postnorm coordinate and every generation first/final coordinate.',
        'primary': 'At radius0.10 compare natural vs within-cohort-permuted target directions separately by seed and their paired average. Compare natural directions against coordinate shuffle and reversed directions. Report all radii without outcome-selected primary radius.',
        'interpretation': 'Cumulative global BF16 norm is matched; per-coordinate support and per-matrix norms may still differ and are measured. This reuses actual learned parameter directions but adds no training trajectory and cannot reconstruct pretraining.'}
    immutable(path, value)
    return value


def calibration(model, material, prior, unseen):
    import torch
    temps = torch.tensor(TEMPERATURES, device='cuda', dtype=torch.float64)
    records, fields, grids = [], [], []
    for row in material['natural']:
        o = model.model(input_ids=torch.tensor([row['ids']], device='cuda'), use_cache=False)
        h = o.last_hidden_state[0, -1]
        logits = model.lm_head(h).float()
        lp = (logits.double()[None]/temps[:, None]).log_softmax(-1)
        prob, target = lp.exp(), row['target']
        qtarget = prior.get(target, unseen)
        grid = torch.stack([-lp[:, target] if alpha == 0 else
            -torch.logaddexp(lp[:, target]+np.log1p(-alpha), torch.full_like(lp[:, target], np.log(alpha*qtarget)))
            for alpha in MIXTURES], -1)
        fields.append(bits(h))
        grids.append(grid.cpu().numpy())
        records.append({k: row[k] for k in ['sample_id', 'source_group', 'cohort', 'split']} | {
            'raw_NLL': float(-lp[2, target]), 'argmax_id': int(logits.argmax()),
            'argmax_correct': int(logits.argmax()) == target, 'entropy': float(-(prob[2]*lp[2]).sum())})
        del o, h, logits, lp, prob, grid
    return {'postnorm': np.stack(fields), 'temperature_prior_mixture_NLL': np.stack(grids)}, records


def relation_fields(model, rows):
    import torch
    data, collected, records, handles = {}, defaultdict(list), [], []
    def put(name, tensor):
        data[name] = bits(tensor[0, -1])
    for layer_index in [16, 17, 36]:
        layer = model.model.layers[layer_index-1]
        handles.append(layer.register_forward_hook(lambda m, a, o, name=f'H{layer_index}': put(name, o)))
    for index in [16, 35]:
        mlp = model.model.layers[index].mlp
        for name in ['gate_proj', 'up_proj']:
            handles.append(getattr(mlp, name).register_forward_hook(lambda m, a, o, k=f'L{index}_{name}': put(k, o)))
        handles.append(mlp.down_proj.register_forward_pre_hook(lambda m, a, k=f'L{index}_activation': put(k, a[0])))
    try:
        for row in rows:
            data.clear()
            o = model.model(input_ids=torch.tensor([row['prompt_ids']], device='cuda'), use_cache=False)
            h = o.last_hidden_state[0, -1]
            lp = model.lm_head(h).float().double().log_softmax(-1)
            binary = lp[row['candidate_ids']].log_softmax(-1)
            for name, array in data.items():
                collected[name].append(array)
            collected['postnorm'].append(bits(h))
            records.append({k: row[k] for k in ['sample_id', 'source_group', 'pair_id', 'world', 'family', 'language', 'truth']} | {
                'full_vocabulary_NLL': float(-lp[row['target_ids'][0]]),
                'conditional_yes_probability': float(binary[0].exp()),
                'conditional_answer_NLL': float(-binary[0 if row['truth'] else 1]),
                'candidate_total_probability': float(lp[row['candidate_ids']].exp().sum()),
                'argmax_id': int(lp.argmax()), 'argmax_correct': int(lp.argmax()) == row['target_ids'][0]})
            del o, h, lp, binary
    finally:
        for h in handles:
            h.remove()
    return {k: np.stack(v) for k, v in collected.items()}, records


def own_history(model, tok, rows, single_fields, single_records):
    import torch
    stop = model.generation_config.eos_token_id or tok.eos_token_id
    stop = set(stop if isinstance(stop, list) else [stop])
    pad, cap = tok.pad_token_id or tok.eos_token_id, 128
    records, endpoints = [], []
    for begin in range(0, len(rows), 8):
        batch = rows[begin:begin+8]
        length, size = max(len(r['prompt_ids']) for r in batch), len(batch)
        ids = torch.full((size, length), pad, device='cuda', dtype=torch.long)
        mask = torch.zeros_like(ids)
        for b, row in enumerate(batch):
            ids[b, -len(row['prompt_ids']):] = torch.tensor(row['prompt_ids'], device='cuda')
            mask[b, -len(row['prompt_ids']):] = 1
        pos = (mask.cumsum(-1)-1).clamp_min(0)
        cache, emitted, done, first, last = None, [[] for r in batch], [False]*size, {}, {}
        for step in range(cap):
            o = model.model(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True)
            cache, h = o.past_key_values, o.last_hidden_state[:, -1]
            chosen = model.lm_head(h).float().argmax(-1)
            for b in range(size):
                if done[b]:
                    continue
                token = int(chosen[b])
                emitted[b].append(token)
                if step == 0:
                    first[b] = bits(h[b])
                done[b] = token in stop or step+1 == cap
                if done[b]:
                    last[b] = bits(h[b])
            del o, h
            if all(done):
                break
            active = torch.tensor([not x for x in done], device='cuda', dtype=torch.long)
            mask = torch.cat([mask, active[:, None]], -1)
            pos = (mask.sum(-1)-1).clamp_min(0)[:, None]
            ids = chosen[:, None]
            ids[active == 0] = pad
        for b, row in enumerate(batch):
            text = tok.decode(emitted[b], skip_special_tokens=True)
            old = read(OLD / 'identifiability/behavior/native/commits' / (row['sample_id']+'.json'))
            divergence = next((j for j, (a, v) in enumerate(zip(emitted[b], old['generated_ids'])) if a != v), None)
            if divergence is None and len(emitted[b]) != len(old['generated_ids']):
                divergence = min(len(emitted[b]), len(old['generated_ids']))
            records.append({k: row[k] for k in ['sample_id', 'source_group', 'pair_id', 'world', 'family', 'language', 'target']} | {
                'generated_ids': emitted[b], 'generated_text': text,
                'answer_scoring': language_score(row, text, emitted[b], stop, cap),
                'first_divergence_from_native_B8': divergence, 'batch_ids': [r['sample_id'] for r in batch],
                'B1_first_token': single_records[begin+b]['argmax_id'], 'B8_first_token': emitted[b][0],
                'B8_vs_B1_first_postnorm_MSE': float(np.mean((unbits(first[b]).astype(float)-unbits(single_fields['postnorm'][begin+b]))**2))})
            endpoints.append(np.stack([first[b], last[b]]))
        del cache, ids, mask
    return {'first_final_postnorm': np.stack(endpoints)}, records


def main():
    import torch
    from phase2742_rdc_query_formation import evaluate
    protocol = freeze_norm()
    _, material = freeze()
    if (OUT / 'result.json').exists():
        assert read(OUT / 'result.json')['all_passed']
        return
    pilot = read(BASE / 'norm_pilot/result.json')
    assert pilot['all_passed']
    for name in ['phase2745_rdc_construction_norm.py', 'rdc_construction_direction.py']:
        assert pilot['sources'][name]['sha256'] == sha(Path(__file__).with_name(name))
    start, model = time.monotonic(), None
    source = snapshot(__file__)
    versions = {p.name: snapshot(p) for p in [Path(__file__), Path(__file__).with_name('rdc_construction_direction.py'),
                Path(__file__).with_name('rdc_construction_common.py'), Path(__file__).with_name('phase2744_rdc_query_identifiability.py')]}
    try:
        guard(1500*1024**2)
        model, tok = load('qwen4', OUT)
        source_rows = material['models']['qwen4']['rows']
        old_material = gzread(OLD / 'identifiability/material.json.gz')
        counts, vocabulary = old_material['train_token_counts'], model.config.vocab_size
        denominator = sum(counts.values())+vocabulary
        prior = {int(k): (v+1)/denominator for k, v in counts.items()}
        controls = gzread(OLD / 'formation/material.json.gz')['panel'][:4]
        with np.load(OLD / 'formation/native_baseline.npz') as z:
            expected = z['loss'][:4]
        completed = []
        with torch.inference_mode():
            baseline = evaluate(model, controls)['loss']
            assert np.array_equal(baseline, expected), 'Original formation baseline changed'
            manager = Direction(model)
            for variant in protocol['variants']:
                folder = OUT / variant['name']
                if (folder / 'result.json').exists():
                    assert read(folder / 'result.json')['all_passed']
                    completed.append(read(folder / 'result.json'))
                    continue
                if variant['kind'] == 'reuse':
                    old = variant['old_variant']
                    references = {
                        'calibration_fields': OLD / 'identifiability/calibration' / old / 'all_fields.npz',
                        'calibration_observations': OLD / 'identifiability/calibration' / old / 'observations.json.gz',
                        'relation_root': OLD / 'identifiability/relations' / old,
                        'behavior_root': OLD / 'identifiability/behavior' / old}
                    result = {'timestamp': stamp(), 'all_passed': True, 'variant': variant,
                        'reused_immutable_baseline': True,
                        'references': {k: str(p.relative_to(ROOT)) for k, p in references.items()},
                        'reference_file_hashes': {k: sha(p) for k, p in references.items() if p.is_file()},
                        'new_trajectories': 0, 'seconds': 0.}
                    save(folder / 'result.json', result)
                    completed.append(result)
                    continue
                tick = time.monotonic()
                match = manager.match(variant)
                save(folder / 'parameter_matching.json', match)
                natural_arrays, natural_records = calibration(model, material, prior, 1/denominator)
                npz(folder / 'natural_fields.npz', **natural_arrays)
                compressed(folder / 'natural.json.gz', natural_records)
                rel_arrays, rel_records = relation_fields(model, source_rows)
                npz(folder / 'relation_fields.npz', **rel_arrays)
                compressed(folder / 'relations.json.gz', rel_records)
                behavior_arrays, behavior_records = own_history(model, tok, source_rows, rel_arrays, rel_records)
                npz(folder / 'behavior_fields.npz', **behavior_arrays)
                compressed(folder / 'behavior.json.gz', behavior_records)
                result = {'timestamp': stamp(), 'all_passed': True, 'variant': variant, 'reused_immutable_baseline': False,
                    'execution_sources': versions,
                    'natural_positions': len(natural_records), 'relation_expressions': len(rel_records),
                    'new_trajectories': len(behavior_records), 'actual_BF16_norm': match['actual_BF16_norm'],
                    'actual_changed_scalars': match['actual_changed_scalars'],
                    'seconds': time.monotonic()-tick,
                    'artifact_hashes': {p.name: sha(p) for p in folder.iterdir() if p.is_file()}}
                save(folder / 'result.json', result)
                completed.append(result)
                save(OUT / 'progress.json', {'timestamp': stamp(), 'variants_completed': len(completed), 'variants_total': len(protocol['variants']), 'last_variant': variant['name']})
                print('CONSTRUCTION_NORM', variant['name'], len(completed), len(protocol['variants']), round(result['seconds'], 1), flush=True)
                del natural_arrays, natural_records, rel_arrays, rel_records, behavior_arrays, behavior_records
                guard(128*1024**2)
            manager.restore()
            reset = evaluate(model, controls)['loss']
            assert np.array_equal(reset, expected), 'Restored native baseline changed'
            result = {'timestamp': stamp(), 'source': source, 'sources': versions,
                'all_passed': True, 'variants': len(completed), 'new_trajectories': sum(r['new_trajectories'] for r in completed),
                'baseline_reuse_count': sum(r['reused_immutable_baseline'] for r in completed),
                'restored_native_first4_exact': True, 'parser_regression': language_checks(),
                'seconds': time.monotonic()-start, 'peak_cuda_allocated': torch.cuda.max_memory_allocated(),
                'scope': 'Matched actual scalar displacement and native own-history outcomes. Existing five baselines are reused, not counted as new model replicas; no disk checkpoint was updated.'}
            save(OUT / 'result.json', result)
            ledger('construction_norm_matched_directions', result['seconds'])
            print('CONSTRUCTION_NORM_DONE', result['seconds'], flush=True)
    except Exception as exc:
        failure(OUT, start, exc)
        raise
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        print('NORM_CONTROL_FROZEN', len(freeze_norm()['variants']), flush=True)
    else:
        main()
