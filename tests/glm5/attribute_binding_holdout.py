"""Freeze and execute the independent pre-question writer transfer experiment."""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
from scipy.stats import binomtest

import mlp_factor_steer_common as previous
from tools.power_analysis import check_phase

RUN = 'ATTRIBUTE-PREFIX-TRANSFER-20261106-v1'
SEED = 611062805
SIZES = {'discovery': 96, 'confirm': 128, 'lockbox': 128}
FRAMES = {
    'discovery': ('Signed occupations', 'The official record gives', 'The requested code is', 'Which occupation is signed?'),
    'confirm': ('Certified register', 'The auditor certified', 'Use the code', 'What occupation does the certified record assign?'),
    'lockbox': ('Authenticated work orders', 'The final order authorizes', 'Resolve the order for code', 'Which activity does the authorized order specify?'),
}
BACKGROUND = {
    'discovery': ('museum', 'workshop', 'courtyard', 'library', 'passage'),
    'confirm': ('terrace', 'conservatory', 'stairwell', 'gallery', 'pavilion'),
    'lockbox': ('observatory', 'auditorium', 'vestibule', 'colonnade', 'atrium'),
}


def freeze(out):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(previous.MODEL, local_files_only=True)
    enc = lambda text: tok.encode(text, add_special_tokens=False)
    opening = enc('<|im_start|>user\n')
    ending = enc('\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
    rng = random.Random(SEED)
    rows = []
    for split, size in SIZES.items():
        heading, clause, selector, question = FRAMES[split]
        words = ('mining', 'fishing') if split != 'lockbox' else ('painting', 'cooking')
        for word in words:
            assert len(enc(' '+word)) == 1, word
        design = [(a, b, order, display) for a in range(2) for b in range(2) for order in range(2) for display in range(2)]
        assignments = design * (size // len(design))
        rng.shuffle(assignments)
        for i, (target, other, reverse, display) in enumerate(assignments):
            key = 'K' if display else 'T'
            alt_key = 'T' if display else 'K'
            label = ('Ravel', 'Morden') if split == 'discovery' else ('Velmont', 'Carmond') if split == 'confirm' else ('Tervan', 'Selwick')
            a, b = [f'{name}-{i}' for name in label]
            records = [(key, a, words[target]), (alt_key, b, words[other])]
            if reverse:
                records.reverse()
            text = f'\n\nIgnore the preceding tour notes. {heading}. Only the signed rows are authoritative.\n'
            spans = {}
            for code, name, value in records:
                text += f'{code} identifies {name}. {clause} {name}:'
                if code == key:
                    spans['attribute'] = (len(text), len(text)+len(value)+1)
                text += f' {value}.\n'
            text += f'A preliminary suggestion mentions {words[1-target]}.\n{selector}:'
            spans['selector'] = (len(text), len(text)+2)
            text += f' {key}.\nRead the signed evidence.\n'
            encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            base = previous.CUT-len(encoded.input_ids)
            background = ' '.join(
                f'Tour card {j+1} for the {rng.choice(BACKGROUND[split])} notes {rng.randint(11, 97)} visitors. '
                f'The group waited beside the {rng.choice(BACKGROUND[split])} before moving to the next room.'
                for j in range(48)
            )
            n_background = base-len(opening)
            bg = enc(background)[:n_background]
            assert len(bg) == n_background and n_background >= 800
            anchors = {'tail_a': previous.CUT-2, 'tail_b': previous.CUT-1}
            for role, (start, end) in spans.items():
                offsets = [base+j for j, (lo, hi) in enumerate(encoded.offset_mapping) if lo < end and hi > start]
                assert len(offsets) == 1, (split, role, offsets)
                anchors[role] = offsets[0]
            ordered = list(reversed(words)) if display else words
            query = enc(f'{question} Begin with one word: {ordered[0]} or {ordered[1]}.')
            assert len(query) <= 32
            tokens = opening+bg+encoded.input_ids+[198]*(32-len(query))+query+ending
            assert len(tokens) == 1034
            rows.append({'id': f'{split}:{i:03}', 'split': split, 'expected': words[target], 'contrast': words[1-target],
                         'orientation': 1 if target == 0 else -1, 'other_matches': target == other,
                         'order': reverse, 'display': display, 'anchors': anchors, 'tokens': tokens,
                         'background_sha256': previous.digest(bg), 'background_tokens': n_background})
    assert len({r['background_sha256'] for r in rows}) == len(rows)
    corpus = out/'inputs.json.gz'
    previous.write_gz(corpus, {'kind': 'controlled_generated_material', 'rows': rows})
    prior = previous.OUT/'frozen-recipe.json'
    power = check_phase(.10, 128, alpha=.025)
    assert power['pass']
    previous.save(out/'prereg.json', {
        **previous.identity(), 'run_id': RUN, 'seed': SEED, 'schema': 'attribute_prefix_prereg.v1', 'splits': SIZES,
        'inputs_sha256': previous.file_sha(corpus), 'runner_sha256': previous.file_sha(__file__),
        'recipe_sha256': previous.file_sha(prior), 'directions_sha256': previous.file_sha(previous.OUT/'directions.npz'),
        'prediction': 'Fixed B35 last pre-question V at common tail. Ridge lambda=10 fit on discovery centered label-oriented vectors and gap residual over OLS label, competing record match, order and display baseline. Gap uses maximum first-token logits across surface forms. Confirmation and new-label lockbox evaluated once, pass if per split RMSE improves >=5% and paired mean binary KL improves.',
        'intervention': 'Unchanged pre-question writer from Phase 2801, alpha=4, mapped to expected/alternative role by fixed answer order. Actual four-token greedy, equal-norm signed permutation, exact KV replay and rescue, all rows. Lockbox role orientation transferred without using new label activations to refit.',
        'gates': {'clean_accuracy': .8, 'alternate_gain': .1, 'over_equal_norm': .05, 'alpha_per_split': .025,
                  'replay_rescue': 'all rows, exact first logits and four generated tokens', 'matched_pre_rounding_tolerance': 1e-5},
        'power': power, 'scope': 'Controlled generated backgrounds, fixed-length instructions, one sample per document. The new words test transfer of the old writer without assuming the new words share its native direction.',
    })
    print('Frozen', RUN, len(rows), previous.file_sha(out/'prereg.json'), flush=True)


def collect(out, parts):
    import torch
    from mlp_factor_steer_edit import load_vectors, positions, prefill
    from mlp_factor_steer_holdout import norm_error
    from mlp_functional_trace import load_model
    from mlp_kv_mediation import metrics
    from mlp_read_route_causal import branch, changed
    from shared.dist_protocol.mechanism_trace import GpuLease

    prereg = previous.read(out/'prereg.json')
    frozen = previous.OUT/'frozen-recipe.json'
    for key, path in [('runner', __file__), ('inputs', out/'inputs.json.gz'), ('recipe', frozen), ('directions', previous.OUT/'directions.npz')]:
        assert previous.file_sha(path) == prereg[f'{key}_sha256']
    recipe = previous.read(frozen)['recipe']
    rows = previous.read_gz(out/'inputs.json.gz')['rows']
    torch.set_num_threads(4)
    with GpuLease(), torch.inference_mode():
        model, tok, eos = load_model()
        vectors = load_vectors('cuda')
        for part in parts:
            subset = [r for r in rows if r['split'] == part]
            t = time.monotonic()
            outcomes, native = [], []
            for lo in range(0, len(subset), 8):
                batch = subset[lo:lo+8]
                words = {w: sorted({tok.encode(form, add_special_tokens=False)[0] for form in (w, w.capitalize(), ' '+w, ' '+w.capitalize())})
                         for w in (batch[0]['expected'], batch[0]['contrast'])}
                assert not (set(words[batch[0]['expected']]) & set(words[batch[0]['contrast']]))
                x = torch.tensor([r['tokens'] for r in batch], device='cuda')
                clean = model(input_ids=x[:, :previous.CUT], use_cache=True, logits_to_keep=1).past_key_values
                native.append(clean.layers[35].values[:, :, -1, :].flatten(1).contiguous().view(torch.int16).cpu().numpy())
                conditions = {'clean': branch(model, x, clean, eos)}
                if part != 'discovery':
                    pos = positions(batch, 'cuda')
                    sign = torch.tensor([r['orientation'] for r in batch], device='cuda', dtype=torch.float32)
                    edited, norms = prefill(model, x, pos, sign, recipe, vectors, recipe['alpha'])
                    forward, restore, changes = changed(clean, edited)
                    conditions['targeted'] = branch(model, x, edited, eos)
                    conditions['replay'] = branch(model, x, clean, eos, forward)
                    conditions['rescue'] = branch(model, x, edited, eos, restore)
                    del edited, forward, restore
                    matched, control = prefill(model, x, pos, sign, recipe, vectors, recipe['alpha'], 'matched', norms)
                    conditions['matched'] = branch(model, x, matched, eos)
                    pre, post = norm_error(norms, control, len(batch))
                    del matched
                outcome = metrics(batch, conditions, tok, words)
                if part != 'discovery':
                    for j, row in enumerate(outcome):
                        row['replay_exact'] = bool(torch.equal(conditions['targeted'][0][j], conditions['replay'][0][j])) and conditions['targeted'][1][j] == conditions['replay'][1][j]
                        row['rescue_exact'] = bool(torch.equal(conditions['clean'][0][j], conditions['rescue'][0][j])) and conditions['clean'][1][j] == conditions['rescue'][1][j]
                        row['changed_slots'] = {f'{b}:{k}': diff[j].nonzero().flatten().tolist() for (b, k), diff in changes.items()}
                        row['norm_error'] = [float(pre[j].max()), float(post[j].max())]
                outcomes.extend(outcome)
                if (lo+len(batch)) % 32 == 0:
                    print(part, lo+len(batch), '/', len(subset), flush=True)
                del clean
            np.savez_compressed(out/f'{part}-native.npz', value_bits=np.concatenate(native))
            previous.write_gz(out/f'{part}-result.json.gz', {'rows': outcomes, 'seconds': time.monotonic()-t,
                'native_sha256': previous.file_sha(out/f'{part}-native.npz'), 'prereg_sha256': previous.file_sha(out/'prereg.json')})


def analyze(out):
    parts = {}
    for part in SIZES:
        data = previous.read_gz(out/f'{part}-result.json.gz')
        native = out/f'{part}-native.npz'
        assert previous.file_sha(native) == data['native_sha256']
        assert previous.file_sha(out/'prereg.json') == data['prereg_sha256']
        rows = data['rows']
        with np.load(native) as raw:
            value = (raw['value_bits'].astype(np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        direction = np.array([r['orientation'] for r in rows])
        features = value*direction[:, None]
        base = np.array([[1, r['orientation'], r['other_matches'], r['order'], r['display']] for r in rows], dtype=float)
        gap = np.array([r['conditions']['clean']['gap'] for r in rows])
        parts[part] = (rows, features, base, gap)
    _, x, base, y = parts['discovery']
    b = np.linalg.lstsq(base, y, rcond=None)[0]
    center = x.mean(0)
    scale = max(float(x.std()), 1e-8)
    x = (x-center)/scale
    weights = x.T @ np.linalg.solve(x @ x.T + 10*np.eye(len(x)), y-base@b)
    np.savez_compressed(out/'predictor.npz', center=center, scale=scale, baseline=b, residual_weights=weights)
    summary, row_output = {}, []
    for part, (rows, x, base, y) in parts.items():
        baseline = base@b
        predicted = baseline + (x-center)/scale@weights
        prob = 1/(1+np.exp(-np.clip(y, -50, 50)))
        kl = lambda v: float(np.mean(np.logaddexp(0, v)-np.logaddexp(0, y)-prob*(v-y)))
        rmse = lambda v: float(np.sqrt(np.mean((y-v)**2)))
        r0, r1 = rmse(baseline), rmse(predicted)
        summary[part] = s = {'n': len(rows), 'clean_accuracy': float(np.mean([r['conditions']['clean']['correct'] for r in rows])),
            'baseline_rmse': r0, 'prediction_rmse': r1, 'baseline_KL': kl(baseline), 'prediction_KL': kl(predicted),
            'prediction_gate': bool(part != 'discovery' and r1 <= .95*r0 and kl(predicted) < kl(baseline))}
        for j, row in enumerate(rows):
            row['prediction'] = {'baseline_gap': float(baseline[j]), 'predicted_gap': float(predicted[j]), 'actual_gap': float(y[j])}
        if part != 'discovery':
            alt = {k: np.array([r['conditions'][k]['word'] == r['contrast'] for r in rows]) for k in rows[0]['conditions']}
            a, c = alt['targeted'], alt['clean']
            gains, losses = int((a & ~c).sum()), int((~a & c).sum())
            p = binomtest(gains, gains+losses).pvalue if gains+losses else 1.
            s.update(alternates={k: int(v.sum()) for k, v in alt.items()}, gains=gains, losses=losses, p=p,
                     mean_KL={k: float(np.mean([r['conditions'][k]['KL'] for r in rows])) for k in alt},
                     mean_gap={k: float(np.mean([r['conditions'][k]['gap'] for r in rows])) for k in alt},
                     replay_exact=sum(r['replay_exact'] for r in rows), rescue_exact=sum(r['rescue_exact'] for r in rows),
                     max_norm_error=max(r['norm_error'][0] for r in rows))
            s['causal_gate'] = bool(s['clean_accuracy'] >= .8 and (gains-losses)/len(rows) >= .1 and p < .025 and
                (int(a.sum())-int(alt['matched'].sum()))/len(rows) >= .05 and s['max_norm_error'] <= 1e-5 and
                s['replay_exact'] == s['rescue_exact'] == len(rows))
        row_output.extend(rows)
    previous.save(out/'summary.json', {**previous.identity(), 'run_id': RUN, 'summary': summary,
                  'prereg_sha256': previous.file_sha(out/'prereg.json'), 'predictor_sha256': previous.file_sha(out/'predictor.npz')})
    previous.write_gz(out/'scored.json.gz', {'rows': row_output})
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'collect', 'analyze'])
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'freeze':
        freeze(args.out)
    elif args.stage == 'collect':
        collect(args.out, SIZES)
    else:
        analyze(args.out)
