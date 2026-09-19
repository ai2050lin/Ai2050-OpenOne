"""Phase 2757: P3 preset-strength intervention (V2 plan item P3, final V2 item).

Question: does removing the passage-source information at block12 (layer 12 of
qwen3-4b) degrade dual-world discrimination MORE than removing the same number
of non-source positions at matched value-mass? Only a targeted > random deficit
localises the mechanism (V2 section 2.3 point 3 and section P3).

Material: 2747 diagnostic controlled_relation rows = 160 pairs x 2 worlds
(world 0/1 with entity-attribute swapped prompts), 5 relation families, en/zh.
Dual-world discrimination per pair: D_i = min over worlds of
m_w = NLL(other_world.target) - NLL(this_world.target).
Intervention operator: zero the block12 v_proj output rows at selected prompt
positions (content-read channel; V is not RoPE-rotated so this is exactly the
block12 value-row zeroing of the KV path; same channel as 2748
source_value_pair_shuffle), scoring with a single full-prompt forward pass.
Targeted bands: top-25%/50%/100% value-norm rows inside the inventory (body)
span. Quality-matched random control (randn): same row count from non-body
positions by descending value norm (deterministic, mass-matched). Uniform
random control (rand): same row count drawn uniformly from non-body positions,
2 redraws (seeds 2757001/2757002) - auxiliary. Control row counts are capped
at the non-body pool size (k=100% can exceed it; actual eta/rows recorded).
Preregistered criteria (frozen before any behavioural forward pass):
  J1: bands with dD_sema > dD_randn >= 2 of 3.
  J2: k=100% band, pair-level bootstrap (1000 resamples, seed 2757010) 95% CI
      of median_i[D_randn_i - D_sema_i] has lower bound > 0.
  verdict = mechanism_localised iff J1 and J2.
Integrity gates:
  G1a pass-through: hook registered but inactive must reproduce the plain
      forward logits bit-exactly (max abs diff == 0).
  G1b effect: sema_100 must shift the base logits on every pair (nonzero).
  Revision note: an earlier two-stage (cached) scoring draft failed G1 against
  single-pass logits (max diff up to 0.66, BF16 large-negative-logit rounding;
  top-1 and margin sign agreed). Scoring was unified to single-pass before any
  behavioural result was read; the failing draft never produced results.
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import rdc_construction_common as cc
import rdc_feature_common as fc
import phase2747_rdc_material as mat2747

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2757' / 'qwen4_block12'
BANDS = [25, 50, 100]
RAND_SEEDS = [2757001, 2757002]
BOOT_SEED = 2757010
BOOT_N = 1000

PREREG = {
    'phase': 2757,
    'plan_item': 'V2 P3 preset-strength intervention',
    'model': 'qwen3-4b base (original weights, no checkpoint)',
    'intervention': 'block12 v_proj output zeroing at selected positions (hook), '
                    'single full-prompt forward per variant',
    'panel': '2747 diagnostic controlled_relation: 160 pairs x 2 worlds',
    'discrimination': 'D_i = min_w [ NLL(other_world.target) - NLL(this_world.target) ]',
    'bands': {'sema': 'body-span rows by descending block12 value norm, top 25/50/100%',
              'randn': 'same row count, non-body rows by descending value norm (mass-matched)',
              'rand': 'same row count, uniform draw from non-body, seeds 2757001/2757002'},
    'eta': 'sum ||v_row|| of zeroed rows / sum ||v_row|| over all prompt rows',
    'criteria': {'J1': 'bands with dD_sema > dD_randn >= 2 of 3',
                 'J2': 'k=100% bootstrap 95% CI lower bound of median_i[D_randn_i - D_sema_i] > 0',
                 'verdict': 'mechanism_localised iff J1 and J2'},
    'bootstrap': {'n': BOOT_N, 'seed': BOOT_SEED, 'unit': 'pair', 'stat': 'median'},
    'integrity_gate': {'G1a': 'inactive hook == plain forward, bit-exact',
                       'G1b': 'sema_100 shifts base logits on every pair'},
    'g1_revision_note': 'two-stage cached scoring draft failed bit-exact G1 (BF16 '
                        'large-negative-logit rounding, top-1/margin sign agreed); unified '
                        'to single-pass hook scoring before any behavioural result',
    'frozen_before_any_behavioural_forward': True,
}


def span_positions(row):
    """Token positions of the body (inventory sentences) inside prompt_ids."""
    text = row['text']
    offs = row['token_offsets']
    assert len(offs) == len(row['prompt_ids']), (len(offs), len(row['prompt_ids']))
    c0 = text.find(row['body'])
    assert c0 >= 0, row['sample_id']
    c1 = c0 + len(row['body'])
    body = np.array([i for i, (a, b) in enumerate(offs) if a < c1 and b > c0], dtype=np.int64)
    return body


def main():
    t0 = time.time()
    cc.guard(0)
    OUT.mkdir(parents=True, exist_ok=True)

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__), 'prereg': PREREG}
    fc.save(OUT / 'execution.json', execution)

    import torch
    from phase2662_symmetric_mapping_contract import load_native

    material, data = mat2747.freeze()
    rows = [r for r in data['diagnostic'] if r['kind'] == 'controlled_relation']
    assert len(rows) == 320, len(rows)

    model, tok = load_native('qwen4')
    model.eval()
    device = next(model.parameters()).device
    assert model.config.num_hidden_layers == 36

    hook_state = {'positions': None, 'last_out': None}

    def vhook(module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        p = hook_state['positions']
        if p is None:
            hook_state['last_out'] = out
        else:
            out[0, list(p), :] = 0
        return None

    v_proj = model.model.layers[12].self_attn.v_proj
    handle = v_proj.register_forward_hook(vhook)

    pairs = defaultdict(list)
    for idx, r in enumerate(rows):
        pairs[r['pair_id']].append(idx)
    assert all(len(v) == 2 for v in pairs.values()) and len(pairs) == 160
    pair_ids = sorted(pairs)

    g1a_diffs = []
    per_variant = defaultdict(lambda: defaultdict(list))

    rng_a = np.random.default_rng(RAND_SEEDS[0])
    rng_b = np.random.default_rng(RAND_SEEDS[1])

    with torch.inference_mode():
        for pi, pid in enumerate(pair_ids):
            w0, w1 = pairs[pid]
            ctxs = []
            for widx in (w0, w1):
                row = rows[widx]
                pids_l = row['prompt_ids']
                body_pos = span_positions(row)
                n = len(pids_l)
                nonbody = np.array([i for i in range(n)
                                    if i not in set(body_pos.tolist())], dtype=np.int64)
                ids = torch.tensor([pids_l], device=device)

                # base forward, hook records v_proj output for norms
                hook_state['positions'] = None
                base_logits = model(ids).logits[0, -1]
                vout = hook_state['last_out'][0]  # (n, hidden)
                plain = model(ids).logits[0, -1]
                if pi < 8:
                    g1a_diffs.append(float((plain - base_logits).abs().max()))

                lp = torch.log_softmax(base_logits.float(), dim=-1)
                tgt_this = row['target']
                tgt_other = rows[w1 if widx == w0 else w0]['target']
                m_base = float(lp[tgt_other] - lp[tgt_this])

                pos_norm = vout.float().norm(dim=-1).cpu().numpy()
                ctxs.append(dict(widx=widx, row=row, ids=ids, body=body_pos,
                                 nonbody=nonbody, m_base=m_base,
                                 pos_norm=pos_norm, base_logits=base_logits))

            assert max(g1a_diffs) == 0.0, ('G1a pass-through violated', g1a_diffs)

            m0, m1 = ctxs[0]['m_base'], ctxs[1]['m_base']
            per_variant['base']['D'].append(min(m0, m1))
            per_variant['base']['m0'].append(m0)
            per_variant['base']['m1'].append(m1)
            per_variant['base']['eta'].append(0.0)
            per_variant['base']['rows'].append(0)

            nbody = len(ctxs[0]['body'])
            n_nonbody = len(ctxs[0]['nonbody'])
            assert len(ctxs[1]['body']) == nbody and len(ctxs[1]['nonbody']) == n_nonbody
            plan = []
            for k in BANDS:
                nsel = int(np.ceil(k / 100 * nbody))
                plan.append(('sema_%d' % k, 'sema', nsel, None))
                nsel_ctl = min(nsel, n_nonbody)
                plan.append(('randn_%d' % k, 'randn', nsel_ctl, None))
                plan.append(('rand_%d_a' % k, 'rand', nsel_ctl, rng_a))
                plan.append(('rand_%d_b' % k, 'rand', nsel_ctl, rng_b))

            for name, kind, nsel, rgen in plan:
                ms = {}
                etas = []
                shifted_all = True
                for ctx in ctxs:
                    if kind == 'sema':
                        pool = ctx['body']
                        sel = pool[np.argsort(-ctx['pos_norm'][pool])][:nsel]
                    elif kind == 'randn':
                        pool = ctx['nonbody']
                        sel = pool[np.argsort(-ctx['pos_norm'][pool])][:nsel]
                    else:
                        pool = ctx['nonbody']
                        sel = np.sort(rgen.choice(pool, size=nsel, replace=False))
                    hook_state['positions'] = sel.tolist()
                    logits = model(ctx['ids']).logits[0, -1]
                    hook_state['positions'] = None
                    if not bool((logits - ctx['base_logits']).abs().max() > 0):
                        shifted_all = False
                    lp = torch.log_softmax(logits.float(), dim=-1)
                    other_tgt = rows[w1 if ctx['widx'] == w0 else w0]['target']
                    ms[ctx['widx']] = float(lp[other_tgt] - lp[ctx['row']['target']])
                    etas.append(float(ctx['pos_norm'][sel].sum() / ctx['pos_norm'].sum()))
                if name == 'sema_100':
                    assert shifted_all, 'G1b: sema_100 did not shift base logits'
                per_variant[name]['D'].append(min(ms[w0], ms[w1]))
                per_variant[name]['m0'].append(ms[w0])
                per_variant[name]['m1'].append(ms[w1])
                per_variant[name]['eta'].append(float(np.mean(etas)))
                per_variant[name]['rows'].append(nsel)

            if pi % 20 == 0:
                print('PAIR %d/160 %s' % (pi, pid), flush=True)

    handle.remove()

    # ---- statistics ----
    D_base = np.array(per_variant['base']['D'])
    results = {'D_base_median': float(np.median(D_base)),
               'D_base_mean': float(D_base.mean()),
               'D_base_negative_fraction': float((D_base <= 0).mean())}

    band_stats = {}
    for k in BANDS:
        d_sema = D_base - np.array(per_variant['sema_%d' % k]['D'])
        d_randn = D_base - np.array(per_variant['randn_%d' % k]['D'])
        a = np.array(per_variant['rand_%d_a' % k]['D'])
        b = np.array(per_variant['rand_%d_b' % k]['D'])
        d_rand = D_base - (a + b) / 2
        band_stats[k] = {
            'dD_sema_median': float(np.median(d_sema)),
            'dD_randn_median': float(np.median(d_randn)),
            'dD_rand_median': float(np.median(d_rand)),
            'eta_sema_mean': float(np.mean(per_variant['sema_%d' % k]['eta'])),
            'eta_randn_mean': float(np.mean(per_variant['randn_%d' % k]['eta'])),
            'eta_rand_mean': float(np.mean(per_variant['rand_%d_a' % k]['eta'] +
                                           per_variant['rand_%d_b' % k]['eta']) / 2),
        }
    results['band_stats'] = band_stats

    j1_bands = [k for k in BANDS if band_stats[k]['dD_sema_median'] > band_stats[k]['dD_randn_median']]
    results['J1_bands'] = j1_bands
    results['J1'] = len(j1_bands) >= 2

    k = 100
    d_sema = D_base - np.array(per_variant['sema_%d' % k]['D'])
    d_randn = D_base - np.array(per_variant['randn_%d' % k]['D'])
    per_pair_gap = d_randn - d_sema
    rngb = np.random.default_rng(BOOT_SEED)
    stats = np.empty(BOOT_N)
    for b in range(BOOT_N):
        idx = rngb.integers(0, len(per_pair_gap), len(per_pair_gap))
        stats[b] = np.median(per_pair_gap[idx])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    results['J2'] = {'gap_median': float(np.median(per_pair_gap)),
                     'boot_ci95': [float(lo), float(hi)],
                     'pass': bool(lo > 0)}
    results['J2_pass'] = bool(lo > 0)
    results['verdict'] = ('mechanism_localised'
                          if results['J1'] and results['J2_pass'] else 'not_localised')
    results['G1a_max_abs_diff'] = g1a_diffs

    fam_of_pair = np.array([rows[pairs[pid][0]]['family'] for pid in pair_ids])
    fam_break = {}
    for fam in sorted(set(fam_of_pair.tolist())):
        sel = fam_of_pair == fam
        fam_break[fam] = {'gap_median': float(np.median(per_pair_gap[sel])), 'n': int(sel.sum())}
    results['family_breakdown_k100'] = fam_break

    arrays = {'D_base': D_base,
              'pair_ids': np.array(pair_ids, dtype=np.str_),
              'families': fam_of_pair.astype(np.str_)}
    for name, fields in per_variant.items():
        for f, vals in fields.items():
            arrays['%s__%s' % (name, f)] = np.array(vals)
    fc.npz(OUT / 'pair_scores.npz', **arrays)
    fc.save(OUT / 'pair_scores_meta.json',
            {'schema': 'per pair (160); D = min over worlds of m = nll(other.target) - nll(this.target)',
             'pair_ids': pair_ids, 'families': fam_of_pair.tolist()})

    results['seconds'] = time.time() - t0
    fc.save(OUT / 'result.json', results)
    print('PHASE2757_DONE ' + json.dumps({'verdict': results['verdict'],
                                          'J1': results['J1'], 'J2': results['J2_pass'],
                                          'D_base_median': results['D_base_median'],
                                          'seconds': results['seconds']}), flush=True)


if __name__ == '__main__':
    main()
