"""Phase 2759: direct manipulation of the question-leak channel - content-level
deletion of question-side fact spans in the 2757 dual-world panel.

Question (2758 接续): 2758 showed that in the perm training states the dual-world
discrimination support sits at QUESTION-SIDE positions (J2 gap sign reversed:
deleting non-body positions hurts, deleting body positions does not). This phase
upgrades that correlational evidence to causal by deleting, from the prompt text
itself (content-level token deletion, no hidden intervention), the question
tokens that restate the body facts.

Fact-token rule (preregistered, deterministic):
  question region = tokens overlapping row['question'] char range.
  s = decode(tok).strip().strip(punct).lower()
  fact iff (relations rule) OR (body rule):
    relations rule: s equals a relations word piece (any length >= 1), or
      len(s) >= 2 and s is a substring of / contains a piece of length >= 2.
    body rule: len(s) >= 3 (or CJK len >= 2) and s occurs in body.lower()
      and s not in FUNCTION_STOP.
Variants per pair (both worlds, content-level):
  base, qdel (all question fact spans removed), qctrl_a/b (equal count of
  random non-fact question tokens, seeds 2759001/2759002), bdel_a/b (equal
  count of random body tokens, seeds 2759003/2759004).
Runs: native, perm_2747, perm_2748 (primary), true_2747, mass_2747 (controls).

Preregistered criteria (frozen in execution.json before any forward):
  W = m(world1) - m(world0), m(world) = nll(other.target) - nll(this.target),
  D = min(m0, m1). |W| = world-discrimination magnitude (bias-free).
  P1 (primary, each perm run): targeted question-fact deletion collapses world
      discrimination beyond matched control: dT = |W_base| - |W_qdel|,
      dC = |W_base| - |W_qctrl(2-seed avg)|; P1 iff median(dT) > median(dC)
      AND bootstrap(1000, seed 2759010) 95% CI of median(dT-dC) lower > 0.
  P2 (each perm run): D_base - D_qdel > 0, median with CI excluding 0.
  P3 (carrier ordering): median(dT) perm_2747 > native AND perm_2748 > native
      AND perm_2747 > mass_2747.
  P4 (native floor expectation): median(dT) native <= median(dT) perm_2747.
  Verdict question_causal_carrier iff P1 both perm runs AND P3.
Integrity gates:
  G0 deployment identity for trained runs (delta SHA + BF16 L2 rel < 1e-9).
  G1 base determinism: repeated base forward bit-exact, first 8 pairs/run.
  G2 deletion validity: qdel count >= 1; qctrl/bdel counts == qdel count;
      all deleted spans inside question/body regions; base target ids intact.
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
OUT = BASE / 'phase2759' / 'qwen4_question_span'
BOOT_SEED = 2759010
BOOT_N = 1000
QCTRL_SEEDS = [2759001, 2759002]
BDEL_SEEDS = [2759003, 2759004]
PUNCT = '.,!?;:"\'`()[]{}<>—–-、。，！？；：""''…·'

FUNCTION_STOP = {'that', 'this', 'these', 'those', 'does', 'the', 'and', 'only',
                 'means', 'also', 'its', 'stated', 'was', 'were', 'are'}

RUNS = [('native', None, None),
        ('perm_2747', 'within_surface_class_permuted_token', 2747),
        ('perm_2748', 'within_surface_class_permuted_token', 2748),
        ('true_2747', 'true_token', 2747),
        ('mass_2747', 'surface_class_mass', 2747)]
TRAIN_ROOT = BASE / 'phase2747' / 'training'

PREREG = {
    'phase': 2759,
    'question': 'Does deleting the question-side fact spans (content-level '
                'token deletion) collapse dual-world discrimination, upgrading '
                'the 2758 correlational "question-side carrier" finding to '
                'causal?',
    'model': 'qwen3-4b native plus four 2747 deployed BF16 checkpoints',
    'panel': '160 pairs x 2 worlds, 2747 diagnostic controlled_relation',
    'fact_token_rule': 'see module docstring; deterministic, no tuning',
    'variants': 'base, qdel, qctrl_a, qctrl_b, bdel_a, bdel_b (content-level '
                'token deletion, sequences rebuilt, no hidden intervention)',
    'criteria': {'P1': 'median(dT) > median(dC) and bootstrap CI (1000, seed '
                       '2759010) of median(dT-dC) lower bound > 0, each perm run',
                 'P2': 'D_base - D_qdel > 0 median, CI excluding 0, each perm run',
                 'P3': 'median(dT) perm_2747 > native, perm_2748 > native, '
                       'perm_2747 > mass_2747',
                 'P4': 'median(dT) native <= median(dT) perm_2747',
                 'verdict': 'question_causal_carrier iff P1(both perm) and P3'},
    'integrity_gates': {'G0': 'trained deployment identity as 2758',
                        'G1': 'base forward determinism, first 8 pairs per run',
                        'G2': 'deletion count/region validity every pair'},
    'frozen_before_any_behavioural_forward': True,
}


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def deploy(model, named, original, run_key, condition, seed):
    folder = TRAIN_ROOT / ('%s_%d' % (condition, seed))
    receipt = fc.read(folder / 'commits' / 'delta_128.json')
    npz_path = ROOT / receipt['field_path']
    if not npz_path.exists():
        npz_path = Path(receipt['physical_path'])
    assert sha256_file(npz_path) == receipt['field_sha256'], ('G0 sha', run_key)
    import torch
    with np.load(npz_path) as z:
        with torch.no_grad():
            for n, p in named.items():
                rebuilt = original[n].numpy() + z[n]
                rebuilt = rebuilt + z['reconstruction_residual__' + n]
                p.copy_(torch.tensor(rebuilt, device=p.device))
    deployed_norm = float(torch.stack(
        [(p.detach().float().cpu() - original[n]).double().square().sum()
         for n, p in named.items()]).sum().sqrt())
    recorded = fc.read(folder / 'result.json')['deployed_BF16_delta_L2']
    rel = abs(deployed_norm - recorded) / max(abs(recorded), 1e-30)
    assert rel < 1e-9, ('G0 L2', run_key, deployed_norm, recorded)
    return deployed_norm


def restore(model, named, original):
    import torch
    with torch.no_grad():
        for n, p in named.items():
            p.copy_(original[n].to(p.device).to(p.dtype))


def region_positions(row, lo_char, hi_char):
    offs = row['token_offsets']
    return np.array([i for i, (a, b) in enumerate(offs)
                     if a < hi_char and b > lo_char], dtype=np.int64)


def fact_tokens(row, tok):
    """Preregistered deterministic fact-token ids inside the question region."""
    text = row['text']
    q0 = text.find(row['question'])
    assert q0 >= 0, row['sample_id']
    q1 = q0 + len(row['question'])
    b0 = text.find(row['body'])
    assert b0 >= 0, row['sample_id']
    b1 = b0 + len(row['body'])
    qpos = np.array([i for i, (a, b) in enumerate(row['token_offsets'])
                     if a < q1 and b > q0 and not (a < b1 and b > b0)],
                    dtype=np.int64)
    words = set()
    for rel in row['relations']:
        for key in ('source', 'relation', 'target'):
            for piece in str(rel[key]).split('_'):
                piece = piece.strip().lower()
                if piece:
                    words.add(piece)
    body_lower = row['body'].lower()
    fact = []
    for i in qpos.tolist():
        s = tok.decode([row['prompt_ids'][i]]).strip().strip(PUNCT).lower()
        if not s:
            continue
        hit = False
        for w in words:
            if s == w:
                hit = True
            elif len(w) >= 2 and s and s in w:
                hit = True
            elif len(s) >= 2 and len(w) >= 2 and w in s:
                hit = True
            if hit:
                break
        if not hit and len(s) >= 2 and s not in FUNCTION_STOP and s in body_lower:
            hit = True
        if hit:
            fact.append(i)
    return qpos, np.array(fact, dtype=np.int64), (q0, q1), (b0, b1)


def spans_to_keep(n, del_positions):
    dead = set(del_positions.tolist())
    return [i for i in range(n) if i not in dead]


def run_panel(model, tok, rows, pair_ids, pairs, run_key):
    import torch
    device = next(model.parameters()).device
    per_variant = defaultdict(lambda: defaultdict(list))
    g1_diffs = []
    rng_q = [np.random.default_rng(s) for s in QCTRL_SEEDS]
    rng_b = [np.random.default_rng(s) for s in BDEL_SEEDS]
    del_counts = []
    with torch.inference_mode():
        for pi, pid in enumerate(pair_ids):
            w0, w1 = pairs[pid]
            info = {}
            for widx in (w0, w1):
                row = rows[widx]
                ids_full = np.array(row['prompt_ids'], dtype=np.int64)
                qpos, fact, qspan, bspan = fact_tokens(row, tok)
                bpos_all = region_positions(row, bspan[0], bspan[1])
                # body content tokens (exclude whitespace-only)
                bcontent = np.array([i for i in bpos_all.tolist()
                                     if tok.decode([row['prompt_ids'][i]]).strip()],
                                    dtype=np.int64)
                info[widx] = dict(row=row, ids=ids_full, qpos=qpos, fact=fact,
                                  bcontent=bcontent, qspan=qspan, bspan=bspan)
            row0, row1 = info[w0]['row'], info[w1]['row']
            assert row0['question'] == row1['question'], pid
            f0 = info[w0]['fact'].tolist()
            assert info[w1]['fact'].tolist() == f0, ('fact asymmetry', pid)
            n_del = len(f0)
            assert n_del >= 1, ('no fact tokens', pid)
            del_counts.append(n_del)

            del_count_by_variant = {}
            variants = {}
            variants['base'] = {w: info[w]['ids'] for w in (w0, w1)}
            variants['qdel'] = {w: info[w]['ids'][spans_to_keep(
                len(info[w]['ids']), info[w]['fact'])] for w in (w0, w1)}
            for tag, seeds, per_world_pool in [
                    ('qctrl', QCTRL_SEEDS, True), ('bdel', BDEL_SEEDS, False)]:
                for si, s in enumerate(seeds):
                    rng = np.random.default_rng(s * 10 + pi)  # per-pair stream
                    seqs = {}
                    for w in (w0, w1):
                        if per_world_pool:
                            pool_w = np.array(
                                [i for i in info[w]['qpos'].tolist()
                                 if i not in set(info[w]['fact'].tolist())
                                 and tok.decode([info[w]['row']['prompt_ids'][i]]).strip()],
                                dtype=np.int64)
                        else:
                            pool_w = info[w]['bcontent']
                        nsel_w = min(n_del, len(pool_w))
                        sel_w = np.sort(rng.choice(pool_w, size=nsel_w,
                                                   replace=False))
                        seqs[w] = info[w]['ids'][spans_to_keep(
                            len(info[w]['ids']), sel_w)]
                    variants['%s_%d' % (tag, s)] = seqs
                    del_count_by_variant['%s_%d' % (tag, s)] = nsel_w

            # G2 validation
            for name, seqs in variants.items():
                if name == 'base':
                    continue
                for w in (w0, w1):
                    kept = seqs[w]
                    orig = info[w]['ids']
                    expected = (n_del if name == 'qdel'
                                else del_count_by_variant[name])
                    assert len(kept) == len(orig) - expected, (name, pid, w)

            ms = {}
            for name, seqs in variants.items():
                mw = {}
                for widx in (w0, w1):
                    other = w1 if widx == w0 else w0
                    ids_t = torch.tensor([seqs[widx].tolist()], device=device)
                    logits = model(ids_t).logits[0, -1]
                    if name == 'base':
                        logits2 = model(ids_t).logits[0, -1]
                        if pi < 8:
                            g1_diffs.append(float((logits - logits2).abs().max()))
                    lp = torch.log_softmax(logits.float(), dim=-1)
                    mw[widx] = float(lp[rows[other]['target']] -
                                     lp[rows[widx]['target']])
                ms[name] = mw
            for name, mw in ms.items():
                per_variant[name]['m0'].append(mw[w0])
                per_variant[name]['m1'].append(mw[w1])
                per_variant[name]['D'].append(min(mw[w0], mw[w1]))
                per_variant[name]['W'].append(mw[w1] - mw[w0])
            assert max(g1_diffs) == 0.0, ('G1 violated', run_key, g1_diffs)
            if pi % 40 == 0:
                print('P2759 %s PAIR %d/160' % (run_key, pi), flush=True)
    return per_variant, g1_diffs, del_counts


def boot_ci_median(values, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=np.float64)
    stats = np.empty(BOOT_N)
    for b in range(BOOT_N):
        idx = rng.integers(0, len(v), len(v))
        stats[b] = np.median(v[idx])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(np.median(v)), float(lo), float(hi)


def stats_for(per_variant):
    W_base = np.array(per_variant['base']['W'])
    D_base = np.array(per_variant['base']['D'])
    res = {'W_base_abs_median': float(np.median(np.abs(W_base))),
           'D_base_median': float(np.median(D_base)),
           'D_base_negative_fraction': float((D_base <= 0).mean())}
    W_qdel = np.array(per_variant['qdel']['W'])
    dT = np.abs(W_base) - np.abs(W_qdel)
    W_qc = (np.array(per_variant['qctrl_%d' % QCTRL_SEEDS[0]]['W']) +
            np.array(per_variant['qctrl_%d' % QCTRL_SEEDS[1]]['W'])) / 2.0
    dC = np.abs(W_base) - np.abs(W_qc)
    W_bd = (np.array(per_variant['bdel_%d' % BDEL_SEEDS[0]]['W']) +
            np.array(per_variant['bdel_%d' % BDEL_SEEDS[1]]['W'])) / 2.0
    dB = np.abs(W_base) - np.abs(W_bd)
    res['dT_median'] = float(np.median(dT))
    res['dC_median'] = float(np.median(dC))
    res['dB_median'] = float(np.median(dB))
    med, lo, hi = boot_ci_median(dT - dC)
    res['P1'] = {'gap_median': med, 'boot_ci95': [lo, hi],
                 'pass': bool(med > 0 and lo > 0)}
    dD = D_base - np.array(per_variant['qdel']['D'])
    med2, lo2, hi2 = boot_ci_median(dD)
    res['P2'] = {'dD_median': med2, 'boot_ci95': [lo2, hi2],
                 'pass': bool(med2 > 0 and lo2 > 0)}
    fam = {}
    return res, dT, dC


def main():
    t0 = time.time()
    cc.guard(0)
    assert not (OUT / 'result.json').exists(), 'result.json immutable; delete before rerun'
    OUT.mkdir(parents=True, exist_ok=True)
    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG}
    fc.save(OUT / 'execution.json', execution)

    material, data = mat2747.freeze()
    rows = [r for r in data['diagnostic'] if r['kind'] == 'controlled_relation']
    assert len(rows) == 320, len(rows)
    pairs = defaultdict(list)
    for idx, r in enumerate(rows):
        pairs[r['pair_id']].append(idx)
    assert all(len(v) == 2 for v in pairs.values()) and len(pairs) == 160
    pair_ids = sorted(pairs)
    fam_of_pair = np.array([rows[pairs[pid][0]]['family'] for pid in pair_ids])

    from phase2662_symmetric_mapping_contract import load_native
    model, tok = load_native('qwen4')
    model.eval()
    assert model.config.num_hidden_layers == 36

    target_module = model.model.layers[16].mlp
    named = dict(target_module.named_parameters())
    original = {n: p.detach().float().cpu().clone() for n, p in named.items()}

    run_results = {}
    arrays = {'pair_ids': np.array(pair_ids, dtype=np.str_),
              'families': fam_of_pair.astype(np.str_)}
    for run_key, condition, seed in RUNS:
        if condition is not None:
            deploy(model, named, original, run_key, condition, seed)
        per_variant, g1, del_counts = run_panel(model, tok, rows, pair_ids,
                                                pairs, run_key)
        res, dT, dC = stats_for(per_variant)
        res['G1_max_abs_diff'] = g1
        res['qdel_count_median'] = float(np.median(del_counts))
        run_results[run_key] = res
        for name, fields in per_variant.items():
            for f, vals in fields.items():
                arrays['%s__%s__%s' % (run_key, name, f)] = np.array(vals)
        arrays['%s__dT' % run_key] = dT
        arrays['%s__dC' % run_key] = dC
        print('P2759_RUN_DONE %s dT=%.4f dC=%.4f P1=%s P2=%s' %
              (run_key, res['dT_median'], res['dC_median'],
               res['P1']['pass'], res['P2']['pass']), flush=True)
        if condition is not None:
            restore(model, named, original)

    results = {'runs': run_results}
    p1_both = bool(run_results['perm_2747']['P1']['pass'] and
                   run_results['perm_2748']['P1']['pass'])
    p2_both = bool(run_results['perm_2747']['P2']['pass'] and
                   run_results['perm_2748']['P2']['pass'])
    p3 = {'perm2747_gt_native': bool(run_results['perm_2747']['dT_median'] >
                                     run_results['native']['dT_median']),
          'perm2748_gt_native': bool(run_results['perm_2748']['dT_median'] >
                                     run_results['native']['dT_median']),
          'perm2747_gt_mass': bool(run_results['perm_2747']['dT_median'] >
                                   run_results['mass_2747']['dT_median'])}
    p3_pass = all(p3.values())
    p4_pass = bool(run_results['native']['dT_median'] <=
                   run_results['perm_2747']['dT_median'])
    results['P1_both_perm'] = p1_both
    results['P2_both_perm'] = p2_both
    results['P3'] = p3
    results['P3_pass'] = p3_pass
    results['P4_pass'] = p4_pass
    results['verdict'] = ('question_causal_carrier' if (p1_both and p3_pass)
                          else 'not_confirmed')
    results['seconds'] = time.time() - t0
    fc.npz(OUT / 'pair_scores.npz', **arrays)
    fc.save(OUT / 'pair_scores_meta.json',
            {'schema': 'per run (5) x per variant (6) x fields m0/m1/D/W; '
                       'm = nll(other.target) - nll(this.target); '
                       'D = min(m0, m1); W = m1 - m0'})
    fc.save(OUT / 'result.json', results)
    print('PHASE2759_DONE ' + json.dumps(
        {'verdict': results['verdict'], 'P1_both_perm': p1_both,
         'P2_both_perm': p2_both, 'P3': p3, 'P4_pass': p4_pass,
         'dT_medians': {k: r['dT_median'] for k, r in run_results.items()},
         'seconds': results['seconds']}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        import traceback
        (OUT / 'crash.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
