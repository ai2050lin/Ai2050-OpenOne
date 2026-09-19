"""Frozen conditional native-coordinate operator comparison; local prediction is not full language closure."""
import gc
from collections import Counter
from rdc_operator_common import *
from rdc_native_conditional_operator import weights, metadata, load_block, fit_bank, save_bank, apply_operator, exact_decomposition, NAMES


def main():
    import torch
    assert (BASE / 'capture/main/result.json').exists()
    out = BASE / 'operators'
    if (out / 'frozen.json').exists():
        print('OPERATORS_ALREADY_FROZEN', flush=True)
        return
    torch.set_num_threads(2)
    start = time.monotonic()
    selected = [r for r in rows() if r['split'] != 'confirmation']
    meta = metadata(selected)
    train = [r for r in meta if r['split'] == 'train']
    train_tokens = {r['token_id'] for r in train}
    for r in meta:
        r['token_seen_in_train_anchors'] = r['token_id'] in train_tokens
    compressed(out / 'row_index.json.gz', meta)
    immutable(out / 'protocol.json', {'timestamp': stamp(), 'source': snapshot(Path(__file__)), 'names': NAMES,
        'blocks_zero_index': [6,16,34], 'train_anchor_rows': len(train), 'validation_anchor_rows': sum(r['split']=='validation' for r in meta),
        'test_anchor_rows': sum(r['split']=='test' for r in meta), 'available_input': 'Actual current post-attention RMS-normalized full2560 x, token ID/class, absolute position bin and causal-prefix lexical cues. No current target MLP output, future token or QA answer is supplied.',
        'scope': 'Local MLP-output forecast, not early-layer or next-token forecast until separately compiled. Uses native full matrices in factored form; not a compressed replacement model.',
        'prototypes': 'Train anchor means, global/piece/cue/identity/position;32-observation shrinkage toward global, rare token IDs(<8train anchors) global fallback. Same train rows for every candidate.',
        'selection': 'Per block, minimize validation mean per-token relative coordinate MSE; all11candidates reported on test. Confirmation capture absent until freeze.',
        'negative_controls': 'Global and position conditions, constant output and per-coordinate affine; group-specific parameter counts disclosed. Extra condition groups are not equal-capacity semantic evidence.',
        'arithmetic_oracle': 'Original BF16 weights, FP32 evaluation of SiLU(Wg x)*(Wu x) and Wd; numerical floor separate from learned operators.',
        'decomposition': 'Four exact finite product terms with TRAIN mean phi/up: shared, value variation, gate variation, gate-value interaction. Full2560 writes and9728 units retained; cross terms evaluated, not ignored.'})
    reports, choices, examples = [], {}, []
    for block in (6,16,34):
        data = load_block(selected, block)
        bank, capacity = fit_bank(data, meta)
        save_bank(out / f'L{block}_bank', bank)
        save(out / f'L{block}_capacity.json', capacity)
        w = weights(block)
        evalix = [i for i,r in enumerate(meta) if r['split'] != 'train']
        evalmeta = [meta[i] for i in evalix]
        preds = {name: [] for name in NAMES}
        oracle, decomposed, terms_energy, cross_energy = [], [], [], []
        identities = []
        unit_sums = np.zeros((4,9728), dtype=np.float64)
        unit_sq = np.zeros_like(unit_sums)
        for at in range(0, len(evalix), 64):
            ix = evalix[at:at+64]
            mm = [meta[i] for i in ix]
            x = torch.as_tensor(data['x'][ix], device='cuda:0')
            with torch.inference_mode():
                for name in NAMES:
                    preds[name].append(apply_operator(name, x, mm, bank, w).cpu().numpy())
                write, reference, terms = exact_decomposition(x, mm, bank, w)
                error = float((write.sum(1)-reference).square().mean()/reference.square().mean().clamp_min(1e-20))
                identities.append(error)
                oracle.append(reference.cpu().numpy())
                decomposed.append(write.cpu().numpy())
                terms_energy.append(write.square().mean(-1).cpu().numpy())
                cross_energy.append((2 * (write.sum(1).square().mean(-1)-write.square().sum(1).mean(-1))/2).cpu().numpy())
                unit = torch.stack(terms, 1).cpu().numpy().astype(np.float64)
                unit_sums += unit.sum(0)
                unit_sq += (unit*unit).sum(0)
            del x, write, reference, terms, unit
        target = data['mlp'][evalix]
        fullpred = {name: np.concatenate(v) for name,v in preds.items()}
        fullpred['FP32_native_oracle'] = np.concatenate(oracle)
        npz(out / f'L{block}_predictions.npz', **fullpred, native=target, decomposition=np.concatenate(decomposed))
        npz(out / f'L{block}_full_unit_decomposition.npz', mean=unit_sums/len(evalix), mean_square=unit_sq/len(evalix),
            term_energy=np.concatenate(terms_energy), all_cross_terms_sum=np.concatenate(cross_energy))
        for name, pred in fullpred.items():
            error = np.mean((pred.astype(np.float64)-target)**2, 1)
            energy = np.mean(target.astype(np.float64)**2, 1)
            rel = error / np.maximum(energy,1e-20)
            cos = np.sum(pred.astype(np.float64)*target,1) / np.maximum(np.linalg.norm(pred,axis=1)*np.linalg.norm(target,axis=1),1e-20)
            for split in ('validation','test'):
              for stratum in ('all','ordinary','event','unseen_token'):
                ii = [i for i,r in enumerate(evalmeta) if r['split']==split and (stratum=='all' or stratum=='ordinary' and not r['event'] or stratum=='event' and r['event'] or stratum=='unseen_token' and not r['token_seen_in_train_anchors'])]
                if not ii:
                    continue
                rr = {'block': block, 'name': name, 'split': split, 'stratum': stratum, 'rows': len(ii),
                      'raw_MSE': float(error[ii].mean()), 'relative_MSE': float(rel[ii].mean()), 'median_relative_MSE': float(np.median(rel[ii])),
                      'cosine_mean': float(cos[ii].mean()), 'relative_MSE_cluster': clustered(rel[ii], [evalmeta[i]['source_group'] for i in ii])}
                reports.append(rr)
        candidates = [r for r in reports if r['block']==block and r['split']=='validation' and r['stratum']=='all' and r['name'] in NAMES]
        chosen = min(candidates, key=lambda r:(r['relative_MSE'], r['name']))['name']
        choices[str(block)] = chosen
        examples.append({'block': block, 'all_finite': all(np.isfinite(p).all() for p in fullpred.values()),
            'exact_four_term_FP32_max_relative_discrepancy': max(identities), 'learned_parameter_counts': capacity,
            'selection': chosen, 'full_rows': len(evalix), 'confirmation_values_not_loaded': True})
        assert examples[-1]['all_finite'] and max(identities) < 1e-10
        print('OPERATOR_FIT_BLOCK', block, chosen, 'test', [r for r in reports if r['block']==block and r['split']=='test' and r['stratum']=='all'], flush=True)
        del w, data, bank, preds, fullpred, decomposed, oracle
        gc.collect()
        torch.cuda.empty_cache()
        guard()
    save(out / 'result.json', {'timestamp': stamp(), 'reports': reports, 'checks': examples,
        'limits': 'Native formula identities are not newly discovered semantics. Conditions are observable piece/cue labels with unequal group counts; forecast is local at supplied current x. No full-language closure, cross-model equivalence or universal discrete pulse is inferred.'})
    immutable(out / 'frozen.json', {'timestamp': stamp(), 'choices': choices, 'bank_files': {p.name: sha(p) for p in sorted(out.glob('*bank.*'))},
        'main_material_sha': sha(BASE/'material.json.gz'), 'fit_result_sha': sha(out/'result.json'),
        'confirmation_capture_exists': (BASE/'capture/confirmation/result.json').exists(), 'rule': 'Validation per-token relative coordinate MSE among11predeclared candidates; separate global baseline kept regardless of winner.'})
    ledger('native_conditional_operator_fit', time.monotonic()-start, anchors=len(meta))


if __name__ == '__main__':
    main()
