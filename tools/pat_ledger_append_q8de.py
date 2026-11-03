# -*- coding: utf-8 -*-
"""追加 Q8-D3 + Q8-E 两条测量 + negative（幂等）。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'research' / 'gpt5' / 'atlas' / 'atlas_ledger.json'

E_D3 = {
    'phase': 'Q8-D3', 'name': 'pat_q8d3', 'line': 'G', 'date': '2026-11-01',
    'verdict': 'pos0_register_is_distribution hotspot_but_carries_no_frequency_knob_weak_inverse_coupling_both_models',
    'res_sha8': '4a59421f',
    'res_sha8_glm': 'ce5d7dbb',
    'prev_post': None,
    'note': (
        'Closes the Q8-D open item: the frequency direction was tested at LAST (content) and '
        'ALLPOS but never at pos0, the format/register position where Q5-A found 46% of the '
        'next-token competition. Chat-templated prompts (pos0 = <|im_start|> for Qwen, [gMASK] '
        'for GLM), same w_freq and dose calibration as Q8-D2, template gate re-evaluated on '
        'baseline only (15/36 Qwen, 24/36 GLM), 200 prompts, n_forward 25/77. '
        'MAIN RESULT D3-1 WEAK_COUPLING IN BOTH MODELS with the SAME INVERSE SIGN: pushing the '
        'format token toward high frequency makes the output slightly LOWER frequency. Qwen '
        'rho_dose=-0.857, dE(+2)=-0.0089 (MDE 0.0257); GLM rho_dose=-0.964, dE(+2)=-0.0142 '
        '(MDE 0.0241). Magnitude 0.01-0.06 zipf per 2-zipf push (~2% efficiency), Qwen curve '
        'non-monotone (+2 bounces back to -0.009). Per D10 this is a CLUE, not a law; the '
        'defensible statement is "pos0 responds to the perturbation (KL) but the response is not '
        'organized along the frequency axis". '
        'D3-2 REGISTER_SENSITIVE IS A MODEL FINGERPRINT: KL(pos0)/KL(LAST) at delta=+2 is 19.2x '
        'in Qwen (0.0625 vs 0.0032) but 1.93x in GLM (0.0398 vs 0.0206) => REGISTER_SENSITIVE vs '
        'POSITION_NEUTRAL. Q5-A pos0-dominance reproduces under perturbation in Qwen only; the '
        'register structure is a Qwen-family training product, not a universal architectural '
        'property (third instance of the FUNCTION_MODEL_SPECIFIC pattern: Q8-B/B5 knob-turning, '
        'Q8-D2/C3 direction specificity, now D3/C4 position sensitivity). '
        'KL SHAPE DIFFERS: Qwen KL is monotone in |delta| (0.006->0.06, symmetric U); GLM KL is '
        'dose-flat (0.034-0.040) - GLM is equally perturbed however you push. '
        'C3 with the EMB control: GENERIC_PERTURBATION both models (Qwen |dE| 0.0089 below RAND '
        'p95; GLM 0.0142 < p95 0.0236). C5 gate 7/7 both. '
        'Q8-D LINE CLOSED: LAST + ALLPOS + POS0, two models, input arm - the static frequency '
        'subspace is not an input-side frequency control anywhere. '
        'MATERIAL TRAPS (both template-completion, baseline-only): Qwen3 default chat template '
        'forces <think> at 1.000 (p_M=0) - needs enable_thinking=False (top1 becomes ni 0.776); '
        'HF GLM-4 template ends at <|assistant|> where the model predicts \\n at 1.000 (official '
        'format is <|assistant|>\\n) - appending \\n gives top1 zhejuhua 0.468. '
        'VERIFY: Qwen 21/0; verifier itself had a dE() that bypassed the wrap() (shift on raw, '
        'baseline on wrapped => constant -0.55 offset) and needed chat-template kernel-noise '
        'calibration (per-prompt max|dE|=0.147 mean 0.0019 from the batch-invariance check; '
        'dE tolerance 0.02, gate p_M tolerance 0.10 with a saturation rule: flips of candidates '
        'with p_M>=0.90 are legal because 8-bit/kernel noise can push a 0.963 p_M below gate - '
        'GLM measured max|dp_M|=0.0934 flipping yi-zhang). GLM verify re-run final.'
    ),
}

E_E = {
    'phase': 'Q8-E', 'name': 'pat_q8e', 'line': 'G', 'date': '2026-11-01',
    'verdict': 'freq_subspace_shape_is_scale_stable_third_point_but_scalar_content_is_only_weakly_shared',
    'res_sha8': 'ca6a7f94',
    'res_sha8_glm': None,
    'prev_post': None,
    'note': (
        'Cross-scale stability of the frequency subspace, pure static (no forward). Qwen3-14B '
        '(d=5120) newly downloaded (hf-mirror; shards validated - 3 shards needed a '
        'size-checked resume loop after two silent curl truncations). Same pool construction as '
        'Q8-B (n=24421, CHAR 8049 + WORD 16372). Runtime 114 s. '
        'E1 PRESENT: 14B full-space held-out R^2 0.7515 (verify 0.7493) vs 4B 0.8266. '
        'E2 LOG_SCALE: raw-frequency axis gives 0.023 (4B) / -0.106 (14B) => gap 0.804 / 0.858. '
        'The raw axis is a NOISE-FLOOR quantity: its point value depends on solver numerics '
        '(cv_fast -0.10 vs hand-written normal equations -0.97 on the same folds); reported as '
        '3-seed mean with the verdict set by INEQUALITY (r2<0.10 and gap>=0.30), not point '
        'reproduction. New rule R-MTH-30. '
        'E3 LOW_DIM_DENSE: Krylov k*=2 both models (4B k1 0.529->k2 0.740; 14B k1 0.629->k2 '
        '0.736). With Q8-B this makes the shape family (log scale, 2-4 dim dense subspace, '
        'orthogonal to norm and semantics, no single-coordinate carrier: best coord -0.4414 4B / '
        '+0.4990 14B against R^2 0.83/0.75 full-space) a THREE-MODEL TWO-FAMILY reference-grade '
        'regularity spanning a 3.5x parameter gap. '
        'E4 SCALAR_WEAKLY_SHARED (the new cross-model instrument): coordinates do not align '
        'across models, so cos(w_freq) is meaningless; the only clean shared object is the '
        'per-token scalar t=(E-mu)@w on the shared 151936 rows. Raw Spearman 0.9141 is trivial '
        '(both predict zipf); after regressing zipf out of each t, the residual correlation is '
        '0.4268 against a shuffle null p99 of 0.0175 (24x) - significantly non-zero but below '
        'the pre-registered 0.50 SHARED threshold. The two models put genuinely shared but '
        'minority structure on the same tokens. '
        'E5 SEM_ORTHOGONAL: COLOR/FRUIT within-set similarity retains 1.00 after removing the '
        'frequency component in 14B (0.1825->0.1823, 0.0973->0.0973; nulls 0.013/0.014). '
        'SYNTHESIS with Q8-D2: the frequency subspace is a universal by-product (attractor) of '
        'training, not a functional component - shape universal, usage null, token-level content '
        'majority-accidental with a real shared minority (0.43). '
        'VERIFY 18/0 after two verifier bugs were caught: (1) the verifier hanzi class missed CJK '
        'extension A (200 chars; pool 24221 vs 24421 - every number drifted), (2) it tried to '
        'point-reproduce the noise-floor raw-axis R^2 in float32 Gram. Cache-collision hazard '
        'empirically confirmed: the diagnostic hit a Gram cache poisoned across models (2560-dim '
        'G served to a 5120-dim solve) because the key lacks X identity - ns must carry the '
        'model name (extension of R-MTH-24).'
    ),
}

NEGATIVES = [
    {
        'name': 'pat_q8d3_neg_pos0_carries_freq_knob', 'phase': 'Q8-D3', 'date': '2026-11-01',
        'claim_refuted': 'If the static frequency direction is used anywhere, it is at the '
                         'format/register position (pos0), where next-token competition '
                         'concentrates (Q5-A: 46%).',
        'why_refuted': 'Shifting pos0 embedding along w_freq (chat-templated prompts, dose-'
                       'calibrated) gives WEAK_COUPLING in both models with INVERSE sign '
                       '(Qwen rho=-0.857 dE(+2)=-0.0089; GLM rho=-0.964 dE(+2)=-0.0142), an '
                       'order of magnitude below the causal threshold 0.10, and the direction '
                       'specificity fails the EMB control in both (GENERIC_PERTURBATION). With '
                       'Q8-D2 (LAST, ALLPOS) this closes every input-side entry point.',
        'what_stands': 'pos0 IS the distributional hotspot in Qwen (KL ratio 19.2x vs LAST) - '
                       'the register concentrates perturbation response, it just does not '
                       'organize that response along the frequency axis.',
    },
    {
        'name': 'pat_q8e_neg_freq_scalar_shared_across_scale', 'phase': 'Q8-E',
        'date': '2026-11-01',
        'claim_refuted': 'The frequency subspace is a shared functional structure: same family, '
                         'larger scale, same tokens get the same frequency scalar.',
        'why_refuted': 'After removing zipf from each model scalar t, the 4B-14B residual '
                       'Spearman is 0.4268 - above the shuffle null (p99 0.0175) but below the '
                       'pre-registered SHARED threshold 0.50; the majority of token-level '
                       'variance is model-specific. Shape invariants (log scale, k*=2, semantic '
                       'orthogonality) DO reproduce, so the claim is refuted only at the '
                       'content level.',
        'what_stands': 'A real shared minority component exists (24x null) - the attractor has '
                       'structure worth dissecting (Q8-F); shape is reference-grade across three '
                       'models and two families.',
    },
]


def main() -> int:
    doc = json.loads(LEDGER.read_text(encoding='utf-8'))
    names = {m.get('name') for m in doc['measurements']}
    prev = doc['ledger_sha256_8']
    if E_D3['name'] not in names:
        e = dict(E_D3)
        e['prev_post'] = prev
        doc['measurements'].append(e)
        print('appended %s (prev=%s)' % (e['name'], e['prev_post']))
    # 追加后必须重放中间链头：prev_post 链 = 逐条文档头哈希，禁用陈旧的 ledger_sha256_8
    if E_E['name'] not in names:
        ms_wo_e = [m for m in doc['measurements'] if m.get('name') != E_E['name']]
        assert ms_wo_e and ms_wo_e[-1]['name'] == E_D3['name']
        tmp = dict(doc)
        tmp['measurements'] = ms_wo_e
        tmp['ledger_sha256_8'] = ms_wo_e[-1]['prev_post']  # E_D3.prev = 旧链头
        mid = hashlib.sha256(json.dumps(tmp, ensure_ascii=False, indent=1)
                             .encode('utf-8')).hexdigest()[:8]
        e = dict(E_E)
        e['prev_post'] = mid
        doc['measurements'].append(e)
        print('appended %s (prev=%s)' % (e['name'], e['prev_post']))
    have = {n.get('name') for n in doc['negatives']}
    for n in NEGATIVES:
        if n['name'] not in have:
            doc['negatives'].append(n)
            print('appended negative %s' % n['name'])

    last_prev = doc['measurements'][-1]['prev_post']
    doc['ledger_sha256_8'] = last_prev
    doc['ledger_sha256_8'] = hashlib.sha256(
        json.dumps(doc, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    LEDGER.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                      encoding='utf-8', newline='')
    chk = json.loads(LEDGER.read_text(encoding='utf-8'))
    tmp = dict(chk)
    tmp['ledger_sha256_8'] = last_prev
    rec = hashlib.sha256(json.dumps(tmp, ensure_ascii=False, indent=1)
                         .encode('utf-8')).hexdigest()[:8]
    print('n=%d neg=%d head=%s recomputed=%s match=%s'
          % (len(chk['measurements']), len(chk['negatives']), chk['ledger_sha256_8'], rec,
             rec == chk['ledger_sha256_8']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
