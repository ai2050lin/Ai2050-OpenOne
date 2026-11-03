# -*- coding: utf-8 -*-
"""追加 Q8-D 一条测量 + 三条 negative（幂等）。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'research' / 'gpt5' / 'atlas' / 'atlas_ledger.json'

ENTRY = {
    'phase': 'Q8-D', 'name': 'pat_q8d', 'line': 'G', 'date': '2026-11-01',
    'verdict': 'static_frequency_subspace_is_not_used_by_the_forward_pass_null_bounded_in_both_models',
    'res_sha8': 'ca7fcc07',
    'res_sha8_glm': '401a884c',
    'note': (
        'Causal test of the frequency direction that Q8-B found in the static table (held-out R^2 '
        '0.826/0.721). Two-phase design: Q8-D wide grid (delta up to +-8 zipf units) exposed the '
        'tolerance window (|delta|=8, 24% of median token norm, breaks the model: top-1 agreement '
        '0.055-0.17, entropy +1.96) and exposed the RAND-control flaw; Q8-D2 narrow grid '
        '(delta in {-2,-1,-0.5,0,0.5,1,2}) is the registered test (PAT_Q8D2_PREREG_v1.md). '
        'Dose calibration measured BEFORE freezing: 1 zipf unit of displacement is 3.0% (Qwen) / '
        '4.9% (GLM) of median token norm. Materials: 36 candidate templates x 50 nouns, template '
        'gate (baseline only: p_M>=0.80, H>=1.00, syntax-incomplete prompts ending in a '
        'frequency-bearing content word - full syntactic prompts collapse to predicting '
        'punctuation, p_M 0.033) keeps 32 (Qwen) / 25 (GLM) templates, first 200 prompts, LAST-'
        'position readout, n_forward=141 (Qwen) / 77 (GLM 8-bit). '
        'MAIN RESULT C1 NULL_BOUNDED IN BOTH MODELS: deltaE[zipf](+2) = -0.00114 (Qwen, CI '
        '[-0.0074,+0.0058], MDE 0.00948, rho_dose=-0.107) and +0.00426 (GLM, CI [-0.0013,+0.0106], '
        'MDE 0.00856, rho_dose=+0.25). The exact zero is BOUNDED, not just undetected. The R^2-0.8 '
        'static frequency subspace is NOT used by the forward pass to set output frequency. '
        'C3 DIRECTION-SPECIFICITY SPLIT BY THE MANIFOLD CONTROL: with isotropic RAND only, FREQ '
        'looks special (KL 20x RAND in the wide-grid run); adding the ON-MANIFOLD control EMB '
        '(difference of two random real token embeddings) kills it in Qwen (FREQ |dE| 0.00114 is '
        'below the RAND p95 0.0038 and the EMB p95 0.0027 => GENERIC_PERTURBATION) but not in GLM '
        '(|dE| 0.00426 > RAND med 0.0015, EMB p95 0.0023; KL 0.0297 vs 0.0151/0.0155 => '
        'DIRECTION_SPECIFIC). GLM w_freq is a REAL sensitivity direction of the output distribution '
        'that does not push frequency. Sensitive is not load-bearing. '
        'C4 INPUT_DRIVEN both models (IN/OUT effect ratio 21x Qwen, 1847x GLM): the displacement '
        'only propagates from the input embedding, not applied to the pre-head hidden state - '
        'consistent with Q2-C1 (ablation measures capacity, patch measures readout). '
        'C6 GLOBAL_OFFSET both models (ALLPOS/LAST effect ratio 6.2x / 3.4x, dE still zero). '
        'C8 semantic probe (40 knowledge questions, forced choice): Qwen SEMANTICS_INTACT (0.600 -> '
        'worst 0.525 at delta=8), GLM SEMANTICS_HURT (0.575 -> 0.45 at delta=8; delta=8 is a '
        'breaking dose; 8-bit background may amplify). C5 gate 7/7 both models. '
        'CONVERGENCE WITH Q8-B: the full proposition is now "frequency structure is written into the '
        'static geometry (B1) but computation does not route through it (D1)". Q2 family already '
        'hit this wall twice (Q2-L injection = readout not mechanism; Q2-C1 ablate/patch split); '
        'Q8-D is the third independent instance, now on the STATIC side: geometric symmetry is not '
        'computational dependence. Because the subspace is unused, its cross-model isomorphism '
        '(Q8-B SHAPE_ISOMORPHIC) points to a shared TRAINING-GEOMETRY property (a fossil record), '
        'not a functional constraint. All E1 (null causal result with quantitative bounds). '
        'PROCESS: D2 gate thresholds were first implemented with D1 defaults (0.5/0.75) and '
        're-run with the registered 0.30/0.60 (stale results deleted; new rule R-MTH-27). GLM '
        'full-precision offload segfaulted repeatedly under external memory pressure; switched to '
        'bitsandbytes 8-bit full-GPU with lm_head/embed_tokens skipped (material change recorded). '
        'GLM offload also leaves lm_head on the meta device with garbage values (|o2|=115761 vs '
        '|ref|=764) - calibrate_head assertion correctly aborted that run before any numbers were '
        'read; fixed by safetensors direct read (new rule R-MTH-26). Independent verify: Qwen '
        '21 PASS / 0 FAIL; GLM verify re-run after fixing a hardcoded model path in the verifier '
        'itself (its 8 FAILs were verifier error, not result error).'
    ),
}

NEGATIVES = [
    {
        'name': 'pat_q8d_neg_forward_pass_uses_freq_direction', 'phase': 'Q8-D',
        'date': '2026-11-01',
        'claim_refuted': 'Because the static table encodes frequency as a strongly linear direction '
                         '(Q8-B B1, R^2 0.826/0.721), the model uses that direction during the '
                         'forward pass to set the frequency of the next token.',
        'why_refuted': 'Shifting token embeddings along w_freq at the input (LAST position, dose-'
                       'calibrated, healthy prompts p_M~0.98, tolerance-gated |delta|<=2) moves '
                       'E[zipf of next token] by |delta|<=0.004 with CI excluding nothing but '
                       'bounded by MDE 0.0095/0.0086 in BOTH models (NULL_BOUNDED, not '
                       'PENDING_POWER). The dose-response shape is noise (linear R^2 0.26-0.29).',
        'what_stands': 'The static subspace exists and is isomorphic across models; Qwen turning '
                       'the knob moves static neighbours (Q8-B B5). What is refuted is only the '
                       'route into computation: geometric encoding without functional use.',
    },
    {
        'name': 'pat_q8d_neg_isotropic_rand_control_suffices', 'phase': 'Q8-D',
        'date': '2026-11-01',
        'claim_refuted': 'A direction whose KL effect on the output distribution exceeds isotropic '
                         'random directions is direction-specific (functionally distinguished).',
        'why_refuted': 'With the on-manifold control EMB (random difference of two real token '
                       'embeddings), the apparent specificity of FREQ in Qwen disappears (|dE| '
                       '0.00114 below both the RAND p95 0.0038 and EMB p95 0.0027); in the wide '
                       'grid the "KL 20x RAND" impression was just the perturbation living on the '
                       'embedding manifold.',
        'what_stands': 'GLM FREQ passes even the EMB control (|dE| 0.00426 vs EMB p95 0.0023, KL '
                       '0.030 vs 0.015), so on-manifold specificity is achievable and meaningful; '
                       'the rule is that the control set must CONTAIN an on-manifold member.',
    },
    {
        'name': 'pat_q8d_neg_syntax_complete_prompts_for_freq_readout', 'phase': 'Q8-D',
        'date': '2026-11-01',
        'claim_refuted': 'Natural complete sentences ("她很喜欢吃苹果") are the right material for '
                         'reading output frequency after an embedding intervention.',
        'why_refuted': 'At a syntactically complete boundary the model predicts punctuation with '
                       'mass 0.849 and mass over frequency-bearing tokens p_M collapses to 0.033; '
                       'E[zipf] over the full vocabulary is then dominated by punctuation and '
                       'cannot register any frequency shift. Baseline-healthy incomplete prompts '
                       '("他把苹果") hold p_M=0.97.',
        'what_stands': 'With the material gate (p_M>=0.80, H>=1.00, final token a frequency-'
                       'bearing content word), the readout is well-posed; the gate looks only at '
                       'the BASELINE distribution, never at intervention results.',
    },
]


def main() -> int:
    doc = json.loads(LEDGER.read_text(encoding='utf-8'))
    names = {m.get('name') for m in doc['measurements']}
    if ENTRY['name'] not in names:
        prev = doc['ledger_sha256_8']
        e = dict(ENTRY)
        e['prev_post'] = prev
        doc['measurements'].append(e)
        print('appended measurement %s (prev=%s)' % (e['name'], prev))
    else:
        print('measurement already present, skipped')

    have = {n.get('name') for n in doc['negatives']}
    for n in NEGATIVES:
        if n['name'] not in have:
            doc['negatives'].append(n)
            print('appended negative %s' % n['name'])
        else:
            print('negative %s already present' % n['name'])

    # 所有内容改动完成后再算链头
    prev = doc['measurements'][-1]['prev_post']
    doc['ledger_sha256_8'] = prev
    doc['ledger_sha256_8'] = hashlib.sha256(
        json.dumps(doc, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    LEDGER.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                      encoding='utf-8', newline='')
    chk = json.loads(LEDGER.read_text(encoding='utf-8'))
    tmp = dict(chk)
    tmp['ledger_sha256_8'] = prev
    rec = hashlib.sha256(json.dumps(tmp, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    print('n=%d neg=%d head=%s recomputed=%s match=%s'
          % (len(chk['measurements']), len(chk['negatives']), chk['ledger_sha256_8'], rec,
             rec == chk['ledger_sha256_8']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
