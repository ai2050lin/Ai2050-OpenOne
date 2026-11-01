# -*- coding: utf-8 -*-
"""追加 Q3-T1 / Q3-T2 两条测量 + 一条 negative（上下文几乎不改 key）。幂等。"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'research' / 'gpt5' / 'atlas' / 'atlas_ledger.json'

ENTRIES = [
    {
        'phase': 'Q3-T1', 'name': 'pat_q3t_t1', 'line': 'G',
        'date': '2026-11-01',
        'verdict': 'identity_is_fully_readable_at_every_layer_but_its_absolute_share_peaks_mid_stack',
        'res_sha8': '7f671a03',
        'note': (
            'T1 of the preregistered identity-fingerprint probe. Key side is GQA (32 query heads '
            'share 8 KV sets), so the fingerprint is defined on 8 kv heads, not 32 - prereg correction '
            'recorded before running. Position alignment is the whole measurement: key is a '
            'post-RoPE quantity, so comparing across positions yields pure artifact. Two templates '
            'with a pinned theme slot (P9 = active voice, W at token 24; P0 = passive, W at token 17), '
            'person pool restricted to modal token length so the slot index never moves, 30 single-token '
            'nouns x 20 (A,B) contexts = 1200 sequences. RESULT: identity retrieval match = 1.0000 at '
            'ALL 36 layers for BOTH templates, 30x the 1/30 chance level - no layer degrades. '
            'cos(same word across contexts): 1.0000 at L0 -> 0.9979 at L35 (P9). '
            'cos(different words): 0.9976 at L0 -> 0.9559 at L35 (P9); 0.9977 -> 0.9472 (P0). '
            'sep = cos_same - cos_diff: 0.002 at L0, peaks L13 at 0.230 (P9) / 0.260 (P0), decays to '
            '0.042 / 0.053 at L35. THREE FACTS: (1) the fingerprint is near-constant - same token across '
            '20 different sentences has key direction cos 0.998; (2) identity is fully decodable at '
            'every layer even where sep is 0.04, because a small absolute difference along a stable '
            'direction still separates perfectly; (3) at L0 the identity difference is EXACTLY zero '
            '(sep=0.002), i.e. the first-layer key carries position only. PREREG PREDICTION FAILED: '
            'the four-way shape test (U / monotone up / monotone down / flat) is missed on all counts '
            '(front-mid = 0.007, rho = -0.169 / +0.194, range 0.227) - verdict NULL. What survives is '
            'shape invariance across position: P9 vs P0 sep profile Spearman = 0.835, max|d| = 0.080, '
            'an invariant candidate under the t4 transform. '
            'RoPE reproduction gate: self-computed attention vs output_attentions max|d| = 8.7e-3; '
            'traced to the model storing attention weights in bf16 (softmax(...).to(query.dtype)), '
            'not to an implementation error - our s is the more precise one (residual 7e-3 on a '
            'quantity whose range is ~30, i.e. 2e-4 relative).'
        ),
    },
    {
        'phase': 'Q3-T2', 'name': 'pat_q3t_t2', 'line': 'G',
        'date': '2026-11-01',
        'verdict': 'context_specific_share_of_key_is_near_zero_so_context_enters_through_q_not_k',
        'res_sha8': 'bd82af6e',
        'note': (
            'T2: exact algebraic decomposition of a single dot product in one forward pass, no ablation, '
            'no injection, no transplant - so none of the three confounds that killed the previous nine '
            'rounds (coverage, linear readout, transplant matching) can apply. Three-step decomposition '
            'after SMOKE exposed that a naive "project onto the cross-context mean direction" lets the '
            'identity term swallow everything (all keys are collinear at cos 0.998, so the mean direction '
            'IS the principal direction; share_id came out a spurious 1.00): g = direction shared by all '
            'words and contexts -> k_cmn; perp = k - k_cmn; u_w = the word\'s cross-context mean within '
            'perp -> k_id; k_ctx = perp - k_id. s = q.k_cmn + q.k_id + q.k_ctx exactly. '
            'Scope limited to the theme slot because the identity direction must be measured on a '
            'template isomorphic and position-aligned to the target sentence (RoPE). '
            'RESULT on 114 cells (19 triples x 2 voices x 3 queries), behaviour gate 113/114 '
            '(ag 37/38, re 38/38, th 38/38). share_id: 0.95 at L0 -> 0.89 at L12 -> 0.21 at L24 -> '
            '0.60 at L35. share_cmn: 0.37 -> 0.60 (median 0.543). share_ctx: <=0.03 everywhere, median '
            '0.009. Share sum median 1.050 (slightly above 1, the three terms are positively '
            'correlated, so shares are not strictly orthogonal - discount accordingly). '
            'VERDICT: J1 boundary fails by 0.089 (front 0.822 > 0.5 but back 0.289 > 0.2), J2 split '
            'fails (0.154 < 0.3), J3 equal is literally true yet contradicts the obvious monotone trend '
            '(rho(layer, share_id) = -0.838) - the four-way criterion has no bucket for "trend present '
            'but under threshold". Prereg verdict NULL on criterion grounds. '
            'INSTABILITY, must not be quoted away: recomputing per query gives ag 0.94->0.68, re '
            '0.94->0.71, th 0.94->0.68, i.e. a drop of 0.23-0.26, whereas pooling all 114 cells gives '
            '0.95->0.29, a drop of 0.66. Pooling across queries inflates the drop 2.5x. share is a '
            'conditional quantity relative to the chosen sample set. The robust statement is the 0.25 '
            'drop, independently reproduced in all three groups - NOT 0.66. '
            'THE ONE THING WORTH KEEPING: share_ctx ~= 0.01 means sentence-to-sentence variation barely '
            'touches the key, corroborating the cos 0.998 of T1. Context does not enter attention '
            'through k. The lock barely moves; what moves is the keyhole (q). This reframes the next '
            'question and explains why nine rounds of head-side and k-side search all failed.'
        ),
    },
]

NEGATIVE = {
    'id': 'pat_q3t_neg_context_key_shift',
    'date': '2026-11-01',
    'claim': (
        '上下文通过改变 key 来改变注意力落点（"attention forms stable patterns by re-encoding the '
        'key according to context"）。'
    ),
    'status': 'refuted',
    'evidence': ['Q3-T1 sha8 7f671a03', 'Q3-T2 sha8 bd82af6e'],
    'detail': (
        'Same token across 20 different sentences: key direction cos = 0.998-1.000 across all 36 '
        'layers. In the three-way decomposition the context-SPECIFIC share of the attention score is '
        'median 0.009 and never exceeds 0.03 at any layer. Caveat recorded in the same breath: k_ctx is '
        'defined as the deviation from the word\'s cross-context mean, so it measures the specific '
        'contribution, not the total contextual influence - the shared part is absorbed into k_cmn. '
        'The claim that survives intact is therefore narrower and still decisive: sentence-to-sentence '
        'differences do not re-encode the key. Implication: whatever selects the role must live on the '
        'q side (or downstream), which is exactly where the next phase is pointed.'
    ),
}


def main() -> int:
    doc = json.loads(LEDGER.read_text(encoding='utf-8'))
    have = {(m.get('phase'), m.get('name')) for m in doc['measurements']}
    bak = LEDGER.with_suffix('.json.bak_q3t')
    if not bak.exists():
        shutil.copy2(LEDGER, bak)
        print('备份 ->', bak.name)

    if not any(n.get('id') == NEGATIVE['id'] for n in doc['negatives']):
        doc['negatives'].append(NEGATIVE)
        print('negatives +=1 (共 %d)' % len(doc['negatives']))

    added = 0
    for e in ENTRIES:
        if (e['phase'], e['name']) in have:
            print('跳过（已存在）%s' % e['name'])
            continue
        prev = doc['ledger_sha256_8']
        e = dict(e)
        e['prev_post'] = prev
        doc['measurements'].append(e)
        doc['ledger_sha256_8'] = prev
        head = hashlib.sha256(
            json.dumps(doc, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
        doc['ledger_sha256_8'] = head
        added += 1
        print('#%d %s  prev=%s  post=%s' % (len(doc['measurements']), e['name'], prev, head))

    LEDGER.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                      encoding='utf-8', newline='')
    print('n=%d  head=%s  新增=%d' % (len(doc['measurements']), doc['ledger_sha256_8'], added))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
