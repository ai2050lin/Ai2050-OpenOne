# -*- coding: utf-8 -*-
"""追加 Q8-A 一条测量 + 四条 negative（幂等）。"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'research' / 'gpt5' / 'atlas' / 'atlas_ledger.json'

ENTRY = {
    'phase': 'Q8-A', 'name': 'pat_q8a', 'line': 'G', 'date': '2026-11-01',
    'verdict': 'vocab_is_not_one_table_and_category_backbone_is_mostly_one_frequency_scalar',
    'res_sha8': '1b9ec38c',
    'note': (
        'First direct measurement of the embedding table itself (Q1-Q7 all measured the residual '
        'stream after context; hs[0] had only ever served as a free baseline). Ten pre-registered '
        'criteria (PAT_Q8A_PREREG_v1.md), 48 s on Qwen3-4B (d=2560, tied), 106 s on GLM-4-9B '
        '(d=4096, untied), independent verify 15/15 on both. '
        'STRUCTURE: the table is not homogeneous - 151936 rows = 16374 whole words / 8806 single '
        'chars / 1457 BYTE FRAGMENTS / 125299 other. 12404 of the 20902 CJK basic-block characters '
        '(59.3%) have NO embedding row at all (GLM 79.8%); 26/74 hand-picked rare characters in '
        'Qwen and 73/74 in GLM are split into 2-3 byte fragments, so "the embedding of a rare '
        'character" mostly does not exist. The one rare character that does (龘) has its own two '
        'byte fragments as nearest neighbours (0.717 / 0.699), third neighbour 龟 at 0.369. '
        'P1 fragment-vs-word within-cosine 0.0298 vs 0.0062 = 4.8x the matched-resample p95 '
        '(0.0064), p<0.005, but the absolute gap +0.0237 misses the pre-registered 0.15 gate => '
        'NO_BYTE_COLLAPSE (direction right, threshold mis-calibrated; see the threshold lesson). '
        'Anisotropy is nearly zero in this model (raw mean pairwise cos 0.0916, centred 0.0005, '
        '||mu||=0.31 vs mean norm 1.10), unlike GPT-2-class models. '
        'P2 norm-vs-frequency is an INVERTED U, not monotone: rho(log||e||,zipf)=+0.272 (n=8049), '
        'held-out +0.264, but the 8 frequency bands are 0.091 0.162 0.197 0.211 0.222 0.220 0.216 '
        '0.176 - mid-frequency content words are largest, both tails smaller. Same shape in GLM. '
        'P3 colour vs fruit PASSES with three separate controls: within(COLOR)=0.1988, '
        'within(FRUIT)=0.0975, between 0.0668, frequency-matched random two-character words '
        '0.0153 => +0.0822 (p=0); leave-one-out 2AFC 0.96 / 1.00; shared-character decomposition '
        '(added after the run, the pre-registration only did it for FRUIT) drops colour to 0.1504 '
        '(-24.4%) and fruit to 0.0738 (-24.3%), so colour stays 2.0x fruit; frequency-matched 2AFC '
        '0.992 [0.955,1.000]. Colour is a STAR: same-modifier families have within-family cos '
        '0.558 (red) / 0.585 (green) / 0.553 (yellow) / 0.536 (blue). '
        'P4 hue is weak-but-real and vanishes in context: Mantel rho=+0.169 p=0.0046 (GLM +0.157), '
        'but MDE at n=45 is 0.386 > the 0.30 gate, so D23 forces PENDING_power, NOT refuted; '
        'contextual readout (colour template, L20 last token) rho=0.030. Lightness-vs-norm '
        'rho=-0.037 => NO_LIGHTNESS_LINK. '
        'P5 byte composition refuted on both counts: codepoint distance vs cosine rho=-0.091 over '
        '79800 pairs (mean pooling destroys the byte ORDER, which is where UTF-8 information lives), '
        'and the collapse prediction has the WRONG SIGN - within(byte-fallback 26)=0.109, '
        'within(in-vocab rare 48)=0.028, within(high-frequency single chars 65)=0.143, i.e. the '
        'function words are the tightest cluster, not the rare characters. Only neighbour '
        'enrichment holds (0.99 vs pool base 0.33). '
        'P6 radicals are in the embedding and HALF OF IT IS NOT SEMANTIC: all 8 groups z=5.0-15.2, '
        'and the semantically incoherent 石 group (硬碰碗破碎砖砂码磁) still sits at +0.0251 '
        '(z=5.0); heterogeneous/homogeneous ratio 0.55 in Qwen, 0.72 in GLM. This is the only '
        'positive reuse result of the round. '
        'P7 SINGLE COORDINATE IS FREQUENCY: raw top-k held-out 4-way accuracy k=1 already 0.780 '
        '(chance 0.25) and k=32 0.751, which looks like a one-parameter backbone, but the top '
        'coordinate correlates with zipf at |rho|=0.802 (GLM 0.774) and the four groups span four '
        'orders of magnitude of frequency. After per-coordinate residualisation on '
        '[1, zipf, byte-length, log-norm] the same curve gives k=32 0.411 (GLM 0.531) and it '
        'DEGRADES with more coordinates (full 0.357) => category information survives frequency '
        'removal by about +0.16 but is diffuse, not concentrated. Frequency-stratified null '
        '(shuffle labels inside zipf deciles) drops the count of significant coordinates from 364 '
        'to 3 (GLM 236 to 3). Main verdict (absolute level) CONFOUNDED_BY_FREQUENCY; attribution '
        '(relative gap) +0.16 diffuse. Colour/fruit top-32 coordinate overlap is 4 in Qwen '
        '(hypergeometric p=5.7e-4, below the 8 threshold) and 9 in GLM (p=7.9e-13) => inconsistent, '
        'no claim. '
        'P8 embedding-versus-output (readout only, tied embedding means the same-table effect '
        'cannot be excluded): partial rho(cos, logprob | zipf) for fruit +0.362 p=0.013 (GLM '
        '+0.351), for colour +0.040 (GLM +0.150 n.s.). '
        'EXPLORATORY (post-hoc, E0 candidates, not part of the verdict): P9 - for 40 Chinese/English '
        'pairs the translation token ranks median 1 of 151936 (top-5 40/40) against a random-English '
        'baseline of 67472, i.e. Qwen static embeddings are cross-lingually folded (红-> red 0.799, '
        '苹果-> Apple 0.718); GLM is completely different (median rank 11.5, top-5 17%, neighbours '
        'are Chinese collocations 红的/的红/红了). P10 - 36 simplified/traditional pairs have median '
        'cos 0.797 and in 94% of cases the traditional form is the FIRST neighbour of the simplified '
        'one (GLM 0.440 / 34%). These two give the Q6 puzzle a new reading: in Qwen the static layer '
        'already folds meaning onto a cross-lingual point, so relational directions can only grow '
        'in context. '
        'CROSS-MODEL: 8 of 8 pre-registered verdicts coincide between Qwen3-4B and GLM-4-9B, but '
        'the two exploratory results do not, so the structure replicates while the organisation '
        'strategy is a model fingerprint. All E1: no intervention, nothing here says the model USES '
        'these structures.'
    ),
}

NEGATIVES = [
    {
        'name': 'pat_q8a_neg_hue_ring_in_static_embedding', 'phase': 'Q8-A', 'date': '2026-11-01',
        'claim_refuted': 'The hue circle of colour words is legible in the static embedding: circular '
                         'hue distance predicts embedding distance at rho >= 0.30.',
        'why_refuted': 'Mantel rho = +0.169 (Qwen, p=0.0046) and +0.157 (GLM, p=0.0055) on n=45 '
                       'chromatic colour words whose Lab hue was computed from explicit sRGB codes. '
                       'The signal is real but less than half the pre-registered effect size, and '
                       'at n=45 the 80%-power MDE is 0.386, so the 0.30 line is undecidable (D23 '
                       '=> PENDING_power, not refuted). In context the signal disappears entirely '
                       '(template readout at L20: rho=0.030 / 0.045).',
        'what_stands': 'Colour words do cluster far more tightly than fruit (0.199 vs 0.098) and the '
                       'clustering is organised as modifier-root stars, not as a hue ring.',
    },
    {
        'name': 'pat_q8a_neg_codepoint_order_in_byte_composition', 'phase': 'Q8-A', 'date': '2026-11-01',
        'claim_refuted': 'Rare characters assembled from byte fragments inherit a geometry ordered by '
                         'Unicode codepoint, because UTF-8 bytes are codepoint-ordered.',
        'why_refuted': 'Over 79800 pairs of 400 byte-fallback CJK characters, rho(codepoint distance, '
                       'cosine) = -0.091 (Qwen) / -0.102 (GLM), with eight distance bands flat at '
                       '0.131 -> 0.107 (MDE 0.009, so 0.09 is a real but negligible effect). The '
                       'mechanism fails by construction: mean-pooling 2-3 byte embeddings destroys '
                       'the ORDER of the bytes, and UTF-8 carries all of its information in the order.',
        'what_stands': 'Byte-fallback characters are more mutually similar than whole words (0.109 vs '
                       '0.006) and 99% of their ten nearest neighbours are other fallback characters.',
    },
    {
        'name': 'pat_q8a_neg_rare_char_collapse', 'phase': 'Q8-A', 'date': '2026-11-01',
        'claim_refuted': 'Rare characters, having no dedicated vector, collapse into one blob so that '
                         'within(RARE) exceeds within(COMMON) by at least 0.10.',
        'why_refuted': 'The sign is opposite and the ordering is the reverse of the intuition: '
                       'within(RARE)=0.0188, within(high-frequency single chars)=0.1433, and splitting '
                       'RARE gives 0.109 for the 26 byte-fallback members vs 0.028 for the 48 members '
                       'that do have a row. The tightest cluster in the vocabulary is the function-word '
                       'class (的/一/是/不/了), whose members are used in identical contexts, not the '
                       'rare characters. Pre-registered delta -0.125 against a +0.10 gate.',
        'what_stands': 'Rare characters have no semantic neighbours (neighbour enrichment 0.99 against a '
                       '0.33 base rate) but they are not similar to each other.',
    },
    {
        'name': 'pat_q8a_neg_one_coordinate_carries_category', 'phase': 'Q8-A', 'date': '2026-11-01',
        'claim_refuted': 'The four-way category structure (colour / fruit / rare / common) is carried '
                         'by a small number of single embedding coordinates, i.e. by single parameters.',
        'why_refuted': 'Raw single coordinates look decisive (k=1 held-out accuracy 0.780, k=32 0.751, '
                       'chance 0.25) but the winning coordinate correlates with zipf at |rho|=0.802 '
                       '(GLM 0.774) and the four material sets span four orders of magnitude of '
                       'frequency. After residualising every coordinate on [1, zipf, byte-length, '
                       'log-norm], k=32 falls to 0.411 (GLM 0.531) and full-dimensional accuracy is '
                       'WORSE than k=64 (0.357 vs 0.415), i.e. the residual category information is '
                       'diffuse and does not concentrate. A frequency-stratified null removes 361 of '
                       '364 apparently significant coordinates.',
        'what_stands': 'Frequency itself IS a single-coordinate scalar (|rho|=0.80) - the single-parameter '
                       'claim is true for a scalar statistic, false for semantic categories.',
    },
]

THRESHOLD_LESSON = {
    'phase': 'Q8-A', 'date': '2026-11-01',
    'claim': 'Thresholds written as absolute cosine differences are meaningless without scaling by the '
             'zero distribution of the same model.',
    'evidence': 'P1 gate 0.15 was missed by an effect that is 4.8x the matched-resample p95 and 4x the '
                'population baseline (0.0237 vs 0.0062); P4 gate 0.30 was missed by a design whose MDE '
                'was 0.386. Both gates were written before seeing any result but without calibrating '
                'against the model\'s cosine scale (Qwen3-4B baseline 0.006, not 0.5).',
    'rule': 'Write gates as "N x the null p95" or as an MDE-relative quantity, never as a bare cosine gap.',
}


def main() -> int:
    doc = json.loads(LEDGER.read_text(encoding='utf-8'))
    bak = LEDGER.with_suffix('.json.bak_q8a')
    if not bak.exists():
        shutil.copy2(LEDGER, bak)
        print('备份 ->', bak.name)

    added_neg = 0
    for n in NEGATIVES:
        if not any(x.get('name') == n['name'] for x in doc['negatives']):
            doc['negatives'].append(n)
            added_neg += 1
    print('negatives +%d (共 %d)' % (added_neg, len(doc['negatives'])))

    if any(m.get('name') == ENTRY['name'] for m in doc['measurements']):
        print('测量已存在，跳过')
    else:
        prev = doc['ledger_sha256_8']
        e = dict(ENTRY)
        e['prev_post'] = prev
        doc['measurements'].append(e)

    # 所有内容改动都完成后再算链头，否则 head 与内容不自洽（本轮踩过）
    if not any(e.get('phase') == 'Q8-A' for e in doc.get('errata_ledger', [])):
        doc['errata_ledger'].append(dict(corrects='Q8-A', phase='Q8-A',
                                         note=THRESHOLD_LESSON['rule']))
    if any(m.get('name') == ENTRY['name'] for m in doc['measurements']):
        prev = doc['measurements'][-1]['prev_post']
        doc['ledger_sha256_8'] = prev
        doc['ledger_sha256_8'] = hashlib.sha256(
            json.dumps(doc, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
        print('head 复算 -> %s' % doc['ledger_sha256_8'])

    LEDGER.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                      encoding='utf-8', newline='')
    print('n=%d  neg=%d  head=%s' % (len(doc['measurements']), len(doc['negatives']),
                                     doc['ledger_sha256_8']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
