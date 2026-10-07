# -*- coding: utf-8 -*-
"""Static pre-check: every REPS old-string of
p3087_patch_full.py must occur EXACTLY once in
phase3085 (they all live in the code section,
outside docstring/PREREG which get rewritten
first). Output -> file (bash stdout lossy)."""
import io

SRC = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\phase3085_omega_p82_l34_full_arbitration.py'
OUTF = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3087_check_reps_out.txt'

src = io.open(SRC, encoding='utf-8').read()

OLDS = [
    "PHASE = 3085",
    "NAME = 'omega_p82_l34_full_arbitration'",
    "                    'qwen2.5-3b-instruct')",
    "NL, HID, KV_HEAD, NQ = 36, 2048, 2, 16",
    "SEED_MAIN = 3085",
    "L_INJ = 34",
    "L_POST = 35",
    "SEED = 3085",
    "assert NVOC == 151936",
    ("log('tie_word_embeddings=%s (tied '\n"
     "    'embeddings: forwards/logits-only '\n"
     "    'protocol unaffected, recorded as an '\n"
     "    'adaptation)' % TIED)"),
    "log('qwen2.5-3b (qwen2.5-3b-instruct) loaded '",
    "# ==== L34 deterministic reproduction",
    "# anchor (vs 3084 layer-scan npz) ====",
    "    z84 = np.load(os.path.join(",
    "        'phase3084',",
    "        'omega_p81_3b_layer_scan',",
    "        'omega_p81_3b_layer_scan.npz'),",
    "        'K3/NP_USE differ from the 3084 '",
    "- float(z84['MED_C_L34_'",
    "== int(z84['N_NEG_L34_' + fk]))",
    "in z84['TOP8_L34_' + fk]])",
    "- z84['CS1H_L34_' + fk])))",
    "- float(z84['R_ALL_L34_'",
    "log('repro[%s] vs 3084 L34: '",
    "'run': 'run1 authoritative (qwen2.5-3b bf16 '",
]
GSUB_OLD = ["third_", "seed 3085", "(16 heads)",
            "'before any qwen2.5-3b '"]

lines = []
fail = 0
for i, o in enumerate(OLDS):
    c = src.count(o)
    flag = 'OK' if c == 1 else 'FAIL'
    if c != 1:
        fail += 1
    lines.append('%2d cnt=%d %s %r'
                 % (i, c, flag, o[:60]))
lines.append('--- global subs (pre-patch counts, '
             'informational) ---')
for o in GSUB_OLD:
    lines.append('cnt=%d %r' % (src.count(o), o))
lines.append('SUMMARY fail=%d/%d'
             % (fail, len(OLDS)))
with io.open(OUTF, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('written', OUTF)
