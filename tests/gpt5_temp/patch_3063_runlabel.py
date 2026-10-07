# phantom-edit countermeasure patch: run label
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3063_omega_p60_adversarial_trace_qwen.py')
s = io.open(P, encoding='utf-8').read()

old = ("          'run': 'run1 authoritative (fp32; "
       "'\n"
       "                 'capture bank from the "
       "phase3048 '\n"
       "                 'npz; four arms re-run "
       "with the '\n"
       "                 '3054 forward_stage "
       "machinery; '\n"
       "                 'chain anchors vs "
       "z51/z53/z54; '\n"
       "                 'same-source stats "
       "offline in '\n"
       "                 'the exact readout "
       "metric)',")
new = ("          'run': 'run2 authoritative (fp32; "
       "'\n"
       "                 'capture bank from the "
       "phase3048 '\n"
       "                 'npz; four arms re-run "
       "with the '\n"
       "                 '3054 forward_stage "
       "machinery; '\n"
       "                 'chain anchors vs "
       "z51/z53/z54; '\n"
       "                 'same-source stats "
       "offline in '\n"
       "                 'the exact readout "
       "metric; '\n"
       "                 'run1 crashed "
       "pre-verdict at the '\n"
       "                 'G-metric axis null: "
       "wud is '\n"
       "                 'vocab-major so wud @ "
       "wud.T tried '\n"
       "                 'to allocate "
       "(151936,151936) - '\n"
       "                 'fixed to wud.T @ "
       "wud)',")
cnt = s.count(old)
assert cnt == 1, 'anchor count=%d' % cnt
s = s.replace(old, new)
assert "run1 authoritative" not in s
assert "run2 authoritative" in s
io.open(P, 'w', encoding='utf-8').write(s)
print('PATCH_OK')
