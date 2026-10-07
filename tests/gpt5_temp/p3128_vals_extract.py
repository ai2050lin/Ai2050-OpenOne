"""3128 vals extract: read result.json ->
print vals checklist (closeout input)."""
import io
import json

OUTD = (r'D:\AI2050\Ai2050-OpenOne'
        r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3128'
        r'\omega_p126_joint_swap_coord_'
        r'inject_interaction_s0match')
r = json.load(io.open(OUTD
                      + r'\result.json',
                      encoding='utf-8'))
o = []
o.append('PHASE %s smoke=%s runtime=%.0fs'
         % (r['phase'], r['smoke'],
            r['runtime_s']))
v = r['verdict']
o.append('VERDICT(%d): %s'
         % (len(v.split('|')), v))
pa = r['part_a']
o.append('A repl: q %s g %s'
         % ({k: round(x, 6) for k, x
             in pa['repl_gates_q'].items()},
            {k: round(x, 6) for k, x
             in pa['repl_gates_g'].items()}))
pb = r['part_b']
o.append('B joint: ratio=%.4f ctrl=%.4f '
         'write=%.4f delta=%.4f'
         % (pb['joint_ratio_q'],
            pb['med_ctrl'], pb['med_write'],
            pb['delta_q']))
for k in sorted(pb['coord_q']):
    c = pb['coord_q'][k]
    o.append('B coord %s: flip=%.4f '
             'med_dm=%.4f'
             % (k, c['flip'], c['med_dm']))
pc = r['part_c']
o.append('C joint: ratio=%.4f ctrl=%.4f '
         'write=%.4f delta=%.4f'
         % (pc['joint_ratio_g'],
            pc['med_ctrl'], pc['med_write'],
            pc['delta_g']))
for k in sorted(pc['coord_g']):
    c = pc['coord_g'][k]
    o.append('C coord %s: flip=%.4f '
             'med_dm=%.4f'
             % (k, c['flip'], c['med_dm']))
o.append('C s0: agree=%.4f bit_mism=%d '
         'repro=%.2e'
         % (pc['s0_agree'], pc['s0_bit_mism'],
            pc['repro_max_abs']))
pd_ = r['part_d']
o.append('D interact: peak=L%02d sep=%.4f '
         'r=%.4f' % (pd_['peak_layer'],
                     pd_['peak_sep'],
                     pd_['peak_r']))
o.append('B repro=%.2e' % pb['repro_max_abs'])
io.open(r'D:\AI2050\Ai2050-OpenOne'
        r'\gpt5_temp\p3128_vals.txt',
        'w', encoding='utf-8').write(
    chr(10).join(o) + chr(10))
print('VALS_OK')
