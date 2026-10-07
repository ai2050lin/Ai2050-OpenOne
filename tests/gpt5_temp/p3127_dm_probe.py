import io, json
import numpy as np

RDIR = (r'D:\AI2050\Ai2050-OpenOne'
        r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUTD = RDIR + (r'\phase3127'
               r'\omega_p125_writechain_port_'
               'crossmodel_a1closure_fullregen')
D25 = RDIR + (r'\phase3125'
              r'\omega_p123_third_comp_qwen_'
              'inputstream')
D26 = RDIR + (r'\phase3126'
              r'\omega_p124_glm4_anchoredlast_'
              'regen_writechain')
res = json.load(io.open(OUTD + r'\result.json',
                        encoding='utf-8'))
z = np.load(OUTD + r'\p125_readout.npz',
            allow_pickle=False)
z25 = np.load(D25 + r'\p123_readout.npz',
              allow_pickle=False)
z26 = np.load(D26 + r'\p124_readout.npz',
              allow_pickle=False)
out = []
for tag, zbase, NL, side in (
        ('B-qwen', z25, 36, 'dmq'),
        ('C-glm4', z26, 40, 'dmg')):
    pb = res['part_b'] if tag.startswith('B') \
        else res['part_c']
    key = 'A1_L02' if tag.startswith('B') \
        else 'A1_L04'
    field = z[side + '_' + key][:, -1] \
        .astype(np.float64)
    for lbl, base in (
            ('mlgNL_lastcol',
             zbase['mlg_s0_A1'][:field.size,
                                NL, -1]
             .astype(np.float64)),
            ('mlgNL_lastcol_full',
             zbase['mlg_s0_A1'][:, NL, -1]
             .astype(np.float64))):
        dm = field - base
        lst = np.array(pb['dm_final'][key])
        out.append('%s %s n=%d' % (tag, lbl,
                                   dm.size))
        out.append('  recomputed[:4] %s'
                   % np.round(dm[:4], 6).tolist())
        out.append('  result    [:4] %s'
                   % np.round(lst[:4], 6).tolist())
        out.append('  max|dm-lst|=%.6g'
                   ' max|dm-|lst||=%.6g'
                   % (float(np.abs(dm - lst).max()),
                      float(np.abs(dm
                                   - np.abs(lst))
                             .max())))
        # stats
        out.append('  dm mean=%.4f med|dm|=%.4f'
                   ' lst mean=%.4f'
                   % (float(dm.mean()),
                      float(np.median(np.abs(dm))),
                      float(lst.mean())))
    # try base = mean over layers? check shape
    out.append('%s mlg_s0 shape=%s'
               % (tag, zbase['mlg_s0_A1'].shape))
    out.append('%s dmq field shape=%s'
               % (tag, z[side + '_' + key].shape))
io.open(r'D:\AI2050\Ai2050-OpenOne'
        r'\gpt5_temp\p3127_dm_probe.txt',
        'w', encoding='utf-8').write(
    chr(10).join(out))
print('PROBE_OK')
