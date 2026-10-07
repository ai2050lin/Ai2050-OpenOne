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
res = json.load(io.open(OUTD + r'\result.json',
                        encoding='utf-8'))
z = np.load(OUTD + r'\p125_readout.npz',
            allow_pickle=False)
z25 = np.load(D25 + r'\p123_readout.npz',
              allow_pickle=False)
out = []
key = 'A1_L02'
field = z['dmq_' + key][:, -1] \
    .astype(np.float64)
base = z25['mlg_s0_A1'][:field.size, 36, -1] \
    .astype(np.float64)
dm = field - base
lst = np.array(res['part_b']['dm_final'][key])
out.append('len dm=%d len lst=%d'
           % (dm.size, lst.size))
d1 = np.abs(dm - lst)
d2 = np.abs(np.abs(dm) - lst)
out.append('max|dm-lst|=%.6f max||dm|-lst|=%.6f'
           % (float(d1.max()), float(d2.max())))
# which records differ for abs version
bad = np.where(d2 > 1e-6)[0]
out.append('n_bad(abs match)=%d' % bad.size)
out.append('bad idx[:20]=%s'
           % bad[:20].tolist())
for i in bad[:8]:
    out.append('  rec%d field=%.6f base=%.6f'
               ' dm=%.6f |dm|=%.6f lst=%.6f'
               % (i, field[i], base[i], dm[i],
                  abs(dm[i]), lst[i]))
# maybe result dm uses base at DIFFERENT layer col
# try base = mlg_s0[:, 36, :] full row minus? or
# base from same swap forward? test field row -1
out.append('field[:4] %s'
           % np.round(field[:4], 6).tolist())
out.append('base[:4] %s'
           % np.round(base[:4], 6).tolist())
# hypothesis: result = |field_final - base_final|
# where base_final = base[:, -1] of mlg_s0 row at
# SAME record but result sorted? check sorted match
s_dm = np.sort(np.abs(dm))
s_ls = np.sort(lst)
out.append('sorted max|diff|=%.6f'
           % float(np.abs(s_dm - s_ls).max()))
io.open(r'D:\AI2050\Ai2050-OpenOne'
        r'\gpt5_temp\p3127_dm_probe2.txt',
        'w', encoding='utf-8').write(
    chr(10).join(out))
print('PROBE2_OK')
