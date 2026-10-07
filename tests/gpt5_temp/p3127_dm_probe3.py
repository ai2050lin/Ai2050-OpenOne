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
# f64 subtraction path (verify)
dm64 = (z['dmq_' + key][:, -1]
        .astype(np.float64)
        - z25['mlg_s0_A1'][:672, 36, -1]
        .astype(np.float64))
# f32 subtraction path (main script)
dm32 = (z['dmq_' + key][:, -1]
        - z25['mlg_s0_A1'][:672, 36, -1]
        ).astype(np.float64)
lst = np.array(res['part_b']['dm_final'][key])
out.append('max||dm64|-lst| = %.3e'
           % float(np.abs(np.abs(dm64)
                          - lst).max()))
out.append('max||dm32|-lst| = %.3e'
           % float(np.abs(np.abs(dm32)
                          - lst).max()))
out.append('|dm64-dm32| max = %.3e'
           % float(np.abs(dm64 - dm32).max()))
# gates recompute both ways
LAY_Q = {'write': [26, 28, 30, 32, 34],
         'port': [20, 21], 'ctrl': [2, 8, 14]}

def gates(base, dcfield):
    dmf = {}
    for l in (LAY_Q['write'] + LAY_Q['port']
              + LAY_Q['ctrl']):
        for dc in ('P', 'A1'):
            k = '%s_L%02d' % (dc, l)
            dmf[k] = (z[dcfield + '_' + k][:, -1]
                      - base['A1' if dc == 'A1'
                             else 'P'])
    mc = np.median(np.abs(np.concatenate(
        [dmf['P_L%02d' % l]
         for l in LAY_Q['ctrl']]
        + [dmf['A1_L%02d' % l]
           for l in LAY_Q['ctrl']]))).astype(
        np.float64)
    g = {}
    for grp in ('write', 'port'):
        meds = [np.median(np.abs(
            dmf['%s_L%02d' % (dc, l)]))
            for l in LAY_Q[grp]
            for dc in ('P', 'A1')]
        g[grp] = float(np.median(meds)) / max(
            float(mc), 1e-12)
    return g, mc

base32 = {'P': z25['mlg_s0_P'][:672, 36, -1],
          'A1': z25['mlg_s0_A1'][:672, 36, -1]}
g32, mc32 = gates(base32, 'dmq')
base64 = {k: v.astype(np.float64)
          for k, v in base32.items()}
g64, mc64 = gates(base64, 'dmq')
pb = res['part_b']
out.append('gates32 %s vs result %s' %
           ({k: round(v, 6)
             for k, v in g32.items()},
            {k: round(v, 6)
             for k, v in
             pb['gates'].items()}))
out.append('gates32 diff: w %.3e p %.3e'
           % (abs(g32['write']
                  - pb['gates']['write']),
              abs(g32['port']
                  - pb['gates']['port'])))
out.append('gates64 diff: w %.3e p %.3e'
           % (abs(g64['write']
                  - pb['gates']['write']),
              abs(g64['port']
                  - pb['gates']['port'])))
out.append('mc32 diff %.3e mc64 diff %.3e'
           % (abs(float(mc32)
                  - pb['ctrl_median']),
              abs(float(mc64)
                  - pb['ctrl_median'])))
io.open(r'D:\AI2050\Ai2050-OpenOne'
        r'\gpt5_temp\p3127_dm_probe3.txt',
        'w', encoding='utf-8').write(
    chr(10).join(out))
print('PROBE3_OK')
