# -*- coding: utf-8 -*-
"""Extract all numbers from phase3125 result.json
for closeout script writing."""
import json, io

RJ = (r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result"
      r"\rdc_query_construction_20260913"
      r"\phase3125"
      r"\omega_p123_third_comp_qwen_inputstream"
      r"\result.json")
out = io.StringIO()
w = out.write

with open(RJ, 'r', encoding='utf-8') as f:
    ds = json.load(f)

w("=== scalars ===\n")
for k in ('phase', 'name', 'smoke', 'n_pairs',
          'np_b', 'runtime_s'):
    w("%s: %r\n" % (k, ds.get(k)))
w("verdict: %s\n\n" % ds.get('verdict'))

pa = ds['part_a']
w("=== part_a ===\n")
w("refit: %r\n" % pa['refit'])
for dc in ('P', 'A1'):
    d = pa['decomp3'][dc]
    w("decomp3[%s]: %r\n" % (dc, d))
w("cm_verdict: %r\n" % pa['cm_verdict'])
w("trail_verdict: %r\n" % pa['trail_verdict'])
w("ar_verdict: %r\n" % pa['ar_verdict'])
w("traj_verdict: %r\n" % pa['traj_verdict'])
for dc in ('P', 'A1'):
    w("trail_G[%s] (10x3):\n" % dc)
    for row in pa['trail_G'][dc]:
        w("  %r\n" % row)
    w("ar_params[%s]: %r\n" % (dc,
                               pa['ar_params'][dc]))
    w("traj_stat[%s]: %r\n" % (dc,
                               pa['traj_stat'][dc]))
w("sim3.r_sim3: %r\n" % pa['sim3']['r_sim3'])
w("sim3.verdict: %r\n" % pa['sim3']['verdict'])
w("sim3.auc_sim3: %r\n\n" % pa['sim3']['auc_sim3'])

pb = ds['part_b']
w("=== part_b ===\n")
w("interference: %r\n" % pb['interference'])
w("ids: %r\n" % pb['ids'])
w("path: %r\n" % pb['path'])
w("repro: %r\n" % pb['repro'])
w("readout_auc: %r\n" % pb['readout_auc'])
w("readout_verdict: %r\n"
  % pb['readout_verdict'])
w("n_span: %r\n" % pb['n_span'])
w("spans_verdict: %r\n" % pb['spans_verdict'])
w("lstar: %r\n" % pb['lstar'])
w("sign: %r\n" % pb['sign'])
for k in sorted(pb['curves'].keys()):
    w("curves.%s = %r\n" % (k, pb['curves'][k]))

with open(r"D:\AI2050\Ai2050-OpenOne"
          r"\tests\gpt5_temp"
          r"\p3125_vals.txt", "w",
          encoding="utf-8") as f:
    f.write(out.getvalue())
print("WROTE_OK")
