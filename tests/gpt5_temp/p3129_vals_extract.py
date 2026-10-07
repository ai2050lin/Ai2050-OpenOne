# -*- coding: utf-8 -*-
"""Extract p3129 result.json full dump to txt."""
import io
import json

OUTD = (r'D:\AI2050\Ai2050-OpenOne\tests'
        r'\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3129'
        r'\omega_p127_dose_sweep_gen_'
        r'decouple_s0full_symbolfield')

r = json.load(io.open(OUTD + r'\result.json',
                      encoding='utf-8'))
txt = json.dumps(r, ensure_ascii=False,
                 indent=1, sort_keys=True,
                 default=str)
with io.open(r'D:\AI2050\Ai2050-OpenOne'
             r'\tests\gpt5_temp'
             r'\p3129_vals.txt', 'w',
             encoding='utf-8') as f:
    f.write(txt)
print('VALS_OK %d chars' % len(txt))
