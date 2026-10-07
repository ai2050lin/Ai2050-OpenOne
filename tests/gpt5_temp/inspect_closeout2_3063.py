# inspect ledger measurement 201 + linkage tail (3063 closeout prep v2)
import json

OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\ledger_state2_3063.txt'
lines = []

lp = (r'D:\AI2050\Ai2050-OpenOne'
      r'\research\gpt5\atlas\atlas_ledger.json')
with open(lp, encoding='utf-8') as f:
    d = json.load(f)
ms = d['measurements']
lines.append('--- last measurement (full) ---')
lines.append(json.dumps(ms[-1], ensure_ascii=False, indent=1))
lines.append('--- meas_id of last 3: %s ---' %
             [m.get('meas_id') for m in ms[-3:]])
lines.append('--- phases of last 3: %s ---' %
             [m.get('phase') for m in ms[-3:]])

lk = d.get('linkage', {})
lines.append('--- linkage type/keys ---')
if isinstance(lk, dict):
    lines.append('linkage keys=%s' % sorted(lk.keys()))
    for k, v in lk.items():
        if isinstance(v, list):
            lines.append('  %s: list len=%d' % (k, len(v)))
            if v:
                lines.append('    last=%s' % json.dumps(
                    v[-1], ensure_ascii=False)[:500])
        else:
            lines.append('  %s: %s' % (k, type(v).__name__))
elif isinstance(lk, list):
    lines.append('linkage list len=%d' % len(lk))
    if lk:
        lines.append('last=%s' % json.dumps(
            lk[-1], ensure_ascii=False)[:500])

# L14 block?
l14 = d.get('L14')
lines.append('--- L14 ---')
lines.append('L14=%s' % json.dumps(l14, ensure_ascii=False)[:800])

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('WROTE', OUT)
