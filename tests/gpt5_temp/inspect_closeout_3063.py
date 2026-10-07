# inspect ledger state + memory dir (closeout prep for 3063)
import json
import os

OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\ledger_state_3063.txt'
lines = []

lp = (r'D:\AI2050\Ai2050-OpenOne'
      r'\research\gpt5\atlas\atlas_ledger.json')
with open(lp, encoding='utf-8') as f:
    d = json.load(f)
ms = d.get('measurements', [])
cn = d.get('L14', {}).get('connects', [])
lines.append('n_meas=%d' % len(ms))
lines.append('last_meas_id=%s' % (ms[-1].get('id') if ms else None))
lines.append('last_meas_keys=%s' % (sorted(ms[-1].keys())
                                    if ms else None))
lines.append('n_conn=%d' % len(cn))
if cn:
    lines.append('last_conn=%s' % json.dumps(
        cn[-1], ensure_ascii=False)[:400])
lines.append('ledger_sha=%s' % d.get('ledger_sha256_8'))
lines.append('top_keys=%s' % sorted(d.keys()))

md = (r'D:\AI2050\Ai2050-OpenOne'
      r'\.workbuddy\memory')
lines.append('memdir_exists=%s' % os.path.isdir(md))
if os.path.isdir(md):
    lines.append('memdir_files=%s' % sorted(os.listdir(md)))

# audit file check
ap = (r'D:\AI2050\Ai2050-OpenOne'
      r'\research\gpt5\docs'
      r'\hdmcc_knowledge_map_review_20260921.md')
lines.append('audit_exists=%s' % os.path.exists(ap))
if os.path.exists(ap):
    with open(ap, encoding='utf-8') as f:
        atxt = f.read()
    lines.append('audit_len=%d' % len(atxt))
    lines.append('audit_tail=%s' % atxt[-500:].replace('\n', ' | '))

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('WROTE', OUT)
