# -*- coding: utf-8 -*-
# 3162 sha 补丁: 三个文件中的 stale sha (ff46fd86/56771169) -> 真实值 (00f15e98/fc7bdd80), 重算 ledger chain
import io, json, hashlib, os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
MP = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
OUTP = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3162_shafix_out.txt')
out = []

# 真实值来源: summary result
sres = json.load(io.open(os.path.join(ROOT, 'tests', 'glm5', 'result',
                 'rdc_query_construction_20260913', 'phase3162', 'g5a1_atlas_foundation',
                 'summary', 'result_summary.json'), encoding='utf-8'))
R_NEW, H_NEW = sres['registry_sha8'], sres['html_sha8']
assert (R_NEW, H_NEW) == ('00f15e98', 'fc7bdd80'), (R_NEW, H_NEW)
OLD_R, OLD_H = 'ff46fd86', '56771169'

def patch_text(p, enc='utf-8'):
    t = io.open(p, encoding=enc).read()
    cr = t.count(OLD_R)
    ch = t.count(OLD_H)
    t = t.replace(OLD_R, R_NEW).replace(OLD_H, H_NEW)
    with io.open(p, 'w', encoding=enc) as f:
        f.write(t)
    return cr, ch

# MEMO (bytes-safe: 全文 decode -> replace -> encode, BOM 保留)
b = open(MEMO, 'rb').read()
t = b.decode('utf-8')
cr, ch = t.count(OLD_R), t.count(OLD_H)
assert (cr, ch) == (1, 1), ('MEMO stale counts', cr, ch)
t = t.replace(OLD_R, R_NEW).replace(OLD_H, H_NEW)
open(MEMO, 'wb').write(t.encode('utf-8'))
out.append('MEMO: ff46fd86->%s x%d, 56771169->%s x%d' % (R_NEW, cr, H_NEW, ch))

# ledger (json 字段级替换)
led = json.loads(io.open(LEDGER, encoding='utf-8').read())
n_fix = 0
for m in led['measurements']:
    if m.get('phase') == 3162 and isinstance(m.get('detail'), str):
        assert m['detail'].count(OLD_R) == 1 and m['detail'].count(OLD_H) == 1
        m['detail'] = m['detail'].replace(OLD_R, R_NEW).replace(OLD_H, H_NEW)
        n_fix += 1
assert n_fix == 1, n_fix
blob = json.dumps(led, ensure_ascii=False, indent=1, sort_keys=False).encode('utf-8')
led['ledger_sha256_8'] = hashlib.sha256(blob).hexdigest()[:8]
json.dump(led, io.open(LEDGER, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
out.append('ledger: detail patched, chain=%s, n=%d' % (led['ledger_sha256_8'], len(led['measurements'])))

# workspace MEMORY
cr2, ch2 = patch_text(MP)
assert (cr2, ch2) == (1, 1), ('MEMORY stale counts', cr2, ch2)
out.append('MEMORY: ff46fd86->%s x%d, 56771169->%s x%d' % (R_NEW, cr2, H_NEW, ch2))

# 复核
t2 = io.open(MEMO, encoding='utf-8').read()
m2 = io.open(MP, encoding='utf-8').read()
led2 = json.loads(io.open(LEDGER, encoding='utf-8').read())
ok = (OLD_R not in t2 and OLD_H not in t2 and R_NEW in t2 and H_NEW in t2
      and OLD_R not in m2 and H_NEW not in m2 and R_NEW in m2
      and any(m.get('phase') == 3162 and R_NEW in m.get('detail', '') for m in led2['measurements'])
      and led2['ledger_sha256_8'] == led['ledger_sha256_8']
      and len(led2['measurements']) == 313)
out.append('VERIFY: %s' % ('OK' if ok else 'FAIL'))
io.open(OUTP, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('SHAFIX DONE')
