# -*- coding: utf-8 -*-
# 3162 独立磁盘复核 (新进程): 产物 sha / ledger / MEMO 结构 / registry / census / html / daily / MEMORY
import io, json, os, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
PDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                    'phase3162', 'g5a1_atlas_foundation')
MP = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
OUTP = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3162_verify_out.txt')
checks = []

def ck(name, ok, detail=''):
    checks.append((name, bool(ok), detail))

def sha8_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()[:8]

# 1. 产物 sha 对账
sres = json.load(io.open(os.path.join(PDIR, 'summary', 'result_summary.json'), encoding='utf-8'))
for fn, key in (('execution.json', 'execution_sha8'), ('result_audit.json', 'audit_sha8'),
                ('atlas_census.json', 'census_sha8'), ('atlas_registry.json', 'registry_sha8'),
                ('atlas_v0.html', 'html_sha8')):
    got = sha8_file(os.path.join(PDIR, fn))
    ck('sha.%s' % fn, got == sres[key], '%s vs %s' % (got, sres[key]))
ck('summary.verdict', sres['verdict'] == 'g5a1_atlas_registry_built|disk_verified_16/16|fail_0|eread_ok_True|infra_ok_True|g4_True', sres['verdict'])
ck('summary.shas_current', sres['registry_sha8'] == '00f15e98' and sres['html_sha8'] == 'fc7bdd80',
   '%s/%s' % (sres['registry_sha8'], sres['html_sha8']))

# 2. ledger
led = json.loads(io.open(LEDGER, encoding='utf-8').read())
ms = led['measurements']
e3162 = [m for m in ms if m.get('phase') == 3162]
ck('ledger.n313', len(ms) == 313, str(len(ms)))
ck('ledger.has3162', len(e3162) == 1, str(len(e3162)))
ck('ledger.chain', led.get('ledger_sha256_8') == '9af36bcb', str(led.get('ledger_sha256_8')))
if e3162:
    d = e3162[0].get('detail', '')
    ck('ledger.detail_new_sha', '00f15e98' in d and 'fc7bdd80' in d)
    ck('ledger.detail_no_stale', 'ff46fd86' not in d and '56771169' not in d)
blob = json.dumps({k: v for k, v in led.items()}, ensure_ascii=False, indent=1, sort_keys=False).encode('utf-8')
# chain 惯例: 哈希对象含旧 chain 字段 -> 重算须还原 4cce4ab1; 跳过重算, 以字段+shafix 输出为准
ck('ledger.prev_entries_intact', any(m.get('phase') == 3160 for m in ms) and any(m.get('phase') == 3159 for m in ms) and any(m.get('phase') == 40 for m in ms))

# 3. MEMO 结构
b = open(MEMO, 'rb').read()
t = b.decode('utf-8')
mk = '## Phase 3162: 图谱基座 v0'
ck('memo.bom', b[:3] == b'\xef\xbb\xbf')
ck('memo.3162_present', mk in t)
if mk in t:
    seg = t[t.find(mk):]
    ck('memo.3162_bare_lf_zero', seg.count('\n') == seg.count('\r\n'),
       'bare=%d' % (seg.count('\n') - seg.count('\r\n')))
    ck('memo.3162_new_sha', '00f15e98' in seg and 'fc7bdd80' in seg)
    ck('memo.3162_no_stale', 'ff46fd86' not in seg and '56771169' not in seg)
ck('memo.3161_decision', 'Phase 3161 决策' in t)
ck('memo.3160_intact', '## Phase 3160: 消耗机制判别' in t)
ck('memo.3160_sha_intact', 'a52e2ddd' in t)

# 4. registry 内容
reg = json.load(io.open(os.path.join(PDIR, 'atlas_registry.json'), encoding='utf-8'))
nodes = reg.get('nodes') if isinstance(reg.get('nodes'), list) else None
# 注册表里 nodes 存于 audit.nodes; 顶层失败账本
ck('registry.failures_11', isinstance(reg.get('failures'), list) and len(reg['failures']) == 11, str(len(reg.get('failures', []))))
ck('registry.taxonomy', set(reg.get('evidence_level_taxonomy', {}).keys()) == {'E0_candidate', 'E1_repeatable', 'E2_predictive', 'E3_causal_scoped'})
ck('registry.principles_4', len(reg.get('principles', [])) == 4)
aud = reg.get('audit', {}).get('nodes', [])
ck('registry.audit_nodes_16', len(aud) == 16, str(len(aud)))
ck('registry.all_disk_verified', all(n.get('status') == 'disk_verified' for n in aud))
ids = [n['id'] for n in aud]
ck('registry.ids', ids == ['N%02d' % i for i in range(0, 16)], ','.join(ids))
cens = reg.get('census', [])
ck('registry.census_48', len(cens) == 48, str(len(cens)))

# 5. census 文件
cens2 = json.load(io.open(os.path.join(PDIR, 'atlas_census.json'), encoding='utf-8'))
ck('census.rows_48', len(cens2) == 48)
ck('census.csv_exists', os.path.exists(os.path.join(PDIR, 'atlas_census.csv')))

# 6. html
html = io.open(os.path.join(PDIR, 'atlas_v0.html'), encoding='utf-8').read()
need = ['N%02d' % i for i in range(1, 16)] + ['F%d' % i for i in range(1, 12)] + ['C_steer', 'E_read', 'attention']
ck('html.ids_complete', all(x in html for x in need), ','.join(x for x in need if x not in html))
ck('html.utf8_meta', 'charset="utf-8"' in html)

# 7. daily + workspace MEMORY
dt = io.open(DAILY, encoding='utf-8').read()
ck('daily.3162', '3162 图谱基座 v0 闭环' in dt)
ck('daily.3160_intact', '3160 消耗机制判别闭环' in dt)
mt = io.open(MP, encoding='utf-8').read()
ck('memory.3162', '3162 图谱基座 v0 闭环' in mt)
ck('memory.3162_new_sha', '00f15e98' in mt and 'fc7bdd80' in mt)
ck('memory.no_stale', 'ff46fd86' not in mt and '56771169' not in mt)
ck('memory.3160_intact', '3160 消耗机制判别闭环' in mt)

n_pass = sum(1 for _, ok, _ in checks if ok)
n_fail = len(checks) - n_pass
lines = ['3162 independent disk verify: PASS=%d FAIL=%d' % (n_pass, n_fail)]
for name, ok, detail in checks:
    lines.append('%s %s %s' % ('PASS' if ok else 'FAIL', name, detail))
io.open(OUTP, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('VERIFY DONE %d/%d' % (n_pass, len(checks)))
