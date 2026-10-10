# -*- coding: utf-8 -*-
# 3160 独立磁盘复核: 独立进程 re-hash + 内容抽查, 结果写 txt
import io, json, os, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
PDIR = os.path.join(RDIR, 'phase3160', 'g4p3_consumption_mechanism')
OUTP = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3160_verify_out.txt')
P = []
F = []

def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

def check(tag, cond, detail=''):
    (P if cond else F).append('%s %s' % (tag, detail))

# 1. ledger
led = json.loads(io.open(LEDGER, encoding='utf-8').read())
ms = led['measurements']
e3160 = [m for m in ms if m.get('phase') == 3160]
check('ledger.n=312', len(ms) == 312, 'n=%d' % len(ms))
check('ledger.3160.unique', len(e3160) == 1)
check('ledger.3160.verdict', e3160 and e3160[0]['verdict'].startswith('g4p3_attention_reallocation_primary'))
check('ledger.chain', led.get('ledger_sha256_8') == '9417b14f', led.get('ledger_sha256_8'))
n_earlier = [m.get('phase') for m in ms[:311]]
check('ledger.earlier_intact', 3159 in n_earlier and 3158 in n_earlier and 40 in n_earlier)

# 2. MEMO
b = open(MEMO, 'rb').read()
t = b.decode('utf-8')
check('memo.bom', b[:3] == b'\xef\xbb\xbf')
check('memo.3160sec', '## Phase 3160: 消耗机制判别（G4-P3）' in t)
check('memo.3161prereg', '预注册 Phase 3161' in t)
check('memo.eol_crlf', b.count(b'\n') == b.count(b'\r\n'),
      'lf=%d crlf=%d' % (b.count(b'\n'), b.count(b'\r\n')))
for key, frag in (('memo.4b.sha', '4b res **6853c671** seal 3c4a601b'),
                  ('memo.summary.sha', 'summary res **a52e2ddd** seal 3ce5cc3b'),
                  ('memo.zero.sha', 'zero res **a9435ded** seal 3e1501f0'),
                  ('memo.3160verdict', 'g4p3_attention_reallocation_primary|fp_ok')):
    check(key, frag in t)

# 3. 产物 re-hash vs MEMO 记录
disk_expect = {
    'res_qwen3-4b': '022ce3b3', 'npz_qwen3-4b': '26da4d18',
    'res_qwen3-14b': '889ccffd', 'npz_qwen3-14b': '4231760d',
    'res_glm4': '67200f8d', 'npz_glm4': '725de043',
    'res_summary': '2d97d24b', 'res_zero': '8795648d', 'npz_zero': '2ce05f06',
}
paths = {
    'res_qwen3-4b': ('qwen3-4b', 'result.json'), 'npz_qwen3-4b': ('qwen3-4b', 'collect.npz'),
    'res_qwen3-14b': ('qwen3-14b', 'result.json'), 'npz_qwen3-14b': ('qwen3-14b', 'collect.npz'),
    'res_glm4': ('glm4', 'result.json'), 'npz_glm4': ('glm4', 'collect.npz'),
    'res_summary': ('summary', 'result_summary.json'),
    'res_zero': ('zero', 'result_zero.json'), 'npz_zero': ('zero', 'collect_zero.npz'),
}
for k, exp in disk_expect.items():
    p = os.path.join(PDIR, paths[k][0], paths[k][1])
    got = sha8_file(p) if os.path.exists(p) else 'MISSING'
    check('rehash.' + k, got == exp, '%s vs %s' % (got, exp))

# 4. result 内容抽查（三模型 verdict + summary gates）
for m in ('qwen3-4b', 'qwen3-14b', 'glm4'):
    r = json.load(open(os.path.join(PDIR, m, 'result.json'), encoding='utf-8'))
    check('result.%s.class' % m, r['mech_class'] == 'attention_reallocation_primary', r['mech_class'])
    check('result.%s.anchor' % m, r['det']['anchor_bitwise_3157'] is True)
    rec = r['recover']
    check('result.%s.recover_gate' % m, abs(rec['mlp_mid1']) < 0.1 and abs(rec['mlp_mid1_2']) < 0.1,
          'mid1=%.4f mid12=%.4f' % (rec['mlp_mid1'], rec['mlp_mid1_2']))
rs = json.load(open(os.path.join(PDIR, 'summary', 'result_summary.json'), encoding='utf-8'))
check('summary.verdict', rs['verdict'].startswith('g4p3_attention_reallocation_primary|fp_ok'), rs['verdict'])
check('summary.fpmin', rs['fpmin_q50_none'] >= 0.8, '%.4f' % rs['fpmin_q50_none'])
check('summary.gates', rs['gates']['fingerprint'] and rs['gates']['class_agreement'])
rz = json.load(open(os.path.join(PDIR, 'zero', 'result_zero.json'), encoding='utf-8'))
check('zero.fpmin', rz['fpmin_q50'] >= 0.8, '%.4f' % rz['fpmin_q50'])
check('zero.bigdrop', rz['big_drop_slots'] == {'qwen3-4b': 18, 'qwen3-14b': 20, 'glm4': 20},
      str(rz['big_drop_slots']))

# 5. daily + workspace MEMORY
daily = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
check('daily.3160', '3160 消耗机制判别闭环' in io.open(daily, encoding='utf-8').read())
mp_txt = io.open(os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md'), encoding='utf-8').read()
check('memory.3160', '3160 消耗机制判别闭环' in mp_txt)
check('memory.orphan1', '。：unembed 谱平坦' not in mp_txt)
check('memory.orphan2', '。：k∈{0,1,2,4' not in mp_txt)
check('memory.3159.kept', '**✅ 3159 等价类动力学闭环（2026-10-09）**' in mp_txt)

out = ['PASS %d' % len(P)] + ['  ' + x for x in P] + ['FAIL %d' % len(F)] + ['  ' + x for x in F]
io.open(OUTP, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('VERIFY DONE')
