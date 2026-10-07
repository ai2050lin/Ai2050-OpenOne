# -*- coding: utf-8 -*-
"""
verify_q04.py —— Q04 独立进程复核（防幻影写；另写独立记录）
================================================================
复核项：
  V1  execution.json 的 design_sha 可复算（content-excluding-self 规范）
  V2  result.json 的 res_sha8 可复算
  V3  面板与 E_read 同一 held-out 族：split_s1 seeds[7,8,9] 的 test_pairs_sha8
      与 q03_result.json 的 fingerprint 逐项相同
  V4  E_ar(k) 与 per_seed 的三 seed 均值自洽
  V5  装置门 D1/D2/D3/S1 与 result 内数值自洽（可失败性检查）
  V6  Q05 预注册（S_rel）存在且已冻结
  V7  E_read 口径未被本 Phase 改动（metric_dict 的 E_read.current 与 Q03 一致）
  V8  产物齐全（report / result / execution）
输出: tests/deepseek/result/verify_q04.txt  (PASS/FAIL 计数)
"""
import os, json, hashlib, sys
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(ROOT, 'tests', 'deepseek', 'result')
SMOKE = True
QA = 'q04_smoke' if SMOKE else 'q04'
EX = os.path.join(OUT, QA + '_execution.json')
RE = os.path.join(OUT, QA + '_result.json')
RP = os.path.join(OUT, QA + '_report.txt')
Q03 = os.path.join(OUT, 'q03_result.json')
MD = os.path.join(ROOT, 'research', 'deepseek', 'atlas', 'metric_dict.json')

R = []
PASS = [0]; FAIL = [0]
def chk(name, ok, detail=''):
    (PASS if ok else FAIL)[0] += 1
    R.append('%s  %-58s %s' % ('PASS' if ok else 'FAIL', name, detail))

def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]

# ---------- V8 产物齐全 ----------
for p, nm in [(EX, 'execution'), (RE, 'result'), (RP, 'report')]:
    chk('V8 exists %s' % nm, os.path.exists(p), os.path.basename(p))

ex = json.loads(open(EX, 'rb').read().decode('utf-8-sig'))
re_ = json.loads(open(RE, 'rb').read().decode('utf-8-sig'))

# ---------- V1 design_sha 复算 ----------
d = dict(ex); claimed = d.pop('design_sha', None)
blob = json.dumps(d, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
calc = hashlib.sha256(blob).hexdigest()
chk('V1 design_sha recomputable', calc == claimed, '%s' % (claimed or 'MISSING')[:8])

# ---------- V2 res_sha8 复算 ----------
r2 = dict(re_); rc = r2.pop('res_sha8', None)
blob2 = json.dumps(r2, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
calc2 = hashlib.sha256(blob2).hexdigest()[:8]
chk('V2 res_sha8 recomputable', calc2 == rc, '%s' % rc)

# ---------- V3 与 E_read 同族 ----------
CLASSES = ['水果', '动物', '交通工具', '家具', '金属', '颜色']
ENT = {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
}
SEEDS = [7, 8, 9]; FRAC = 0.2
ENTS = [e for cl in CLASSES for e in ENT[cl]]
PAIRS = [(i, c) for i in range(len(ENTS)) for c in range(len(CLASSES))]
NP_ = len(PAIRS)
q3 = json.loads(open(Q03, 'rb').read().decode('utf-8-sig'))
fp3 = q3['fingerprint']
allm = True; det = []
for s in SEEDS:
    rng = np.random.RandomState(s)
    idx = rng.permutation(NP_)
    n_test = int(round(FRAC * NP_))
    test = set([PAIRS[j] for j in idx[:n_test]])
    h = sha8(json.dumps(sorted(map(list, test)), ensure_ascii=False).encode('utf-8'))
    ok = (h == fp3[str(s)]['test_pairs_sha8']) and (len(test) == fp3[str(s)]['n_test_pairs'])
    allm = allm and ok
    det.append('s%d=%s' % (s, 'ok' if ok else 'MISMATCH'))
chk('V3 held-out fold == Q03/E_read fingerprint', allm, ' '.join(det))

# ---------- V4 E_ar 与 per_seed 自洽 ----------
E = re_['E_ar']; PS = re_['per_seed']
agg_ok = True; mx = 0.0
for k in range(re_['K'] + 1):
    m = float(np.mean([PS[str(s)][str(k)]['mae_b4'] for s in SEEDS]))
    mx = max(mx, abs(m - E[str(k)]))
chk('V4 E_ar == mean_3seed(per_seed)', mx < 1e-12, 'max_dev=%.2e' % mx)

# ---------- V5 门自洽 + 可失败性 ----------
g = re_['gates']
chk('V5a D1 recorded True', g['D1'] is True)
chk('V5b D2 recorded True', g['D2'] is True)
chk('V5c D3 recorded True', g['D3'] is True)
s1_should = max(E[str(k)] for k in range(1, re_['K'] + 1)) >= g['S1_thr']
chk('V5d S1 recomputable', bool(g['S1']) == bool(s1_should),
    'max_{k>=1}=%.4f thr=%.2f' % (g['S1_max_k_ge1'], g['S1_thr']))
# 可失败性：S1 门必须存在一个“FAIL 可能的”读数 —— 阈值 > 0
chk('V5e S1 threshold non-trivial (>0)', float(g['S1_thr']) > 0.0)

# ---------- V6 Q05 预注册 ----------
qp = re_.get('q05_prereg', {})
chk('V6 q05_prereg.S_rel frozen', 'S_rel' in qp and
    qp['S_rel'].get('rule', '').find('E_ar_rel') >= 0,
    qp.get('S_rel', {}).get('rule', 'MISSING'))

# ---------- V7 E_read 未被动 ----------
md = json.loads(open(MD, 'rb').read().decode('utf-8-sig'))
cur = md['global_kpis']['E_read']['current']
q3c = q3['summary']['E_read']
same = all(abs(cur[m] - q3c[m]) < 1e-15 for m in cur)
chk('V7 E_read.current unchanged vs Q03', same,
    '/'.join('%.6f' % cur[m] for m in cur))

txt = ['Q04 独立复核报告（%s）' % QA, '=' * 78]
txt += R
txt.append('')
txt.append('TOTAL: %d PASS / %d FAIL %s' % (PASS[0], FAIL[0],
                                             'ALL_PASS' if FAIL[0] == 0 else 'HAS_FAIL'))
open(os.path.join(OUT, 'verify_q04.txt'), 'w', encoding='utf-8').write('\n'.join(txt) + '\n')
print('\n'.join(txt))
print('EXIT_OK')
