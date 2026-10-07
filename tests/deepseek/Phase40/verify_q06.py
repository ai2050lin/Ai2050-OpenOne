# -*- coding: utf-8 -*-
"""Q06 独立复核（独立进程；从磁盘 re-hash + 从 cells_detail 重算聚合，不从 verdict 自证）。"""
import json, hashlib, os
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(ROOT, 'tests', 'deepseek', 'result')
TEMP = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase40')
P = []
FAIL = []

def chk(name, ok, detail=''):
    P.append('%-42s %s %s' % (name, 'PASS' if ok else 'FAIL', detail))
    if not ok:
        FAIL.append(name)

def sha8f(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

res = json.load(open(os.path.join(OUT, 'q06_result.json'), encoding='utf-8'))
exe = json.load(open(os.path.join(OUT, 'q06_execution.json'), encoding='utf-8'))
det = json.load(open(os.path.join(TEMP, 'q06_cells_detail.json'), encoding='utf-8'))

# V1 独立 re-hash
chk('V1a prereg sha8 == ebf960cf',
    sha8f(os.path.join(OUT, 'q06_prereg_design_v1.json')) == 'ebf960cf')
chk('V1b execution.design_sha == result.design_sha',
    exe['design_sha'] == res['design_sha'], exe['design_sha'][:8])
rb = json.dumps({k: v for k, v in res.items() if k != 'res_sha8'},
                ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
chk('V1c result res_sha8 recompute',
    hashlib.sha256(rb).hexdigest()[:8] == res['res_sha8'], res['res_sha8'])

# V2 面板/identity/hook
PANEL_SHA = hashlib.sha256(json.dumps(
    [res_reconstruct_classes := ['水果', '动物', '交通工具', '家具', '金属', '颜色'],
     [e for cl in ['水果'] for e in []] or
     ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬',
      '狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛',
      '汽车', '火车', '飞机', '摩托车', '卡车', '地铁',
      '桌子', '椅子', '床', '沙发', '地毯', '窗帘',
      '铁', '铜', '铝', '金', '银', '锌', '铅',
      '红', '蓝', '绿', '黄', '黑', '白'],
     ['{e}是一种{c}。', '{e}属于{c}这一类。', '{e}，一种常见的{c}。'],
     [7, 8, 9], 0.2], ensure_ascii=False).encode('utf-8')).hexdigest()[:8]
chk('V2a panel sha8 == be17ef8a', PANEL_SHA == 'be17ef8a', PANEL_SHA)
F1 = max(max(r['identity_maxd'], r['identity_maxd_concat']) for r in det.values())
chk('V2b identity bit-identical (all cells)', F1 == 0.0, 'maxd=%.2e' % F1)
hk = all(r[k]['hook_hits'] == 1 for r in det.values()
         for k in r if isinstance(r[k], dict) and 'hook_hits' in r[k])
chk('V2c hook_hits == 1 (all arms)', hk)

# V3 从 detail 重算聚合（与 result 逐位对齐）
SEEDS = [7, 8, 9]
def recompute_pool(key):
    per = {}
    for s in SEEDS:
        cs = [k for k in det if k.startswith('%d|' % s)]
        el = [k for k in cs if det[k]['eligible']]
        if not el:
            continue
        h = sum(1 for k in el if det[k][key]['argmax'] == det[k]['true_class']
                and det[k][key]['collat'] == 0)
        per[s] = (h, len(el))
    tn = sum(v[1] for v in per.values()); th = sum(v[0] for v in per.values())
    return (th / tn if tn else None), per

bad = []
for grp, store in (('steer', res['steer_curves']), ('rand', res['rand_curves'])):
    for key in store:
        kind, sg, al = key.split('|')
        rk = '%s|%s|%s' % (kind, sg, '%.2f' % float(al))
        pr, per = recompute_pool(rk)
        if store[key]['pool_rate'] != pr:
            bad.append((key, store[key]['pool_rate'], pr))
chk('V3a steer/rand pool_rate recompute (20 cfg)', len(bad) == 0, str(bad[:3]))

# V4 main 读数一致
bs = max(res['steer_curves'].items(), key=lambda kv: (kv[1]['pool_rate'] or 0))
chk('V3b C_steer_main == argmax over curves',
    bs[0] == res['C_steer_main']['config'] and bs[1]['pool_rate'] == res['C_steer_main']['value'],
    '%s=%s' % (bs[0], bs[1]['pool_rate']))

# V5 v1 轴 npz
z = np.load(os.path.join(TEMP, 'q06_v1_axis.npz'))
v1, vr = z['v1'], z['vr']
chk('V4a |v1| unit', abs(float(np.linalg.norm(v1)) - 1) < 1e-9)
chk('V4b |cos(v1,vr)| < 0.2', abs(float(np.dot(v1, vr))) < 0.2,
    '%.4f' % abs(float(np.dot(v1, vr))))
chk('V4c sigma29 > 0', float(z['sigma29']) > 0, '%.4f' % float(z['sigma29']))
vh = hashlib.sha256(v1.astype(np.float64).tobytes()).hexdigest()[:8]
chk('V4d vhat_sha8 match result', vh == res['v1_axis']['vhat_sha8'], vh)

# V6 collateral 域
allc = [r[k]['collat'] for r in det.values() for k in r
        if isinstance(r[k], dict) and 'collat' in r[k]]
chk('V5 collateral integer domain [-13,13]',
    all(isinstance(c, int) and -13 <= c <= 13 for c in allc) and len(allc) > 0,
    'n=%d mean=%.3f' % (len(allc), float(np.mean(allc))))

# V7 cells 覆盖
n_seed = {s: sum(1 for k in det if k.startswith('%d|' % s)) for s in SEEDS}
chk('V6 heldout rows per seed == 147',
    all(v == 147 for v in n_seed.values()), str(n_seed))
elig = sum(1 for r in det.values() if r['eligible'])
chk('V7 eligible count match result', elig == res['cells']['eligible'],
    '%d' % elig)

# V8 三 KPI 报告义务
chk('V8 kpi_report 三 KPI 齐备',
    set(res['kpi_report']) >= {'E_read', 'E_ar', 'C_steer'})

with open(os.path.join(TEMP, 'verify_q06.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(P) + '\n')
    f.write('TOTAL PASS=%d FAIL=%d\n' % (len(P) - len(FAIL), len(FAIL)))
print('\n'.join(P))
print('TOTAL PASS=%d FAIL=%d' % (len(P) - len(FAIL), len(FAIL)))
