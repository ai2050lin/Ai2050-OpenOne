# -*- coding: utf-8 -*-
# Phase 3175 pre-freeze probe: measure all anchors, confirm 3172 smoke result
# structure, sanity-check the out-panel vocabulary (string level).
import io
import json
import os
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
P3169 = os.path.join(RDIR, 'phase3169', 'g5a6_oov_panel')
P3172 = os.path.join(RDIR, 'phase3172', 'g5a9_port_calibration')
P3174 = os.path.join(RDIR, 'phase3174', 'g5a11_audit')
OUT = []

def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

# ---- 1. file sha8 of everything we will anchor ----
files = {
    'p3169_result': os.path.join(P3169, 'result.json'),
    'p3169_smoke': os.path.join(P3169, 'smoke_result.json'),
    'p3169_4b': os.path.join(P3169, 'collect_qwen3-4b.npz'),
    'p3169_14b': os.path.join(P3169, 'collect_qwen3-14b.npz'),
    'p3169_glm4': os.path.join(P3169, 'collect_glm4-9b.npz'),
    'p3169_smoke_npz': os.path.join(P3169, 'collect_smoke_qwen3-4b.npz'),
    'p3172_result': os.path.join(P3172, 'result.json'),
    'p3172_smoke': os.path.join(P3172, 'smoke_result.json'),
    'p3172_exec': os.path.join(P3172, 'execution.json'),
    'p3174_prereg': os.path.join(P3174, 'prereg_3175_structural_residual_draft.json'),
    'p3174_result': os.path.join(P3174, 'result.json'),
}
for kk, p in files.items():
    OUT.append('file %s = %s (%d B)' % (kk, sha8(p), os.path.getsize(p)))

# ---- 2. content sha fields ----
R69 = json.load(io.open(files['p3169_result'], encoding='utf-8'))
S69 = json.load(io.open(files['p3169_smoke'], encoding='utf-8'))
R72 = json.load(io.open(files['p3172_result'], encoding='utf-8'))
S72 = json.load(io.open(files['p3172_smoke'], encoding='utf-8'))
R74 = json.load(io.open(files['p3174_result'], encoding='utf-8'))
OUT.append('')
OUT.append('p3169 result content: res=%s seal=%s' % (R69['res_sha8'], R69['seal_sha8']))
OUT.append('p3169 smoke content: res=%s seal=%s' % (S69['res_sha8'], S69['seal_sha8']))
OUT.append('p3172 result content: res=%s seal=%s design=%s' % (R72['res_sha8'], R72['seal_sha8'], R72['design_sha8']))
OUT.append('p3172 smoke content: res=%s seal=%s' % (S72['res_sha8'], S72['seal_sha8']))
OUT.append('p3174 result content: res=%s seal=%s' % (R74['res_sha8'], R74['seal_sha8']))
OUT.append('p3169 gate: %s' % json.dumps(R69['gate']))
OUT.append('p3169 per_model keys: %s' % ','.join(R69['per_model'].keys()))
for mk in ('qwen3-4b', 'qwen3-14b', 'glm4-9b'):
    b = R69['per_model'][mk]['B']
    OUT.append('p3169 %s B: %s' % (mk, json.dumps({k: b[k] for k in ('E_oov', 'E_seen', 'E_newent')})))
OUT.append('p3169 per_model readout: %s' % json.dumps({mk: R69['per_model'][mk]['readout'] for mk in R69['per_model']}))

# ---- 3. 3172 pooled + per-model curves (in-arm replay reference) ----
OUT.append('')
OUT.append('p3172 verdict: %s' % R72['verdict'])
OUT.append('p3172 pooled keys: %s' % sorted(R72['pooled'].keys()))
for k in sorted(R72['pooled'].keys(), key=int):
    OUT.append('p3172 pooled[%s]: %s' % (k, json.dumps(R72['pooled'][k])))
for mk in R72['per_model']:
    kc = R72['per_model'][mk]['kcurves']
    OUT.append('p3172 %s kout=%s NL=%s' % (mk, R72['per_model'][mk].get('kout'), R72['per_model'][mk].get('NL')))
    for k in sorted(kc.keys(), key=int):
        v = kc[k]
        OUT.append('  k%s: E_seen=%.9f E_oov=%.9f E_newent=%.9f ratio=%.9f' % (
            k, v['E_seen'], v['E_oov'], v['E_newent'], v['ratio']))
OUT.append('')
OUT.append('p3172 smoke verdict: %s' % S72['verdict'])
OUT.append('p3172 smoke k_grid: %s' % json.dumps(S72['k_grid']))
for k in sorted(S72['pooled'].keys(), key=int):
    OUT.append('p3172 smoke pooled[%s]: %s' % (k, json.dumps(S72['pooled'][k])))
for mk in S72['per_model']:
    kc = S72['per_model'][mk]['kcurves']
    for k in sorted(kc.keys(), key=int):
        v = kc[k]
        OUT.append('p3172 smoke %s k%s: E_seen=%.9f E_oov=%.9f E_newent=%.9f ratio=%.9f' % (
            mk, k, v['E_seen'], v['E_oov'], v['E_newent'], v['ratio']))

# ---- 4. out-panel vocabulary sanity (string level) ----
CLASSES_SEEN = ['水果', '动物', '交通工具', '家具', '金属', '颜色']
ENT_SEEN = {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
}
CLASSES_OOV = ['乐器', '天气', '运动', '电器']
ENT_OOV_PANEL = {
    '乐器': ['钢琴', '小提琴', '吉他', '鼓', '笛子', '二胡', '琵琶', '口琴'],
    '天气': ['雨', '雪', '雷', '雾', '冰雹', '台风', '露水', '霜'],
    '运动': ['足球', '篮球', '乒乓球', '游泳', '跑步', '体操', '拳击', '围棋'],
    '电器': ['电视', '冰箱', '洗衣机', '空调', '微波炉', '电饭煲', '吸尘器', '风扇'],
}
ENT_OOV_OUT = {
    '乐器': ['长笛', '竖琴', '唢呐', '大提琴', '手风琴', '萨克斯', '木琴', '锣', '钹', '竖笛'],
    '天气': ['彩虹', '闪电', '暴雨', '微风', '寒潮', '热浪', '沙尘暴', '霜冻', '梅雨', '阴天'],
    '运动': ['网球', '排球', '跳水', '滑雪', '射箭', '击剑', '马拉松', '瑜伽', '跳高', '举重'],
    '电器': ['烤箱', '豆浆机', '加湿器', '电吹风', '热水器', '打印机', '电熨斗', '榨汁机', '路由器', '电磁炉'],
}
allwords = []
for d in (ENT_SEEN, ENT_OOV_PANEL, ENT_OOV_OUT):
    for v in d.values():
        allwords.extend(v)
dups = sorted(set(w for w in allwords if allwords.count(w) > 1))
OUT.append('')
OUT.append('vocab total=%d unique=%d dups=%s' % (len(allwords), len(set(allwords)), dups))
OUT.append('out pool sizes: %s' % json.dumps({c: len(v) for c, v in ENT_OOV_OUT.items()}))
# per-class first-token distinctness across 10 classes needs the tokenizer;
# string-level checks here:
assert len(set(allwords)) == len(allwords), 'word collision'
for c in CLASSES_OOV:
    assert c not in CLASSES_SEEN
panel_words = set(w for d in (ENT_SEEN, ENT_OOV_PANEL) for v in d.values() for w in v)
out_words = [w for v in ENT_OOV_OUT.values() for w in v]
coll = [w for w in out_words if w in panel_words]
assert not coll, coll
OUT.append('string-level vocab sanity OK (no dup, no panel overlap)')
OUT.append('out words: %s' % ' '.join(out_words))

with io.open(os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3175_probe.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(OUT) + '\n')
print('done')
