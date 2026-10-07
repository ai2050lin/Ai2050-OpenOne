# -*- coding: utf-8 -*-
"""生成 Phase 8 执行冻结档 execution_phase8.json（观测前）：面板分层划分 + 配对 + 配置哈希。"""
import os, io, json, hashlib
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MDIR = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
OUT = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase8', 'execution_phase8.json')

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
def ids_of(s):
    return tok.encode(s, add_special_tokens=False)

GROUPS = {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
}
SUP_ID = {s: ids_of(s)[0] for s in GROUPS if len(ids_of(s)) == 1}
SUPS = [s for s in GROUPS if s in SUP_ID]
CLASS_TOKENS = set(SUP_ID.values())

INST = []
for sup in SUPS:
    for wd in GROUPS[sup]:
        ii = ids_of(wd)
        if len(ii) != 1:
            continue
        # Phase 7 教训：实例 token 与类 token 撞车须剔除类别 token 后再判
        if ii[0] in CLASS_TOKENS:
            continue
        INST.append((wd, sup))
assert len(INST) == 41, len(INST)

# 分层划分：每类前 4 个 -> discovery，其余 -> confirmation（顺序即 GROUPS 顺序，确定性）
discovery, confirmation = [], []
seen = {}
for wd, sup in INST:
    k = seen.get(sup, 0)
    (discovery if k < 4 else confirmation).append([wd, sup])
    seen[sup] = k + 1
assert len(discovery) == 24 and len(confirmation) == 17, (len(discovery), len(confirmation))

# 配对（在 41 全表上，规则同 N2h1）：i -> j=(i+n//2)%n 跨类推进
IDX = {wd: i for i, (wd, s) in enumerate(INST)}
n = len(INST)
PAIRS = []
for i, (rw, rs) in enumerate(INST):
    j = (i + n // 2) % n
    while INST[j][1] == rs:
        j = (j + 1) % n
    dw, ds = INST[j]
    same = [x for x, s in INST if s == rs and x != rw]
    sw = same[(i * 3) % len(same)] if same else None
    PAIRS.append([rw, rs, dw, ds, sw])

def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()

cfg = io.open(os.path.join(MDIR, 'config.json'), encoding='utf-8').read()
cfgj = json.loads(cfg)

execu = {
    'phase': 8,
    'frozen_at': '2026-10-01 21:40',
    'model': 'qwen3-4b',
    'model_dir': 'models/hf/qwen3-4b',
    'config_sha256': sha(os.path.join(MDIR, 'config.json')),
    'tok_sha256': sha(os.path.join(MDIR, 'tokenizer.json')) if os.path.isfile(
        os.path.join(MDIR, 'tokenizer.json')) else None,
    'expected_cfg': {k: cfgj.get(k) for k in ['num_hidden_layers', 'hidden_size', 'num_attention_heads',
                                              'head_dim', 'intermediate_size', 'vocab_size',
                                              'tie_word_embeddings']},
    'o_proj_in_features': cfgj['num_attention_heads'] * cfgj['head_dim'],
    'template': '%s是一种',
    'seed': 20261001,
    'sup_id': SUP_ID,
    'classes': SUPS,
    'instances_all': [[a, b] for a, b in INST],
    'discovery': discovery,
    'confirmation': confirmation,
    'pairs_all': PAIRS,
    'PATCH_L': [1, 3, 5, 6, 7, 9, 12, 15, 17, 20, 24, 27, 30, 33, 34],
    'primary_layer': 6,
    'control_layer': 7,
    'pre_layer': 5,
    'n_heads': cfgj['num_attention_heads'],
    'head_dim': cfgj['head_dim'],
    'arms': ['T_transport', 'Z_necessity', 'LOO', 'L7_aggregate', 'W_weight_space',
             'V_rand', 'M_mismatch', 'R_rand', 'S_self'],
    'smoke_env': 'SMOKE=1 -> 2 实例 x 3 臂 + 形状/维数/NaN 断言',
    'result_keys': ['base_ok', 'T', 'Z', 'LOO', 'L7', 'W', 'controls', 'shares', 'gate',
                    'verdict', 'flags', 'elapsed_s'],
}
assert cfgj['num_attention_heads'] * cfgj['head_dim'] == 4096
io.open(OUT, 'w', encoding='utf-8').write(json.dumps(execu, ensure_ascii=False, indent=1))
print('wrote', OUT, os.path.getsize(OUT), 'bytes')
print('discovery', len(discovery), 'confirmation', len(confirmation), 'pairs', len(PAIRS))
print('config_sha256', execu['config_sha256'][:16])
