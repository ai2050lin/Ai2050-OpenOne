# -*- coding: utf-8 -*-
"""
Phase 14 / N2h1-alpha-7 : seal amend1 —— 模型配置字段勘误（SMOKE 触发，正式运行前冻结）
=============================================================================
不改动原 seal 字节（保住审计链），另立 amend 文件记录纠正。
原 seal：tests/deepseek_temp/Phase14/N2h1a7_design_seal.json（sha8 074fc963）
输出：  tests/deepseek_temp/Phase14/N2h1a7_design_seal_amend1.json
"""
import os, io, json, hashlib, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P14T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase14')
SEAL = os.path.join(P14T, 'N2h1a7_design_seal.json')
OUT = os.path.join(P14T, 'N2h1a7_design_seal_amend1.json')


def sha(p):
    return hashlib.sha256(io.open(p, 'rb').read()).hexdigest()


seal_b = io.open(SEAL, 'rb').read()
SEAL_SHA = hashlib.sha256(seal_b).hexdigest()
S = json.load(io.open(SEAL, encoding='utf-8'))

# 运行探针得到的地面真值（本 amend 的唯一依据，来自 config/module 直读）
import torch
from transformers import AutoModelForCausalLM
MDIR = os.path.join(ROOT, 'models', 'hf', S['model']['name'])
m = AutoModelForCausalLM.from_pretrained(MDIR, dtype=torch.bfloat16, trust_remote_code=True,
                                         attn_implementation='eager')
CFG = m.config
_core = getattr(m.model, 'language_model', m.model)
A = _core.layers[S['model'].get('n_layers', 36) // 5].self_attn
OPROJ = None
for nm in ['o_proj', 'dense', 'out_proj']:
    if hasattr(A, nm):
        OPROJ = getattr(A, nm); break
FACTS = {
    'num_hidden_layers': int(CFG.num_hidden_layers),
    'hidden_size': int(CFG.hidden_size),
    'num_attention_heads': int(CFG.num_attention_heads),
    'num_key_value_heads': int(getattr(CFG, 'num_key_value_heads', CFG.num_attention_heads)),
    'head_dim': int(CFG.head_dim),
    'intermediate_size': int(CFG.intermediate_size),
    'vocab_size': int(CFG.vocab_size),
    'tie_word_embeddings': bool(CFG.tie_word_embeddings),
    'o_proj_module': type(OPROJ).__name__,
    'o_proj_in_features': int(OPROJ.in_features),
    'o_proj_out_features': int(OPROJ.out_features),
    'derived_n_heads_times_head_dim': int(CFG.num_attention_heads * CFG.head_dim),
    'is_gqa': bool(int(getattr(CFG, 'num_key_value_heads', CFG.num_attention_heads)) != int(CFG.num_attention_heads)),
}

amend = {
    'phase': 14,
    'kind': 'schema_amend (no design change)',
    'name': 'N2h1-alpha-7 amend1：模型配置字段勘误（SMOKE 触发）',
    'amend_of_seal_sha256': SEAL_SHA,
    'amend_of_seal_sha8': SEAL_SHA[:8],
    'trigger': 'SMOKE（正式运行前）——装置前置断言 drift=[\'o_proj_in\'] 首次触发，未产生任何实验数据',
    'discovered_ground_truth': FACTS,
    'old_seal_values': {'head_dim': S['model']['head_dim'], 'n_heads': S['model']['n_heads']},
    'defect_root_cause': (
        '可行性探针 _feas_probe.py 把 head_dim 写成 HID // CFG.num_attention_heads（=2560/32=80），'
        '即【用 hidden/n_heads 反推】，而不是直读 CFG.head_dim（=128）；seal 与 exec 照抄了该派生值。'
        'qwen3-4b 是 GQA（num_key_value_heads=8 != num_attention_heads=32），'
        'o_proj.in_features = n_heads*head_dim = 32*128 = 4096，并不等于 hidden_size。'
        '=> 新装置坑（#47）：GQA 模型【不得】用 hidden/n_heads 反推 head_dim；'
        'o_proj 输入维必须由模块直读。'
    ),
    'corrected_fields': {
        'head_dim': FACTS['head_dim'],
        'n_heads': FACTS['num_attention_heads'],
        'n_kv_heads': FACTS['num_key_value_heads'],
        'o_proj_in_features': FACTS['o_proj_in_features'],
        'expected_cfg': {
            'num_hidden_layers': FACTS['num_hidden_layers'],
            'hidden_size': FACTS['hidden_size'],
            'num_attention_heads': FACTS['num_attention_heads'],
            'num_key_value_heads': FACTS['num_key_value_heads'],
            'head_dim': FACTS['head_dim'],
            'intermediate_size': FACTS['intermediate_size'],
            'vocab_size': FACTS['vocab_size'],
            'tie_word_embeddings': FACTS['tie_word_embeddings'],
        },
    },
    'what_is_NOT_changed': [
        '预注册 5 条预测 P1-P5 —— 不变',
        '判决表（同坐标 6 行 + 跨族 4 行）—— 不变',
        '14 条 floor F24-F37 —— 不变（仅 F26/F27 的期望常量随配置字段一并校正）',
        'arms 定义、α 网格、mask 语义、统计量定义、bootstrap 方案 —— 全部不变',
        'inheritance anchors 全部不变',
        'honesty 10 条不变（新增第 11 条记录本次勘误）',
    ],
    'why_no_HARKing_concern': (
        '本次勘误只把【config.json 的事实字段】从错误派生值改为直读值，'
        '不触及任何假设、判据、阈值或统计量选择；且发生在任何实验观测之前（SMOKE 阶段、零实验产物）。'
    ),
    'added_honesty_11': (
        '11. 本 Phase 的模型配置字段（head_dim/n_kv_heads/o_proj_in_features）由 SMOKE 触发勘误后冻结于 amend1；'
        '原始 seal 的错误派生值（head_dim=80）保留在案，供审计。勘误不涉及任何设计元素。'
    ),
    'frozen_at': time.strftime('%Y-%m-%d %H:%M:%S'),
}

with io.open(OUT, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(amend, f, ensure_ascii=False, indent=1)
    f.write('\n')

b = io.open(OUT, 'rb').read()
print('WROTE %s' % OUT)
print('  bytes = %d ; sha8 = %s' % (len(b), hashlib.sha256(b).hexdigest()[:8]))
print('  amend_of_seal_sha8 = %s' % SEAL_SHA[:8])
print('  ground truth: %s' % json.dumps({k: FACTS[k] for k in
      ['head_dim', 'num_attention_heads', 'num_key_value_heads', 'o_proj_in_features', 'is_gqa']},
      ensure_ascii=False))
