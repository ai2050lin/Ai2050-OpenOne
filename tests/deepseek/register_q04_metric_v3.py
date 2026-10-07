# -*- coding: utf-8 -*-
"""
register_q04_metric_v3.py —— 把 Q04 解析出的 E_ar 口径登记回 metric_dict（v2 -> v3）
=====================================================================================
纪律：
  - 只改 global_kpis.E_ar 与版本字段；E_read / C_steer / metrics / meta_rules **逐字不动**
    （保住 p3150_disk_verify 的 len(metrics)==7 与 meta_rules 键断言）
  - content_sha256_8 按规定复算：json.dumps(content_excluding_self, ensure_ascii=False, indent=1)
  - 数字一律从 q04_smoke_result.json 现场渲染
  - v2 先备份
"""
import os, json, hashlib, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
ATL = os.path.join(ROOT, 'research', 'deepseek', 'atlas')
MD = os.path.join(ATL, 'metric_dict.json')
MDV2 = os.path.join(ATL, 'metric_dict_v2_backup.json')
SMO = os.path.join(ROOT, 'tests', 'deepseek', 'result', 'q04_smoke_result.json')
SMEX = os.path.join(ROOT, 'tests', 'deepseek', 'result', 'q04_smoke_execution.json')

TS = time.strftime('%Y-%m-%d %H:%M')

def content_hash(d):
    c = dict(d); c.pop('content_sha256_8', None)
    return hashlib.sha256(json.dumps(c, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]

raw = open(MD, 'rb').read()
d = json.loads(raw.decode('utf-8-sig'))
v2_file_sha = hashlib.sha256(raw).hexdigest()[:8]
v2_content = d['content_sha256_8']
assert content_hash(d) == v2_content, 'v2 self-hash mismatch'
assert len(d['metrics']) == 7, 'metrics count changed'
assert d['global_kpis']['E_ar']['status'] == 'not_built', 'E_ar already built'
E_READ_BEFORE = dict(d['global_kpis']['E_read']['current'])

# 备份 v2（只备份一次，避免覆盖首次备份）
if not os.path.exists(MDV2):
    with open(MDV2, 'wb') as f:
        f.write(raw)
    print('backup v2 ->', os.path.basename(MDV2))

smo = json.loads(open(SMO, 'rb').read().decode('utf-8-sig'))
sex = json.loads(open(SMEX, 'rb').read().decode('utf-8-sig'))
g = smo['gates']

# ---------------- 登记 E_ar（resolved） ----------------
d['global_kpis']['E_ar'] = {
    "name_cn": "k 步自回归 logit-margin 误差",
    "space": "behavior-autoregressive",
    "definition": "把模型自身前 k 步生成内容回喂后，对第 k 步 logit-margin（目标 token 减竞争 token）的预测误差",
    "formula": "E_ar(k) = mean_cells | margin_pred(k; i, c) - margin_true(k; i, c) |，margin = logit(t_target) - logit(t_competitor)",
    "status": "device_built",
    "device_built_in": "Q04",
    "to_measure_in": "Q05（全面板三模型曲线）",
    "operationalization": {
        "margin": "logit(t_target) - logit(t_competitor)；t_target = 该 cell 真实类名首 token；t_competitor = k=0 处 logit 最高的非该类类首 token（每 cell 冻结）",
        "margin_true(k)": "模型自身前 k 步贪心生成内容回喂后，第 k 步 last-position 的 margin（k=0 = 模板前缀本身，无回喂）；纯自回归 rollout，无 teacher forcing",
        "margin_pred(k)": "与 E_read 同一 B4 加性族（one-hot[entity 41] + one-hot[class 6] + one-hot[template 3] + bias, ridge λ=1e-3）在 held-out 行上的预测",
        "predictor_family": "B4 = ridge_primal（与 E_read 同实现）",
        "target_units": "logit（原始 L1，不归一化）",
        "aggregation": "per seed: mean over held-out rows；then mean over 3 seeds",
        "rollout": "ctx0 = 模板前缀 token ids；每步读 6 类 logit；k<K 时 append argmax(logits_last) 回喂"
    },
    "data": {
        "panel": "与 E_read 同一 held-out 面板族：41 实体 × 6 类 = 246 pairs × 3 模板 = 738 行",
        "fold": "S1 seeds [7,8,9]，frac=0.2 -> 49 test pairs × 3 模板 = 147 held-out 行/seed",
        "carriers": "不需要既有 collect.npz；现场 logits（Q04 自采）",
        "fingerprint_cross_check": "test_pairs_sha8 与 Q03 E_read 指纹逐项相同（复核 V3 PASS）"
    },
    "k_range": {"K": 16, "reported": "0..K（k=0 为装置确定性锚）", "smoke_K": 4},
    "gate": {
        "device": {
            "D1": "k=0 确定性锚：同 cell 两遍 6 类 logit 逐位相同",
            "D2": "非退化：held-out margin_true std > 0",
            "D3": "存活性：全部 E_ar(k) 有限"
        },
        "S_smoke": {"rule": "max_{k>=1} E_ar(k) >= 0.05（**原始 logit**）", "role": "装置灵敏度门（非科学否证门）",
                    "observed": g['S1_max_k_ge1'], "thr": g['S1_thr'], "pass": g['S1']},
        "S_rel_q05_prereg": {
            "rule": "min_{k=1..K} E_ar_rel(k) <= 0.05，E_ar_rel = E_ar(k) / scale(k)",
            "scale": "held-out margin_true 的 std",
            "frozen_before": "Q05 任何观测（2026-10-03 Q04 冻结）",
            "rationale": "Q04 SMOKE 发现原始 logit 单位下 0.05 被 88× 平凡满足、不具否证力；按冻结纪律不追溯改 S1，另立相对形式为 Q05 科学门"
        }
    },
    "ci": "Q04 smoke 为装置验证，不报 CI；Q05 随全面板三模型报 per-seed 全列（3 seed 不折叠）",
    "current": {
        "smoke_only": True,
        "model": smo['model'],
        "cells": smo['n_cells'],
        "K": smo['K'],
        "forwards": smo['sum_fwd'],
        "E_ar": smo['E_ar'],
        "E_ar_rel": smo['E_ar_rel'],
        "gates": g,
        "verdict": smo['verdict'],
        "note": "SMOKE 子面板仅 10 实体（PAIRS[:60]），E_ar 偏大且 per-seed 方差大，数字**不可科学解读**；仅证明装置端到端可用"
    },
    "device_artifacts": {
        "device_script": "tests/deepseek/q04_e_ar_device.py",
        "execution": {"path": "tests/deepseek/result/q04_smoke_execution.json", "design_sha": sex['design_sha']},
        "result": {"path": "tests/deepseek/result/q04_smoke_result.json", "res_sha8": smo['res_sha8']},
        "verify": {"path": "tests/deepseek/result/verify_q04.txt", "tally": "14 PASS / 0 FAIL"}
    }
}

# ---------------- 版本字段 ----------------
d['supersedes'] = {
    "format": "metric_dict_v2",
    "sha8": v2_file_sha,
    "content_sha256_8": v2_content,
    "created": d.get('created'),
    "backup": "research/deepseek/atlas/metric_dict_v2_backup.json"
}
d['format'] = 'metric_dict_v3'
d['version'] = 3
d['frozen_at'] = TS
d['generated_by'] = 'tests/deepseek/register_q04_metric_v3.py'

# ---------------- 自洽断言（守住兼容） ----------------
assert len(d['metrics']) == 7
assert 'F2_verdict_grading' in d['meta_rules'] and 'F7_bit_anchor_status' in d['meta_rules']
assert all(abs(d['global_kpis']['E_read']['current'][m] - E_READ_BEFORE[m]) < 1e-15 for m in E_READ_BEFORE)
assert d['global_kpis']['C_steer']['status'] == 'not_measured'
ch = content_hash(d)
d['content_sha256_8'] = ch

with open(MD, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(d, f, ensure_ascii=False, indent=1)

rb = open(MD, 'rb').read()
d2 = json.loads(rb.decode('utf-8-sig'))
print('metric_dict v2 %s (%d B) -> v3 file_sha8=%s content=%s (%d B)'
      % (v2_file_sha, len(raw), hashlib.sha256(rb).hexdigest()[:8], d2['content_sha256_8'], len(rb)))
print('  self-hash reproducible =', content_hash(d2) == d2['content_sha256_8'])
print('  metrics=%d  E_ar.status=%s  E_read.current=%s'
      % (len(d2['metrics']), d2['global_kpis']['E_ar']['status'],
         '/'.join('%.6f' % d2['global_kpis']['E_read']['current'][m]
                  for m in ['qwen3-4b', 'qwen3-14b', 'glm4-9b'])))
print('DONE')
