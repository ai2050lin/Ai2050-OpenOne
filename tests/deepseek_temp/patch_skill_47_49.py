# -*- coding: utf-8 -*-
"""向 rdc-phase-closeout/SKILL.md 追加教训 47–49（Q05 实证）。文件为 LF-only，保持之。"""
import hashlib

P = r'C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md'
b = open(P, 'rb').read()
print('before bytes', len(b), 'sha8', hashlib.sha256(b).hexdigest()[:8],
      'bare_lf', b.count(b'\n') - b.count(b'\r\n'))
t = b.decode('utf-8')

anchor = '\n\n## 参照实现（Phase 3125'
assert t.count(anchor) == 1, 'anchor count=%d' % t.count(anchor)

NEW = '''
47. **4-bit 量化臂必须配「同模型双精度桥」，且门阈要在量化观测前预注册（Q05 实证，2026-10-03）**
   (a) 目标模型 bf16 无法常驻（显存/内存）时，量化是唯一路径；但量化会改 logit ⇒ 与既有 **bf16 基线（如 E_read）不可直接比**。
   (b) **固定做法**：留一个**同模型**的 bf16 vs 量化 对照臂（Q05 用 qwen3-4b 双跑），把 `max_k |Δrel|` 当作**精度桥**，
       门阈在**量化臂观测之前**冻结进 `*_prereg_*.json`（非事后订）。
   (c) 通过 ⇒ 量化臂可与 bf16 同轴定量并排；不通过 ⇒ 量化臂**降级 descriptive-only**，只报形状与量级。
   (d) Q05 实测 `max|Δrel|=0.0489`（门 0.05）——**紧贴门**。结论再"PASS"也必须**同时报出与门的余量**，否则看不出脆弱性。

48. **「bf16 常驻失败」时先探 RAM —— CPU-offload 可能同样不可行，别默认它是 fallback（Q05 实证）**
   (a) 14B bf16 ≈ 29.55 GB > 16 GB 显存，直觉是 offload；但主机**总 RAM 仅 33.7 GB（可用 18.6）** ⇒ 29.55 GB 权重的
       CPU-offload **同样装不下**。offload 不是免费午餐：它需要 ≥ 权重体量的可寻址内存。
   (b) **固定做法**：开跑前先跑资源探针（模型体量 + `GlobalMemoryStatusEx` 可用 RAM + 量化/offload 后端版本 +
       **实测前向延迟**），再决定量化/offload。
   (c) 探针顺带用 `cells×(K+1)×latency` 换算全面板耗时，避免"跑一半才发现要 10 h"。
   (d) Q05 实测：4b=8.06 GB、14B=29.55 GB、9B=18.82 GB；14B nf4 加载 31 s、9.97 GB VRAM、0.045 s/fwd。

49. **字典哈希域必须 JSON 往返稳定：整数键 vs 字符串键会让 `res_sha8` 无法从产物复算（Q05 实证）**
   (a) 生产者在内存里用**整数键**建 `{k: v}`（k=0..16）；`json.dumps(sort_keys=True)` 按**数值**排 0,1,2,…,16。
       落盘后 JSON 键变**字符串**，复核再读按**字典序**排 0,1,10,11,…,16,2,… ⇒ 字节不同 ⇒ 复核**伪报**「res_sha8 不符」。
   (b) **固定做法**：建库即用字符串键（`{str(k): v}`）；或入哈希域前先 `json.loads(json.dumps(obj))` 归一化。
   (c) 复核脚本必须**显式记录**哈希域的键型约定；**数据级复核须独立于该约定**（如由 `per_seed` 重算聚合量，Q05 max_dev=0）。
   (d) 排查信号：self-hash 不符、但文件 sha8 稳定、且字段里**恰有 k 索引字典** ⇒ 先查键型而不是查文件。
'''

t2 = t.replace(anchor, NEW + anchor)
assert t2 != t
open(P, 'w', encoding='utf-8', newline='\n').write(t2)
b2 = open(P, 'rb').read()
print('after  bytes', len(b2), 'sha8', hashlib.sha256(b2).hexdigest()[:8],
      'bare_lf', b2.count(b'\n') - b2.count(b'\r\n'))
assert b2[:len(b)] == b, 'prefix changed!'
import re
print('lessons now:', re.findall(r'^\s*(4[0-9])\.\s+\*\*', b2.decode('utf-8'), re.M))
print('SKILL_PATCH_OK')
