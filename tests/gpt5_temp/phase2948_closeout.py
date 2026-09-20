# -*- coding: utf-8 -*-
"""Phase 2948 closeout: seal + ledger + MEMO + logs. Idempotent."""
import hashlib
import json
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2948', 'wov_head_gain')
GPT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5'
        r'\docs\AGI_GPT5_MEMO.md')
LEDGER = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5'
          r'\atlas\atlas_ledger.json')
WLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\2026-09-19.md')
WMEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\MEMORY.md')
SCRIPT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
          r'\phase2948_wov_head_gain.py')

res = json.load(open(os.path.join(OUT, 'result.json'),
                     encoding='utf-8'))


def sha8(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


paths = {
    'execution.json': os.path.join(OUT, 'execution.json'),
    'result.json': os.path.join(OUT, 'result.json'),
    'wov_head_gain.npz':
        os.path.join(OUT, 'wov_head_gain.npz'),
    'script': SCRIPT,
}
hashes = {k: sha8(v) for k, v in paths.items()}
stamp = json.load(open(paths['execution.json'],
                       encoding='utf-8'))['created']
verdict = res['final_verdict']
runtime = res['runtime_s']

# ---------- 1. seal ----------
seal = ['Phase 2948 seal', 'verdict: ' + verdict,
        'created: ' + stamp, 'runtime_s: ' + str(runtime)]
seal += [k + ' sha256-8 ' + v for k, v in hashes.items()]
seal.append('weights: L16/L17 v_proj+o_proj from '
            'model-00002-of-00003.safetensors')
open(os.path.join(GPT, 'phase2948_seal_report.txt'), 'w',
     encoding='utf-8').write('\n'.join(seal) + '\n')
print('seal written')

# ---------- 2. ledger ----------
led = json.load(open(LEDGER, encoding='utf-8'))
mid = 'meas_2948_wov_head_gain'
if not any(m['meas_id'] == mid
           for m in led['measurements']):
    led['measurements'].append({
        'meas_id': mid,
        'type': 'zero_forward_linear_head_gain',
        'verdict': verdict,
        'source': 'phase2948/wov_head_gain'})
    l14 = [lk for lk in led['linkage']
           if lk['link_id'] == 'L14_readout_spectrum_cross_model'
           ][0]
    if mid not in l14['connects']:
        l14['connects'].append(mid)
led.pop('ledger_sha256_8', None)
new_h = hashlib.sha256(json.dumps(
    led, sort_keys=True, ensure_ascii=False)
    .encode('utf-8')).hexdigest()[:8]
led['ledger_sha256_8'] = new_h
json.dump(led, open(LEDGER, 'w', encoding='utf-8'),
          indent=2, ensure_ascii=False)
print('ledger done, hash ' + new_h)

# ---------- 3. MEMO ----------
memo = open(MEMO, encoding='utf-8').read()
if '## Phase 2948' not in memo:
    section = '''## Phase 2948: W_ov 线性头增益与头级秩序判决 [''' + stamp + ''']

**为什么做**：2947 定位开关到少数高杠杆头（L17 集中 h22/h19；L16 量大而分散 h17/h19/13/27；共享促进头 19/22）。遗留问题：逐头效应秩序是否由线性 W_ov 读出增益预测？是 → 头层级是线性特征输运属性；否 → 由非线性/注意力项承载。

**方法**：零前向。g_h = median_w u35·(Wov_h @ xdir_w)，Wov_h = Wo[:, h] @ Wv[kv(h)]（**GQA：32 query 头共享 8 KV 头，h//4 同组**）；perm null（20000 次置换 D，seed 2904）定阈。锚 **7/7**：a1 GQA 形状门（v_proj (1024,2560) / o_proj (2560,4096)）、a2/a3 冻结产物重载 bit 级 0、a4 D 以 2947 npz 全精度为权威源 + result.json round(v,2) 一致性门 5.01e-3（实测 4.98e-03）、a5 xdir 自检 9.95e-14、a7 显式循环交叉核对 rel 8.11e-10。runtime 10.8 s。

**勘误（correction_note 入 PREREG）**：run1 a4 误把全精度 npz 与 round(v,2) 的 json 值做 bit 级比较（4.98e-03 不可达）；run2 a7 绝对阈 1e-9 撞 float64 2560 项累加顺序噪声（实测 1.73e-09）——改为相对阈 1e-8。两次均为判据可达性错误（纪律 10 同族），run3 权威。

**T 检验（判决 `linear_head_gain_confirmed`，quasi-post-hoc 机制整合，纪律 9）**：
| 检验 | rho | perm-p | null p95 | 判定 |
|---|---|---|---|---|
| T1 L17 | **0.5594** | **5.0e-4** | 0.2969 | ✓ |
| T2 L16 | **0.3776** | 1.6e-2 | 0.2966 | ✓ |

**结构发现**：
- top5 重叠：L17 4/5（g: 0,7,24,22,19 vs D: 22,19,0,7,10）；L16 3/5（g: 13,16,1,17,6 vs D: 17,19,13,27,16）。
- 共享促进头 g 排名：h19 在 L17 rank5 / L16 rank6；h22 在 L17 rank4 但 L16 rank17——h22 的 L16 主导地位**超出线性 W_ov 预测**（非线性/注意力贡献）。
- cos(ov(xdir_mean), u35) 仅 0.02-0.09、cos(v3) ≈ 0——单头增益是对 u35 的小投影聚合，与 v3 旋转目标无关。

**结论**：**头级秩序的骨架是线性的**——W_ov 特征输运逐头增益显著预测实测消融效应（两独立层 perm 显著），头层级不是任意非线性涌现；但 L16-h22 类偏离（幅度主导但线性排名低）表明**幅度仍由非线性/注意力门控放大**。与 2930 呼应：rho/秩序约定不变（线性骨架），maxT/幅度约定相对（非线性放大）。机制链 2936→2948 补上第十二环：头级秩序环。

**文件+SHA256-8**：execution @EXE@ / result @RES@ / npz @NPZ@ / script @SCR@。产物 `tests/glm5/result/rdc_query_construction_20260913/phase2948/wov_head_gain/`。runtime @RT@ s。

**接续**：2949 候选——A 头级剂量充分性：仅增强 top-g 头（h0/7/22/19 at L17）的定向放大是否复现开关（一次前向）；B 头级秩序跨层泛化（L14/L15/L18 的 g vs D，零前向）；C 承重带跨模型复现（glm4）；D L16-h22 的非线性来源分解（attn 权重捕获，一次前向）。

'''
    section = (section
               .replace('@EXE@', hashes['execution.json'])
               .replace('@RES@', hashes['result.json'])
               .replace('@NPZ@', hashes['wov_head_gain.npz'])
               .replace('@SCR@', hashes['script'])
               .replace('@RT@', str(runtime)))
    with open(MEMO, 'a', encoding='utf-8') as f:
        f.write(section)
memo2 = open(MEMO, encoding='utf-8').read()
print('memo 2948 present:', '## Phase 2948' in memo2,
      '| placeholders left:',
      '@EXE@' in memo2[memo2.find('## Phase 2948'):])

# ---------- 4. workspace log ----------
wl = open(WLOG, encoding='utf-8').read()
if 'Phase 2948' not in wl:
    entry = ('- Phase 2948 闭环：W_ov 线性头增益 vs 2947 实测 D（零前向，GQA 修正后 Wov_h=Wo[:,h]@Wv[kv(h)]，perm 20000 定阈，锚 7/7）。'
             '判决 linear_head_gain_confirmed：L17 rho 0.5594 perm-p 5.0e-4、L16 rho 0.3776 perm-p 1.6e-2，双双超 null p95≈0.297；'
             'top5 重叠 L17 4/5、L16 3/5；h22 在 L16 的主导地位超线性预测（非线性放大）。勘误 run1 a4 round(v,2) 口径、run2 a7 累加噪声阈值，run3 权威。'
             'Ledger 87 条 / L14 connects 55 / ledger ' + new_h + '。runtime ' + str(runtime) + ' s。\n')
    with open(WLOG, 'a', encoding='utf-8') as f:
        f.write(entry)
print('wlog done:',
      'Phase 2948' in open(WLOG, encoding='utf-8').read())

# ---------- 5. MEMORY.md ----------
mem = open(WMEM, encoding='utf-8').read()
if 'max=2948' not in mem:
    mem = mem.replace('max=2947', 'max=2948')
    mem = mem.replace('下一个 2948', '下一个 2949')
    marker = '判据函数必须带结构门（如 `o_proj.in_features == NH*HD` 断言）。'
    if marker in mem:
        mem = mem.replace(
            marker,
            marker + '2948 延伸：零前向锚的阈值要按量纲设——全精度 npz vs round(v,2) json 不能 bit 级比（rounding 门 5.01e-3）；大向量内积的显式/矩阵累加顺序噪声 ~1e-9，交叉核对用相对阈。',
            1)
    with open(WMEM, 'w', encoding='utf-8') as f:
        f.write(mem)
mem2 = open(WMEM, encoding='utf-8').read()
print('wmem max=2948:', 'max=2948' in mem2,
      '| 2948 ext:', '2948 延伸' in mem2)
