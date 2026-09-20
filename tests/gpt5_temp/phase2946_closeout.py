# -*- coding: utf-8 -*-
"""Phase 2946 closeout: seal + ledger + MEMO + logs. Idempotent."""
import hashlib
import json
import os
import time

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2946', 'dose_allocation')
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
          r'\phase2946_dose_allocation.py')

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
    'dose_allocation.npz':
        os.path.join(OUT, 'dose_allocation.npz'),
    'script': SCRIPT,
}
hashes = {k: sha8(v) for k, v in paths.items()}
stamp = json.load(open(paths['execution.json'],
                       encoding='utf-8'))['created']
verdict = res['final_verdict']
runtime = res['runtime_s']
chk = res['anchors']

# ---------- 1. seal report ----------
seal = ['Phase 2946 seal', 'verdict: ' + verdict,
        'created: ' + stamp, 'runtime_s: ' + str(runtime)]
seal += [k + ' sha256-8 ' + v for k, v in hashes.items()]
seal += ['anchor a1 ' + str(chk['a1_diff']),
         'a8 ' + str(chk['a8_max_spread']),
         'a9 ' + str(chk['a9_diff'])]
open(os.path.join(GPT, 'phase2946_seal_report.txt'), 'w',
     encoding='utf-8').write('\n'.join(seal) + '\n')
print('seal written')

# ---------- 2. ledger ----------
led = json.load(open(LEDGER, encoding='utf-8'))
mid = 'meas_2946_dose_allocation'
if not any(m['meas_id'] == mid
           for m in led['measurements']):
    led['measurements'].append({
        'meas_id': mid,
        'type': 'forward_family_dose_allocation',
        'verdict': verdict,
        'source': 'phase2946/dose_allocation'})
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
if '## Phase 2946' not in memo:
    section = '''## Phase 2946: 剂量分配交互与干扰判决 [''' + stamp + ''']

**为什么做**：2944 证明开关要求单层峰值浓度（同总量摊薄 5 层失效）；2945 测得各层独立阈值 s_c(L17)=0.656 / s_c(L15)=0.845 / s_c(L16)=1.843 且与传导量级解耦。遗留问题：双层联合注入固定总量按 (alpha, 1-alpha) 分配时，联合阈值服从局部浓度规则 min(s_c17/a17, s_c16/a16) 还是加权平均规则 a17*s_c17+a16*s_c16？判别点 J25（alpha17=0.25：预测 2.4573 vs 1.5462）。

**方法**：一次前向族。配置 J75/J50/J25（L17:L16 = 0.75:0.25 / 0.5:0.5 / 0.25:0.75）+ 单层参照 S17/S16；xdir 注入（v1/v2/v5，2942/2945 verbatim）；s 网格 {0.5..3.0} 八点；K=3 同 session 重复取中位；execution.json 先冻结，预测值取自 2945 T2（预注册前无任何 2946 观测）。锚 **9/9**：a1 dirs 重建 2.17e-08（**第 19 次连续前向锚定**）、a3 Vt8 bit 级 0、a7 9.95e-14、a8 同 session 确定性 2.84e-14、**a9（新）S17 参照曲线 vs 2945 D1_sep[L17] 共享 s 点最大差 0.0046**——陡降单调层跨 session 高度稳定。runtime 19.5 s。

**T1 主检验（判决 `switch_interaction_nonlinear`）**：两预注册规则全部失败——
| 配置 | 实测 s_c | pred_local | err_local | pred_avg | err_avg |
|---|---|---|---|---|---|
| J75 | 0.9448 | 0.8747 | 0.0701 | 0.9527 | 0.0079 |
| J50 | 1.6388 | 1.3120 | 0.3268(<0.35) | 1.2495 | 0.3893 |
| J25 | 1.8946 | 2.4573 | 0.5628 | 1.5462 | 0.3483(<0.35) |

规则通过要求全部 3 点 < 0.35：local 在 J25 爆炸（0.563），avg 在 J50 爆炸（0.389）。**无简单规则**。

**关键发现（干扰而非协同）**：实测联合阈值 0.945/1.639/1.895 全部**劣于最优单层**（L17 单独 0.656）——向次敏感层 L16 分配任何剂量都单调拉高联合阈值（25%: 0.945 → 50%: 1.639 → 75%: 1.895，趋近 L16 单独的 1.843）。两层竞争同一开关资源：亚阈剂量的共存不是"再加一条触发路径"（局部最小值规则预测的 J25 2.457 意味着 L17 份额被稀释后需要更大总量——实测 1.895 比 local 预测**更敏感**但比 avg 预测更迟钝），而是非线性干扰。sep 曲线形状佐证：J75 在 s=1.0 已达 88.9（深穿），J25 在 s=2.0 才 88.9；S16 参照曲线 vs 2945 描述性对比 max diff 仅 0.0（s=0.5..2.0 全部 bit 级一致，包括中增益区——2942 的跨 session 不稳定在本次两 session 间未再现，登记描述性）。

**硬伤与勘误**：run1 KeyError（sc_interp 内部用浮点键查字符串键字典，教训 18 同款键型混用）——run2 权威；sc_interp 已注明单一键规范。

**结论**：开关的完整操作画像最终成型：带内冗余(2944) + 每层独立浓度阈值与量级解耦(2945) + **双层联合非线性干扰、联合阈值被次敏感层单调拉高、无局部/平均简单规则**(2946)。单层强扰动仍是唯一已知的有效操作化方式——null 重编码的因果通道对任何分布式注入模式保持关闭。

**文件+SHA256-8**：execution @EXE@ / result @RES@ / npz @NPZ@ / script @SCR@。产物 `tests/glm5/result/rdc_query_construction_20260913/phase2946/dose_allocation/`。runtime @RT@ s。

**接续**：2947 候选——A 三层分配 (0.6,0.2,0.2) 检验干扰是否随层数累积；B 头级解剖 L16 渐变型 vs L17 开关型（层内头分解传导）；C 承重带跨模型复现（glm4）；D 2942 s=2 跨 session 不稳定源（多进程同机重跑）。

'''
    section = (section
               .replace('@EXE@', hashes['execution.json'])
               .replace('@RES@', hashes['result.json'])
               .replace('@NPZ@', hashes['dose_allocation.npz'])
               .replace('@SCR@', hashes['script'])
               .replace('@RT@', str(runtime)))
    with open(MEMO, 'a', encoding='utf-8') as f:
        f.write(section)
memo2 = open(MEMO, encoding='utf-8').read()
print('memo 2946 present:', '## Phase 2946' in memo2,
      '| placeholders left:',
      '@EXE@' in memo2[memo2.find('## Phase 2946'):])

# ---------- 4. workspace log ----------
wl = open(WLOG, encoding='utf-8').read()
if 'Phase 2946' not in wl:
    entry = ('- Phase 2946 闭环：剂量分配交互（L17,L16 双层联合注入 J75/J50/J25 + 单层参照，'
             's 网格 8 点 K=3，锚 9/9 含新 a9 跨 phase 参照 0.0046）。判决 '
             'switch_interaction_nonlinear：局部/平均两预注册规则均失败，联合阈值'
             '（0.945/1.639/1.895）全部劣于最优单层 0.656——次敏感层共存单调拉高阈值，'
             '干扰非协同。run1 sc_interp 键型混用 KeyError（教训 18 复发），run2 权威。'
             'Ledger 85 条 / L14 connects 53 / ledger ' + new_h + '。runtime ' + str(runtime) + ' s。\n')
    with open(WLOG, 'a', encoding='utf-8') as f:
        f.write(entry)
print('wlog done:',
      'Phase 2946' in open(WLOG, encoding='utf-8').read())

# ---------- 5. MEMORY.md ----------
mem = open(WMEM, encoding='utf-8').read()
if 'max=2946' not in mem:
    mem = mem.replace('max=2945', 'max=2946')
    mem = mem.replace('下一个 2946', '下一个 2947')
    if '2946 延伸' not in mem:
        old18 = mem.find('2945 延伸')
        # append extension line after the lesson-18 item
        marker = '（复合键写入/读取必须一次统一）'
        if marker in mem:
            mem = mem.replace(marker,
                              marker + '；2946 延伸：判据函数内部的键访问也要单一规范（sc_interp 浮点键查字符串键字典 KeyError）',
                              1)
        else:
            mem = mem.replace('2945 延伸',
                              '2946 延伸：判据函数内部键访问单一规范（sc_interp KeyError）。2945 延伸', 1)
    with open(WMEM, 'w', encoding='utf-8') as f:
        f.write(mem)
mem2 = open(WMEM, encoding='utf-8').read()
print('wmem max=2946:', 'max=2946' in mem2,
      '| 2946 ext:', '2946 延伸' in mem2)
