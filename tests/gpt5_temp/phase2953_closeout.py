# -*- coding: utf-8 -*-
"""Phase 2953 closeout: seal + Ledger + MEMO + worklog + MEMORY.
Idempotent, placeholder replacement."""
import hashlib
import json
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2953', 'a11_s_response')
LEDGER = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\2026-09-19.md')
WMEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\MEMORY.md')
SCRIPT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
          r'\phase2953_a11_s_response.py')

res = json.load(open(os.path.join(OUT, 'result.json'),
                     encoding='utf-8'))
verdict = res['final_verdict']
created = json.load(open(os.path.join(OUT, 'execution.json'),
                         encoding='utf-8'))['created']


def sha8(path):
    return hashlib.sha256(
        open(path, 'rb').read()).hexdigest()[:8]


npz_name = 'a11_s_response.npz'
hashes = {
    'execution.json': sha8(os.path.join(OUT, 'execution.json')),
    'result.json': sha8(os.path.join(OUT, 'result.json')),
    npz_name: sha8(os.path.join(OUT, npz_name)),
    'script': sha8(SCRIPT),
}

# ---------------- Ledger ----------------
led = json.load(open(LEDGER, encoding='utf-8'))
meas_id = 'meas2953_a11_s_response_sigmoid'
if not any(m['meas_id'] == meas_id
           for m in led['measurements']):
    led['measurements'].append({
        'meas_id': meas_id,
        'type': 'forward_family_sigmoid_response',
        'verdict': verdict,
        'source': ('phase2953/a11_s_response; '
                   'keep-head aggregation caveat per '
                   'discipline: keep set excludes '
                   'high-leverage heads'),
    })
    l14 = [lk for lk in led['linkage']
           if lk['link_id'] == 'L14_readout_spectrum_cross_model'][0]
    if meas_id not in l14['connects']:
        l14['connects'].append(meas_id)
led.pop('ledger_sha256_8', None)
new_h = hashlib.sha256(json.dumps(
    led, sort_keys=True, ensure_ascii=False)
    .encode('utf-8')).hexdigest()[:8]
led['ledger_sha256_8'] = new_h
json.dump(led, open(LEDGER, 'w', encoding='utf-8'),
          indent=2, ensure_ascii=False)
chk = new_h

# ---------------- MEMO ----------------
memo = open(MEMO, encoding='utf-8').read()
if '## Phase 2953' not in memo:
    t1 = res['T1_sigmoid_fit']
    t2 = res['T2_threshold_lock']
    t3 = res['T3_steepness']
    d3 = res['D3_vs_2945']
    anc = res['anchors']
    section = '''## Phase 2953: A11 路由 sigmoid 与开关阈值解耦判决 [@STAMP@]

**判决：`@VERDICT@`** —— A11(s) 两层均为真实 sigmoid（T1 全过：L17 R²=0.9992 k=3.291 / L16 R²=0.9995 k=2.623），陡度分化 T3 过（L17 3.291 > L16 2.623，开关型更陡）；但路由中点 s_t 与 sep 阈值 s_c 仅在 L16 重合（|Δ|=0.1969 < 0.3），L17 显著滞后（s_t 1.2219 vs s_c 0.6567，|Δ|=0.5653 > 0.3）——**开关型层的读出塌缩在路由 sigmoid 早期侧翼触发，不是集体路由中点事件**。

**设计（一次前向族，runtime @RT@s）**：L17/L16 单层 xdir 注入（2942/2945/2952 verbatim），细网格 GRID17={0.25..2.0 十点、近 s_c≈0.66 加密} / GRID16={0.75..2.0 十点、近 s_c≈1.84 加密} + s=0 共 11 曲线点；K=3 同 session 中位 sep（2945 口径），A11 线模型恢复（2952 recover verbatim）。锚 **13/13**：a1 2.17e-08（**第 24 次连续前向锚定**）、a10/a11 base 与共享点 A11 vs 2952 **bit 级 0**、a13 共享 12 点 sep vs 2945 **bit 级 0**、a8 确定性 2.84e-14、a12 恢复残差 1.62e-01、a4/a5 vs 2935 7.21e-06/6.26e-06。

**主检验**：
- T1 sigmoid：L17 s_t=1.2219（a_lo 0.0224 / a_hi 0.8576）；L16 s_t=1.6445（a_lo 0.0266 / a_hi 0.3600）
- T2 lock：s_c 同 session 复算 0.6567/1.8413（vs 2945 的 0.656/1.843，插值口径 bit 级复现）；L17 fail / L16 pass
- T3 陡度：k_L17=3.291 > k_L16=2.623 pass

**解剖（D2 逐头，keep 集）**：
- **头集口径警告（教训 24）**：keep_L17 不含高杠杆头（h22/h19/h0 被 2951 非退化门排除）——"keep-中位路由"是排除最强头后的口径；2952/2953 的 A11 中位一致（s=1.0 均 0.3066，bit 级）但均非全体头中位。
- **早翻转头 = 2947 抵抗头**：h20（s=0.375 已 0.61、s=0.5 已 0.96）、h21（0.71/0.98）在 s_c=0.657 处已完全翻转，而 keep-中位仅完成跳变 ~15-20%（s_c 处中位 A11≈0.16，全程 0.047→0.80）；h1（最强抵抗 −50.7）为慢 sigmoid（0.03→0.96 跨全网格）。2947 的"抵抗"（消融→塌缩加深）与"最早翻转"自洽：这些头最早读入注入词值，但其 u35 读出方向对抗塌缩。
- L16：10/18 keep 头 s_t 落在 s_c±0.3；h26（2950 重平衡载体）s=0.25 已 0.37 极早翻转；h31（抵抗）几乎不翻转（0.04→0.16 缓升）；h27（促进）base A11 已 0.79（高基线自注意）。
- L17 中位曲线 s=2 达 0.80、接近拟合平台 0.858——外推可信；h20/h21/h1 三头 curve_fit 数值失败（极端陡/非标准形），登记描述性。

**结论（重复 3 次）**：**开关的头级载体拆解完成：路由跳变是真实 sigmoid，但开关型层（L17）的 sep 阈值由早期侧翼的少数早翻转头（h20/h21，恰为 2947 抵抗头）触发——读出塌缩不需要集体路由中点；渐变型层（L16）阈值才与路由中点重合。路由跳变（微观）与 sep 开关（宏观）之间是"特定头子集先翻、读出聚合响应"的因果链，不是同一事件的两面。与 2947/2949/2950 合读：任何"中位/集体"层面的重合都不能外推到头级因果。**

**硬伤与勘误**：无 run 失败（一次通过）。收尾期曲线探针曾报 FileNotFoundError 假象——排查为探针路径漏产物子目录（`\\phase2953\\a11_s_response.npz` 应为 `\\phase2953\\a11_s_response\\a11_s_response.npz`），非文件系统缺陷、非沙箱问题；教训：产物路径必须含 `\\phase{N}\\{arm}\\` 完整两层。Ledger 92 条 / L14 connects 60 / ledger @LEDHASH@。

**文件+SHA256-8**：execution @HEXE@ / result @HRES@ / npz @HNPZ@ / script @HSCR@。产物 `tests/glm5/result/rdc_query_construction_20260913/phase2953/a11_s_response/`。

**接续**：机制链第十六环（路由-阈值关系环）闭合。候选 2954：A 早翻转头读出极性验证（h20/h21 的 W_ov 方向与 u35 符号，零前向——解释"最早翻转却对抗塌缩"）；B q·k 来源分解（2952 遗留：注入对 q/k 的直接改动 vs softmax 增益，一次前向）；C 承重带跨模型复现（glm4）；D v3 解码器方向重启（2940 遗留）。
'''
    stamp = created.replace('T', ' ')[:16]
    reps = [('@STAMP@', stamp), ('@VERDICT@', verdict),
            ('@RT@', str(res['runtime_s'])),
            ('@LEDHASH@', chk),
            ('@HEXE@', hashes['execution.json']),
            ('@HRES@', hashes['result.json']),
            ('@HNPZ@', hashes[npz_name]),
            ('@HSCR@', hashes['script'])]
    for k, v in reps:
        section = section.replace(k, v)
    memo += section
    open(MEMO, 'w', encoding='utf-8').write(memo)

# ---------------- worklog ----------------
wl = open(WLOG, encoding='utf-8').read()
if 'Phase 2953' not in wl:
    wl += ('- Phase 2953 闭环：a11_sigmoid_threshold_decoupled。'
           'A11(s) 两层均真实 sigmoid（R²≥0.999，k 2.6/3.3），'
           '但 L17 路由中点 s_t=1.22 显著滞后 sep 阈值 s_c=0.66'
           '（开关在路由早期侧翼触发、由早翻转头 h20/h21 即 2947 '
           '抵抗头承载），L16 才重合（|Δ|=0.20）；T3 L17 更陡过。'
           '教训 24：keep 集排除高杠杆头，聚合口径必须登记。'
           'Ledger 92 / hash ' + chk + '。\n')
    open(WLOG, 'w', encoding='utf-8').write(wl)

# ---------------- MEMORY ----------------
mem = open(WMEM, encoding='utf-8').read()
if '当前 max=2952' in mem:
    mem = mem.replace('当前 max=2952，下一个 2953',
                      '当前 max=2953，下一个 2954')
if '24. **2953' not in mem:
    lesson = ('24. **2953（聚合头集口径门）**：median/中位类聚合'
              '统计必须显式登记头集口径——keep 集（非退化门）系统性'
              '排除高杠杆/极端头（keep_L17 不含 h22/h19/h0），'
              '"keep-中位"≠"全体-中位"；跨相位交叉引用（2947 促进/'
              '抵抗头 ↔ 2953 早翻转头）必须先对齐头集。科学结论：'
              'A11(s) 路由跳变是真实 sigmoid（R²≥0.999），但开关型层'
              '（L17）sep 阈值在早期侧翼由少数早翻转头触发（h20/h21'
              ' 即 2947 抵抗头，s_c 处中位路由仅完成 ~15-20%），'
              '渐变型层（L16）阈值才与路由中点重合——微观路由与宏观'
              '开关是"子集先翻、聚合响应"因果链，非同一事件。\n')
    i23 = mem.find('23. **2952')
    if i23 >= 0:
        iend = mem.find('\n', i23)
        mem = mem[:iend + 1] + lesson + mem[iend + 1:]
    else:
        mem = mem.rstrip() + '\n' + lesson
open(WMEM, 'w', encoding='utf-8').write(mem)

print('closeout done: ledger=%s n_meas=%s' % (
    chk, len(led['measurements'])))
