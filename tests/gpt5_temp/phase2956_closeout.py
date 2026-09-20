# -*- coding: utf-8 -*-
"""Phase 2956 closeout: seal + Ledger + MEMO + worklog + MEMORY.
Idempotent, placeholder replacement."""
import hashlib
import json
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2956',
                   'rebalance_module_localization')
LEDGER = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\2026-09-19.md')
WMEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\MEMORY.md')
SCRIPT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
          r'\phase2956_rebalance_module_localization.py')

res = json.load(open(os.path.join(OUT, 'result.json'),
                     encoding='utf-8'))
verdict = res['final_verdict']
created = json.load(open(os.path.join(OUT, 'execution.json'),
                         encoding='utf-8'))['created']


def sha8(path):
    return hashlib.sha256(
        open(path, 'rb').read()).hexdigest()[:8]


npz_name = 'rebalance_module_localization.npz'
hashes = {
    'execution.json': sha8(os.path.join(OUT, 'execution.json')),
    'result.json': sha8(os.path.join(OUT, 'result.json')),
    npz_name: sha8(os.path.join(OUT, npz_name)),
    'script': sha8(SCRIPT),
}

# ---------------- Ledger ----------------
led = json.load(open(LEDGER, encoding='utf-8'))
meas_id = 'meas2956_rebalance_module_localization'
if not any(m['meas_id'] == meas_id
           for m in led['measurements']):
    led['measurements'].append({
        'meas_id': meas_id,
        'type': 'forward_family_module_localization',
        'verdict': verdict,
        'source': ('phase2956/rebalance_module_localization; '
                   'competitive rebalancing (dSep + D_abl) '
                   'split by module via true-residual chain: '
                   'MLP carries the largest share (L17 75%, '
                   'L16 62%), attention beyond the exact '
                   'passive loss (S_att[dose] = -D_abl) is '
                   'secondary and late-concentrated (L34-35); '
                   'band mixed (top3 0.47/0.59, argmax L35)'),
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
if '## Phase 2956' not in memo:
    section = '''## Phase 2956: 重平衡模块定位判决 [@STAMP@]

**判决：`@VERDICT@`** —— 2950 竞争重平衡（D_nonlin，占组消融塌缩 70-74%）的下游载体定位：**MLP 模块承载最大份额（L17 族 75%、L16 族 62%），attention 在扣除精确被动损失后仅次要贡献且集中晚层（L34-35）**；带状轴两层分裂（top3 份额 0.47/0.59，argmax 均 L35）→ mixed_modules_mixed_band。

**设计（2950 verbatim 协议，K=3 同 session 重复，runtime @RT@s）**：B0 + {I0,I1}×{L17 s=1.0, L16 s=2.0} 五条件；捕获三层量：x_l = `input_layernorm` pre-hook 输入（**真残差**，pos1）、a_l = self_attn 输出（post-hook 取 tuple[0]）、m_l = mlp 输出；剂量层 o_proj 输入捕获做逐头快照 sc map。分账：dSep_pre = csep(真残差差分)（fin_cap 捕获的是 model.norm 的 **pre-hook 输入**，无需重构）；S_att_l/S_mlp_l = csep(Δa_l)/csep(Δm_l)；被动损失全部住在剂量层 attention 项（o_proj 线性）→ **R_att = ΣS_att + D_abl、R_mlp = ΣS_mlp、R_tot = dSep_pre + D_abl**（2950 D_nonlin 的模块分账）。

**锚 13/13**：a1 dirs 重建 2.17e-08（**第 27 次连续前向锚定**）、a3 Vt8 **bit 0**、a2/a8 确定性 **bit 0**、a13 消融头捕获零 **bit 0**、a18 上游因果零剖面 **bit 0**（l<剂量层 C_l 精确为零）、a16 残差链 bf16 界归一 ratio 0.979（原始 rel 6.25e-3 = 纯 bf16 ulp 噪声）、a17 相对残差 4.7e-4/4.2e-3（<1%）、a10/a11 dSep vs 2950 复现 **0.002/0.006**（sep 21.46/−9.45/84.81/48.12 与 2950 完全一致）。

**主检验**：
| 族 | R_att | R_mlp | R_tot | T1 模块轴 | T2 带轴 |
|---|---|---|---|---|---|
| L17（开关型） | **−5.28** | **−16.26** | −21.54 | mlp_carried（75%） | distributed（top3 0.468，argmax L35） |
| L16（渐变型） | **−10.40** | **−16.76** | −27.16 | mixed（62%） | concentrated（top3 0.592，argmax L35） |

**关键发现（D1 逐层剖面）**：
1. **被动损失精确对账**：S_att[17] = −9.34 vs D_abl = +9.35（L16 同）——o_proj 线性下消融头快照贡献被逐字移除，验证捕获/分账链正确。
2. **MLP 重平衡沿深度渐增、L35 最大**（L17 族 S_mlp[35] = −4.93；L16 族 = −9.34），且 **R_mlp 跨两族几乎恒定（−16.26/−16.76）**——MLP 重平衡对剂量层身份不敏感。
3. **attention 超被动部分晚层集中**（L17 族 L34/35 = −1.17/−1.37；L16 族 = −1.36/−3.19），中层贡献微弱。
4. **与 2952 合读**：注入响应的"注意力增益"是剂量层局部机制（A11 路由），而**消融诱发的竞争重平衡主要由 MLP 承载**——两种扰动的补偿载体不同模块；"全层分布式涌现"的图像进一步细化：attention 开关在 L17 局部、补偿在 MLP 全局。

**结论（重复 3 次）**：**竞争重平衡不是注意力专属现象——MLP 承载最大份额（75%/62%），attention 只在晚层（L34-35）有次要贡献；重平衡沿下游层分布式展开但向最深层（L35）倾斜；被动损失与快照线性预测精确对账（−9.34 vs −9.35），分账链可信。2952 注意力增益与 2956 MLP 重平衡是不同扰动的不同载体。**

**硬伤与勘误（7 轮运行）**：run1 pass1 捕获门未开（KeyError）；run2 **幻影编辑**（Edit 报成功未落盘，probe 证实 hook 本身健康，重编辑后 Grep 复核——本机已知缺陷第 3 次复现）；run3 a1 失配 3.7e-1（pass1 误用真残差口径，2927/2950 dirs_word 定义在 **post-layernorm attnin** 上）+ self_attn 输出 tuple 取 [0]；run4 pre_oproj 捕获放错分支（I1 永不捕获）；run5 a16 判据 5e-3 未按 bf16 界标定（cancellation 下分母失准）→ a16 v2 界归一 ratio≤2.0（实测 0.979）；run6 a17 fp64 恒等式不可达（bf16 残差链 36 层累积舍入投影 ~0.4%）→ a17 v2 相对阈 1%。**教训 27（判据）：bf16 前向下的跨层望远镜恒等式一律用相对阈（~1%）；单链路舍入误差用 2^-7·(|x|+|a|+|m|) 界归一；cancellation 场景禁用 max|结果| 作分母。** Ledger 95 条 / L14 connects 63 / ledger @LEDHASH@。

**文件+SHA256-8**：execution @HEXE@ / result @HRES@ / npz @HNPZ@ / script @HSCR@。产物 `tests/glm5/result/rdc_query_construction_20260913/phase2956/rebalance_module_localization/`。

**接续**：机制链第十九环（模块定位环）闭合。候选 2957：A（主选）R_mlp 恒定性解剖（两族 R_mlp ≈ −16.3/−16.8 几乎相同——MLP 重平衡对剂量层不敏感的机制：逐层 MLP Δ 剖面是否族间逐点相同，一次前向族）；B 交叉项代数结构（2955 遗留：Δq·Δk 主导是否可由注入方向预测——零前向+一次前向）；C 承重带跨模型复现（glm4）；D v3 解码器方向重启（2940 遗留）。
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
if 'Phase 2956' not in wl:
    wl += ('- Phase 2956 闭环：mixed_modules_mixed_band。竞争重平衡'
           '按模块分账（真残差链）：MLP 承载最大份额（L17 75%、L16 '
           '62%），attention 扣除精确被动损失（S_att[17]=−9.34 vs '
           'D_abl=+9.35）后仅晚层 L34-35 次要贡献；R_mlp 跨族恒定'
           '（−16.26/−16.76）；argmax 均 L35。2952 注意力增益（剂量'
           '层局部）与 MLP 重平衡（下游全局）是不同扰动的不同载体。'
           '勘误：a16 bf16 界归一 + a17 相对阈（fp64 望远镜恒等式在 '
           'bf16 链下不可达，~0.4%）；幻影编辑缺陷第 3 次复现。'
           'Ledger 95 / hash ' + chk + '。\n')
    open(WLOG, 'w', encoding='utf-8').write(wl)

# ---------------- MEMORY ----------------
mem = open(WMEM, encoding='utf-8').read()
if '当前 max=**2955**' in mem:
    mem = mem.replace(
        '当前 max=**2955**，下一个 **2956**（候选 A 重平衡时间定位）',
        '当前 max=**2956**，下一个 **2957**（候选 A R_mlp 恒定性解剖）')
if 'bf16 恒等链判据规范（2956）' not in mem:
    anchor_line = '- 聚合头集口径门：'
    add = ('- bf16 恒等链判据规范（2956）：跨层望远镜恒等式（x_{l+1}'
           '=x_l+a_l+m_l）在 bf16 前向下非 fp64 精确——单链路舍入用 '
           '2^-7·(|x|+|a|+|m|) 界归一（ratio≤2），累积链式判据一律用'
           '相对阈 ~1%；cancellation 场景禁用 max|结果| 作分母；'
           'self_attn forward 输出是 tuple（post-hook 取 [0]）；'
           'model.norm 的 forward_PRE_hook 捕获的是 final-norm 输入'
           '（真残差）——历史 "final-norm 输出" 读出实为 pre-norm '
           '残差（各相位口径一致，跨相位锚不受影响）；dirs_word 口径'
           '是 post-layernorm attnin（2927/2950 verbatim），与真残差'
           '口径不可混用。科学结论：消融竞争重平衡由 MLP 主导承载'
           '（75%/62%），attention 仅晚层次要（被动损失 S_att[剂量层]'
           '=-D_abl 精确对账）；2952 注意力增益（剂量层局部路由）与 '
           'MLP 重平衡（下游全局补偿）是不同扰动的不同载体。\n')
    i = mem.find(anchor_line)
    if i >= 0:
        mem = mem[:i] + add + mem[i:]
    else:
        mem = mem.rstrip() + '\n' + add
if '早翻转头极性否定(2954)。' in mem:
    mem = mem.replace(
        '早翻转头极性否定(2954)。',
        '早翻转头极性否定(2954)→路由来源大 logit/交叉项(2955)→'
        '模块定位 MLP 主导(2956)。')
if '## 机制链状态（17 环）' in mem:
    mem = mem.replace('## 机制链状态（17 环）',
                      '## 机制链状态（19 环）')
if '锚 a1 已连续 25 次前向锚定。' in mem:
    mem = mem.replace('锚 a1 已连续 25 次前向锚定。',
                      '锚 a1 已连续 27 次前向锚定。')
open(WMEM, 'w', encoding='utf-8').write(mem)

print('closeout done: ledger=%s n_meas=%s' % (
    chk, len(led['measurements'])))
