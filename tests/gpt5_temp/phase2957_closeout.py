# -*- coding: utf-8 -*-
"""Phase 2957 closeout: seal + Ledger + MEMO + worklog + MEMORY.
Idempotent, placeholder replacement."""
import hashlib
import json
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2957',
                   'rebalance_mlp_constancy')
LEDGER = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\2026-09-19.md')
WMEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\MEMORY.md')
SCRIPT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
          r'\phase2957_rebalance_mlp_constancy.py')

res = json.load(open(os.path.join(OUT, 'result.json'),
                     encoding='utf-8'))
verdict = res['final_verdict']
created = json.load(open(os.path.join(OUT, 'execution.json'),
                         encoding='utf-8'))['created']


def sha8(path):
    return hashlib.sha256(
        open(path, 'rb').read()).hexdigest()[:8]


npz_name = 'rebalance_mlp_constancy.npz'
hashes = {
    'execution.json': sha8(os.path.join(OUT, 'execution.json')),
    'result.json': sha8(os.path.join(OUT, 'result.json')),
    npz_name: sha8(os.path.join(OUT, npz_name)),
    'script': sha8(SCRIPT),
}

# ---------------- Ledger ----------------
led = json.load(open(LEDGER, encoding='utf-8'))
meas_id = 'meas2957_rebalance_mlp_constancy'
if not any(m['meas_id'] == meas_id
           for m in led['measurements']):
    led['measurements'].append({
        'meas_id': meas_id,
        'type': 'forward_family_profile_constancy',
        'verdict': verdict,
        'source': ('phase2957/rebalance_mlp_constancy; R_mlp '
                   'cross-family near-equality (2956: -16.26 vs '
                   '-16.76) is a total-level coincidence, not '
                   'profile identity: per-layer S_mlp profiles '
                   'partially shared (cos 0.868, rho 0.573), '
                   'scale divergent (b 0.549, R2 0.754), and '
                   'ablation-specific (cos vs injection MLP '
                   'response 0.569/0.049 - the ablation '
                   'rebalancing is a genuine interaction, not '
                   'an injection echo; injection M_inj sums '
                   '-133.6/-88.2 vs ablation S_mlp '
                   '-15.2/-18.8)'),
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
if '## Phase 2957' not in memo:
    section = '''## Phase 2957: R_mlp 恒定性解剖判决 [@STAMP@]

**判决：`@VERDICT@`** —— 2956 的"R_mlp 跨族几乎恒定（−16.26/−16.76）"是**总量层面的巧合，不是逐层剖面恒等**：族间 MLP 剖面仅部分共享（cos 0.868、spearman 0.573），尺度发散（过原点斜率 b=0.549、R²=0.754），且消融 MLP 响应与注入 MLP 响应**不对齐（ablation_specific）**——L17 族 cos 0.569、L16 族 cos 0.049（近正交）。

**设计（2956 verbatim 协议 + B0 捕获，K=3，runtime @RT@s）**：B0 + {I0,I1}×{L17 s=1.0, L16 s=2.0}；逐层剖面 S_mlp_f = csep(m^I1−m^I0)（2956 verbatim）、M_inj_f = csep(m^I0−m^B0)（注入响应，新增）；共同下游区 l≥18。判据冻结：T1 cos≥0.95 且 rho≥0.9 → shared；T2 b∈[0.8,1.25] 且 R²≥0.85 → unity；T3 双族 cos(S_mlp, M_inj)≥0.8 → injection_echo。

**锚 15/15（一次通过，无 run 失败）**：a1 dirs 重建 2.17e-08（**第 28 次连续前向锚定**）、a3/a9/a12/a2/a8/a13/a18 全 **bit 0**、a16 界归一 ratio 0.979、a17 相对 4.7e-4/4.2e-3、a10/a11 vs 2950 复现 0.002/0.006、**a14 S_att/S_mlp vs 2956 npz bit 级 0、a15 D1 标量 vs 2956 bit 级 0**（同 batch 组成跨相位确定性）。

**主检验**：
| 检验 | 冻结阈 | 实测 | 判定 |
|---|---|---|---|
| T1 族间剖面恒等 | cos≥0.95 & rho≥0.9 | cos **0.868** / rho **0.573** | profile_partial |
| T2 尺度 | b∈[0.8,1.25] & R²≥0.85 | b **0.549** / R² **0.754** | scale_divergent |
| T3 机制（B0 分解） | 双族 cos≥0.8 | L17 **0.569** / L16 **0.049** | ablation_specific |

**关键发现（D3 三方分解）**：
1. **消融 MLP 重平衡是真交互，不是注入回声**：注入的 MLP 响应总量巨大（sum M_inj[18:] = −133.6/−88.2）但消融差分只取 −15.2/−18.8 且剖面不对齐——2950 的"竞争重平衡"是消融扰动特有的下游动力学，不能由注入响应线性外推。
2. **R_mlp 恒定 = 总量巧合**：两族逐层剖面在 L18-20 与 L30-35 形状相近（均负、深层增强），但中层（L21-29）符号与幅度分化，总量上相互抵消至近似相等——"对剂量层身份不敏感"的表象不成立。
3. **与 2956 合读修正**：MLP 承载最大份额（75%/62%）仍成立，但"MLP 重平衡是固定下游回声"的机制假设被否定；重平衡载体是**消融特异的分布式 MLP 响应**，其族间相似性仅限深层趋势。
4. B0 基线 sep 185.70（无注入基线读出，供后续归一口径）。

**结论（重复 3 次）**：**R_mlp 跨族恒定是总量巧合——逐层剖面仅部分共享（cos 0.868）、尺度发散（b 0.549）、且与注入响应机制不同（ablation_specific，L16 近正交 cos 0.049）；消融竞争重平衡是消融特异的分布式 MLP 动力学，不可由注入响应外推。**

**硬伤与勘误**：无（run1 一次通过）。教训 reinforcement：跨相位 bit 级锚（a14/a15）在同 batch 组成 + 同 session 下可靠复现（2934 教训的正确用法面）。

**文件+SHA256-8**：execution @HEXE@ / result @HRES@ / npz @HNPZ@ / script @HSCR@。产物 `tests/glm5/result/rdc_query_construction_20260913/phase2957/rebalance_mlp_constancy/`。Ledger 96 条 / L14 connects 64 / ledger @LEDHASH@。

**接续**：机制链第二十环（恒定性证伪环）闭合。候选 2958：A（主选）消融特异性来源定位——消融 MLP 响应（S_mlp）与"剂量层快照损失的空间印记"（D_abl 的下游传播剖面）对比，检验重平衡是否是对被动损失印记的主动抵消（一次前向族，2956 npz 已有 x/a/m 捕获可复用口径）；B 交叉项代数结构（2955 遗留）；C 承重带跨模型复现（glm4）；D v3 解码器方向重启（2940 遗留）。
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
if 'Phase 2957' not in wl:
    wl += ('- Phase 2957 闭环：profile_partial_scale_divergent_'
           'ablation_specific。2956 的 R_mlp 跨族恒定（−16.26/'
           '−16.76）是总量巧合：族间逐层剖面 cos 0.868/rho 0.573'
           '（partial）、尺度发散 b 0.549/R² 0.754、消融响应与注入'
           '响应不对齐（cos 0.569/0.049，L16 近正交）——竞争重平衡'
           '是消融特异的分布式 MLP 动力学，不可由注入响应外推（注入 '
           'M_inj −133.6/−88.2 vs 消融 S_mlp −15.2/−18.8）。run1 '
           '一次通过 15/15 锚（a14/a15 vs 2956 bit 级 0；a1 第 28 次'
           '连续前向锚定）。Ledger 96 / hash ' + chk + '。\n')
    open(WLOG, 'w', encoding='utf-8').write(wl)

# ---------------- MEMORY ----------------
mem = open(WMEM, encoding='utf-8').read()
if '当前 max=**2956**' in mem:
    mem = mem.replace(
        '当前 max=**2956**，下一个 **2957**（候选 A R_mlp 恒定性解剖）',
        '当前 max=**2957**，下一个 **2958**（候选 A 消融特异性来源定位）')
if '总量巧合非剖面恒等(2957)' not in mem:
    old = '模块定位 MLP 主导(2956)。'
    if old in mem:
        mem = mem.replace(old, old[:-1] + '→R_mlp 恒定性证伪：总量巧合非剖面恒等(2957)。')
if '## 机制链状态（19 环）' in mem:
    mem = mem.replace('## 机制链状态（19 环）',
                      '## 机制链状态（20 环）')
if '连续 27 次前向锚定' in mem:
    mem = mem.replace('连续 27 次前向锚定',
                      '连续 28 次前向锚定')
add = ('- 跨相位 bit 级锚规范（2957）：同 batch 组成 + 同 session 下，'
       '逐层剖面/标量复现可达 bit 级（a14/a15 = 0）——跨相位复现锚'
       '优先用"与上一 Phase npz/result 的 max|Δ|"而非自设阈；总量'
       '相等不蕴含剖面相等，跨族/跨条件恒定性主张必须做逐点检验'
       '（T1 cos+rho / T2 过原点斜率 / T3 与上游响应的对齐分解）。\n')
if '跨相位 bit 级锚规范（2957）' not in mem:
    anchor_line = '- bf16 恒等链判据规范（2956）：'
    i = mem.find(anchor_line)
    if i >= 0:
        mem = mem[:i] + add + mem[i:]
    else:
        mem = mem.rstrip() + '\n' + add
open(WMEM, 'w', encoding='utf-8').write(mem)

print('closeout done: ledger=%s n_meas=%s' % (
    chk, len(led['measurements'])))
