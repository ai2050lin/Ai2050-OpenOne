# -*- coding: utf-8 -*-
"""Phase 2943 closeout: Ledger + MEMO + workspace log + MEMORY."""
import json
import os

LEDGER = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\2026-09-19.md')
WMEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\MEMORY.md')

OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913\phase2943'
       r'\gamma_anatomy')

MEAS_ID = 'meas_2943_gamma_anatomy'
LINK_ID = 'L14_readout_spectrum_cross_model'

MEMO_BLOCK = """
## Phase 2943: gamma 负偏移解剖与 regime 开关签名 [2026-09-19 19:14]

**原理**：2937 P2 发现 null 条件 OLS 截距 gamma in [-16.1,-11.6]（same +3.94），2942 发现联合注入逐词中位位移≈0 但 sep 过冲。本 Phase 零前向解剖：sep 塌缩有多少是线性收缩几何的必然？gamma 是否独立机制成分？注入是否携带独立形状？输入量（beta/gamma/R1）曾在 2937/2942 并排展示，组合统计量准注册新造但按纪律 9 标注 quasi-post-hoc，判决登记为机制链整合（discovery-grade 权重有限）。

**方法**：ZERO FORWARD，冻结产物 2937 proj/2939 coords/2942 注入 npz。锚 a1 proj 跨 2937/2939/2942 bit 级一致（0.0，三相位同批次基线交叉验证）；a2 OLS 重建 vs 2937 result 4.78e-05；a3 dcks=c8_null0-c8_func0 bit 级 0；a4 sep_func 185.6975（9.44e-06）；a5 words/labels 一致。5/5 过。

**判决 `regime_signature_confirmed`**（run5 权威，script eeb59d73）：
- **T1 残差类结构占比**：sep(y)=beta*sep_f+sep(resid) 为构造恒等式（identity_dev ~1e-16 仅作校验）；经验量 resid_share=|sep_resid|/(beta*|sep_f|)：null0-3 = 0.030/0.041/0.042/0.027，全 < 0.10。**sep 塌缩 96-97% 由 beta 收缩承载，残差无类结构**。
- **T2 gamma 独立于 U8**：gamma_pred_from_U8 = P_u8-(beta-1)*mean_f 与实际 gamma gap = 7.03/10.84/10.41/9.11/11.63（same/null0-3），median gamma -12.55 < -5。**gamma 是 U8 子空间外的独立截距成分**（U8 重构逐词 pearson 0.9975 但均值差 ~10）。
- **T3 注入无独立形状**：partial Spearman(d_inj, d_null0 | f) s=1/2/3/4 = -0.1245/0.1006/0.1093/0.3066，median 0.1049 < 0.3（raw 0.7129）。**注入位移的逐词共变几乎全部由"对 s_func 的收缩"解释，无独立 null 形状成分**（s=2 跨 session 不稳定性已注记，取 4 点稳健中位）。

**描述性**：D2 类条件 gamma 分裂大（null1 gamma_L0 +91.4 vs gamma_L1 -17.3）——类间斜率/截距重分配，但合成残差类结构小（T1）；D3 截距缺口：mean(d_inj) -23.97 vs mean(d_null) -54.03，+gamma 后仍缺 18.5（斜率差承载；2942 R2 中位 +0.874 与均值 -23.97 的差异=类不对称位移）。

**硬伤与勘误**：run3 a5 words 格式不一致（2937 三列数组 vs 2942 全串）假阴；run4 T1 写成 rel_err 判据——**构造恒等式不携带证据**（resid:=y-beta*f-gamma 使 sep(y)=beta*sep_f+sep(resid) 永真，rel_err 恒 0），run5 修正为残差类结构占比（correction_note 入 PREREG）。全部判决零前向 quasi-post-hoc，不翻转 2937 rewrite 判决，而是给出其内容：**rewrite = 均匀收缩(beta<1) + 独立负截距(gamma) + 可忽略类残差**。

**结论（机制链第八环：签名整合环）**："null 上下文重编码"的完整签名 = (1) 方向重写（2937 能量不塌方向塌）+ (2) 子空间保持（2938）+ (3) 固定目标 v3（2939）+ (4) v3 词属性盲（2940）+ (5) 单方向注入阻尼（2941）+ (6) 联合注入陡降过冲（2942）+ (7) **sep 塌缩=收缩几何必然 + gamma 独立截距 + 注入无独立形状（2943）**。2942 的"regime 开关"统计签名坐实：读出失败不是逐词重写场，而是收缩×截距的线性外壳 + 不稳定开关动态。2936→2943 闭环完成：scale 塌缩的全部四层（量级/方向/子空间/坐标）+ 因果操作化否定 + 签名整合。

**文件+SHA256-8**：execution e03ee93a / result 4961189e / npz 8c95d730 / script eeb59d73；源 s2937 c6747439 等 execution.json sources。产物 `tests/glm5/result/rdc_query_construction_20260913/phase2943/gamma_anatomy/`。runtime 0.0s（零前向）。

**接续（2944 候选）**：A（主选）逐层定位 regime 开关（一次前向：L14-L18 单层联合 v1/v2/v5 注入，定位陡降触发的最小层集，检验 L16 单点 vs 多层协同）；B 承重带跨模型复现（glm4，一次前向）；C 类条件 OLS 重参数化（零前向：类条件 beta/gamma 联合拟合的稳定性与 AIC 比较）；D 2942 注入 s 网格细扫 + 同 session 重复（一次前向，解决跨 session 不稳定标定）。
"""


def sha8(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def main():
    out = []

    # ---------- Ledger ----------
    led = json.load(open(LEDGER, encoding='utf-8'))
    ids = [m['meas_id'] for m in led['measurements']]
    if MEAS_ID not in ids:
        led['measurements'].append({
            'meas_id': MEAS_ID,
            'type': 'zero_forward_anatomy',
            'verdict': 'regime_signature_confirmed',
            'source': 'phase2943/gamma_anatomy'})
        for lk in led['linkage']:
            if lk['link_id'] == LINK_ID:
                if MEAS_ID not in lk['connects']:
                    lk['connects'].append(MEAS_ID)
        led['ledger_sha256_8'] = sha8(LEDGER)
        json.dump(led, open(LEDGER, 'w', encoding='utf-8'),
                  indent=2, ensure_ascii=False)
    led2 = json.load(open(LEDGER, encoding='utf-8'))
    n_meas = len(led2['measurements'])
    l14 = [lk for lk in led2['linkage']
           if lk['link_id'] == LINK_ID][0]
    out.append('ledger: n_meas=%d last=%s L14=%d ledger8=%s'
               % (n_meas, led2['measurements'][-1]['meas_id'],
                  len(l14['connects']),
                  led2['ledger_sha256_8']))

    # ---------- MEMO ----------
    memo = open(MEMO, encoding='utf-8').read()
    if '## Phase 2943' not in memo:
        with open(MEMO, 'a', encoding='utf-8') as f:
            f.write(MEMO_BLOCK)
    memo2 = open(MEMO, encoding='utf-8').read()
    i = memo2.find('## Phase 2943')
    out.append('memo: 2943 at char %d | total %d | title: %s'
               % (i, len(memo2),
                  memo2[i:i + 70].split('\n')[0]))
    out.append('memo 2943 present: %s | no 2944: %s'
               % ('## Phase 2943' in memo2,
                  '## Phase 2944' not in memo2))

    # ---------- workspace log ----------
    entry = ('\n## Phase 2943 (2026-09-19 19:15)\n'
             '- 判决 regime_signature_confirmed：sep 塌缩 96-97% '
             '由 beta 收缩承载（残差类占比 2.7-4.2%）；gamma 独立'
             '截距（U8 gap 7.0-11.6）；注入偏相关崩塌 0.105（raw '
             '0.71）。机制链第八环（签名整合环）闭合。\n'
             '- 勘误：run4 T1 恒等式判据（rel_err 恒 0 无证据），'
             'run5 修正为残差类结构占比；run3 a5 words 格式不一致。\n'
             '- 产物 phase2943/gamma_anatomy：execution e03ee93a '
             '/ result 4961189e / npz 8c95d730 / script eeb59d73。'
             'Ledger 82 条 / L14 connects 50。\n')
    wl = open(WLOG, encoding='utf-8').read()
    if 'Phase 2943' not in wl:
        with open(WLOG, 'a', encoding='utf-8') as f:
            f.write(entry)
    wl2 = open(WLOG, encoding='utf-8').read()
    out.append('wlog 2943 present: %s' % ('Phase 2943' in wl2))

    # ---------- MEMORY.md ----------
    mem = open(WMEM, encoding='utf-8').read()
    if 'max=2943' not in mem:
        mem = mem.replace('max=2942，下一个 2943',
                          'max=2943，下一个 2944')
    if '2943 教训' not in mem:
        old17 = [ln for ln in mem.split('\n')
                 if ln.startswith('17. ')]
        new17 = (
            '17. **恒等式判据禁令（2943 教训）**：判据若由分解定义'
            '自身保证成立（如 resid:=y-beta*f-gamma 使 '
            'sep(y)=beta*sep_f+sep(resid) 永真），则其"通过"不携带'
            '任何证据——预注册前必须问"这个判据可能失败吗"；经验'
            '内容必须落在可失败的量上（如残差类结构占比）。'
            '2943 延伸：机制整合类 Phase 的输入量若已并排展示，'
            '判决按 quasi-post-hoc 登记（纪律 9），结论定位为整合'
            '而非新发现。')
        if old17:
            mem = mem.replace(old17[0], new17)
        else:
            mem = mem.replace(
                '\n## 本机环境缺陷与对策',
                '\n' + new17 + '\n\n## 本机环境缺陷与对策')
    with open(WMEM, 'w', encoding='utf-8') as f:
        f.write(mem)
    mem2 = open(WMEM, encoding='utf-8').read()
    out.append('wmem max=2943: %s | lesson17 id: %s'
               % ('max=2943' in mem2, '2943 教训' in mem2))

    print('\n'.join(out))


if __name__ == '__main__':
    main()
