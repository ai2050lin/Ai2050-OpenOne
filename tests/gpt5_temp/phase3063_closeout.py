# -*- coding: utf-8 -*-
"""Phase 3063 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md (project workspace)."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3063'
     r'\omega_p60_adversarial_trace_qwen')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
WLOG_DIR = ROOT + r'\.workbuddy\memory'
LOGF = R + r'\closeout_log.txt'
o = []

res = json.load(io.open(R + r'\result.json',
                        encoding='utf-8'))
seal = json.load(io.open(R + r'\seal.json',
                         encoding='utf-8'))
exe = json.load(io.open(R + r'\execution.json',
                        encoding='utf-8'))
created = exe['created']
verdict = res['verdict']
assert verdict == 'adversarial_same_source_qwen', \
    verdict
assert seal['verdict'] == verdict
assert seal['anchor_core_ok'] is True
st = res['stats']
an = st['anchors']
assert an['a220_seals_ok'] is True
assert an['anchor_core_ok'] is True
for kk in ('a221_tt_diff', 'a222_recapture_diff',
           'a223_gamma_stats_diff', 'a224_top64_diff',
           'a225_dtan_diff', 'a226_bbody_pc1_diff',
           'a227_coal_diff', 'a228_varm_diff',
           'a229_h0k_diff', 'a230_h7j_diff',
           'a231_h3k_diff', 'a232_sham_diff',
           'a233_pre_max'):
    assert an[kk] == 0.0, (kk, an[kk])
t2 = st['T2_v_mechanism']
assert t2['v_adv'] is True
assert abs(t2['med_cos_V']
           + 0.3408632071271014) < 1e-12
assert abs(t2['med_cos_d']
           + 0.3004718959809629) < 1e-12
assert abs(t2['med_cos_dtan']
           + 0.05689211189823873) < 1e-12
assert abs(t2['med_cos_dtan_only']
           + 0.3411984072467673) < 1e-12
assert abs(t2['med_cos_lg_pred']
           - 0.998461338737201) < 1e-12
assert abs(t2['med_frac_rad']
           - 0.2921037321026409) < 1e-12
assert abs(t2['med_tan_frac']
           - 0.95635534911639) < 1e-12
t3 = st['T3_same_source']
assert abs(t3['ss1_obs']
           + 0.0014695757428706546) < 1e-12
assert abs(t3['ss1_p'] - 0.526736631684158) < 1e-12
assert abs(t3['c_mean_vk0']
           - 0.30836146730724256) < 1e-12
assert abs(t3['p_ax_mean']
           - 0.0034982508745627187) < 1e-12
assert abs(t3['c_pc1_vk0']
           - 0.3409874824243583) < 1e-12
assert abs(t3['ss2_p']
           - 0.001999000499750125) < 1e-12
assert abs(t3['jaccard_obs']
           - 0.13777777777777778) < 1e-12
assert t3['jaccard_null_med'] == 0.0
assert abs(t3['ss3_p']
           - 0.0004997501249375312) < 1e-12
assert t3['ss_count'] == 2
t4 = st['T4_specificity']
assert abs(t4['c_pc1_vk3']
           - 0.1378651578375219) < 1e-12
assert abs(t4['c_pc1_vj']
           - 0.019495714599979938) < 1e-12
assert abs(t4['c_pc1_k0k3']
           - 0.08595417369673013) < 1e-12
assert t4['axis_dominance'] is True
assert abs(t4['med_cos_K0']
           + 0.24919460797697768) < 1e-12
assert abs(t4['med_cos_K0_dtan_only']
           + 0.24920459177363602) < 1e-12
t5 = st['T5_channel_identity']
assert t5['V']['ov_s16'] == 9
assert t5['V']['ov_t64'] == 36
assert abs(t5['V']['e16']
           - 0.046525877471087336) < 1e-12
assert abs(t5['V']['e64']
           - 0.11796006299782784) < 1e-12
assert t5['K0']['ov_s16'] == 16
assert t5['K0']['ov_t64'] == 51
assert abs(t5['K0']['e16']
           - 0.14653973307666454) < 1e-12
assert abs(t5['K0']['e64']
           - 0.23916255607851875) < 1e-12
t6 = st['T6_known_axis']
assert abs(t6['med_cos_dv_vmat']
           + 0.038415646339217256) < 1e-12
assert abs(t6['p_kx']
           - 0.001999000499750125) < 1e-12
assert abs(t6['med_abs_cos_dv_pc1']
           - 0.05275889791937896) < 1e-12
assert abs(t6['med_abs_cos_dk0_pc1']
           - 0.042289878499813305) < 1e-12
assert abs(t6['cos_meanDv_pc1D']
           + 0.14832464623868186) < 1e-12

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3063
           for m in led['measurements']):
    claim = (
        'Omega-P60 (plan 3063 A) - adversarial-'
        'source tracing: is the 3051 V-only '
        '-0.3409 effect tangential-adversarial '
        'and SAME-SOURCE with the h0 exact-K '
        'tangential adversarial component, run2 '
        'fp32 authoritative (77.5s, ~120 '
        'forwards; 14 anchors a220-a233 all bit '
        '0.0). run1 crashed pre-verdict at the '
        'G-metric axis null (wud is vocab-major '
        'so wud @ wud.T tried to allocate '
        '(151936,151936); fixed to wud.T @ wud). '
        'RESULTS: (1) VERDICT '
        'adversarial_same_source_qwen. (2) T2: '
        'V-only d_tan through the exact norm '
        'counterfactual reproduces the FULL '
        'effect (med cos -0.3412 vs raw '
        '-0.3409); tan_frac 0.956; Jacobian '
        'prediction cos 0.9985 - tangential '
        'adversarial, not radial dilution. (3) '
        'T3 same-source: SS2 shared axis '
        '|cos mean| 0.3084 (p 0.0035) and '
        '|cos PC1| 0.3410 (p 0.0020) vs the '
        'W_U-image random-direction null; SS3 '
        'top-256 channel-support Jaccard 0.1378 '
        'vs null med 0.0 (p 0.0005); SS1 per-'
        'pair pairing NULL (p 0.53) - family-'
        'level shared axis, not per-pair '
        'correspondence; ss_count 2. (4) T4 '
        'specificity: V<->h3K |cos PC1| 0.138, '
        'V<->h7J 0.019, h0K<->h3K 0.086, '
        'axis_dominance True. (5) T5: both '
        'adversarial arms live in the S16/TOP64 '
        'pipe support (V: S16 9, TOP64 36, E16 '
        '0.047, E64 0.118; K0: S16 16, TOP64 '
        '51, E16 0.147, E64 0.239; all p '
        '0.0005) and NOT in the relay coalition '
        'or voting dims. (6) T6: the adversarial '
        'axis is near-orthogonal to both the '
        'payload (med cos(D_V,Vmat) -0.038, '
        'p 0.002 slight negative tilt) and the '
        'readout axis (|cos PC1_ro| 0.053/0.042)'
        ' - a distinct third (competition) axis '
        'inside the pipe infrastructure.')
    meas = {
        'meas_id': 'meas3063_omega_p60_adversarial_'
                   'trace_qwen',
        'phase': 3063,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a220 source seals 3044-3062; '
                   'a221 TT diff 0.0; a222 re-'
                   'capture diff 0.0; a223 gamma '
                   'stats diff 0.0; a224 TOP64/'
                   'FLAT64 diff 0.0 vs z58; a225 '
                   'D_TAN diff 0.0 vs z59; a226 '
                   'Vmat/PC1/B_BODY/C_PREFIX '
                   'diff 0.0 vs z59/z60/z61; '
                   'a227 COAL_TOP64 diff 0.0; '
                   'a228 V arm diff 0.0 vs z51 '
                   'COS_V; a229 h0K arm diff '
                   '0.0 vs z54 (5 arrays); '
                   'a230 h7J arm diff 0.0 vs '
                   'z51 COS_H[7]+z53 STAGE; '
                   'a231 h3K arm diff 0.0 vs '
                   'z53 COS_KCTRL_h3; a232 sham '
                   'bit identity; a233 stage-pre '
                   'identity 0.0',
        'artifacts': {
            'result': 'phase3063/omega_p60_'
                      'adversarial_trace_qwen/'
                      'result.json',
            'npz': 'phase3063/omega_p60_'
                   'adversarial_trace_qwen/'
                   'omega_p60_adversarial_'
                   'trace_qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run2 authoritative (77.5s, fp32; '
                'run1 crashed pre-verdict at the '
                'G-metric axis null, registered)',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 202
    l14['connects'].append({
        'meas_id': 'meas3063_omega_p60_adversarial_'
                   'trace_qwen',
        'phase': 3063,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P60: V-only write-'
                        'side overwrite and h0 gate-'
                        'side K-overwrite trigger '
                        'the SAME adversarial '
                        'mechanism - family-level '
                        'shared axis (|cos PC1| '
                        '0.341, specificity '
                        'controls 0.138/0.019) + '
                        'shared S16/TOP64 channel '
                        'support (Jaccard 0.138, '
                        'both arms E16/E64 p '
                        '0.0005); the axis is '
                        'neither payload (cos '
                        'Vmat -0.04) nor the '
                        'readout axis (cos PC1_ro '
                        '0.05) - a distinct '
                        'competition axis inside '
                        'the gamma/S16 pipe; '
                        'closes the 3051 B-item '
                        'question '
                        '(adversarial_same_source_'
                        'qwen)'})
    led.pop('ledger_sha256_8')
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w', encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False, indent=1)
    o.append('ledger appended n=%d l14=%d sha=%s'
             % (len(led['measurements']),
                len(l14['connects']),
                led['ledger_sha256_8']))
else:
    o.append('ledger already upserted n=%d l14=%d'
             % (len(led['measurements']),
                len(l14['connects'])))

# ---------- MEMO append ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3063:' not in memo:
    sec = u'''## Phase 3063: Ω-P60 h0 对抗成分溯源——V-only 与 h0 切向对抗同源、管道内竞争轴（adversarial_same_source_qwen） [%(created)s]

**判决：`adversarial_same_source_qwen`**（run2 fp32 权威 77.5s，约 120 次前向 + 离线统计；14 锚 a220–a233 全 bit 0.0，anchor_core_ok=True）。run1 崩溃如实入册：T3 轴 null 的读出度量矩阵写成 `wud @ wud.T`——wud 是 vocab-major (151936,2560)，试图分配 (151936,151936) fp64 = 172 GiB 直接 OOM；修正为 `wud.T @ wud`（(2560,2560)，即 G = W_U·W_Uᵀ 的正确形式）。run1 已完成全部前向与全部锚（与 run2 bit 一致），在离线统计起点崩溃，无判决。

### 问题与设计
**问题**（3051 遗留 B 项，挂起至今）：① 3051 V-only 全行替换的 −0.3409 负对齐是切向对抗（过最终 RMSNorm 仍存活）还是径向稀释？② V 臂对抗成分与 h0 精确 K 切向对抗成分（3054：d_tan 过 norm 仍 −0.2492）是否**同源**（同方向/同子空间/同通道支撑）？——以正臂 h3K/h7J 作特异性对照。**协议**：4 臂 × 24 canonical diagonal 对（c-major），全部用 3054 forward_stage 机械复刻（K 块替换走 RoPE 偏移旋转 rotK；V 全行走 srcV 全行），捕获每对 4 个 d = post_i − post_b 的切向分量 → D_V/D_K0/D_K3/D_J (24×2560)。离线统计：**T2** V 机制（精确 norm 反事实）；**T3** 同源三检验（logit 空间 Y = D·W_Uᵀ，精确 G 度量）：SS1 逐对 Gram 对角支配（标签置换 seed 9998）、SS2 族均值/PC1 共享轴 vs W_U 像空间随机方向 null $$|x^T G y|/sqrt((x^T G x)(y^T G y)), G = W_U W_U^T$$（seed 9999）、SS3 top-256 通道支撑 Jaccard（seed 10002）；**T4** 特异性（V↔h3K seed 10000、V↔h7J seed 10001 + axis_dominance）；**T5** 通道身份（top-256 廓线 vs S16/TOP64/COAL/sig2802 重叠 + E16/E64 能量，V seeds 10003-10008、K0 seeds 10009-10014）；**T6** 已知轴关系（反载荷检验 cos(D_V[k], Vmat[k])、读出轴对齐、旋转 null seed 10015）。

### 核心结果（重复三遍）
**① 同源结论（一）**：V-only 写侧对抗与 h0 门侧切向对抗**同源**——两族对抗向量共享族级轴：|cos mean|=0.3084（p=0.0035）、|cos PC1|=0.3410（p=0.0020，W_U 像空间随机方向 null），共享通道支撑 Jaccard=0.1378 vs null med 0.0（p=0.0005），ss_count=2，axis_dominance=True（正对照 V↔h3K 仅 0.138、V↔h7J 仅 0.019、h0K↔h3K 0.086）。**② 同源结论（二）**：该共享轴是**管道内的独立第三轴**——与载荷方向近正交（med cos(D_V,Vmat)=−0.0384，反载荷置换 p=0.002 仅微弱负倾）、与读出单轴近正交（med |cos(D_V,PC1_ro)|=0.0528、|cos(D_K0,PC1_ro)|=0.0423），但两臂均住 S16/TOP64 读出管道支撑（V：S16 重叠 9、TOP64 36、E16=0.0465/E64=0.1180；K0：S16 16、TOP64 51、E16=0.1465/E64=0.2392；全 p=0.0005），且都不在中继联盟（COAL p=0.96/0.46）与 2802 投票维（sig p=0.94/0.62）。**③ 同源结论（三）**：V 臂机制定性为**切向对抗而非径向稀释**——精确 norm 反事实下 d_tan 单独过 norm 复现全部效应（med −0.3412 vs 原始 −0.3409），tan_frac=0.956（径向占范数 29pct 但对 logit 效应贡献可忽略），雅可比线性预测 cos=0.9985。**④ SS1 逐对配对失败**（obs=−0.0015，p=0.53）：同源是族级共享轴，不是逐对 token 对应——两臂在每个样本对上的对抗向量互不相同，但都投影到同一共享竞争轴。

### 机制链定版
Ω-P2 链最后一个溯源问题关闭：L35 的两类干预——改写载荷（V-only 全行替换，−0.34）与改写门位路由（h0 K-only，−0.249 切向）——触发的是**同一个对抗机制**，其方向签名是一条族级共享的**竞争轴**，住在 γ/S16 管道基建内，既不携带载荷也不承担读出传输。与 3060–3062 合并的完整图像：**门位易感（承载）= 语境组装方向（写入侧 body 身份 ~5 维）与候选竞争轴（门侧对抗签名）共存于同一 S16 管道支撑，读出单轴选择性传输行为相关分量**。Cmp(o,r,v) 的候选竞争效应有了明确几何载体：一条可检验的共享轴 + 可分支撑，而非逐对对应向量。

### 硬伤与边界
- **SS1 null（p=0.53）**：逐对配对不成立，"同源"只能主张族级（共享轴+共享支撑）；不能说两臂逐样本触发同一向量。
- **共享轴绝对强度温和**：|cos PC1|=0.341 只解释对抗方差的一部分；族内仍有大量独立成分。
- **n=24 对**，族级统计样本小；单模型 qwen3-4b，跨模型普适性未检验（DS7B 复刻为下一优先）。
- **T6 微弱负倾**（−0.038，p=0.002）：存在极小反载荷成分，不可忽略但不主导。
- frac_rad（0.292，范数占比）与 tan_frac（0.956）是范数分解量；"切向对抗"是**效应级**结论（反事实复现），不是范数占比结论。

### 方法论入册
- **G 度量矩阵方向（3063）**：wud 为 vocab-major (V,2560) 时读出度量 G = W_U·W_Uᵀ = `wud.T @ wud`（(2560,2560)）；写反方向即 172 GiB OOM（run1 教训）。
- **家族级 vs 逐对同源（3063）**：共享轴/共享支撑（族级）与逐对 Gram 对角支配（逐对）是两个独立检验，可分离成立——本期即"族级成立、逐对不成立"。
- 幻影 Edit 本会话再现：Edit 报成功但未落盘（run 标签修正）；Python 补丁 assert count==1 仍是唯一可靠修正通道；cmd.exe 经 bash shim 调用也损坏（del/dir 参数被截断），删除/列目录用 Python。

### 智能理论洞察（第一性原理）
本期把"条件化齿轮组"的一个候选齿轮钉到了几何上：**竞争轴**。语言能力的组合性要求系统在多候选解释间仲裁——本期证明这种仲裁在 LLM 内部有可测的方向载体：一条独立于"写了什么"（载荷 Vmat）与"读出什么"（PC1_ro）的轴，写侧与门侧两条干预通路汇聚其上。这支持 RDC 的核心主张：**语义传递不是单轴广播，而是"管道基建（γ/S16）+ 管内多轴分工（读出轴/身份指纹/竞争轴）"**。下一层第一性原理问题：竞争轴的通道支撑从哪里来（L35 之前的哪些头/MLP 写入这些通道）？竞争轴与 body 身份指纹（3061 的 ~5 维）在支撑内的关系是正交、嵌套还是分层？跨模型这条轴是否守恒（DS7B 复刻）？——若守恒，则"竞争轴"从 qwen 特异性上升为语言编码的候选普适结构。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3063/omega_p60_adversarial_trace_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3064 菜单**——A（主选）**跨模型 DS7B 复刻全链**（KV 阶梯+门区+γ 管道+PC1 身份+写入分解，检验机制链普适性）；B 门区 2D 易感图（门位易感在 (layer, head) 平面的完整地图）；C body 指纹下游消费定位（3016 放大追踪法：谁在后续层读取路由签名）；D（新增）**竞争轴源头定位**：共享竞争轴的通道支撑在 L35 之前的写入来源（头/MLP 贡献分解）。"好的，继续"即进 3064 A。
''' % {'created': created,
           'script8': seal['script_sha256_8'],
           'result8': seal['result_sha256_8'],
           'npz8': seal['npz_sha256_8'],
           'exec8': seal['exec_sha256_8'],
           'n': len(led['measurements']),
           'l14': len(l14['connects'])}
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    o.append('memo +%d chars' % len(sec))
else:
    o.append('memo already appended')

# ---------- HDMCC audit addendum ----------
aud = io.open(AUDIT, encoding='utf-8').read()
if '## 二十五、3063 增补' not in aud:
    add = u'''

---

## 二十五、3063 增补：h0 对抗成分溯源——写侧与门侧对抗同源、管道内竞争轴（Omega-P60，判决 adversarial_same_source_qwen）

1. **V-only 对抗是切向的**：精确 norm 反事实下 d_tan 单独过 norm 复现全部效应（−0.3412 vs 原始 −0.3409）、tan_frac 0.956、雅可比预测 cos 0.9985——排除径向稀释解释，3051 的 −0.3409 是真对抗信号。
2. **写侧与门侧对抗同源**：V 臂与 h0K 臂对抗族共享族级轴（|cos PC1|=0.341, p=0.002；正对照 h3K/h7J 仅 0.138/0.019，axis_dominance 成立）与共享通道支撑（Jaccard 0.138 vs null 0.0, p=0.0005）；SS1 逐对配对 null（p=0.53）——同源是族级共享轴而非逐对对应。
3. **竞争轴是管道内第三轴**：与载荷方向近正交（cos Vmat −0.04）、与读出单轴近正交（cos PC1_ro 0.05），但两臂均住 S16/TOP64 管道支撑（E16/E64 全 p=0.0005）且不在中继联盟/投票维——γ/S16 基建内独立于载荷轴与读出轴的候选竞争签名。
4. HDMCC 修正：Cmp(o,r,v) 兼容性机制的两条干预通路（改写载荷 V vs 改写门位路由 K）汇聚于同一竞争机制；门位易感的方向载体是族级共享竞争轴，其通道身份与 3062 的身份指纹同住管道支撑，二者的支撑内关系（正交/嵌套/分层）尚未检验，列为 3064 候选。
'''
    aud += add
    with io.open(AUDIT, 'w', encoding='utf-8') as f:
        f.write(aud)
    o.append('audit addendum +%d chars' % len(add))
else:
    o.append('audit already appended')

# ---------- workspace log ----------
wl = WLOG_DIR + r'\\2026-09-21.md'
try:
    prev = io.open(wl, encoding='utf-8').read()
except IOError:
    prev = ''
if 'Phase 3063' not in prev:
    line = ('- Phase 3063 Omega-P60 adversarial-'
            'source tracing: verdict '
            'adversarial_same_source_qwen (run2 '
            'fp32 77.5s; 14 anchors a220-a233 all '
            'bit 0.0; run1 crashed at G-metric '
            'axis null - wud is vocab-major, '
            'wud @ wud.T would be 172 GiB, fixed '
            'to wud.T @ wud). RESULTS: V-only '
            'd_tan through exact norm '
            'counterfactual reproduces the full '
            'effect (-0.3412 vs -0.3409, '
            'tan_frac 0.956) - tangential '
            'adversarial; V and h0K adversarial '
            'families share a family-level axis '
            '(|cos PC1| 0.341, p 0.002; '
            'specificity controls 0.138/0.019; '
            'axis_dominance) and channel support '
            '(Jaccard 0.138 vs null 0.0, p '
            '0.0005; both arms S16/TOP64 E16/E64 '
            'p 0.0005); SS1 per-pair pairing '
            'null (p 0.53); axis near-'
            'orthogonal to payload (cos Vmat '
            '-0.04) and readout axis (cos '
            'PC1_ro 0.05) - distinct competition '
            'axis inside the gamma/S16 pipe. '
            'Audit addendum 25; ledger 202/L14 '
            '170.\n')
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

# ---------- MEMORY.md (project workspace) ----------
MEMO_W = os.path.join(WLOG_DIR, 'MEMORY.md')
try:
    mem_cur = io.open(MEMO_W, encoding='utf-8').read()
except IOError:
    mem_cur = ''
if 'max=3063' not in mem_cur:
    mem_new = u'''# RDC/LPF 研究项目纪律（工作区长期记忆）

## 项目身份
- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）。
- MEMO append-only：research\\gpt5\\docs\\AGI_GPT5_MEMO.md（glm5 版封存勿写）。
- 脚本 tests\\glm5\\phase{N}_*.py；closeout tests\\gpt5_temp\\；产物 ...\\phase{N}\\{arm}\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14.connects（linkage 列表第 14 项 link_id=L14_readout_spectrum_cross_model）；hash=去 ledger_sha256_8 后 dumps(sort_keys, ensure_ascii=False) sha256 前 8。

## 强制流程
1. 闭环：execution 冻结→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）；占位符 %(key)s 风格；裸百分号写 %%。
3. 重跑先删旧 execution.json 与 result.json；负结果/崩溃如实登记；verdict 单分支赋值。
4. 统计纪律：obs/null 同量纲；负对照带符号解读；阈值预注册；随机对照判特异性。

## 标准锚与精度
- bit 级仅限同文件链/同精度；精确 KV 复放：post-norm K+RoPE 偏移旋转；全长 repl+全 True mask。
- 捕获库跨相位复用（上游 npz+抽样重捕获 bit 锚+派生量逐对复现锚）；锚前 npz 键名-RHS 逐项核对。
- SVD 符号任意性用 |cos|；json 禁 numpy 标量；fp64 W_U 3.1GB 用后即 del；大置换逐行/分块。
- **3063 新纪律**：①wud 是 vocab-major (151936,2560)——读出度量 G=W_U·W_Uᵀ 写成 wud.T @ wud（写反即 172 GiB OOM）；②同源检验分族级（共享轴/支撑）与逐对（Gram 对角支配）两层，结论须注明层级。

## 机制解释审计链（命名前依次检查）
…→KV 五级阶梯→体位均匀→末层 L35→K 场+h7/h6→门位易感+内容自由→norm 投影→γ 预对齐→反方差重加权→通道身份→子空间几何→PC1 身份→写入分解（body 82.7pct）→身份解码（opaque）→**3063 竞争轴（写侧/门侧对抗同源，管道内第三轴）**。禁单点归因：头级 loo、内容自由门、通道 loo 三重非可加签名。

## 工程规范（Qwen3-4B）
- hidden=2560；GQA 32q/8kv；k_norm (1,s,8,128)；v_proj (1,s,1024)；RoPE NeoX；fp32 logit 级测量。
- 读出管道=单共享轴×S16 支撑；γ=反方差重加权；S16=TOP64[:16]。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；长任务 run_in_background。
- cmd.exe 经 bash 调用也损坏（参数截断）——删除/列目录用 Python os.remove/os.listdir。
- 关键写入后必须 Grep/Read 复核磁盘；幻影 Edit 会再现（本会话 3063 run 标签修正即中招）——Python 补丁 assert count==1 是唯一可靠修正通道；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3063）
Ω-P2（3011-3063）：3045-3048 KV 阶梯；3049 体位均匀；3050 末层 L35；3051 K 场+h7/h6；3052 门槽；3053 内容自由门；3054 径向稀释+γ 重整；3055 γ 预对齐；3056 词表 null；3057 反方差重加权；3058 head16 组合承重；3059 子空间几何；3060 PC1=跨体共享单轴+管道；3061 写入分解=body 82.7pct；3062 身份解码 opaque（身份=通道空间路由签名）；**3063 竞争轴：写侧 V-only 与门侧 h0K 对抗同源（|cos PC1| 0.341、Jaccard 0.138、正对照 0.138/0.019）——管道内第三轴，非载荷非读出轴**。终局：承载=门位易感；方向=语境组装（body ~5 维）+竞争轴；读出=任务无关单轴传输；γ=反方差放大。

## 下一步
- max=3063，下一个 3064（A 主选 **DS7B 复刻全链**跨模型检验；B 门区 2D 易感图；C body 指纹下游消费定位；D 竞争轴源头定位）。
'''
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory written %d chars' % len(mem_new))
else:
    o.append('memory already max=3063')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
