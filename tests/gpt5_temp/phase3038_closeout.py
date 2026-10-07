# -*- coding: utf-8 -*-
"""Phase 3038 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log -> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3038'
     r'\omega_p35_reentrant_readout_qwen')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_DIR = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
            r'\.workbuddy\memory')
MEMO_W = WLOG_DIR + r'\MEMORY.md'
LOGF = R + r'\closeout_log.txt'
o = []

res = json.load(io.open(R + r'\result.json',
                        encoding='utf-8'))
seal = json.load(io.open(R + r'\seal.json',
                         encoding='utf-8'))
exe = json.load(io.open(R + r'\execution.json',
                        encoding='utf-8'))
created = exe['created']
verdict = res['final_verdict']
assert verdict == 'reentrant_flattening_qwen', verdict
assert res['anchor_all_ok'] is True
an = res['anchors']
assert an['a62_dup_prefill_bit'] == 0.0
assert an['a63_dup_reentrant_bit'] == 0.0
assert an['a64_max_dp_3036'] == 0.0
assert an['a64_tok_identity'] is True
assert an['a65_attn_flag_diff'] == 0.0
assert an['a66_source_seals'] is True
t1 = res['T1_redistribution']
assert abs(t1['med_abs_dp']
           - 0.09374930034039089) < 1e-12
assert abs(t1['max_abs_dp']
           - 0.8817482671866087) < 1e-12
assert t1['n_tok_match'] == 10
assert abs(t1['dp_per_prompt'][0]
           - (-0.3089043006893806)) < 1e-12
assert abs(t1['dp_per_prompt'][6]
           - (-0.8817482671866087)) < 1e-12
t2 = res['T2_self_attention']
assert abs(t2['self_deep_med']
           - 0.03045654296875) < 1e-12
assert abs(t2['self_early_med']
           - 0.051513671875) < 1e-12
t3 = res['T3_sharpening']
assert abs(t3['med_gain']
           - 0.8683405407546754) < 1e-12
assert abs(t3['med_H_ratio']
           - 1.228930212562358) < 1e-12
assert abs(t3['med_jac10']
           - 0.6666666666666666) < 1e-12

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3038
           for m in led['measurements']):
    claim = (
        'Omega-P35 (plan v5 P35) - re-entrant '
        'readout anatomy: the two-step protocol '
        '(prefill -> step-2 re-feeds last token at '
        'position L) used by all intervention phases '
        'vs the direct prefill readout (position '
        'L-1); 12 GEN prompts, 3 passes each (direct, '
        're-entrant verbatim, re-entrant + '
        'output_attentions).  Anchors: a62/a63 '
        'duplicate passes bit-level 0.0; a64 '
        'cross-phase identity vs 3036 p0_top 11/11 '
        'rows max dp 0.0 (bit) + top-1 identity; a65 '
        'attn-flag pass bit-level 0.0; a66 source '
        'seals 3036/3037.  T1: med|dp| 0.0937, '
        'max|dp| 0.8817 (P6), tok_match 10/12; T3 '
        'verdict reentrant_flattening_qwen: med '
        'gain 0.868 (<0.95) AND med H-ratio 1.229 '
        '(>1.05) - the re-entrant readout is '
        'FLATTER on median (per-prompt mixed: P0 '
        'sharpens H-ratio 0.449, P6 flattens '
        '3.957); top-10 Jaccard med 0.667.  T2 '
        'mechanism: step-2 self-attention mass at '
        'own fresh KV is SMALL and not deep-'
        'enriched (med 3.0pct deep vs 5.2pct early) '
        '- the difference is not a self-loop but '
        'the extra full-depth pass (damping field '
        'mixing, 3018-3019).  IMPLICATION: the '
        're-entrant quantity is the continuation '
        'AFTER prompt+repeated-token, not the '
        'natural continuation; all absolute '
        'probabilities since 3028 are protocol-'
        'conditional (within-phase comparisons '
        'unaffected); direct-protocol replication '
        'of the logistic/curvature findings is the '
        'priority follow-up.  NEXT: direct-protocol '
        'logistic replication, episodic-component '
        'extraction, cross-model.')
    meas = {
        'meas_id': 'meas3038_omega_p35_reentrant_'
                   'readout_qwen',
        'phase': 3038,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a62/a63/a64/a65 bit-level 0.0 '
                   '(incl. cross-phase 3036 11/11); '
                   'a66 seals ok; T1 med|dp| 0.0937 '
                   'max 0.8817 tok 10/12; T3 med '
                   'gain 0.868 H-ratio 1.229',
        'artifacts': {
            'result': 'phase3038/omega_p35_'
                      'reentrant_readout_qwen/'
                      'result.json',
            'npz': 'phase3038/omega_p35_'
                   'reentrant_readout_qwen/'
                   'omega_p35_reentrant_readout_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run1 authoritative (7.4s, first '
                'pass, no corrections); reachability '
                'pre-checked on the 3037 dp preview '
                '(negligible branch dropped)',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 177
    l14['connects'].append({
        'meas_id': 'meas3038_omega_p35_reentrant_'
                   'readout_qwen',
        'phase': 3038,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P35: re-entrant '
                        'readout anatomy - two-step '
                        'protocol readout is FLATTER '
                        'than direct prefill readout '
                        '(med gain 0.868, med H-ratio '
                        '1.229, tok_match 10/12, '
                        'med|dp| 0.0937 max 0.8817); '
                        'per-prompt mixed (P0 '
                        'sharpens 0.449 / P6 flattens '
                        '3.957); step-2 self-'
                        'attention small and not '
                        'deep-enriched (3.0pct vs '
                        '5.2pct early) - difference '
                        '= extra full-depth pass, not '
                        'self-loop; absolute probs '
                        'since 3028 are protocol-'
                        'conditional; '
                        'reentrant_flattening_qwen'})
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
if '## Phase 3038:' not in memo:
    sec = u'''## Phase 3038: Ω-P35 再入读出解剖——两步协议读出系统性偏平（med gain 0.868 / H-ratio 1.229），自注意非机制 [%(created)s]

**判决：`reentrant_flattening_qwen`**（run1 权威 7.4s **一次通过**，锚全过，无崩溃）

### 设计（3037 a60 教训的直接后续）
12 GEN prompts × 3 遍：① 直接 prefill 读出（位置 L−1，注意 0..L−1）；② 再入链 verbatim（prefill → step-2 位置 L 重喂末 token，注意 0..L）＝3028 起所有干预相位的读出协议；③ 同②+output_attentions（T2 解剖用）。float64 softmax；纯观察无干预。判据可达性预检用 3037 dp 预览（negligible 分支剔除）。

### 核心结果（重复三遍）
**① 再入读出系统性偏平**：med gain=**0.868**（<0.95）且 med H-ratio=**1.229**（>1.05）→ flattening 分支；逐 prompt 混合（P0 锐化 0.449 ↔ P6 剧烈扁平 3.957）；med|dp|=0.0937、max|dp|=0.882（P6 逗号 top-1 在再入下 p 从大掉到 0.047）、tok_match=10/12、top-10 Jaccard med 0.667。**② 机制排除自环**：step-2 对自身新 KV 的注意质量 med 仅 **3.0pct**（深带）vs 5.2pct（早带）——**不富集且量小**；差异来自**多过的整深度一遍计算**（阻尼场混合，3018–3019 独立复证），非自注意捷径。**③ 协议条件性（重要警示）**：再入读出回答的是"prompt+重复 token 之后"的延续分布，**非自然延续**——3028 以来的绝对概率都是协议条件量（相位内双臂同协议，比较不受影响）；a64 跨相位 11/11 位级 0.0 证明机器无漂移，差异纯粹是协议本体。

### 机制链更新
种子 → L3 联盟中继 → 阻尼场均衡 → logistic 读出——其中"logistic 读出"确立于**再入协议**；自然直读协议下的曲率/符号是否复刻 = 下一步 A（协议敏感性判决）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3038/omega_p35_reentrant_readout_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3039 菜单——A（主选）**直读协议 logistic/曲率复刻**（直读下定向 ±δ 干预：inflection 符号、曲率符号律、指纹特异性是否协议不变——机制链的协议鲁棒性判决）；B 情景分量提取（去词身份公共方向后的语境编码）；C 深峰头簇公共核心；D 跨模型复刻（DS7B/GLM4 四件套+KV 三分量）。
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

# ---------- workspace log ----------
wl = WLOG_DIR + r'\2026-09-21.md'
try:
    prev = io.open(wl, encoding='utf-8').read()
except IOError:
    prev = ''
if 'Phase 3038' not in prev:
    line = ('- Phase 3038 Omega-P35: verdict '
            'reentrant_flattening_qwen (run1 7.4s '
            'first pass; anchors a62/a63/a64/a65 '
            'bit-level 0.0 incl. cross-phase 3036 '
            '11/11, a66 seals); re-entrant readout '
            '(two-step protocol) systematically '
            'FLATTER than direct prefill readout: '
            'med gain 0.868, med H-ratio 1.229, '
            'tok_match 10/12, med|dp| 0.0937 max '
            '0.8817; per-prompt mixed (P0 sharpen '
            '0.449 / P6 flatten 3.957); self-'
            'attention small not deep-enriched (3.0 '
            'vs 5.2pct early) - mechanism = extra '
            'full-depth pass (damping field), not '
            'self-loop; absolute probs since 3028 '
            'are protocol-conditional; ledger '
            '177/L14 145.\n')
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

# ---------- MEMORY.md rewrite (<=3000 chars) ----------
mem_new = u'''# RDC/LPF 研究项目纪律（工作区长期记忆）

## 项目身份
- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）。
- MEMO append-only：research\\gpt5\\docs\\AGI_GPT5_MEMO.md（glm5 版封存勿写）。
- 脚本 tests\\glm5\\phase{N}_*.py；产物 ...\\phase{N}\\{arm}\\；临时 gpt5_temp\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14；hash=去 ledger_sha256_8 后 dumps(sort_keys) sha256 前 8。

## 强制流程
1. 闭环：execution 冻结（PREREG/锚/判决）→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→present_files→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）。
3. 重跑先删旧 execution/result/npz；负结果如实登记；verdict 判据分支内赋值。
4. MEMO 占位符一律 %(key)s 风格；MEMO 文本内裸百分号写 %%（3033 教训）。

## 标准锚与精度
- a1 dirs 重建 2.17e-08；bit 级仅限同文件链/上游全精度；跨相位 npz 锚均 0.0；剂量两端点锚死。
- GPU 复刻相位：verbatim 拷贝+外科补丁（assert count==1），全套旧锚新 run 复过（3034）。
- 干预相位锚族（3035/3036）：重复基链/门开 m=0/重复臂 rs 位级 0.0+注入比率门 2e-2；手工 norm+lm_head 重算 top2 恒等+0.15 门；源封印校验；读出类锚：重复 prefill/提取/再入链/output_attentions 位级 0.0（3037/3038）。

## 统计判据纪律
- 判据可达性先检；零方差候选先剔；maxT 家族校正；margin n≳40（n=11 探索性）。
- **null 门不得设在接收干预的量上**（3035 run4）；**二阶差分/曲率检验须报操作点 P0 并排除饱和区**（3036）；**跨相位锚定前核对读出协议量纲：再入 step-2 ≠ 直接 prefill**（3037 a60 教训）。
- 比值/凸超额报 (log 基线, gamma) 二元组；集合统计用精确超几何 null；同词相似性用标签置换+maxT（3037）。

## 机制解释审计链（命名前依次检查）
norm 分解→口径标签→子空间→SVD 能量迁移→组内混淆→倍率报分子分母→剂量分单调/峰位→凸响应定位→异质性归因→集合身份对照随机 null→定向干预对照随机方向→曲率符号报操作点→曲率残差对照随机方向底线→KV 相似性分词身份/公共/语境三分量（3037）→**读出协议条件性标注（再入 vs 直读，3038）**→消融差分=直接+重平衡→读出集中对照任意扰动 null。

## 工程规范（Qwen3-4B）
- hidden=2560；头仅 o_proj 输入侧；真残差流=decoder-layer pre-hook；npz dict→0-d .item()；hook 改输出用返回值+active 门；bf16 hook 配 bf16 列；step-2 单 token [0,-1]；**step-2 前向污染 KV cache→每链重新 prefill**；**pre-hook with_kwargs 改输入须返回 (new_args,new_kwargs) 且注册先于捕获 hook**；lm_head 行数=config.vocab_size；cache 读 K/V：past.layers[li].keys[0,kv_head,pos,:]；output_attentions=True 位级不变（3038 a65）。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；反引号/管道/tail 被篡改；rm 损坏→os.remove；`cmd &` 孤儿进程死→run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3038）
Ω-P2（3011-3038）：3011 门控=L3 KV；3014 剂量非单调=K 路由；3018-3019 抵消=通用抑制场；3020 注入特异 944×；3021 MLP 中继 69pct；3022 稀疏联盟 top32=82pct；3024 联盟承重 0.657；3028 剂量凸增长；3029-3030 凸=读出本征；3031/3033 异质性=tag 特异+比值伪影（β 0.649 CI<1）；3032 深峰=头集中 89pct；3034 头集合=部分共享通路（J 0.188）；3035 指纹竞争=logistic 读出（inflection 10/11、差方向特异 13.2×、κ 0.448）；3036 曲率=操作点属性（H2 拒绝 0.564<3）；3037 L3 KV=词身份主导+公共中继+语境调制（ratio 1.72）；3038 **再入读出系统性偏平（med gain 0.868/H-ratio 1.229/tok 10/12；自注意 3pct 非机制=多过整深度一遍；3028 起绝对概率协议条件化）**。核心：重要性=关系属性；统一链=种子→L3 联盟中继→阻尼场均衡→logistic 读出（再入协议确立）。

## 下一步
- max=3038，下一个 3039（A 主选 **直读协议 logistic/曲率复刻**——机制链的协议鲁棒性判决；B 情景分量提取；C 深峰头簇公共核心；D 跨模型复刻 DS7B/GLM4；附件审计 research\\gpt5\\docs\\fingerprint_competition_review_20260921.md）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
