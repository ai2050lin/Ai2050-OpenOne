# -*- coding: utf-8 -*-
"""Phase 3039 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log -> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3039'
     r'\omega_p36_direct_logistic_replication_qwen')
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
assert verdict == 'direct_logistic_breakdown_qwen', \
    verdict
assert res['anchor_all_ok'] is True
an = res['anchors']
assert an['a67_dup_base_bit'] == 0.0
assert an['a68_gate_m0_bit'] == 0.0
assert an['a69_max_dp_3038'] == 0.0
assert an['a69_tok_identity'] is True
assert an['a70_dup_rs_bit'] == 0.0
assert an['a70_ratio_err'] <= 0.02
assert an['a71_top2_ok'] is True
assert an['a71_maxdiff'] <= 0.15
assert an['a72_source_seals'] is True
t1 = res['T1_inflection']
assert t1['n_eligible'] == 9
assert t1['n_excluded_saturation'] == 3
assert t1['n_match'] == 4
assert t1['sham_match'] == 1
t2 = res['T2_specificity']
assert abs(t2['spec_ratio']
           - 16.04828406302903) < 1e-12
t3 = res['T3_competition']
assert abs(t3['med_A_idx']
           - 0.18667137594928157) < 1e-12
assert abs(t3['med_abs_kappa']
           - 0.3532870207555261) < 1e-12
assert t3['n_mono'] == 10

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3039
           for m in led['measurements']):
    claim = (
        'Omega-P36 (plan v5 P36) - direct-protocol '
        'logistic replication (protocol robustness '
        'verdict after 3038): fp_inject verbatim '
        'moved INTO the prefill last position (L-1) '
        'so intervention and readout share the '
        'direct natural-continuation protocol; 12 '
        'logic + 3 sham prompts, M_GRID '
        '{-0.05..0.05} per token A/B at site 35, '
        'random-direction control (rank-100/101 '
        'pair).  Anchors: a67/a68/a69/a70 bit-level '
        '0.0 (incl. cross-phase vs 3038 p_dirA '
        '12/12); a70 ratio_err 0.0021; a71 manual '
        'recompute 0.0453; a72 seals.  T1 verdict '
        'direct_logistic_breakdown_qwen: inflection '
        'sign law COLLAPSES under the direct '
        'protocol - 4/9 eligible matches (chance '
        'level; sham 1/3; 3 rows excluded by the '
        '3036 saturation window) vs 10/11 under '
        're-entrant (3035).  T2: fingerprint '
        'specificity PERSISTS and strengthens - '
        'spec_ratio 16.05 (m01) / 29.01 (m02) vs '
        '13.2 re-entrant.  T3: odd-dominance '
        'robust (med A_idx 0.187 vs 0.202), '
        'multibody kappa same order (med |kappa| '
        '0.353 vs 0.448), mono 10/12.  CONCLUSION: '
        'the logistic curvature SIGN LAW is a '
        're-entrant-protocol property (created by '
        'the extra full-depth pass); first-order '
        'directed migration, fingerprint '
        'specificity and multibody coupling are '
        'protocol-robust.  All curvature claims '
        '(3028-3036) are protocol-conditional; '
        'odd-component claims are protocol-'
        'invariant.  NEXT: episodic-component '
        'extraction, cross-model replication of '
        'the robust triplet (spec/kappa/odd).')
    meas = {
        'meas_id': 'meas3039_omega_p36_direct_'
                   'logistic_replication_qwen',
        'phase': 3039,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a67/a68/a69/a70 bit 0.0 (a70 '
                   'ratio_err 0.0021); a71 0.0453; '
                   'a72 seals; T1 4/9 eligible '
                   'vs sham 1/3; T2 spec 16.05; T3 '
                   'A_idx 0.187 kappa 0.353 mono '
                   '10/12',
        'artifacts': {
            'result': 'phase3039/omega_p36_'
                      'direct_logistic_'
                      'replication_qwen/'
                      'result.json',
            'npz': 'phase3039/omega_p36_'
                   'direct_logistic_'
                   'replication_qwen/'
                   'omega_p36_direct_logistic_'
                   'replication_qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run1 authoritative (17.2s, first '
                'pass, no corrections); a69 '
                'reuses the main-loop base arrays '
                '(same quantity, no duplicate '
                'pass needed beyond a67)',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 178
    l14['connects'].append({
        'meas_id': 'meas3039_omega_p36_direct_'
                   'logistic_replication_qwen',
        'phase': 3039,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P36: direct-protocol '
                        'logistic replication - '
                        'curvature sign law '
                        'COLLAPSES under direct '
                        'protocol (4/9 vs 10/11 '
                        're-entrant; chance level) '
                        '= re-entrant-protocol '
                        'property; fingerprint '
                        'specificity persists and '
                        'strengthens (spec 16.05 vs '
                        '13.2); odd-dominance (A_idx '
                        '0.187) and multibody kappa '
                        '(0.353) protocol-robust; '
                        'curvature claims 3028-3036 '
                        'protocol-conditional; '
                        'direct_logistic_breakdown_'
                        'qwen'})
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
if '## Phase 3039:' not in memo:
    sec = u'''## Phase 3039: Ω-P36 直读协议 logistic 复刻——曲率符号律崩溃（4/9≈机会）而特异性增强（16×）：二阶结构=再入协议属性 [%(created)s]

**判决：`direct_logistic_breakdown_qwen`**（run1 权威 17.2s **一次通过**，锚全过，无崩溃）

### 设计（3038 协议条件性发现的直接判决实验）
fp_inject **verbatim 移入 prefill 末位置（L−1）**——干预与读出同处直读（自然延续）协议；无 step-2、无重喂。12 logic + 3 sham prompts；剂量网格 M_GRID {−0.05..0.05} 逐 token A/B 于 site 35 输入；随机方向对照（rank-100/101 对）verbatim；饱和窗 |P0−0.5|<0.05 剔除（3036 教训）。锚：a67/a68/a69/a70 位级 0.0（a69 跨相位 vs 3038 p_dirA 12/12 位级恒等；a70 重复臂 rs 0.0 + 注入比率误差 0.0021）；a71 手工重算 0.0453；a72 封印校验。

### 核心结果（重复三遍）
**① 曲率符号律崩溃**：eligible 9 行（3 行饱和窗剔除）中 inflection 符号匹配仅 **4/9 ≈ 机会**（sham 1/3）——再入协议下 10/11（3035）的 logistic 符号律**不复刻**。**② 指纹特异性保持且增强**：spec_ratio=**16.05**（m=0.01）/29.01（m=0.02）vs 再入 13.2——差方向迁移的定向性是协议鲁棒的。**③ 奇主导与多体耦合鲁棒**：med A_idx=**0.187**（再入 0.202）、med |κ|=**0.353**（再入 0.448，同量级）、mono 10/12。

### 机制链定版（协议分层标注）
- **协议不变（真机制）**：定向指纹迁移（奇分量主导）、指纹特异 ×13–29、多体场 κ≈0.35–0.45、L8 阻尼 27×、联盟/承重/头载体全链（3032/3034 未涉协议）。
- **再入协议条件（3038–3039 判决）**：logistic 曲率符号律、凸指数 γ 增长的读出段绝对值——由**多过的整深度一遍**（阻尼场再混合）制造。
- 3028–3036 全部曲率类断言标注"协议条件"；奇分量类断言标注"协议不变"。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3039/omega_p36_direct_logistic_replication_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3040 菜单——A（主选）**情景分量提取**（3037 遗留 B：去除词身份公共方向后残差的语境编码——cos 残差 vs 位置/上下文特征回归）；B 阻尼场通道分解（再入 vs 直读的衰减谱差 = 3038 机制的直接验证）；C 深峰头簇公共核心；D 跨模型复刻（鲁棒三件套 spec/κ/奇主导 上 DS7B/GLM4）。
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
if 'Phase 3039' not in prev:
    line = ('- Phase 3039 Omega-P36: verdict '
            'direct_logistic_breakdown_qwen (run1 '
            '17.2s first pass; anchors a67-a70 bit '
            '0.0 incl. cross-phase 3038 12/12, a71 '
            '0.0453, a72 seals); logistic curvature '
            'sign law COLLAPSES under direct '
            'protocol (4/9 eligible vs 10/11 '
            're-entrant, chance level); fingerprint '
            'specificity persists and strengthens '
            '(spec 16.05 vs 13.2); odd-dominance '
            '(A_idx 0.187) and kappa (0.353) '
            'protocol-robust; curvature claims '
            '3028-3036 protocol-conditional, odd-'
            'component claims protocol-invariant; '
            'ledger 178/L14 146.\n')
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
- 干预锚族：重复基链/门开 m=0/重复臂 rs 位级 0.0+注入比率门 2e-2；手工 norm+lm_head 重算+0.15 门；源封印；跨相位基线锚（3039 a69 vs 3038 p_dirA 位级）。**重复臂方向必须与原臂同对象**（3039 修正）。

## 统计判据纪律
- 判据可达性先检；零方差候选先剔；maxT 家族校正；margin n≳40（n=11 探索性）。
- **null 门不得设在接收干预的量上**（3035）；**曲率检验报操作点并排除饱和区 |P0−0.5|<0.05**（3036）；**跨相位锚定前核对读出协议量纲：再入 step-2 ≠ 直接 prefill**（3037）。
- 比值/凸超额报 (log 基线, gamma) 二元组；集合统计用精确超几何 null；同词相似性用标签置换+maxT。

## 机制解释审计链（命名前依次检查）
norm 分解→口径标签→子空间→SVD 能量迁移→组内混淆→倍率报分子分母→剂量分单调/峰位→凸响应定位→异质性归因→集合身份对照随机 null→定向干预对照随机方向→曲率符号报操作点→曲率残差对照随机方向底线→KV 相似性三分量（3037）→**读出协议条件性标注：曲率=再入协议属性，奇分量/特异性=协议不变（3038/3039）**→消融差分=直接+重平衡。

## 工程规范（Qwen3-4B）
- hidden=2560；头仅 o_proj 输入侧；真残差流=decoder-layer pre-hook；npz dict→0-d .item()；hook 改输出用返回值+active 门；bf16 hook 配 bf16 列；step-2 单 token [0,-1]；**step-2 污染 KV cache→每链重新 prefill**；**pre-hook with_kwargs 改输入须返回 (new_args,new_kwargs) 且注册先于捕获 hook**；lm_head 行数=config.vocab_size；fp_inject 可移入 prefill 末位（直读协议干预，3039）；output_attentions 位级不变（3038）。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；反引号/管道/tail 被篡改；rm 损坏→os.remove；`cmd &` 孤儿进程死→run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3039）
Ω-P2（3011-3039）：3011 门控=L3 KV；3018-3019 通用抑制场；3020 注入特异 944×；3021-3022 L3 联盟中继 82pct；3024 承重 0.657；3028 剂量凸增长；3029-3030 凸=读出本征（再入）；3031/3033 异质性=比值伪影（β 0.649）；3032 深峰=头集中 89pct；3034 头集合=部分共享通路；3035 指纹竞争 logistic（10/11、特异 13.2×、κ 0.448）；3036 曲率=操作点属性（H2 拒绝）；3037 L3 KV=词身份主导+公共中继+语境调制；3038 **再入读出偏平（gain 0.868/H 1.229；自注意 3pct 非机制）**；3039 **直读复刻：曲率符号律崩溃 4/9（=再入协议属性）而特异增强 16×、奇主导 0.187/κ 0.353 协议鲁棒**。定版：种子→L3 联盟中继→阻尼场均衡→定向指纹读出（奇分量协议不变；二阶结构=再入多过一遍所生）。

## 下一步
- max=3039，下一个 3040（A 主选 **情景分量提取**——去词身份公共方向后残差的语境编码；B 阻尼场通道分解（再入 vs 直读衰减谱差）；C 深峰头簇公共核心；D 跨模型复刻鲁棒三件套；附件审计 research\\gpt5\\docs\\fingerprint_competition_review_20260921.md）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
