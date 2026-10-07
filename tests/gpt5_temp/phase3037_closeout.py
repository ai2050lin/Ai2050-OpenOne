# -*- coding: utf-8 -*-
"""Phase 3037 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log -> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3037'
     r'\omega_p34_kv_situational_specificity_qwen')
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
assert verdict == 'kv_mixed_qwen', verdict
assert res['anchor_all_ok'] is True
t1 = res['T1_stereotypy']
assert abs(t1['med_diff_cosV3']
           - 0.5542954337685086) < 1e-12
assert abs(t1['grand_ratio']
           - 1.7181572469253465) < 1e-12
assert t1['n_ratio_hi'] == 0
assert abs(t1['p_maxT'] - 0.00315) < 1e-12
pw = {p['word']: p for p in t1['per_word']}
assert abs(pw['although']['med_cosV3']
           - 0.9805093376768814) < 1e-12
assert abs(pw['while']['med_cosV3']
           - 0.8479756172606726) < 1e-12
assert pw['because']['n_pairs'] == 6
t3 = res['T3_layer20']
assert abs(t3['ratio'] - 1.683208544881659) < 1e-12
assert t3['med_diff_cosV20'] > 0.4
an = res['anchors']
assert an['a58_dup_prefill_bit'] == 0.0
assert an['a59_dup_all_bit'] == 0.0
assert an['a60_top2_ok'] is True
assert an['a60_maxdiff'] < 0.15
assert an['a61_source_seals'] is True
reobs = res['a60_reentrant_observation']
assert reobs['tok_match'] == 9
assert abs(reobs['max_dp']
           - 0.7638666384316595) < 1e-12

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3037
           for m in led['measurements']):
    claim = (
        'Omega-P34 (plan v5 P34) - L3 KV situational '
        'specificity (3013 direct test): 28 prompts '
        '(12 GEN + 16 minimal-pair), 8 target words, '
        '25 occurrences located by exact token-id '
        'scan; pure prefill + cache read at layer 3 '
        'kv head 7 (3011 gate head), V primary (K '
        'RoPE-confounded, descriptive).  Anchors: '
        'a58/a59 duplicate-extraction bit-level 0.0; '
        'a60 manual final-norm+lm_head recompute '
        'top-2 identity max|dlogit| 0.0453 (gate '
        '0.15); a61 source seals 3035/3036.  T1 '
        'PRIMARY: same-word cross-context V3 cosine '
        'med 0.848-0.981 per word (8/8 words '
        'p<0.005 uncorrected, maxT p=0.00315) vs '
        'different-word null 0.554; grand ratio '
        '1.718, NO word reaches ratio>=2.0 - '
        'verdict kv_mixed_qwen: L3 word writes = '
        'word-identity component DOMINANT (same-'
        'word cos ~0.95) + large shared relay '
        'component (different-word cos 0.554) + '
        'genuine context modulation (1 - cos ~ '
        '0.05-0.15); the 3013 episodic-KV claim is '
        'PARTIALLY correct - context matters but '
        'word identity dominates.  T3: same mixed '
        'structure at L20 (ratio 1.683).  T2 within-'
        'prompt repeats: empty bank (registered '
        'design gap).  Registered observation: the '
        '3036 two-step re-entrant readout (step-2 '
        're-feeds last token at position L) differs '
        'from the direct prefill readout (position '
        'L-1): tok_match 9/11, max dp 0.764 - the '
        'two-step protocol measures RE-ENTRANT '
        'readout, flagged for follow-up.  NEXT: '
        're-entrant readout anatomy, episodic-'
        'component extraction, deep-head core, '
        'cross-model replication.')
    meas = {
        'meas_id': 'meas3037_omega_p34_kv_'
                   'situational_specificity_qwen',
        'phase': 3037,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a58/a59 bit-level 0.0; a60 '
                   'manual recompute 0.0453; a61 '
                   'seals ok; T1 8/8 words '
                   'p<0.005, maxT 0.00315, ratio '
                   '1.718',
        'artifacts': {
            'result': 'phase3037/omega_p34_'
                      'kv_situational_'
                      'specificity_qwen/'
                      'result.json',
            'npz': 'phase3037/omega_p34_'
                   'kv_situational_'
                   'specificity_qwen/'
                   'omega_p34_kv_situational_'
                   'specificity_qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run2 authoritative (21.5s); run1 '
                'crashed pre-verdict (grand_ratio '
                'shape bug) and its a60 was mis-'
                'specified (re-entrant step-2 '
                'softmax vs direct prefill readout '
                'are different quantities by '
                'design; max dp 0.764 registered '
                'as observation); corrected per '
                '3035-run4 precedent; T1 values '
                'identical across run1/run2',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 176
    l14['connects'].append({
        'meas_id': 'meas3037_omega_p34_kv_'
                   'situational_specificity_qwen',
        'phase': 3037,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P34: L3 KV situational '
                        'specificity - same-word '
                        'cross-context V cos ~0.95 vs '
                        'different-word 0.554 (ratio '
                        '1.718, maxT p 0.00315, no '
                        'word >= 2x): word-identity '
                        'dominant + shared relay '
                        'component + genuine context '
                        'modulation; 3013 episodic '
                        'claim PARTIALLY correct; L20 '
                        'replicates (1.683); re-entrant '
                        'vs direct readout differ '
                        '(9/11 tok, dp 0.764) - '
                        'two-step protocol flagged; '
                        'kv_mixed_qwen'})
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
if '## Phase 3037:' not in memo:
    sec = u'''## Phase 3037: Ω-P34 L3 KV 情景性检验——词身份分量主导 + 语境调制（同词 cos 0.95 vs 异词 0.55，ratio 1.72）[%(created)s]

**判决：`kv_mixed_qwen`**（run2 权威 21.5s；run1 崩溃前修正如实登记：a60 误设 + grand_ratio 形状 bug，T1 统计跨 run 一致不受影响）

### 设计（同词异位 KV 直接检验 3013）
28 prompts（12 GEN + 16 极小对），8 目标词（because×4，however/while/although/therefore/yet/thus/so×3）共 25 次出现，精确 token-id 扫描定位；**纯 prefill + cache 读出**（无干预链），L3 kv head 7（3011 门头），**V 为主判据**（K 含 RoPE 位置混淆，仅描述性）；逐词跨语境同词对 vs 异词对经验 null + 精确标签置换（100k，seed 9037）+ maxT 家族校正。

### 核心结果（重复三遍）
**① 词身份分量主导**：同词跨语境 V3 cos med 0.848–0.981（8/8 词全部 p<0.005，maxT p=0.00315），异词 null 0.554——**L3 的 V 写入有大公共中继分量（异词间仍有 0.55 cos）**，词身份在其上再抬 ~0.4。**② 但非固定刻板方向**：同词 cos 距 1.0 尚差 0.05–0.15 = 真实语境调制分量；grand_ratio=1.718，无一词达 2.0 门 → 既非纯刻板亦非纯情景，**3013 情景式 KV 判决：部分正确**——语境确实调制 L3 KV，但词身份是主导分量。**③ L20 复刻**：ratio 1.683（同词 0.722 vs 异词 0.429）——混合结构非 L3 特有，是中继层的普遍组织。**④ 再入观察（新登记）**：3036 两步协议的再入读出（step-2 位置 L 重喂末 token）与直接 prefill 读出（位置 L−1）显著不同（tok_match 9/11，max dp 0.764）——**两步协议测的是再入读出**，为独立解剖对象。

### 缺陷与修正登记（如实）
run1 预注册 a60 把两步再入 softmax 与直接 prefill 读出对比——**设计性错误（不同量纲）**，实测 dp 0.764 后改判为观察量、换 a51 族手工重算锚（maxdiff 0.0453，门 0.15）；run1 grand_ratio 数组形状 bug 崩溃（判决前）。**教训入册：跨相位锚定前必须核对读出协议量纲（再入 step-2 ≠ 直接 prefill）。** T2 词库内重复字母 token 为空集（词库设计缺口，如实登记）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3037/omega_p34_kv_situational_specificity_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3038 菜单——A（主选）**再入读出解剖**（两步协议 vs 直读差异的系统量化：重复 token 在位置 L 的注意路径与读出增益）；B 情景分量提取（去除词身份公共方向后残差的语境编码：cos 残差 vs 位置/上下文特征）；C 深峰头簇公共核心解剖；D 跨模型复刻（DS7B/GLM4 四件套 + KV 混合结构）。
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
if 'Phase 3037' not in prev:
    line = ('- Phase 3037 Omega-P34: verdict '
            'kv_mixed_qwen (run2 21.5s; run1 '
            'crashed pre-verdict: a60 mis-specified '
            're-entrant vs direct readout + '
            'grand_ratio shape bug, registered); '
            'same-word cross-context V3 cos '
            '0.848-0.981 (8/8 words p<0.005, maxT '
            '0.00315) vs different-word 0.554, '
            'ratio 1.718: word-identity dominant + '
            'shared relay component + context '
            'modulation; 3013 episodic claim '
            'partially correct; L20 replicates '
            '1.683; re-entrant vs direct readout '
            'differ (9/11 tok, dp 0.764) - two-step '
            'protocol flagged; ledger 176/L14 144.\n')
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
- 干预相位锚族（3035/3036）：重复基链/门开 m=0/重复臂 rs 位级 0.0+注入比率门 2e-2；手工 norm+lm_head 重算 top2 恒等+0.15 门；源封印校验；读出类锚：重复 prefill/提取位级 0.0（3037 a58/a59）。

## 统计判据纪律
- 判据可达性先检；零方差候选先剔；maxT 家族校正；margin n≳40（n=11 探索性）。
- **null 门不得设在接收干预的量上**（3035 run4）；**二阶差分/曲率检验须报操作点 P0 并排除饱和区**（3036）；**跨相位锚定前核对读出协议量纲：再入 step-2 ≠ 直接 prefill**（3037 a60 教训，dp 0.764）。
- 比值/凸超额报 (log 基线, gamma) 二元组；集合统计用精确超几何 null；同词相似性用标签置换+maxT（3037）。

## 机制解释审计链（命名前依次检查）
norm 分解→口径标签→子空间→SVD 能量迁移→组内混淆→倍率报分子分母→剂量分单调/峰位→凸响应定位→异质性归因→集合身份对照随机 null→定向干预对照随机方向→曲率符号报操作点→曲率残差对照随机方向底线→KV 相似性分词身份/公共/语境三分量（3037）→消融差分=直接+重平衡→读出集中对照任意扰动 null。

## 工程规范（Qwen3-4B）
- hidden=2560；头仅 o_proj 输入侧；真残差流=decoder-layer pre-hook；npz dict→0-d .item()；hook 改输出用返回值+active 门；bf16 hook 配 bf16 列；step-2 单 token [0,-1]；**step-2 前向污染 KV cache→每链重新 prefill**；**pre-hook with_kwargs 改输入须返回 (new_args,new_kwargs) 且注册先于捕获 hook**；lm_head 行数=config.vocab_size；cache 读 K/V：past.layers[li].keys[0,kv_head,pos,:]。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；反引号/管道/tail 被篡改；rm 损坏→os.remove；`cmd &` 孤儿进程死→run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3037）
Ω-P2（3011-3037）：3011 门控=L3 KV；3014 剂量非单调=K 路由；3018-3019 抵消=通用抑制场；3020 注入特异 944×；3021 MLP 中继 69pct；3022 稀疏联盟 top32=82pct；3023 零消融有毒；3024 联盟承重 0.657；3027 消费=通用读出；3028 剂量凸增长；3029-3030 凸=读出本征；3031/3033 异质性=tag 特异+比值伪影（β 0.649 CI<1）；3032 深峰=头集中 89pct；3034 头集合=部分共享通路（J 0.188，按 ldp 层聚类）；3035 指纹竞争=logistic 读出（inflection 10/11、差方向特异 13.2×、κ 0.448、L8 阻尼 27×）；3036 曲率=操作点属性（H2 拒绝 med_max_z 0.564<3；符号 86/87；指纹低重叠 cos 0.123）；3037 **L3 KV=词身份主导+公共中继分量+语境调制（同词 cos 0.95 vs 异词 0.55，ratio 1.72，maxT 0.00315；L20 复刻 1.68；再入≠直读 dp 0.764）**。核心：重要性=关系属性；统一链=种子→L3 联盟中继→阻尼场均衡→logistic 读出。

## 下一步
- max=3037，下一个 3038（A 主选 **再入读出解剖**——两步协议 vs 直读差异系统量化；B 情景分量提取（去词身份公共方向后的语境编码）；C 深峰头簇公共核心；D 跨模型复刻 DS7B/GLM4；附件审计 research\\gpt5\\docs\\fingerprint_competition_review_20260921.md）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
