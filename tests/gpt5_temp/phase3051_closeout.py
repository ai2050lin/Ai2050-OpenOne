# -*- coding: utf-8 -*-
"""Phase 3051 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3051'
     r'\omega_p48_l35_anatomy_qwen')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
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
verdict = res['verdict']
assert verdict == 'kv35_headfocus_qwen', verdict
st = res['stats']
an = st['anchors']
assert an['anchor_core_ok'] is True
assert an['a116_seals_ok'] is True
assert an['a145_recapture_diff'] == 0.0
assert an['a146a_joint35_diff'] == 0.0
assert an['a146b_front_diff'] == 0.0
assert an['a147_sham_diff'] == 0.0
assert an['a148_fail'] == 0
assert an['a148_checked'] == 24
assert abs(an['a149_max_dlg']
           - 2075.698056401137) < 1e-6
t2 = st['T2_kvsplit']
assert abs(t2['med_joint']
           - 0.702645386234352) < 1e-12
assert abs(t2['med_k_only']
           - 0.6704875944661952) < 1e-12
assert abs(t2['med_v_only']
           - -0.3408632071271014) < 1e-12
assert abs(t2['frac_joint_med']
           - 1.2486363742390907) < 1e-12
assert abs(t2['frac_k_med']
           - 1.0564757920527423) < 1e-12
assert abs(t2['frac_v_med']
           - 0.8062749742615154) < 1e-12
t3 = st['T3_heads']
assert t3['best_head'] == 7
assert abs(t3['med_at_best']
           - 0.6804954382830863) < 1e-12
assert abs(t3['loo_drop_at_best']
           - 0.16325175082513388) < 1e-12
assert abs(t3['dominance_ratio']
           - 0.9684763489731675) < 1e-12
assert abs(t3['med_per_head'][6]
           - 0.6366827361132641) < 1e-12
assert abs(t3['med_loo'][7]
           - 0.5393936354092181) < 1e-12
assert abs(t3['med_loo'][1]
           - 0.7244081882847786) < 1e-12
t4 = st['T4_attn_readout']
assert abs(t4['per_pair_corr_med']
           - -0.05864768508899662) < 1e-12
assert abs(t4['corr_global']
           - -0.08947693391265528) < 1e-12
t5 = st['T5_null']
assert abs(t5['p_null']
           - 0.004975124378109453) < 1e-12
assert abs(t5['med_null']
           - 0.45518673297510037) < 1e-12
assert abs(t5['max_null']
           - 0.666724978960606) < 1e-12

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3051
           for m in led['measurements']):
    claim = (
        'Omega-P48 (plan 3051 A) - L35 final-layer '
        'anatomy, run1 fp32 authoritative (317.8s, '
        'all anchors passed first try). Capture '
        'bank LOADED from the z48 npz (a145 '
        're-capture bit 0.0) with per-pair chain '
        'anchors vs the z50 npz (a146a joint-L35 '
        'repro diff 0.0, a146b FRONT all-layer '
        'repro diff 0.0). RESULTS: (1) K/V split '
        'at L35 x FRONT: joint 0.7026, K-only '
        '0.6705 (95.4 pct of joint, frac 1.056), '
        'V-only NEGATIVE -0.3409 (frac 0.806) - '
        'the carrying payload is a K-FIELD '
        'property; V replacement alone actively '
        'ANTI-ALIGNS (V field carries an '
        'opposing/normalize-away component). '
        '(2) Per-KV-head joint replacement: h7 '
        '0.6805 (dominance 0.968 vs joint) and '
        'h6 0.6367 carry nearly everything; '
        'heads 0-5 weak (0.10-0.54); '
        'leave-one-out: dropping h7 falls to '
        '0.5394 (drop 0.163) while dropping any '
        'other head stays 0.664-0.724 - h1 loo '
        '0.7244 EXCEEDS the joint 0.7026, so '
        'carrying is non-additive (dilution/'
        'redundancy between heads). Verdict '
        'kv35_headfocus_qwen (h7 med >= 0.8 x '
        'joint). (3) Attention readout alignment: '
        'per-GQA-group attention mass from the '
        'last position onto FRONT positions does '
        'NOT predict head carrying (per-pair '
        'corr -0.059, global -0.089) - the readout '
        'routing is independent of which head '
        'carries the direction, consistent with '
        '3027 (head readout = context property). '
        '(4) PRIMARY null restricted to L35 x '
        'FRONT rows: obs 0.7026 vs null med '
        '0.4552 max 0.6667, p = 0.00498.')
    meas = {
        'meas_id': 'meas3051_omega_p48_l35_'
                   'anatomy_qwen',
        'phase': 3051,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a145 re-capture bit 0.0 vs '
                   'z48; a146a joint-L35 diff 0.0 '
                   'and a146b FRONT diff 0.0 vs '
                   'z50 per pair; a147 sham bit '
                   '0.0; a148 integrity fails 0 '
                   '(24 checked); a149 max dlg '
                   '2075.7',
        'artifacts': {
            'result': 'phase3051/omega_p48_'
                      'l35_anatomy_qwen/'
                      'result.json',
            'npz': 'phase3051/omega_p48_'
                   'l35_anatomy_qwen/'
                   'omega_p48_l35_anatomy_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run1 authoritative (317.8s, fp32); '
                'null restricted to L35 x FRONT '
                'rows; head blocks sliced in the '
                '8x128 head-major layout; '
                'attention mass via '
                'output_attentions=True on the '
                'BASE forward (descriptive only)',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 190
    l14['connects'].append({
        'meas_id': 'meas3051_omega_p48_l35_'
                   'anatomy_qwen',
        'phase': 3051,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P48: L35 anatomy - '
                        'K-field carries (K-only '
                        '0.6705 = 95.4 pct of '
                        'joint; V-only NEGATIVE '
                        '-0.3409, opposing V '
                        'component); head '
                        'concentration h7 0.6805 '
                        '+ h6 0.6367 (kv35_'
                        'headfocus_qwen); '
                        'leave-one-out non-'
                        'additive (h1 loo 0.7244 '
                        '> joint 0.7026); '
                        'attention mass does NOT '
                        'predict carrying (corr '
                        '~0) - readout routing '
                        'independent of content '
                        'carrying; restricted '
                        'null p 0.00498'})
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
if '## Phase 3051:' not in memo:
    sec = u'''## Phase 3051: Ω-P48 L35 末层解剖——K 场承载 + 头 7/6 集中 + 注意力读出与承载分离（kv35_headfocus_qwen） [%(created)s]

**判决：`kv35_headfocus_qwen`**（run1 fp32 权威 317.8s，锚核心一次全过、无崩溃）。捕获库复用 z48 npz（a145 重捕获 bit 0.0），链锚 vs z50 npz：a146a joint-L35 逐对复现 COS_SL[35] diff 0.0、a146b FRONT 全层复现 COS_FR diff 0.0。

### 设计
L35×FRONT 带四臂解剖：K/V 分拆（K-only / V-only / joint，3045/3046 协议）+ 8 KV 头逐个 joint 替换（K+V 同头块，8×128 头主序切片）+ leave-one-out + 注意力读出对齐（基线前向 output_attentions=True，GQA 组末位→FRONT 位质量）+ **行限制性 null**（L35×FRONT 行 norm 匹配随机 K+V，R=200）。

### 核心结果（重复三遍）
**① 承载是 K 场属性**：joint=**0.7026**，K-only=**0.6705**（95.4pct，frac 1.056），**V-only=−0.3409（负！）**（frac 0.806）——单独替换 V 场主动反向对齐，V 场携带对抗性/被归一化淹没的成分；KV 通路的方向信息几乎全部经 K 场进入。**② 头集中 h7+h6**：h7=**0.6805**（dominance 0.968 ≥0.8→判决 headfocus）、h6=0.6367，头 0-5 弱（0.10-0.54）；leave-one-out：去 h7 掉到 0.5394（降 0.163），去其余任一头保持 0.664-0.724，且 **h1 loo=0.7244 > joint 0.7026**——承载非可加（头间稀释/冗余）。**③ 注意力读出与承载分离**：GQA 组注意力质量不预测头承载（per-pair corr=−0.059，global=−0.089）——读出路由独立于内容承载，与 3027（头读出=上下文属性）闭环。**④ 主检验 null 显著**：obs 0.7026 vs null med 0.4552 / max 0.6667，**p=0.00498**。

### 机制链定版
**L35 内部解剖完成：K 场主承载 + 头 7/6 双头集中 + V 负成分 + 读出-承载分离**。KV 载荷定位链收口：体位均匀（3049）→层维末层集中（3050）→通道维 K 场+双头（3051）。方向信息在 L35 以 K 场（经 RoPE 旋转的 key 空间）形式写入，由特定 KV 头（h7/h6）承载，而读出注意权重与之无关——承载是内容属性，读出是上下文属性。

### 方法论入册
- **头块切片纪律（3051）**：1024=8×128 头主序；头级替换 = 基场 copy + 头块覆写（rot 后逐头切片）；loo 须把替换块外的头块写回基场值。
- **非可加警示（3051）**：leave-one-out 可超过 joint（h1 loo 0.7244 > 0.7026）——头/通道级归因不可用可加分解叙述，必须报 loo 全谱。
- **读出-承载分离检验（3051）**：注意力质量（output_attentions，基线前向）与干预效应的相关是"路由 vs 内容"的标准检验，corr≈0 即分离。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3051/omega_p48_l35_anatomy_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3052 菜单**——A（主选）**h7/h6 头身份解剖**：h7/h6 的 K 场与 unembed/词表子空间对齐、跨 body×prefix 特异性（是否固定头还是内容依赖）、与 3022 正性稀疏联盟成员身份比对；B V 负成分机制（V-only −0.34：归一化抵消还是主动对抗）；C L12 中层孤峰定位；D 跨模型 DS7B 复刻全链。
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
if '## 十三、3051 增补' not in aud:
    add = u'''
    
---

## 十三、3051 增补：L35 通道维解剖——K 场承载、头 7/6 集中、读出-承载分离（Omega-P48，判决 kv35_headfocus_qwen）

1. **K 场是承载通道**：L35×FRONT 带替换 K-only 0.6705（=joint 的 95.4pct），V-only 反向 −0.3409——方向信息经 post-norm K（RoPE 后 key 空间）写入，V 场携带对抗性成分；KV 五级阶梯的"K/V 通路"叙事在末层收口为 K 主导。
2. **头级双峰集中**：h7 0.6805（dominance 0.968）+ h6 0.6367，其余头弱；loo 非可加（去 h7 降 0.163，h1 loo 0.7244 反超 joint）——头间存在稀释/冗余，禁止可加归因。
3. **读出-承载分离**：GQA 组注意力质量与头承载效应 corr≈0（−0.059/−0.089）——注意力路由（谁被读）与 KV 内容承载（谁带方向）是独立自由度，强化 3027"头读出=上下文属性"结论；HDMCC 的"路由节点"图景在通道维同样不成立。
'''
    aud += add
    with io.open(AUDIT, 'w', encoding='utf-8') as f:
        f.write(aud)
    o.append('audit addendum +%d chars' % len(add))
else:
    o.append('audit already appended')

# ---------- workspace log ----------
wl = WLOG_DIR + r'\2026-09-21.md'
try:
    prev = io.open(wl, encoding='utf-8').read()
except IOError:
    prev = ''
if 'Phase 3051' not in prev:
    line = ('- Phase 3051 Omega-P48 L35 anatomy: '
            'verdict kv35_headfocus_qwen (run1 '
            'fp32 317.8s, all anchors passed '
            'first try). RESULTS: K/V split at '
            'L35 x FRONT - K-only 0.6705 (95.4 '
            'pct of joint 0.7026), V-only '
            'NEGATIVE -0.3409 (opposing V '
            'component); head concentration h7 '
            '0.6805 + h6 0.6367 (heads 0-5 '
            'weak); leave-one-out non-additive '
            '(h1 loo 0.7244 > joint 0.7026); '
            'attention mass does NOT predict '
            'carrying (corr -0.059/-0.089) - '
            'readout routing independent of '
            'content carrying (closes with '
            '3027); restricted null p 0.00498. '
            'Audit addendum 13; ledger 190/L14 '
            '158.\n')
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
- 脚本 tests\\glm5\\phase{N}_*.py；closeout tests\\gpt5_temp\\；产物 ...\\phase{N}\\{arm}\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14.connects；hash=去 ledger_sha256_8 后 dumps(sort_keys, ensure_ascii=False) sha256 前 8。

## 强制流程
1. 闭环：execution 冻结→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→present_files→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）；占位符 %(key)s 风格；裸百分号写 %%。
3. 重跑先删旧产物；负结果/锚失败/崩溃如实登记；verdict 单分支赋值。
4. 统计量纪律：obs 与 null 同量纲同范围；**null 限制在 exact 场同一行集**（3050）；**loo 全谱必报、禁可加归因**（3051）。

## 标准锚与精度
- bit 级仅限同文件链/同精度；精确 KV 复放：post-norm K+RoPE 偏移旋转；全长 repl+全 True mask。
- 捕获库跨相位复用：上游 npz+抽样重捕获 bit 锚+派生统计量逐对复现锚。
- 二维输出索引纪律（3050）：hook out[0] 剥 batch 维后 lm_head 输出 (n,V)——末 token 是 lg[-1]，lg[0,-1] 是标量（块状假剖面）；末层 vs LG bit 锚必须在统计前抓退化。
- transformers 5.14（3050）：decoder layer 返回裸 tensor；output_hidden_states=True 不可信——层捕获一律自建 hook。
- 头块切片纪律（3051）：1024=8×128 头主序；头级替换=基场 copy+头块覆写；loo 把替换块外头块写回基场。

## 统计判据纪律
- 判据可达性先检；构造匹配置换；maxT；镜像必配。

## 机制解释审计链（命名前依次检查）
…→KV 五级阶梯（3045-3048）→载荷定位（3049 体位均匀）→层定位（3050 末层 L35 单层充分、主动写入）→**通道定位（3051：K 场承载 95.4pct、V 负成分 −0.34、头 7/6 双峰集中、loo 非可加、注意力质量不预测承载=读出-承载分离）**。

## 工程规范（Qwen3-4B）
- hidden=2560；GQA 32q/8kv；k_norm 输出 (1,s,8,128)；v_proj 输出 (1,s,1024)；RoPE NeoX 配对、attention_scaling==1；fp32 logit 级测量；注意力质量 output_attentions=True（基线前向，描述性）。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；长任务 run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；补丁锚串禁行尾反斜杠；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3051）
Ω-P2（3011-3051）：3011 门控=L3 KV；3018-3019 抑制场；3020 注入特异 944×；3021-3024 联盟中继/承重；3028 剂量凸增长；3045-3048 KV 五级阶梯；3049 载荷定位（体位均匀）；3050 层定位（末层 L35 单层充分）；**3051 通道定位：K 场主承载+V 负成分+h7/h6 头集中+读出-承载分离（kv35_headfocus_qwen）**。

## 下一步
- max=3051，下一个 3052（A 主选 **h7/h6 头身份解剖**——K 场 unembed/词表子空间对齐+跨条件特异性+与 3022 正性联盟身份比对；B V 负成分机制；C L12 中层孤峰定位；D 跨模型 DS7B 复刻）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
