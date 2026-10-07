# -*- coding: utf-8 -*-
"""Phase 3125 closeout (idempotent):
result asserts -> Ledger -> MEMO Phase 3125 ->
workspace logs (x2 entries) -> MEMORY.md."""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3125'
        r'\omega_p123_third_comp_qwen_'
        'inputstream')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_D = ROOT + r'\.workbuddy\memory'
WLOG_C = (r'C:\Users\Admin\WorkBuddy'
          r'\2026-09-17-01-30-05\.workbuddy\memory')
MEMO_W = WLOG_D + r'\MEMORY.md'
LOGF = OUTD + r'\closeout_log.txt'
NOW = datetime.datetime.now()
NOWS = NOW.strftime('%Y-%m-%d %H:%M')
WDATE = NOW.strftime('%Y-%m-%d')
o = []

V = ('common_mode_partial|trail_present|'
     'ar_absent|trajectory_transient|'
     'third_partial|path_valid|'
     'replay_bit_exact|readout_ok|spans_ok|'
     'qis_syntax_L23|qis_syntax_L22|'
     'qis_content_L28|qis_content_L32|'
     'qis_syn_final_negative|'
     'qis_syn_final_negative|'
     'qis_cont_final_negative|'
     'qis_cont_final_negative')

# ---------- 1. result.json asserts ----------
res = json.load(io.open(OUTD + r'\result.json',
                        encoding='utf-8'))
_n = [0]


def chk(cond):
    assert cond, 'assert #%d failed' % len(_n)
    _n.append(1)


chk(res['phase'] == 3125)
chk(res['name'] == 'omega_p123_third_comp_'
    'qwen_inputstream')
chk(res['verdict'] == V)
chk(res['smoke'] is False)
chk(res['n_pairs'] == 672)
chk(res['np_b'] == 672)
chk(abs(res['runtime_s'] - 245.9) < 0.05)
pa = res['part_a']
rfP = pa['refit']['P']
chk(abs(rfP['S'] - (-0.601611613690753))
    < 1e-12)
chk(abs(rfP['MS'] - (-4.955594255793292))
    < 1e-12)
rfA = pa['refit']['A1']
chk(abs(rfA['S'] - (-0.5540798866844862))
    < 1e-12)
chk(abs(rfA['MS'] - (-7.154421990932929))
    < 1e-12)
dcP = pa['decomp3']['P']
chk(abs(dcP['cm_share']
        - 0.19001305759179912) < 1e-12)
chk(abs(dcP['m_share']
        - 0.0033137063011249046) < 1e-12)
chk(abs(dcP['trail_share']
        - 0.08599867672568678) < 1e-12)
chk(abs(dcP['ar_share']
        - 0.005767845108351762) < 1e-12)
chk(abs(dcP['rem_share']
        - 0.7149067142730373) < 1e-12)
chk(abs(dcP['ss_tot']
        - 69737.75834882268) < 1e-9)
chk(dcP['leak'] < 1e-12)
dcA = pa['decomp3']['A1']
chk(abs(dcA['cm_share']
        - 0.35574441732458434) < 1e-12)
chk(abs(dcA['m_share']
        - 3.0830390620402785e-06) < 1e-12)
chk(abs(dcA['trail_share']
        - 0.05022617239377846) < 1e-12)
chk(abs(dcA['ar_share']
        - 0.007261861528795945) < 1e-12)
chk(abs(dcA['rem_share']
        - 0.5867644657137793) < 1e-12)
chk(abs(dcA['ss_tot']
        - 103458.4879387006) < 1e-9)
chk(dcA['leak'] == 0.0)
chk(pa['cm_verdict'] == 'common_mode_partial')
chk(pa['trail_verdict'] == 'trail_present')
chk(pa['ar_verdict'] == 'ar_absent')
chk(pa['traj_verdict']
    == 'trajectory_transient')
tGP = pa['trail_G']['P']
chk(abs(tGP[0][0] - 1.2613665625792099)
    < 1e-12)
chk(abs(tGP[0][1] - (-1.7570000874491265))
    < 1e-12)
chk(abs(tGP[1][0] - (-1.6769770017180752))
    < 1e-12)
chk(abs(tGP[4][1] - 1.592554632997308)
    < 1e-12)
chk(abs(tGP[9][1] - 1.6959194067754657)
    < 1e-12)
tGA = pa['trail_G']['A1']
chk(abs(tGA[0][1] - (-1.728273987975595))
    < 1e-12)
chk(abs(tGA[4][1] - 2.4345274637276195)
    < 1e-12)
chk(abs(tGA[9][1] - 0.44877744803404757)
    < 1e-12)
arP = pa['ar_params']['P']
chk(abs(arP['phi1'] - (-0.09355245050878877))
    < 1e-12)
chk(abs(arP['phi2'] - (-0.03499462595239264))
    < 1e-12)
chk(abs(arP['c'] - (-0.0072109727195960125))
    < 1e-12)
arA = pa['ar_params']['A1']
chk(abs(arA['phi1'] - (-0.10965806973954667))
    < 1e-12)
chk(abs(arA['phi2'] - (-0.0773445705673305))
    < 1e-12)
chk(abs(arA['c'] - (-0.009942803709820168))
    < 1e-12)
tsP = pa['traj_stat']['P']
chk(abs(tsP['rho_within']
        - (-0.000633567318687721)) < 1e-12)
chk(abs(tsP['rho_perm_std']
        - 0.014041536747155588) < 1e-12)
tsA = pa['traj_stat']['A1']
chk(abs(tsA['rho_within']
        - (-0.018863878190179637)) < 1e-12)
chk(abs(tsA['rho_perm_std']
        - 0.012302487752804975) < 1e-12)
s3 = pa['sim3']
chk(abs(s3['r_sim3'] - 0.3768119309633145)
    < 1e-12)
chk(s3['verdict'] == 'third_partial')
chk(abs(s3['auc_sim3'][0]
        - 0.9809094210600907) < 1e-12)
chk(abs(s3['auc_sim3'][4]
        - 0.739937641723356) < 1e-12)
chk(abs(s3['auc_sim3'][12]
        - 0.7930706136621315) < 1e-12)
chk(abs(s3['r_det_3124_ref']
        - (-0.12297439326221062)) < 1e-12)
pb = res['part_b']
chk(pb['interference']
    == 'input_prompt_span_equal_len_'
    'anchored_last')
chk(pb['ids'] == {'yes': 9834, 'no': 902,
                  'dot': 13, 'n_layers': 36})
chk(abs(pb['path']['r']
        - 0.99997947625086) < 1e-12)
chk(pb['path']['verdict'] == 'path_valid')
chk(pb['repro']['max_diff'] == 0.0)
chk(pb['repro']['verdict']
    == 'replay_bit_exact')
chk(abs(pb['readout_auc']
        - 0.7269898844954649) < 1e-12)
chk(pb['readout_verdict'] == 'readout_ok')
chk(pb['n_span'] == {'P': 672, 'A1': 672})
chk(pb['spans_verdict'] == 'spans_ok')
lsp = pb['lstar']
chk(lsp['P']['syn'] == {'rel': 23, 'abs': 22})
chk(lsp['P']['cont'] == {'rel': 28, 'abs': 27})
chk(lsp['A1']['syn'] == {'rel': 22, 'abs': 20})
chk(lsp['A1']['cont'] == {'rel': 32, 'abs': 30})
chk(pb['sign'] == {'P': {'syn': 'negative',
                         'cont': 'negative'},
                   'A1': {'syn': 'negative',
                          'cont': 'negative'}})
cv = pb['curves']
chk(cv['E_syn_P'][0] == 0.0)
chk(abs(cv['E_syn_P'][23]
        - (-0.2861765851113166)) < 1e-12)
chk(abs(cv['E_syn_P'][36]
        - (-0.8362901988763356)) < 1e-12)
chk(abs(cv['E_syn_A1'][22]
        - (-0.1131437063138701)) < 1e-12)
chk(abs(cv['E_syn_A1'][36]
        - (-0.26489097411120044)) < 1e-12)
chk(abs(cv['E_cont_P'][28]
        - (-0.40839819107443537)) < 1e-12)
chk(abs(cv['E_cont_P'][36]
        - (-0.8324938236459432)) < 1e-12)
chk(abs(cv['E_cont_A1'][32]
        - (-0.17526632380729779)) < 1e-12)
chk(abs(cv['E_cont_A1'][36]
        - (-0.1783839988389185)) < 1e-12)
o.append('asserts ok (%d checks)' % len(_n))

# ---------- 2. Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('phase') == 3125
           for m in led['measurements']):
    claim = (
        'Omega-P123 (3125, T4 eighth phase: '
        'residual third-component localization '
        '[trail kernel + AR + trajectory '
        'persistence] + qwen3-4b input-stream '
        'control, offline 672 + qwen3-4b GPU '
        '672x2, 245.9s) - verdict ' + V + '.  '
        'Part A (offline, frozen 3118/3120, '
        '3124 fit asserted 1e-12): sequential '
        'decomposition step-dummy -> m-struct '
        '-> trail kernel (D=3, 30 params) -> '
        'AR(2) -> remainder gives m ~0, '
        'common-mode P 0.190 / A1 0.356, '
        'TRAIL 0.086 / 0.050 (A1 marginal at '
        'the 0.05 gate), AR 0.006 / 0.007, '
        'remainder still 0.587-0.715; '
        'within-trajectory lag-1 persistence '
        'of the remainder is ZERO (rho -0.001/'
        '-0.019 vs perm std 0.014/0.012, 200 '
        'deterministic perms) -> the third '
        'component is NOT a per-trajectory '
        'persistent anchor-like term; it is '
        'partly a CONTENT-TRAIL kernel '
        '(largest: cls1 d1 -1.68/-0.45, cls0 '
        'd2 -1.76/-1.73, cls4 d3 -2.03/-2.06) '
        'and re-simulation improves auc-shape '
        'r from -0.123 (3124 det) to +0.377 '
        '(third_partial) - still far from '
        'sufficient.  Part B (GPU qwen3-4b 36L '
        'input-stream, anchored-last span, gen '
        'replayed teacher-forced, path r '
        '0.99998, replay bit-exact, n_span '
        '672/672, readout AUC 0.727): E_syn '
        'L*=23(P)/22(A1), E_cont L*=28(P)/'
        '32(A1), final-layer sign NEGATIVE '
        'all four -> combined with 3124 GLM4 '
        'input-stream E_cont final POSITIVE, '
        'the final-layer sign is MODEL-'
        'SPECIFIC (not an interference-'
        'semantics variable); but the layer '
        'LOCALIZATION itself is interference-'
        'semantics dependent (Qwen cont L20 '
        'output-stream -> L28/32 input-'
        'stream) -> cross-model invariance '
        'holds at the relative-depth-band '
        'level, not exact layer index.  '
        'ENGINEERING: (1) transformers 5.14 '
        'qwen3 hidden_states[last] ALREADY '
        'carries the final norm (probe raw '
        'r=0.99999 vs logit diff; re-norm '
        'double-norm r=0.854) -> 3122/3123 '
        'L36 points were double-normed '
        '(L0..35 correct, L*=L21/L20 '
        'unaffected); (2) 3124 GLM4 span '
        'used FIRST occurrence which under P '
        'collides with the identical Facts '
        'line (probe: P occurrences=2 '
        'first=185 anchored=255; A1 unique '
        '255) -> 3124 P-direction '
        'interference replaced the FACTS '
        'ground-truth line, not the query '
        'line; 3125 fixes with suffix-'
        "anchored LAST occurrence (' Is "
        "this').  NEXT 3126: GLM4 anchored-"
        'last re-run + counterfactual '
        'regeneration + write-chain layer id '
        '(GPU); long-range trail kernel D>3 + '
        'permutation significance + higher-'
        'order decomposition of the remaining '
        '0.59-0.71 (offline).')
    meas = {
        'meas_id': 'meas3125_omega_p123_third_'
                   'comp_qwen_inputstream',
        'phase': 3125,
        'claim': claim,
        'verdict': V,
        'anchors': 'design_seal.json frozen '
                   'before computation: A_trail '
                   'share >=0.05 present, A_ar '
                   '>=0.05 present, A_traj rho '
                   '>=0.1 AND > perm_mean+4*std '
                   '(200 deterministic perms '
                   'rng 3125), A_sim r >=0.5 '
                   'sufficient/>=0.3 partial, '
                   'B_path corr >0.9999 FATAL, '
                   'B_repro ==0/<1e-6 FATAL, '
                   'B_readout AUC >0.6, B_spans '
                   '300/300 full (SMOKE '
                   'max(1,NP_B//2)), L*_rel = '
                   'min L>=20 E<=-0.3|E_final| '
                   '(abs -0.05/-0.025 reported); '
                   'deterministic: crc32 seeds '
                   '+ fixed rng 3125, no MC',
        'artifacts': {
            'result': 'phase3125/omega_p123_'
                      'third_comp_qwen_'
                      'inputstream/result.json',
            'seal': 'phase3125/omega_p123_'
                    'third_comp_qwen_'
                    'inputstream/'
                    'design_seal.json',
            'readout': 'phase3125/omega_p123_'
                       'third_comp_qwen_'
                       'inputstream/'
                       'p123_readout.npz'},
        'hashes': {},
        'note': 'GPU qwen3-4b only (36L BF16 '
                'eager, batch1 forwards, '
                '245.9s; Part A offline on '
                'frozen qwen3-4b data); 3 '
                'SMOKE iterations before full '
                'run: (1) logits margin = '
                'yes-no logit diff not w_dn '
                'matmul (vocab-space dim), '
                '(2) D-PATH FATAL -> probe '
                'found qwen3 hs[last] already '
                'final-normed -> norm only '
                'L<NL (r 0.99998), (3) SPANPROBE '
                'exposed the 3124 first-'
                'occurrence Facts-line collision '
                '-> anchored-last fix',
    }
    led['measurements'].append(meas)
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model'][
        0]
    l14['connects'].append(
        'meas3125_omega_p123_third_comp_qwen_'
        'inputstream')
    led.pop('ledger_sha256_8', None)
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w',
                 encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False,
                  indent=1)
    o.append('ledger appended n=%d l14=%d sha=%s'
             % (len(led['measurements']),
                len(l14['connects']),
                led['ledger_sha256_8']))
else:
    o.append('ledger already upserted')

# ---------- 3. MEMO Phase 3125 ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3125:' not in memo:
    sec = u'''## Phase 3125: Ω-P123 残差第三成分定位（trail 核 + AR + 轨迹持久性）+ Qwen 输入流对照与符号分离（T4 第8Phase）——**第三成分=内容尾迹（trail 0.086/0.050，AR≈0、轨迹内持久性否定），sim3 r −0.12→+0.38 仍 third_partial；Qwen 输入流 E_syn L23/22、E_cont L28/32、终层全负——E_cont 终层符号=模型特异非干预语义，层定位本身是干预语义依赖的；工程双发现：transformers 5.14 Qwen3 hidden_states 末项已含 final norm（3122/3123 L36 双重 norm、L* 判定不受影响）+ 3124 P 方向 span=Facts 行碰撞（3125 anchored-last 修复）** [[NOW]]

**性质**：T4 第 8 Phase，3124 MEMO 第 5 节预注册四项中的 ①（offline）+ ③（Qwen 输入流）；门在 seal 观测前冻结（design_seal.json）。Part A offline 全量 672（3118/3120 冻结数据 + 3124 refit 断言 1e-12，纯 numpy 确定性）；Part B GPU qwen3-4b（36 层 BF16 eager batch1）245.9s（5408 forwards + check/repro）。GLM4 两项（② 反事实生成、④ 写入链层识别）按 GPU 逐模型纪律留待 3126。SMOKE 3 轮迭代后全绿。

### 1. 三大发现（重复三遍）
1. **第三成分定性：内容尾迹（trail）而非轨迹持久结构——(m, content一步, anchor, 线性算子, trail核, AR2) 模型族仍不足**。顺序分解（step-dummy → m-结构 → trail 核 D=3 → AR(2) → 余项）：m-结构 P 0.0033/A1 3e-6（复现 3124）、共模 P 0.190/A1 0.356（复现）、**trail 份额 P 0.086/A1 0.050（A1 恰过 0.05 门——边缘判定）**、AR 0.006/0.007（absent）、**余项仍 0.715/0.587**。余项的轨迹内 lag-1 持久性精确为零（ρ −0.001/−0.019 vs 置换 std 0.014/0.012，200 次确定性置换）——**第三成分不是"轨迹特定持久项"（Kalman 锚点族的替代假说被否定）**；内容尾迹核（类条件 × lag 1-3）贡献 5–9%，主系数：cls1 lag1 −1.68/−0.45（语法 token 抑制下一步）、cls0 lag2 −1.76/−1.73、cls4 lag3 −2.03/−2.06（事实 token 长程反相尾迹）、cls9 lag2 +1.70/+0.45（答案 token 正尾迹）。确定性重模拟 r 从 3124 的 −0.123 提升至 +0.377（third_partial）——**方向正确但远不充分，剩余 0.59–0.71 主体未定位**。
2. **Qwen 输入流对照：E_cont 终层符号=模型特异；层定位本身是干预语义依赖的**。输入流（prompt 内 anchored-last 查询行替换、生成流 teacher-forced 重放、path r 0.99998、replay bit-exact、n_span 672/672、readout AUC 0.727）：E_syn L\\*=23(P)/22(A1)、E_cont L\\*=28(P)/32(A1)、**终层符号四项全负**（E_syn_P [36] −0.836、E_syn_A1 −0.265、E_cont_P −0.832、E_cont_A1 −0.178）。三方对照：Qwen 输出流（3123）终层负、Qwen 输入流（3125）终层负、GLM4 输入流（3124）终层正 → **E_cont 终层符号差异（GLM4 +0.34/+0.09 vs Qwen −0.83/−0.18，同为输入流）=模型特异，排除干预语义解释**。同时 Qwen 的层定位在两种干预语义下移动：syn L21→L23/22、**cont L20→L28/32（大幅后移）**——"写入链上游涌现"的精确层号不是跨模型不变量，稳健的是相对深度带（涌现于 0.5–0.9 深度区间、写入链上游）与符号模式指纹。
3. **工程双发现（影响既有结论的解释边界）**：① **transformers 5.14 Qwen3 的 hidden_states 末项已含 final norm**（探针：raw(hs[36]) vs logit-diff r=0.99999、再 norm 双重归一 r=0.854）→ **3122/3123 系列的 L36 读出是双重 norm 版本**（L0–35 单 norm 正确、L\\*=L21/L20 判定不受影响，末层曲线值语义有偏——下游若用 3123 末层值需重算）；② **3124 GLM4 的 P 方向 span 定位碰撞**：find() 首次出现命中与查询行文本相同的 Facts 真值行（SPANPROBE：P occurrences=2 first=185 anchored=255；A1 唯一 255）→ **3124 P 方向干预实际替换的是 Facts 真值行**（"答案依据行替换"语义），A1 方向无混杂（查询行替换）——3124 内 P/A1 曲线形状差异部分来自行身份不对称；3125 用后缀锚定（' Is this' 跟随查询行）的 LAST 出现修复，P/A1 对称（672/672）。

### 2. 关键数值
Part A：decomp3 P {cm 0.19001305759179912, m 0.0033137063011249046, trail 0.08599867672568678, ar 0.005767845108351762, rem 0.7149067142730373, leak 2.2e-16}、A1 {cm 0.35574441732458434, m 3.08e-06, trail 0.05022617239377846, ar 0.007261861528795945, rem 0.5867644657137793, leak 0}；AR φ1 −0.0936/−0.1097、φ2 −0.0350/−0.0773、c −0.0072/−0.0099；traj ρ_within −0.0006/−0.0189（perm mean −0.0018/+0.0012、std 0.0140/0.0123）；sim3 r 0.3768119309633145、auc_sim3 [0] 0.9809/[4] 0.7399/[12] 0.7931。Part B：path r 0.99997947625086、repro 0.0、readout_auc 0.7269898844954649、n_span 672/672；lstar P {syn rel23/abs22, cont rel28/abs27}、A1 {syn rel22/abs20, cont rel32/abs30}；sign 全 negative；E_syn_P [23] −0.2862/[36] −0.8363；E_syn_A1 [22] −0.1131/[36] −0.2649；E_cont_P [28] −0.4084/[36] −0.8325；E_cont_A1 [32] −0.1753/[36] −0.1784；ids yes 9834/no 902/dot 13。

### 3. 硬伤
① trail_share A1 0.0502 恰过 0.05 门（门 0.06 即翻转 absent）——边缘判定未做敏感性；② trail 核 30 参数在 8064 样本上无正则拟合、无置换显著性检验——份额可能含过拟合成分；③ sim3 r 0.377 仍 third_partial：trail+AR 只回收一小部分，剩余 0.59–0.71 主体未定位（候选：D>3 长程尾迹、跨轨迹高阶共模、m 定义外非线性项）；④ Qwen 输入流 readout AUC 0.727 低于输出流 3123 水平——teacher-forced 固定生成流在 prompt 扰动下行为有效性未验证（反事实生成对照仍缺，GLM4/Qwen 都待做）；⑤ lstar 层号是干预语义依赖的（cont L20→L28/32）——跨实验比较层号必须在同干预语义下进行；⑥ 3122/3123 L36 双重 norm——其末层曲线值需重算后方可用于定量比较（本 Phase 未使用其数值）；⑦ 3124 P 方向 Facts 行混杂使 3124 P 与 3125 P 不严格可比（A1 方向可比）——GLM4 anchored-last 重跑列入 3126；⑧ 置换检验 200 次分辨率有限（p≈1/200）。

### 4. 机制拼图更新
内部响应图谱：① 5 维残差分解表（cm/m/trail/ar/rem × 方向）——第三成分首次定性；② trail 核 G 表（10 类 × 3 lag × 2 方向）——**首次内容影响的时序核估计**（语法/事实 token 负尾迹、答案 token 正尾迹）；③ Qwen 全 37 层 × 4 条件 × 672 × 2 lens margin 场（mlg npz）+ E 曲线 + L* + 终层符号；④ span_idx 存档（672×2 × 2 方向）。RDC 更新：① **残差模型族再扩展仍不足：+(trail 核, AR2) 后余项 0.59–0.71——第三成分主体既非轨迹持久也非短程内容尾迹，需要新候选项（长程序列结构 / 跨轨迹条件二阶结构 / 非线性内容交互）**；② **E_cont 终层符号=模型特异指纹（GLM4 正/Qwen 负），跨模型比较从"精确层号"降级为"相对深度带 + 符号模式"**；③ 干预语义 × span 行身份是实验设计变量——未来跨模型对照必须 anchored-last + 同干预语义；④ Qwen3 hs 末项已 norm 修正后，forward_trackL_q 成为 Qwen 系读出标准（3122/3123 末层值留待按需重算）。

### 5. 3126 预注册（T4 继续，观测前冻结框架）
① **GLM4 anchored-last 修正重跑（GPU）**：修复 P 方向 span 定位后复测 GLM4 E_syn/E_cont 曲线、L*、终层符号——与 3125 Qwen 同构比较（判定 3124 曲线形状差异中行身份混杂的贡献）；② **GLM4 反事实生成对照（GPU）**：s1–s3 扰动 prompt 重新生成，检验固定重放结论的行为有效性；③ **GLM4 写入链层识别（GPU）**：对应 Qwen L26–L35 的 GLM4 写入层定位与读出；④ **trail 深化（offline）**：D>3 长程核 + 置换显著性 + G 表跨方向/跨类结构 + 剩余 0.59–0.71 的更高阶分解（跨轨迹条件二阶统计）。具体门在 3126 seal 冻结。

产物：`tests/glm5/result/rdc_query_construction_20260913/phase3125/omega_p123_third_comp_qwen_inputstream/`（result.json、design_seal.json、run_log.txt、p123_readout.npz）；脚本 `tests/glm5/phase3125_omega_p123_third_comp_qwen_inputstream.py`；探针 `tests/gpt5_temp/p3125_path_probe.py`。
'''
    sec = sec.replace('[[NOW]]', '[' + NOWS + ']')
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    _memo_delta = len(sec)
    o.append('memo +%d chars (Phase 3125)'
             % len(sec))
else:
    _memo_delta = 0
    o.append('memo already appended')

# ---------- 4. workspace logs (x2) ----------
line_exp = ('- Phase 3125 Omega-P123 (T4 eighth '
            'phase: residual third-component '
            'localization [trail kernel D=3 + '
            'AR(2) + trajectory persistence] + '
            'qwen3-4b input-stream control, '
            'offline 672 + GPU 672x2, 245.9s): '
            'verdict ' + V + '. '
            '(A) Third component = content '
            'TRAIL (0.086/0.050, A1 marginal), '
            'AR absent (0.006/0.007), within-'
            'trajectory persistence ZERO (rho '
            '-0.001/-0.019 vs perm std 0.014/'
            '0.012) -> NOT a per-trajectory '
            'persistent term; trail kernel: '
            'cls1 lag1 -1.68/-0.45, cls0 lag2 '
            '-1.76/-1.73, cls4 lag3 -2.03/'
            '-2.06, cls9 lag2 +1.70/+0.45; '
            're-sim r -0.123 -> +0.377 (third_'
            'partial), remainder still 0.59-0.71. '
            '(B) Qwen input-stream: E_syn L23/'
            '22, E_cont L28/32, final sign '
            'NEGATIVE x4 -> E_cont final sign '
            'is MODEL-SPECIFIC (GLM4 input-'
            'stream positive vs Qwen input-'
            'stream negative); layer '
            'localization itself is interference-'
            'semantics dependent (Qwen cont '
            'L20 output -> L28/32 input) -> '
            'invariance holds at relative-depth-'
            'band level. (C) Engineering: '
            'transformers 5.14 qwen3 '
            'hidden_states[last] already final-'
            'normed (raw r 0.99999, re-norm '
            '0.854) -> 3122/3123 L36 double-'
            'normed (L0..35 correct, L* '
            'unaffected); 3124 GLM4 P-direction '
            'span collided with the identical '
            'Facts line (first-occurrence bug, '
            'probe P occ=2 first=185 anchored='
            '255) -> 3124 P replaced the FACTS '
            'line; 3125 anchored-last fix '
            '(n_span 672/672, path r 0.99998, '
            'replay bit-exact, readout AUC '
            '0.727). NEXT 3126: GLM4 anchored-'
            'last re-run + counterfactual '
            'regeneration + write-chain layer '
            'id (GPU); long-range trail + '
            'permutation significance + higher-'
            'order decomposition (offline).\n')
line_clo = ('- Phase 3125 closeout finished: '
            'five-write chain ok (ledger n=%LGN% '
            'l14=%L14N% sha=%SHA8%, MEMO +%MEMOC% '
            'chars, dual wlog, MEMORY.md update); '
            'disk verify next. 3 SMOKE iterations '
            '(logits margin = yes-no diff, not '
            'w_dn matmul; qwen3 hs[last] already '
            'final-normed -> norm only L<NL, '
            'D-PATH FATAL fixed r 0.99998; '
            'SPANPROBE exposed 3124 first-'
            'occurrence Facts-line collision -> '
            'anchored-last fix); full run '
            '245.9s.\n')
try:
    led2 = json.load(io.open(LEDGER,
                             encoding='utf-8'))
    _sha8 = led2['ledger_sha256_8']
    _lgn = len(led2['measurements'])
    _l14n = len([l for l in led2['linkage']
                 if l.get('link_id')
                 == 'L14_readout_spectrum_'
                    'cross_model'][0]
                ['connects'])
except Exception:
    _sha8 = 'unknown'
    _lgn = 0
    _l14n = 0
line_clo = (line_clo
            .replace('%LGN%', str(_lgn))
            .replace('%L14N%', str(_l14n))
            .replace('%SHA8%', _sha8)
            .replace('%MEMOC%', str(_memo_delta)))
for wdir in (WLOG_D, WLOG_C):
    for tag, line in (('exp', line_exp),
                      ('clo', line_clo)):
        wl = wdir + '\\' + WDATE + '.md'
        try:
            prev = io.open(wl,
                           encoding='utf-8').read()
        except IOError:
            prev = ''
        marker = ('Phase 3125 Omega-P123' if tag
                  == 'exp'
                  else 'Phase 3125 closeout')
        if marker not in prev:
            try:
                with io.open(wl, 'a',
                             encoding='utf-8') as f:
                    f.write(line)
                o.append('wlog %s appended %s'
                         % (tag, wl))
            except Exception as e:
                o.append('wlog %s fail %s: %r'
                         % (tag, wl, e))
        else:
            o.append('wlog %s already %s'
                     % (tag, wl))

# ---------- 5. MEMORY.md ----------
mem_old = io.open(MEMO_W, encoding='utf-8').read()
if 'max=3124' in mem_old:
    NEW_3125 = (u'- 3125（T4）：第三成分=内容尾迹（trail '
                u'0.086/0.050 边缘过门；AR≈0、轨迹持久性'
                u'否定）；sim3 r −0.12→+0.38；Qwen 输入流 '
                u'E_syn L23/22、E_cont L28/32、终层全负→'
                u'**E_cont 符号=模型特异、层定位是干预语义'
                u'依赖的**；**Qwen3 hs 末项已 norm（3122/'
                u'3123 L36 双 norm）+3124 P 方向 span='
                u'Facts 行混杂**。')
    NEW_3124_COMPACT = (u'- 3124（T4）：残差 m≈0、共模 '
                        u'0.19/0.36、余项 0.65–0.81 主导→'
                        u'(m,锚点,算子) 族不足；L35 释压='
                        u'范数×语义各半；GLM4 语法/内容 '
                        u'L*=20/20 复现写入链上游。')
    NEW_3121_22 = (u'- 3121–3122：token 级替换失效'
                   u'（振荡淹没均值）；写入谱 L28–34 正写+'
                   u'L35 大负写 −11.4；句级替换 E_cont '
                   u'−3.68/−2.23、语法混排恢复一半=**语法'
                   u'使内容可读**。')
    NEW_NEXT = (u'- max=3125，下一 3126：**GLM4 anchored-'
                u'last 修正重跑 + 反事实生成对照 + 写入链'
                u'层识别（GPU）+ trail 长程核/显著性深化'
                u'（offline）**。')
    lines = mem_old.splitlines()
    out = []
    skip = 0
    for ln in lines:
        if skip > 0:
            skip -= 1
            continue
        if ln.startswith(u'## 机制链状态'):
            out.append(u'## 机制链状态（3125）')
        elif ln.startswith(u'- 3124'):
            out.append(NEW_3124_COMPACT)
            out.append(NEW_3125)
        elif ln.startswith(u'- 3121'):
            out.append(NEW_3121_22)
            skip = 1
        elif ln.startswith(u'- max=3124'):
            out.append(NEW_NEXT)
        else:
            out.append(ln)
    mem_new = u'\n'.join(out) + u'\n'
    assert mem_new.count(
        u'## 机制链状态（3125）') == 1
    assert mem_new.count(u'- 3125（T4）') == 1
    assert mem_new.count(u'- 3124（T4）') == 1
    assert mem_new.count(u'max=3125') == 1
    assert mem_new.count(u'- 3122（T4）') == 0
    assert mem_new.count(u'- 3121（T4）') == 0
    assert mem_new.count(u'- 3121–3122：') == 1
    assert mem_new.count(u'- 3123（T4）') == 1
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w',
                 encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory updated %d chars'
             % len(mem_new))
else:
    o.append('memory already updated')

with io.open(LOGF, 'w', encoding='utf-8') as f:
    f.write('\n'.join(o) + '\n')
print('CLOSEOUT_OK (%d steps)' % len(o))
