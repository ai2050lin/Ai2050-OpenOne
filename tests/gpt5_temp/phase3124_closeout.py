# -*- coding: utf-8 -*-
"""Phase 3124 closeout (idempotent):
result asserts -> Ledger -> MEMO Phase 3124 ->
workspace logs (x2 entries) -> MEMORY.md."""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3124'
        r'\omega_p122_resid_cm_kalman_l35rel_'
        'glm4x')
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

V = ('common_mode_partial|det_core_failed|r2_weak|'
     'lag1_negative|anchor_static|'
     'own_anchor_insufficient|'
     'semantic_component_present|path_valid|'
     'replay_bit_exact|readout_ok|spans_ok|'
     'glm_syntax_L20|glm_syntax_L20|'
     'glm_content_L20|glm_content_L20')

# ---------- 1. result.json asserts ----------
res = json.load(io.open(OUTD + r'\result.json',
                        encoding='utf-8'))
_n = [0]


def chk(cond):
    assert cond, 'assert #%d failed' % len(_n)
    _n.append(1)


chk(res['phase'] == 3124)
chk(res['name'] == 'omega_p122_resid_cm_'
    'kalman_l35rel_glm4x')
chk(res['verdict'] == V)
chk(res['smoke'] is False)
chk(res['n_pairs'] == 672)
chk(res['np_d'] == 672)
chk(abs(res['runtime_s'] - 13475.6) < 0.05)
pa = res['part_a']
rfP = pa['refit']['P']
chk(abs(rfP['S'] - (-0.601611613690753)) < 1e-9)
chk(abs(rfP['MS'] - (-4.955594255793292)) < 1e-9)
rfA = pa['refit']['A1']
chk(abs(rfA['S'] - (-0.5540798866844862)) < 1e-9)
chk(abs(rfA['MS'] - (-7.154421990932929)) < 1e-9)
dcP = pa['decomp']['P']
chk(abs(dcP['cm_share']
        - 0.19001305759179912) < 1e-12)
chk(abs(dcP['m_share']
        - 0.0033137063011249046) < 1e-12)
chk(abs(dcP['rem_share']
        - 0.806673236107076) < 1e-12)
chk(abs(dcP['ss_tot']
        - 69737.75834882268) < 1e-9)
chk(dcP['leak'] == 0.0)
dcA = pa['decomp']['A1']
chk(abs(dcA['cm_share']
        - 0.35574441732458434) < 1e-12)
chk(abs(dcA['m_share']
        - 3.0830390620402785e-06) < 1e-12)
chk(abs(dcA['rem_share']
        - 0.6442524996363537) < 1e-12)
chk(abs(dcA['ss_tot']
        - 103458.4879387006) < 1e-9)
chk(abs(pa['cm_min']
        - 0.19001305759179912) < 1e-12)
chk(pa['cm_verdict'] == 'common_mode_partial')
ds = pa['det_sim']
chk(abs(ds['r_det']
        - (-0.12297439326221062)) < 1e-12)
chk(ds['verdict'] == 'det_core_failed')
chk(abs(ds['r2_median']['P']
        - 0.32964950758136624) < 1e-12)
chk(abs(ds['r2_median']['A1']
        - (-0.015582100480399985)) < 1e-12)
chk(ds['r2_verdict'] == 'r2_weak')
chk(abs(ds['auc_det'][0]
        - 0.9809094210600907) < 1e-12)
chk(abs(ds['auc_det'][4]
        - 0.739937641723356) < 1e-12)
chk(abs(ds['auc_det'][12]
        - 0.9714405293367347) < 1e-12)
an = pa['analytic']
chk(abs(an['sig_P'] - 3.206171299571775) < 1e-12)
chk(abs(an['sig_A1']
        - 4.001745467245389) < 1e-12)
chk(abs(an['gap'] - 2.198827735139629) < 1e-12)
chk(abs(an['phi'] - 0.6659700011599246) < 1e-12)
chk(abs(an['phi_3123_plateau']
        - 0.6635588242718963) < 1e-12)
pb = res['part_b']
lg = pb['lag1']
chk(abs(lg['P'] - (-0.1674141723780209)) < 1e-12)
chk(abs(lg['A1']
        - (-0.006200686258676985)) < 1e-12)
chk(pb['lag1_verdict'] == 'lag1_negative')
klP = pb['kalman']['P']
chk(abs(klP['q_hat']
        - 0.00014353249592527267) < 1e-12)
chk(abs(klP['ratio'] - 0.0001) < 1e-12)
chk(abs(klP['sig_a2']
        - 1.4353249592527266) < 1e-12)
klA = pb['kalman']['A1']
chk(abs(klA['q_hat']
        - 0.0001481438608183718) < 1e-12)
chk(abs(klA['ratio'] - 0.0001) < 1e-12)
chk(abs(klA['sig_a2']
        - 1.4814386081837179) < 1e-12)
chk(pb['kalman_verdict'] == 'anchor_static')
os_ = pb['own_sim']
chk(abs(os_['r_own']
        - 0.0062127328674929) < 1e-12)
chk(os_['verdict'] == 'own_anchor_insufficient')
chk(abs(os_['r2_median']['P']
        - 0.4035387815911334) < 1e-12)
chk(abs(os_['r2_median']['A1']
        - 0.016867642371469482) < 1e-12)
chk(abs(os_['auc_own'][12]
        - 0.92605362457483) < 1e-12)
pc = res['part_c']
l35P = pc['l35rel']['P']
chk(abs(l35P['W_ans']
        - (-11.373681399084273)) < 1e-9)
chk(abs(l35P['W_oth']
        - (-11.479271332374513)) < 1e-9)
chk(abs(l35P['norm_share']
        - 0.5009523266638253) < 1e-12)
chk(abs(l35P['t_norm']
        - 2.6518068179125405) < 1e-12)
chk(abs(l35P['t_sem']
        - (-2.641724475918722)) < 1e-12)
chk(abs(l35P['leak']
        - (-0.11567227528405821)) < 1e-12)
l35A = pc['l35rel']['A1']
chk(abs(l35A['W_ans']
        - (-8.029358918468157)) < 1e-9)
chk(abs(l35A['W_oth']
        - (-11.716023877190498)) < 1e-9)
chk(abs(l35A['norm_share']
        - 0.4717725310807821) < 1e-12)
chk(abs(l35A['t_norm']
        - (-1.7147589040668996)) < 1e-12)
chk(abs(l35A['t_sem']
        - (-1.919956538433672)) < 1e-12)
chk(abs(l35A['leak']
        - (-0.051949516221769354)) < 1e-12)
chk(pc['verdict'] == 'semantic_component_present')
pd = res['part_d']
chk(pd['interference']
    == 'input_prompt_span_equal_len')
chk(abs(pd['path']['r']
        - 0.9999857905405052) < 1e-12)
chk(pd['path']['verdict'] == 'path_valid')
chk(pd['repro']['max_diff'] == 0.0)
chk(pd['repro']['verdict']
    == 'replay_bit_exact')
chk(abs(pd['readout_auc']
        - 0.8313137755102041) < 1e-12)
chk(pd['readout_verdict'] == 'readout_ok')
chk(pd['n_span'] == {'P': 672, 'A1': 672})
chk(pd['spans_verdict'] == 'spans_ok')
lsp = pd['lstar']
chk(lsp['P']['syn'] == {'rel': 20, 'abs': 20})
chk(lsp['P']['cont'] == {'rel': 20, 'abs': 20})
chk(lsp['A1']['syn'] == {'rel': 20, 'abs': 20})
chk(lsp['A1']['cont'] == {'rel': 20, 'abs': 20})
cv = pd['curves']
chk(cv['E_syn_P'][0] == 0.0)
chk(abs(cv['E_syn_P'][9]
        - (-0.34420040916078365)) < 1e-12)
chk(abs(cv['E_syn_P'][20]
        - (-0.12357999268414346)) < 1e-12)
chk(abs(cv['E_syn_P'][40]
        - 0.0593891150570533) < 1e-12)
chk(abs(cv['E_syn_A1'][9]
        - (-0.31807351703226805)) < 1e-12)
chk(abs(cv['E_syn_A1'][40]
        - (-0.1140573490683959)) < 1e-12)
chk(abs(cv['E_cont_P'][9]
        - (-0.2808397461834192)) < 1e-12)
chk(abs(cv['E_cont_P'][20]
        - (-0.3004961357086951)) < 1e-12)
chk(abs(cv['E_cont_P'][26]
        - 0.7840392479669615) < 1e-12)
chk(abs(cv['E_cont_P'][40]
        - 0.3408039071573277) < 1e-12)
chk(abs(cv['E_cont_A1'][20]
        - (-0.2806829876836132)) < 1e-12)
chk(abs(cv['E_cont_A1'][26]
        - 0.8675801791301875) < 1e-12)
chk(abs(cv['E_cont_A1'][40]
        - 0.08595561432300791) < 1e-12)
ft = pd['first_tokens']
chk(ft['P'][0] == [9450, 670, 'Yes'])
chk(ft['A1'][0] == [2753, 555, 'No'])
o.append('asserts ok (%d checks)' % len(_n))

# ---------- 2. Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('phase') == 3124
           for m in led['measurements']):
    claim = (
        'Omega-P122 (3124, T4 seventh phase: '
        'residual 3-way decomposition + Kalman '
        'anchor-drift test + L35 release first-'
        'order decomposition + GLM4 cross-model '
        'sentence paradigm and layer tracing, '
        'qwen3-4b offline + glm4-9b-chat-hf '
        'GPU, 13475.6s) - verdict ' + V + '.  '
        'Part A (offline, frozen 3118/3123 '
        'data): 3-way residual decomposition '
        'gives m-structure share ~0 (P 0.0033, '
        'A1 3e-6), common-mode share P 0.190 / '
        'A1 0.356 (A1 near the 0.2 dominant '
        'gate), remainder 0.65-0.81 -> '
        'common_mode_partial; the '
        'deterministic no-noise simulation '
        'm(t+1)=m+S(m-MS)+content FAILS (r '
        '-0.123, R2med P 0.33 / A1 -0.016) '
        'while the analytic AR(1) plateau '
        'prediction Phi(gap/sqrt(sigP^2+'
        'sigA1^2)) = 0.6660 again matches the '
        '3123 plateau 0.6636 -> residual is '
        'NOT i.i.d. (3123) AND NOT explained '
        'by (m-structure + anchor + linear '
        'operator): a THIRD systematic '
        'component dominates (rem 0.65-0.81).  '
        'Part B: implied-anchor lag-1 autocorr '
        '-0.167 (P) / -0.006 (A1) negative; '
        'Kalman random-walk MLE q_hat ~1.4e-4 '
        'at the grid LOWER BOUND (ratio 0.0001 '
        'both dirs) -> anchor_static within '
        'the tested model family; RTS-smoothed '
        'own-anchor simulation r 0.0062 '
        'insufficient -> persistence of a '
        'per-trajectory anchor is unsupported '
        'AND unhelpful for reconstruction.  '
        'Part C (offline 3122 wrec): L35 '
        'release first-order decomposition W_'
        'oth-W_ans = Pn*(Wn_o-Wn_a) [norm] + '
        'Wn_a*(Pn_o-Pn_a) [sem] + leak gives '
        'norm_share 0.501 (P) / 0.472 (A1) -> '
        'semantic_component_present: the '
        'final-layer brake is ~half norm gain '
        'x half genuine semantic suppression '
        '(t_sem -2.64/-1.92 both significant); '
        'A1 release +3.69 replicates 3123 '
        'bit-level (raw dW -8.029 vs -11.716).'
        '  Part D (GPU glm4-9b-chat-hf 40L, '
        'hook-collected emb+L1..L40 pre-norm '
        'outputs after probe-verifying the '
        'transformers 5.14 append-before '
        'hidden_states trap [emb, L1..L39, '
        'norm(h)] with double-norm r 0.847 -> '
        'fixed, path r 0.99999 vs lm_head '
        'logits, replay BIT-EXACT 0.0): GLM4 '
        'answers Yes/No directly (P first-tok '
        'Yes 670/672; A1 No 555/672 + drift '
        '117) and NEVER echoes the query '
        'line, so interference was REDESIGNED '
        'from output-stream to INPUT-prompt '
        'span (equal-length replacement, '
        'pos0 preserved, generated stream '
        'replayed unchanged; n_span 672/672 '
        'BOTH dirs, spans_ok); readout AUC '
        '0.831; 41-layer logit-lens curves '
        'give E_syn L* = 20 and E_cont L* = '
        '20 BOTH directions (rel and abs '
        'gates agree) -> the Qwen 3123 L21/'
        'L20 write-chain-upstream emergence '
        'REPLICATES cross-model (GLM4 L20/40 '
        '= 0.50 relative depth vs Qwen L21/'
        '36 = 0.58); curve SHAPE differs: no '
        'final-layer deepening (E_syn final '
        '+0.059/-0.114 vs Qwen -2.256/-1.707) '
        'and E_cont final POSITIVE +0.341/'
        '+0.086 (vs Qwen -3.677/-2.231) with '
        'a shared mid-layer trough L8-11 - '
        'signs not directly comparable across '
        'interference semantics.  NEXT 3125: '
        'third-component localization (within-'
        'trajectory autocorrelation/spectral '
        'structure + content-trail terms); '
        'GLM4 counterfactual regeneration '
        'control; Qwen input-stream control '
        'to isolate the interference-semantics '
        'variable; GLM4 write-chain layer '
        'identification.')
    meas = {
        'meas_id': 'meas3124_omega_p122_resid_'
                   'cm_kalman_l35rel_glm4x',
        'phase': 3124,
        'claim': claim,
        'verdict': V,
        'anchors': 'design_seal.json frozen '
                   'before computation: A_cm min '
                   'share >=0.2 dominant/>=0.05 '
                   'partial, A_det r 0.7/0.5, '
                   'A_r2 median 0.5, B_lag1 mean '
                   '+/-0.05, B_kalman ratio '
                   'max>0.1 drifts/<0.02 static, '
                   'B_owns r 0.7/0.5, C_l35rel '
                   'norm_share_A1>0.5 dominant '
                   'else semantic, D_path corr '
                   '>0.9999 FATAL, D_repro ==0/'
                   '<1e-6 FATAL, D_readout AUC '
                   '>0.6, D_spans 300/300 full, '
                   'L*_rel = min L>=20 E<=-0.3|E_'
                   'final| (abs gates -0.05/-0.025 '
                   'reported); deterministic '
                   'seeds only (crc32), no MC',
        'artifacts': {
            'result': 'phase3124/omega_p122_'
                      'resid_cm_kalman_l35rel_'
                      'glm4x/result.json',
            'seal': 'phase3124/omega_p122_'
                    'resid_cm_kalman_l35rel_'
                    'glm4x/design_seal.json',
            'readout': 'phase3124/omega_p122_'
                       'resid_cm_kalman_l35rel_'
                       'glm4x/p122_readout.npz'},
        'hashes': {},
        'note': 'GPU used (glm4-9b-chat-hf '
                '40L BF16 eager, gen batch 32, '
                'forward batch 1, 13475.6s; '
                'parts A/B/C offline on frozen '
                'qwen3-4b data); 5 SMOKE '
                'iterations before full run: '
                'fixed bash-shim rm, GLM4 '
                'double-norm hidden_states trap '
                '(probe-verified, hook rewrite, '
                'D-PATH 0.847->0.99999), embed '
                'squeeze, output->input '
                'interference redesign '
                '(GLM4 does not echo the query '
                'line; n_span 0->672/672), '
                'SPANS gate SMOKE adaptation; '
                'A1 first-tok drift 17.4% is '
                'genuine model behavior',
    }
    led['measurements'].append(meas)
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model'][0]
    l14['connects'].append(
        'meas3124_omega_p122_resid_cm_kalman_'
        'l35rel_glm4x')
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

# ---------- 3. MEMO Phase 3124 ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3124:' not in memo:
    sec = u'''## Phase 3124: Ω-P122 残差共模分解 + Kalman 锚点漂移检验 + L35 释压一阶分解 + GLM4 跨模型句级范式与层追踪（T4 第7Phase）——**残差三成分定性：m-结构≈0、共模 A1 0.356/P 0.190、余项 0.65–0.81 主导；确定性重构失败（r −0.12）、Kalman q̂ 撞下界=锚点静态、own 平滑锚点 r 0.006 不足——(m,锚点,线性算子) 族整体不足，存在第三系统成分；L35 释压=范数增益×语义抑制各半（norm_share 0.50/0.47）；GLM4 跨模型复现写入链上游层定位：E_syn L*=20、E_cont L*=20 双方向（Qwen L21/L20），span 672/672、readout AUC 0.831** [[NOW]]

**性质**：T4 第 7 Phase，3123 MEMO 第 5 节预注册、门在 seal 观测前冻结（design_seal.json）。Parts A/B/C offline（3118/3122/3123 冻结数据，纯 numpy 确定性）；Part D GPU：glm4-9b-chat-hf（GlmForCausalLM 40 层 BF16 eager），13475.6s（gen 1344 prompts 批 32 ≈5599s + forwards 5408 批 1 ≈7877s）。Part D 关键工程：transformers 5.14 GLM4 hidden_states 为 append-before 模式 [emb, L1..L39 out, norm 后 final]（探针实测 hs[-1]==last_hidden_state、二次 norm 使 D-PATH r 0.847）→ 改用 forward hooks 收集 emb+L1..L40 norm 前输出（与 Qwen 3123 读出语义同构），D-PATH r 0.99999；GLM4 生成直接答 Yes/No 不回显查询行（Qwen 305/320 复述）→ 干预从输出流重设计为**输入 prompt 查询行等长替换**（pos0 不变、生成流固定重放），n_span 672/672。SMOKE 5 轮迭代后全绿。

### 1. 三大发现（重复三遍）
1. **残差三成分定性：m-结构≈0、共模次之、余项主导——(m,锚点,线性算子) 模型族整体不足**。3-way 顺序分解（12 个 step-dummy 块 → (m−MS) 块 → 余项）：m-结构份额 P 0.0033/A1 3e-6（≈0！m 自身的偏差结构在残差中可忽略）、共模份额 P 0.190/A1 0.356（A1 接近 0.2 dominant 门但 cm_min 取 P 值 → common_mode_partial）、余项 0.65–0.81。确定性无噪声模拟 m(t+1)=m+S(m−MS)+content 完全失败（r=−0.123、R2med P 0.33/A1 −0.016、auc_det 仍 0.72–1.0 振荡）；解析平台预测 Φ(gap/√(σ_P²+σ_A1²))=0.6660 再次与 3123 平台 0.6636 精确吻合（跨相位复现 3123 的扩散伪影定位）。Kalman 随机游走锚点 MLE q̂≈1.4e−4 撞网格下界（ratio 0.0001 双方向）→ **锚点在测试模型族内完全静态**；lag-1 自相关 −0.167/−0.006；RTS 平滑自身锚点重建 r=0.0062 不足。**结论：残差既非 i.i.d.（3123 已证必为共模/系统），也不能由（m-结构+静态/动态锚点+分方向线性算子）解释——存在第三系统成分（候选：轨迹内自相关结构、内容尾迹项、非线性算子），余项 0.65–0.81 是下一步的分解对象**。
2. **L35 释压≈范数增益×语义抑制各半——末层刹车含真实语义分量**。L35 一阶分解 ΔW=Pn·ΔWn[范数项]+Wn_ans·ΔPn[语义项]+leak：norm_share P 0.501/A1 0.472 → semantic_component_present（双方向接近五五开，t_sem −2.64/−1.92 双显著、leak −0.116/−0.052 小）；A1 raw dW −8.029 vs −11.716（+3.69 释压 bit 级复现 3123）。**"no-答案成形"通道候选（3123）不是纯 final-norm 几何效应——语义抑制分量真实存在**。spec_rows（L30/L32/L35 × 7 类 × 2 方向）描述性读出同场保存。
3. **GLM4 跨模型复现写入链上游层定位：E_syn L\*=20、E_cont L\*=20 双方向（相对门与绝对门一致）**。40 层 logit-lens（hook 收集 emb+40 层，path r 0.999986 vs lm_head logits、replay bit-exact 0.0、n_span 672/672 双方向、readout AUC 0.831）：E_syn 与 E_cont 首次持续越限均 L20（P abs 门 −0.05 实越 −0.124/−0.300、A1 门 −0.025 实越 −0.108/−0.281），与 Qwen 3123 的 L21/L20 同位——**相对深度 GLM4 20/40=0.50 vs Qwen 21/36=0.58，"语法门控/内容可读性涌现于写入链上游"跨模型成立**。曲线形状差异同样重要：GLM4 E_syn 无终层深化（末段 P +0.059/A1 −0.114 vs Qwen −2.256/−1.707）、E_cont 终层大幅正（+0.341/+0.086 vs Qwen −3.677/−2.231）、两方向共享 L8–11 中层深谷（E_syn −0.18~−0.34）——干预语义不同（GLM4 输入流 vs Qwen 输出流）下符号不可直接比较，**层定位可比、符号待 Qwen 输入流对照实验分离**。

### 2. 关键数值
Part A：decomp P {cm 0.19001305759179912, m 0.0033137063011249046, rem 0.806673236107076, ss_tot 69737.76}、A1 {cm 0.35574441732458434, m 3.0830390620402785e-06, rem 0.6442524996363537, ss_tot 103458.49}；det r=−0.12297439326221062、R2med P 0.32964950758136624/A1 −0.015582100480399985；analytic σ 3.2062/4.0017、gap 2.1988、φ 0.66597 vs 3123 平台 0.66356。Part B：lag1 −0.1674141723780209/−0.006200686258676985；Kalman P {q̂ 0.00014353249592527267, ratio 0.0001, σ_a² 1.4353}、A1 {q̂ 0.0001481438608183718, ratio 0.0001, σ_a² 1.4814}；own r 0.0062127328674929、R2med P 0.4035/A1 0.0169、auc_own 终 0.9261。Part C：P {W_ans −11.3737, W_oth −11.4793, norm_share 0.50095, t_norm 2.6518, t_sem −2.6417, leak −0.1157}、A1 {W_ans −8.0294, W_oth −11.7160, norm_share 0.47177, t_norm −1.7148, t_sem −1.9200, leak −0.0519}。Part D：path r 0.9999857905405052、repro 0.0、readout_auc 0.8313137755102041、n_span 672/672；first-tok P [Yes 670, No 2]、A1 [No 555, Yes 90, no 16, The 11]；E_syn_P [9] −0.3442/[20] −0.1236/[40] +0.0594；E_syn_A1 [9] −0.3181/[40] −0.1141；E_cont_P [9] −0.2808/[20] −0.3005/[26] +0.7840/[40] +0.3408；E_cont_A1 [20] −0.2807/[26] +0.8676/[40] +0.0860；L* 全 {rel 20, abs 20}。

### 3. 硬伤
① 干预语义变化：GLM4 用输入流干预（输出流无查询行对象），与 Qwen 3123 输出流干预不严格可比——层定位可比性依赖"查询行扰动→读出位置"对称结构，**符号与幅度不可直接比较**（E_cont 终层符号相反可能主要是干预位置差异）；② 生成流固定重放：s1–s3 扰动下 GLM4 的真实生成可能改变，本实验测的是"固定输出下的状态敏感性"而非反事实生成行为；③ E_syn_P 终层 eff 仅 0.059，rel 门 −0.3eff 阈值相应小——L20 判定的稳健性由 abs 门（−0.124 越 −0.05）独立支撑，但 P 方向"语法效应"幅度弱；④ GLM4 hidden_states 陷阱修复依赖 transformers 5.14.1 特定行为，跨平台复现须用 hook 方案（已固化在脚本）；⑤ L8–11 中层深谷的"扰动传播窗口"解释是候选解释非机制证明；⑥ Parts A/B 完全继承 qwen3-4b 冻结数据——第三成分的定位（自相关结构 vs 内容尾迹 vs 非线性）未做；⑦ Kalman q̂ 撞 logspace(−4) 网格下界，"静态"是网格内结论（真实 q̂ 可能更小，即更静态）；⑧ GLM4 A1 首 token 17.4% 漂移（90 Yes+16 no+11 The）——readout AUC 0.831 未对漂移样本分层，部分 AUC 来自漂移子群。

### 4. 机制拼图更新
内部响应图谱：① 残差 3-way 分解表（cm/m/rem × 方向，m-结构≈0 新事实）；② Kalman (q̂, ratio, σ_a²) 表 + lag1 + own 重建——锚点持久性在模型族内被否定；③ L35 一阶分解（norm/sem/leak × 方向）——末层刹车首次机制分解；④ GLM4 全 41 层×4 条件×672×2 lens margin 场（mlg npz）+ E_syn/E_cont 曲线 + L* 定位 + first-tok 分布。RDC 更新：① **(m, cls, anchor, 线性算子) 模型族正式不足以解释轨迹形状（三重否定：det 重构 r −0.12、Kalman q̂→0、own r 0.006）——残差模型必须引入轨迹特定项或非线性结构，余项 0.65–0.81 是明确的分解对象**；② **L35 末层刹车 = 范数增益×语义抑制各半——3123 的 A1 释压含语义分量，"no-答案成形"通道候选升级**；③ **跨模型层定位不变量候选+1：语法/内容涌现层=写入链上游（GLM4 L20/40 与 Qwen L21/L36 相对深度 0.50/0.58），加上 3110 真值全息读出、3123 锚点 gap，跨模型不变量清单至三项**；④ 干预语义（输入流 vs 输出流）本身成为实验变量——Qwen 输入流对照列入 3125。

### 5. 3125 预注册（T4 继续，观测前冻结框架）
① **残差第三成分定位（offline）**：轨迹内 AR(p)/谱自相关结构 + 内容尾迹项（尾迹 token 类条件均值）+ 跨步条件互相关——把余项 0.65–0.81 拆开并重模拟；② **GLM4 反事实生成对照（GPU）**：s1–s3 扰动 prompt 重新生成，检验固定重放结论的行为有效性；③ **Qwen 输入流对照（GPU）**：Qwen 上复刻输入流干预版本，隔离干预语义变量、判定 E_cont 符号差异来源；④ **GLM4 写入链层识别**：spec_rows 已存 L30/L32/L35（GLM4 层号），需先定位 GLM4 的写入链层（对应 Qwen L26–L35）再读出。具体门在 3125 seal 冻结。

产物：`tests/glm5/result/rdc_query_construction_20260913/phase3124/omega_p122_resid_cm_kalman_l35rel_glm4x/`（result.json、design_seal.json、run_log.txt、p122_readout.npz）；脚本 `tests/glm5/phase3124_omega_p122_resid_cm_kalman_l35rel_glm4x.py`；探针 `tests/gpt5_temp/p3124_probe_gpath.py`、`tests/gpt5_temp/p3124_span_probe.py`。
'''
    sec = sec.replace('[[NOW]]', '[' + NOWS + ']')
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    _memo_delta = len(sec)
    o.append('memo +%d chars (Phase 3124)'
             % len(sec))
else:
    _memo_delta = 0
    o.append('memo already appended')

# ---------- 4. workspace logs (x2) ----------
line_exp = ('- Phase 3124 Omega-P122 (T4 seventh '
            'phase: residual 3-way decomposition '
            '+ Kalman anchor-drift test + L35 '
            'release decomposition + GLM4 cross-'
            'model sentence paradigm, glm4-9b-'
            'chat-hf GPU + offline qwen3-4b '
            'data, 13475.6s): verdict ' + V + '. '
            '(A) Residual 3-way decomposition: '
            'm-structure ~0 (P 0.0033/A1 3e-6), '
            'common-mode P 0.190/A1 0.356, '
            'remainder 0.65-0.81 dominant; det '
            'no-noise simulation FAILS (r -0.123, '
            'R2med 0.33/-0.016) while analytic '
            'AR(1) plateau 0.6660 again == 3123 '
            'plateau 0.6636; Kalman q_hat ~1.4e-4 '
            'AT GRID LOWER BOUND (ratio 0.0001 '
            'both) -> anchor_static, lag1 -0.167/'
            '-0.006, own smoothed-anchor rebuild '
            'r 0.0062 insufficient -> (m, anchor, '
            'linear operator) family INSUFFICIENT, '
            'a THIRD systematic component dominates. '
            '(B) L35 release first-order split: '
            'norm_share 0.501 (P) / 0.472 (A1) -> '
            'semantic_component_present, brake = '
            'norm gain x semantic suppression '
            '~half-half (t_sem -2.64/-1.92), A1 '
            '+3.69 release replicates 3123. '
            '(C) GLM4 cross-model (40L, hook-'
            'collected layers after probe-'
            'verifying the transformers 5.14 '
            'append-before hidden_states trap, '
            'path r 0.99999, replay bit-exact): '
            'GLM4 answers Yes/No directly (P Yes '
            '670/672; A1 No 555/672 + drift 117) '
            'and never echoes the query line -> '
            'interference REDESIGNED to input-'
            'prompt span (equal-length, pos0 '
            'preserved, stream replayed; n_span '
            '672/672 BOTH, spans_ok); readout '
            'AUC 0.831; E_syn L*=20 and E_cont '
            'L*=20 BOTH dirs (rel+abs gates '
            'agree) -> Qwen L21/L20 write-chain-'
            'upstream emergence REPLICATES (rel '
            'depth 0.50 vs 0.58); shape differs: '
            'no final deepening (E_syn +0.059/'
            '-0.114), E_cont final POSITIVE '
            '(+0.341/+0.086), shared L8-11 '
            'mid-layer trough; signs not directly '
            'comparable across interference '
            'semantics. NEXT 3125: third-component '
            'localization (autocorr/spectral + '
            'content trail); GLM4 counterfactual '
            'regeneration; Qwen input-stream '
            'control; GLM4 write-chain layer id.\n')
line_clo = ('- Phase 3124 closeout finished: '
            'five-write chain ok (ledger n=%LGN% '
            'l14=%L14N% sha=%SHA8%, MEMO +%MEMOC% '
            'chars, dual wlog, MEMORY.md update); '
            'disk verify next. 5 SMOKE iterations '
            'before full run (bash-shim rm; GLM4 '
            'double-norm hidden_states trap probe-'
            'verified and hook-rewritten, D-PATH '
            '0.847->0.99999; embed squeeze; output-'
            'to-input interference redesign, '
            'n_span 0->672/672; SPANS gate SMOKE '
            'adaptation); full run 13475.6s gen '
            '1344 + 5408 forwards.\n')
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
        marker = ('Phase 3124 Omega-P122' if tag
                  == 'exp'
                  else 'Phase 3124 closeout')
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
if 'max=3123' in mem_old:
    NEW_3124 = (u'- 3124（T4）：残差分解 m-结构≈0、共模 '
                u'A1 0.356/P 0.190、余项 0.65–0.81 主导；'
                u'det 重构 r −0.12、Kalman q̂→0=锚点静态、'
                u'own r 0.006 → **(m,锚点,线性算子) 族不足、'
                u'存在第三系统成分**；L35 释压=范数×语义各半'
                u'（0.50/0.47）；**GLM4 跨模型：语法/内容 '
                u'L*=20/20 双方向（Qwen L21/L20）复现写入链'
                u'上游、span 672/672、readout AUC 0.83**。')
    NEW_3118_20 = (u'- 3118–3120：AUC 0.981→0.672 强振荡、'
                   u'状态补偿 0.52 vs 闭环放大 1.08–1.73；'
                   u'答案 token 仅 t=1、振荡由内容步驱动；'
                   u'重述步上推（断言侵蚀）、标点步恢复 gap；'
                   u'L30/L32 反向于 L26；Δm 线性 slope −0.68。')
    NEW_3113_17 = (u'- 3113–3117：MLP 主写 L20–28（相关峰 '
                   u'+2.48）、L32 负写=擦除相；head=相关影子、'
                   u'三层联合 additive；TOP3=L26/L33/L31；'
                   u'A1 首 token 20.2% 分叉。')
    NEW_NEXT = (u'- max=3124，下一 3125：**残差第三成分定位'
                u'（自相关/谱+内容尾迹+跨步互相关）+ GLM4 反事实'
                u'生成对照 + Qwen 输入流对照（干预语义变量隔离）'
                u'+ GLM4 写入链层识别**。')
    lines = mem_old.splitlines()
    out = []
    skip = 0
    for ln in lines:
        if skip > 0:
            skip -= 1
            continue
        if ln.startswith(u'## 机制链状态'):
            out.append(u'## 机制链状态（3124）')
        elif ln.startswith(u'- 3123'):
            out.append(ln)
            out.append(NEW_3124)
        elif ln.startswith(u'- 3120'):
            out.append(NEW_3118_20)
            skip = 2
        elif ln.startswith(u'- 3114'):
            out.append(NEW_3113_17)
            skip = 1
        elif ln.startswith(u'- max=3123'):
            out.append(NEW_NEXT)
        else:
            out.append(ln)
    mem_new = u'\n'.join(out) + u'\n'
    assert mem_new.count(
        u'## 机制链状态（3124）') == 1
    assert mem_new.count(u'- 3124（T4）') == 1
    assert mem_new.count(u'- 3123（T4）') == 1
    assert mem_new.count(u'max=3124') == 1
    assert mem_new.count(u'- 3119：') == 0
    assert mem_new.count(u'- 3118（T4）') == 0
    assert mem_new.count(u'- 3113：') == 0
    assert mem_new.count(
        u'- 3118–3120：') == 1
    assert mem_new.count(
        u'- 3113–3117：') == 1
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
