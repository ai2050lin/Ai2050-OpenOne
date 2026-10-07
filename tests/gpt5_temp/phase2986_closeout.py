# -*- coding: utf-8 -*-
"""Phase 2986 closeout: seal -> Ledger -> MEMO ->
workspace log -> MEMORY.md. Idempotent guards; no bare %
formatting."""
import hashlib
import io
import json
import os

BASE = r'D:\AI2050\Ai2050-OpenOne'
RES = os.path.join(
    BASE, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913', 'phase2986',
    'length_context_drift')
SCRIPT = os.path.join(
    BASE, 'tests', 'glm5',
    'phase2986_length_context_drift.py')
LEDGER = os.path.join(BASE, 'research', 'gpt5', 'atlas',
                      'atlas_ledger.json')
MEMO = os.path.join(BASE, 'research', 'gpt5', 'docs',
                    'AGI_GPT5_MEMO.md')
WSLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
         r'\.workbuddy\memory\2026-09-20.md')
MEMO_MEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
            r'\.workbuddy\memory\MEMORY.md')
LINK_ID = 'L14_readout_spectrum_cross_model'
OUTLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
          r'\.workbuddy\tmp_closeout2986_log.txt')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


out = []
res = json.load(io.open(os.path.join(RES, 'result.json'),
                        encoding='utf-8'))
exec_path = os.path.join(RES, 'execution.json')
exec_json = json.load(io.open(exec_path, encoding='utf-8'))
verdict = res['final_verdict']
assert verdict == 'drift_present_nonmonotone'
stamp = exec_json['created']
s_exec = sha8(exec_path)
s_res = sha8(os.path.join(RES, 'result.json'))
s_npz = sha8(os.path.join(
    RES, 'length_context_drift.npz'))
s_scr = exec_json.get('script_sha256_8')
if s_scr is None:
    s_scr = sha8(SCRIPT)
    exec_json['script_sha256_8'] = s_scr
    exec_json['script_sha256_8_note'] = (
        'registration-only append at closeout')
    io.open(exec_path, 'w', encoding='utf-8').write(
        json.dumps(exec_json, indent=2, ensure_ascii=False))
    s_exec = sha8(exec_path)
    out.append('script hash appended to execution.json')
out.append('hashes: exec=' + s_exec + ' res=' + s_res
           + ' npz=' + s_npz + ' script=' + s_scr)

# ---------- 1. seal ----------
seal_path = os.path.join(RES, 'seal.json')
if not os.path.exists(seal_path):
    seal = {'phase': 2986,
            'verdict': verdict,
            'sealed_at': stamp,
            'sha256_8': {'execution': s_exec,
                         'result': s_res,
                         'npz': s_npz,
                         'script': s_scr},
            'anchors_ok': res['anchor_all_ok'],
            'note': 'run4 authoritative after 3 '
                    'corrections (2977 path / phantom '
                    'edit .median / a7 ill-posed anchor '
                    'removed + npz preinit gap); '
                    'anchors 7/7 with a2/a3/a5/a6/a8 '
                    'bit-level 0.00; a7 removed: 2945 '
                    's_c uses sep_med<100 rule over '
                    '57-word set with xdir vector - '
                    'incomparable construction'}
    io.open(seal_path, 'w', encoding='utf-8').write(
        json.dumps(seal, indent=2, ensure_ascii=False))
    out.append('seal written')
else:
    out.append('seal already present')

# ---------- 2. Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
meas = [m for m in led['measurements']
        if m.get('phase') == 2986]
if not meas:
    m = {'phase': 2986,
         'name': 'length_context_drift',
         'model': 'qwen3-4b',
         'created': stamp,
         'question': ('Omega-C opening: does context '
                      'length restructure the L17/L34 '
                      'readout manifold - existence, '
                      'monotonicity, saturation of the '
                      'drift; fate of the L34 word-class '
                      'signature and s_c threshold under '
                      'context'),
         'design': ('length bins {2,16,64,256,1024}; '
                    'filler = tiled tokens of a fixed '
                    'neutral sentence, target tail '
                    '[func_tid, word_tid]; L=2 bin = '
                    '2979 protocol verbatim; per-'
                    'condition independent single-sample '
                    'forwards (2934); base sweep 74 '
                    'words x 5 lengths + lang-axis dose '
                    'grid s in {0.25..2.0} x 74 x 5 '
                    '(~2600 forwards); readouts: prof34 '
                    'drift (primary), B band diff '
                    '(2973), s_c 0.5xmax-crossing rule, '
                    'A11 amp = sep(1.0)/sep(0), axis '
                    'rebuild cos/norm, sig34 F/C '
                    'permutation'),
         'anchors': ('7/7 ok (a7 removed by '
                     'correction, ill-posed); a2/a3/a5/'
                     'a6/a8 bit-level 0.00 (determinism, '
                     'norms+coss vs 2973, d_lang_u/d_cls_u '
                     'vs 2979, n17 vs 2979, B_rec vs '
                     '2973); a1 3.04e-08; a4 74/74'),
         'tests': ('T1 D34(1024)=3.046 vs gate '
                   '0.0969 (5% of spread 1.938) '
                   'ok=True; T2 rho=0.0 p_exact=0.525 '
                   'ok=False -> peak at L=16 '
                   '(D34=[0,6.344,3.739,3.366,3.046]); '
                   'T3 saturated=True (moot); T4 '
                   'dB(1024)=1.655 vs gate 0.0436 '
                   'ok=True; T5 s_c_drift, s_c=[1.404,'
                   '0.358,0.0,0.525,0.611], amp=[5.24,'
                   '2.05,1.75,2.72,2.95]; T6 sig34='
                   '[2.343(p.033),-0.300(.678),-0.133'
                   '(.768),-0.159(.767),-0.261(.689)]'),
         'verdict': verdict,
         'sha256_8': {'execution': s_exec,
                      'result': s_res,
                      'npz': s_npz,
                      'script': s_scr}}
    led['measurements'].append(m)
    lk = None
    for l in led['linkage']:
        if l['link_id'] == LINK_ID:
            lk = l
    lk['connects'].append(
        {'phase': 2986,
         'note': ('length-context drift is present but '
                  'NON-monotone with peak at L=16: '
                  'context EXISTENCE (2->16) does the '
                  'bulk of the restructuring, not '
                  'length dose; L34 F/C word-class '
                  'signature exists only in the len-2 '
                  'protocol (p=.033) and vanishes under '
                  'any context (p~.7) - scope caveat '
                  'for the whole len-2 chain 2936-2985; '
                  'cls axis rotates to cos 0.23, lang '
                  'to 0.62, both saturated by L=16; '
                  's_c drops 1.40 -> 0.36-0.61 (switch '
                  'fires more easily under context)'),
         'connects_to': [2962, 2963, 2964, 2945, 2953,
                         2973, 2979]})
    if 'ledger_sha256_8' in led:
        led.pop('ledger_sha256_8')
    newh = hashlib.sha256(json.dumps(
        led, sort_keys=True,
        ensure_ascii=False).encode(
        'utf-8')).hexdigest()[:8]
    led['ledger_sha256_8'] = newh
    io.open(LEDGER, 'w', encoding='utf-8').write(
        json.dumps(led, indent=2, ensure_ascii=False))
    out.append('ledger: n=%d L14=%d newhash=%s'
               % (len(led['measurements']),
                  len(lk['connects']), newh))
else:
    out.append('ledger already has 2986')

# ---------- 3. MEMO ----------
memo_txt = io.open(MEMO, encoding='utf-8').read()
if '## Phase 2986:' not in memo_txt:
    sec = (
        '## Phase 2986: Ω-C 长上下文开题——漂移非单调、上下文存在性主导、'
        '词类签名 context 脆弱 [' + stamp + ']\n\n'
        '**问题**（方案 v3 Ω-C）：全部前链协议 pos=1、上下文≤2 token。'
        '真实篇章上下文长度档 {2,16,64,256,1024} 下，读出流形是否漂移？'
        '漂移的存在性/单调性/饱和性（三检验分开预注册）；s_c 开关阈值、'
        'A11 增益、B 带差分、L34 词类签名的命运。\n\n'
        '**设计（冻结，~2600 前向）**：L=2 档 = 2979 协议 verbatim'
        '（无填充），其余档前置固定中性句平铺截断至 L-2 token，目标尾巴'
        '[the, word]；逐条件独立单样本前向（2934 铁律）。base sweep '
        '74 词 x 5 档 + 语言轴剂量网格 s∈{0.25..2.0}（2953 粗化）x 74 '
        'x 5 档。主读出 D34(L)=median_i|prof34_i(L)-prof34_i(2)|'
        '（L34 位点）；三件套：s_c（0.5xmax 交叉规则）、A11 增益 '
        'amp=sep(1.0)/sep(0)、B 带差分（2973 口径）；轴漂移 cos/范数比、'
        'sig34 F/C 置换（secondary）。\n\n'
        '**产物**：`phase2986/length_context_drift/` execution '
        + s_exec + ' / result ' + s_res
        + ' / length_context_drift.npz ' + s_npz
        + ' / script ' + s_scr + '。\n\n'
        '**锚 7/7**（a7 经 correction 删除）：a2 确定性、a3 norms/coss '
        'vs 2973、a5 轴方向 vs 2979、a6 n17 vs 2979、a8 B_rec vs 2973 '
        '全部 bit 级 0.00——L=2 档与旧协议逐位同一；a1 3.04e-08；'
        'a4 单 token 74/74。\n\n'
        '**结果**：T1 存在性 D34(1024)=3.046，门 0.0969（=词间散布 '
        '1.938 的 5%）——漂移 31 倍于门，存在。T2 单调性 rho=0.000、'
        '精确置换 p=0.525——不单调，D34=[0, 6.344, 3.739, 3.366, '
        '3.046]，**峰在 L=16**。T4 B 漂移 1.655（门 0.044）成立，'
        'B 中位 -1.24→-3.86 后回稳 -3.4。T5 s_c=[1.404, 0.358, 0.0, '
        '0.525, 0.611]（漂移 0.79），amp=[5.24, 2.05, 1.75, 2.72, '
        '2.95]——上下文使开关更易触发、增益近乎减半。轴漂移：cls 轴 '
        'cos 1.0→0.26 后稳 0.23（范数比降至 0.46），lang 轴 cos 0.62 '
        '（范数比 1.47）——均 L=16 饱和。T6（secondary）：sig34 F/C '
        '对比在 L=2 为 2.343（p=0.033），任何上下文下坍缩至 ~-0.3 '
        '（p≈0.7）。\n\n'
        '**判决**：`drift_present_nonmonotone`（冻结映射：T1 pass & '
        'T2 fail；peak_length=16 已登记）。\n\n'
        '**结论（重复三遍）**：读出流形漂移真实存在但**非长度剂量驱动**'
        '——L2→16 一步完成主要重构（D34 峰 6.34 @L16），其后部分回退；'
        '更关键的是 L34 读出的 F/C 词类签名是 **len-2 协议专属现象**，'
        '任何上下文存在即消失——2936-2985 全链（全部 len-2 协议）的'
        '适用域警示。cls 轴旋转至 cos 0.23、lang 轴 0.62，L16 饱和；'
        's_c 从 1.40 降至 0.36-0.61，amp 减半——上下文使 L17 开关'
        '更易触发。\n\n'
        '**硬伤/勘误（三次 correction，均删产物重跑）**：run1 '
        'SRC_2977 路径子目录名错（two_axis_fusion_injection）；run2 '
        'Edit 幻影——.median(axis=1) 行报告已删但磁盘残留（本机已知'
        '缺陷再现，改 Python 补丁+Grep 复核）；run3 a7 锚 ill-posed：'
        '2945 的 s_c 用 sep_med<100 规则 + 57 词集 + xdir 向量，与本 '
        'Phase（74 词集、0.5xmax 交叉、相对剂量 lang 轴）构造不可比'
        '——跨产物锚必须核对构造口径（2985 run3 教训的锚定版）；'
        '另修复 anchor_fail 路径 sig34/p6 预初始化缺口。\n\n'
        '**接续**（插队规则：主线新硬伤优先）：下一候选 2987：'
        'A（主选·插队）context 存在性最小对照 + 关键卡适用域审计'
        '（len∈{2,3,16} 最小阶梯 + 填充物内容对照；复测 2962 词类'
        '签名 / 2963 类效应 / 2964 载体在 L16 下的存活）；'
        'B Ω-C 深化（填充物语义对照、近因位置）；C dose law 分层'
        '复测；D r_lang 方向身份。\n')
    with io.open(MEMO, 'a', encoding='utf-8') as f:
        f.write('\n' + sec)
    out.append('memo appended')

# ---------- 4. workspace log ----------
wl = io.open(WSLOG, encoding='utf-8').read()
if 'Phase 2986' not in wl:
    entry = (
        '\n## Phase 2986（2026-09-20）\n'
        '- Ω-C 长上下文开题：判决 drift_present_nonmonotone'
        '（run4 权威，锚 7/7 含五重 bit 级 0.00）；D34 峰 L16=6.34 '
        '非单调——上下文存在性而非长度剂量主导重构。\n'
        '- 重大发现：L34 F/C 词类签名 len-2 专属（p=.033 → 任何'
        '上下文 p≈.7 消失）——2936-2985 全链适用域警示；cls 轴 '
        'cos 0.23 / lang 0.62，L16 饱和；s_c 1.40→0.36-0.61。\n'
        '- 三次 correction：2977 路径名、Edit 幻影（.median 残留，'
        'Python 补丁修复）、a7 ill-posed 锚删除（2945 构造不可比）。\n'
        '- Ledger 125 / L14 93。产物 '
        'phase2986/length_context_drift/。\n')
    with io.open(WSLOG, 'a', encoding='utf-8') as f:
        f.write(entry)
    out.append('wslog appended')

# ---------- 5. MEMORY.md ----------
mem = io.open(MEMO_MEM, encoding='utf-8').read()
old_chain = ('2985 perp 通道身份=单方向 86% 能量但'
             '词级不稳定+非轴锁定（重写非旋转）。'
             '核心：')
new_chain = ('2985 perp 通道身份=单方向 86% 能量但'
             '词级不稳定+非轴锁定（重写非旋转）；'
             '2986 Ω-C 开题：漂移峰 L16 非单调='
             '上下文存在性主导；L34 词类签名 '
             'len-2 专属（全链适用域警示）。'
             '核心：')
changed = False
if old_chain in mem:
    mem = mem.replace(old_chain, new_chain, 1)
    changed = True
else:
    out.append('WARN: chain anchor not found')
old_next = ('- max=2985，下一个 **2986**（A 主选 Ω-C '
            '长上下文调制开题：长度档漂移；B dose law '
            '分层复测；C r_lang 方向身份；D h12 增益'
            '曲线）。')
new_next = ('- max=2986，下一个 **2987**（A 主选 插队'
            '审计：context 最小对照+关键卡 L16 复测；'
            'B 填充物对照；C dose law；D r_lang）。')
if old_next in mem:
    mem = mem.replace(old_next, new_next, 1)
    changed = True
else:
    out.append('WARN: MEMORY next-anchor not found')

# compress if over limit
COMPRS = [
    ('（2970 transpose 教训）', '（2970）'),
    ('——2971/2972 两次产物行残留教训，收尾后必 '
     'Grep 复核', '（2971/2972 教训，收尾必 Grep）'),
    ('json.dumps(sort_keys,ensure_ascii=False) '
     'sha256 前 8 位', 'json.dumps(sort_keys) '
     'sha256 前 8 位'),
    ('重跑先删旧 execution.json/result.json/npz',
     '重跑先删旧产物'),
]
for o, n in COMPRS:
    if len(mem) <= 3000:
        break
    if o in mem:
        mem = mem.replace(o, n, 1)
        out.append('compressed: %s...' % o[:20])
if changed:
    io.open(MEMO_MEM, 'w', encoding='utf-8').write(mem)
out.append('memory chars=%d ok3000=%s'
           % (len(mem), len(mem) <= 3000))

io.open(OUTLOG, 'w', encoding='utf-8').write(
    '\n'.join(out) + '\n')
print('closeout done')
