# -*- coding: utf-8 -*-
"""Phase 2985 closeout: seal -> Ledger -> MEMO ->
workspace log -> MEMORY.md. Idempotent guards; no bare %
formatting (use concat)."""
import hashlib
import io
import json
import os

BASE = r'D:\AI2050\Ai2050-OpenOne'
RES = os.path.join(
    BASE, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913', 'phase2985',
    'redirect_subspace_identity')
SCRIPT = os.path.join(
    BASE, 'tests', 'glm5',
    'phase2985_redirect_subspace_identity.py')
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
          r'\.workbuddy\tmp_closeout2985_log.txt')


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
assert verdict == 'redirect_concentrated_unstable'
stamp = exec_json['created']
s_exec = sha8(exec_path)
s_res = sha8(os.path.join(RES, 'result.json'))
s_npz = sha8(os.path.join(
    RES, 'redirect_subspace_identity.npz'))
s_scr = exec_json.get('script_sha256_8')
if s_scr is None:
    s_scr = sha8(SCRIPT)
    exec_json['script_sha256_8'] = s_scr
    exec_json['script_sha256_8_note'] = (
        'registration-only append at closeout: script '
        'self-hash was omitted from the frozen doc; '
        'prereg fields untouched')
    io.open(exec_path, 'w', encoding='utf-8').write(
        json.dumps(exec_json, indent=2, ensure_ascii=False))
    s_exec = sha8(exec_path)
    out.append('script hash appended to execution.json')
out.append('hashes: exec=' + s_exec + ' res=' + s_res
           + ' npz=' + s_npz + ' script=' + s_scr)

# ---------- 1. seal ----------
seal_path = os.path.join(RES, 'seal.json')
if not os.path.exists(seal_path):
    seal = {'phase': 2985,
            'verdict': verdict,
            'sealed_at': stamp,
            'sha256_8': {'execution': s_exec,
                         'result': s_res,
                         'npz': s_npz,
                         'script': s_scr},
            'anchors_ok': res['anchor_all_ok'],
            'note': 'run4 authoritative after 3 '
                    'corrections (import / n17 source / '
                    'head-level I_h + median d_bar); '
                    'anchors 10/10, five cross-product '
                    'bit-level 0.00; T1 gate semantics '
                    'nuance registered: sign-flip null '
                    'makes null MORE concentrated for '
                    'near-collinear delta sets, obs e1 '
                    'sits at null LOWER edge, conc '
                    'triggered only at quantile '
                    'boundary - verdict substantively '
                    'driven by T1b instability'}
    io.open(seal_path, 'w', encoding='utf-8').write(
        json.dumps(seal, indent=2, ensure_ascii=False))
    out.append('seal written')
else:
    out.append('seal already present')

# ---------- 2. Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
meas = [m for m in led['measurements']
        if m.get('phase') == 2985]
if not meas:
    m = {'phase': 2985,
         'name': 'redirect_subspace_identity',
         'model': 'qwen3-4b',
         'created': stamp,
         'question': ('what is the geometric identity '
                      'of the perp redirect channel of '
                      'delta_h12: word-invariant '
                      'subspace (concentration + '
                      'stability) and is it locked to '
                      'the injection axes or '
                      'intrinsic?'),
         'design': ('2983 protocol verbatim, 370 '
                    'forwards: 74 intact + 74 words x '
                    '4 conds {00,10,01,11} dose 0.1; '
                    'word-level delta_h12 (74,128); '
                    'T1 SVD energy + sign-flip null '
                    '10000; T1b split-half + '
                    'group-relabel null; T2 axis '
                    'alignment vs empirical response '
                    'dirs r_lang/r_cls/r_sum'),
         'anchors': ('10/10 ok; a5 I_h (74x32) vs '
                     '2983 bit-level 0.00; a6 g_all '
                     '0.00 (n=2368); a7 cos_all 0.00; '
                     'a8 d_bar[h12] median vs 2983 '
                     '0.00; a9 layer I vs 2977 0.00; '
                     'a10 readout identity 5.72e-17; '
                     'a3 n17 rel 1.21e-07'),
         'tests': ('T1 e1=0.8593 e3=0.9001, null '
                   'q95=0.8593, p=1.0 (obs at null '
                   'lower edge; conc at boundary); '
                   'T1b cos_split=0.9830 vs null '
                   'q95=0.9920, p=0.916, '
                   'stable=False; T2 cos lang '
                   '0.8994 (p 0.0515) / cls 0.3590 '
                   '(p 0.1236) / sum 0.6368 (p '
                   '0.0550), aligned=False'),
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
        {'phase': 2985,
         'note': ('perp redirect channel of h12: '
                  'single dominant direction holds '
                  '86% energy but is word-unstable '
                  '(split-half cos below null) and '
                  'not locked to injection axes '
                  '(lang alignment p 0.051 boundary) '
                  '- redirect is word-dependent '
                  'direction rewriting, not a fixed '
                  'subspace'),
         'connects_to': [2983, 2981, 2982, 2980,
                         2977, 2938, 2928]})
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
    out.append('ledger already has 2985')

# ---------- 3. MEMO ----------
memo_txt = io.open(MEMO, encoding='utf-8').read()
if '## Phase 2985:' not in memo_txt:
    sec = (
        '## Phase 2985: perp 通道几何身份——集中但不稳定、非轴锁定 '
        '[' + stamp + ']\n\n'
        '**问题**：2983 证明 L17 头级交互由 h12 的正交重定向'
        '（perp）通道承载 ~85% 后，该通道的几何身份待定：是词'
        '不变的固定子空间（能量集中 + 跨词稳定），且是否锁定到'
        '注入轴（语言/词类）？\n\n'
        '**设计（冻结，370 前向）**：2983 协议 verbatim（74 '
        'intact + 74 词 x 4 条件 {00,10,01,11} 剂量 0.1，2979 '
        '单位方向 + n17 层输入剂量门）。词级非线性残差位移 '
        'delta_h12(w) = dx11-(dx10+dx01) (74,128)。T1 SVD 能量'
        '谱 + 词级符号翻转 null（10000, rng 29851）；T1b '
        'split-half 子空间固定性 + 组重标记 null（rng 29852）；'
        'T2 delta_bar 与经验响应方向 r_lang/r_cls/r_sum 的对齐'
        '（符号翻转 null, rng 29853）。\n\n'
        '**产物**：`phase2985/redirect_subspace_identity/` '
        'execution ' + s_exec + ' / result ' + s_res
        + ' / redirect_subspace_identity.npz ' + s_npz
        + ' / script ' + s_scr + '。\n\n'
        '**锚 10/10**：a5 I_h (74x32) vs 2983 逐词 bit 级 '
        '0.00；a6 g_all 0.00 (n=2368)；a7 cos_all 0.00；'
        'a8 d_bar[h12]（逐维 median）vs 2983 0.00；a9 层级 I '
        'vs 2977 逐词 0.00；a10 读出线性恒等 I_h12 == '
        'M12.delta 5.72e-17；a3 n17 rel 1.21e-07；覆盖率门 '
        'median ||delta||=0.3157。\n\n'
        '**结果**：T1 e1=0.8593、e3=0.9001——单一主方向占 86% '
        '能量；但 sign-flip null 的 q95 恰为 0.8593、p=1.0（obs '
        '位于 null 分布最低端——对近共线 delta 集，逐词符号翻转'
        '破坏词间对消反而使 null 更 rank-1，T1 门仅在分位数边界'
        '成立）。T1b split-half cos=0.9830 低于组重标记 null '
        'q95=0.9920（p=0.916）——高基线量须 null 校准（2928 '
        '教训），词间方向不稳定。T2 对齐 cos(lang)=0.8994（p '
        '0.0515 边界未过）、cos(cls)=0.3590（p 0.124）、'
        'cos(sum)=0.6368（p 0.055）——均不显著。\n\n'
        '**判决**：`redirect_concentrated_unstable`（按冻结映'
        '射：T1 conc & ~T1b）。T1 门语义 nuance 已在 seal '
        '登记；判决实质由 T1b 不稳定性驱动。\n\n'
        '**硬伤/勘误（三次 correction，均删产物重跑）**：'
        'run1 模块级 import 错（native_loader 不存在，实为 '
        'phase2662_symmetric_mapping_contract.load_native）；'
        'run2 n17 误用 o_proj 输入（4096 维）范数——2980 '
        'run1 同款教训复现，正确为 L17 层输入（2560 维）'
        'x17_cap 范数 + anchor_fail 路径容器未初始化'
        '（UnboundLocalError）；run3 I_h 误构为层级 profile '
        '差 (74,) 而非头级 C 矩阵差 (74,32)（contributions '
        '第一输出）+ d_bar 误用 mean 而 2983 为逐维 median——'
        '跨产物锚不仅要核对键名，还必须核对源量的构造口径。\n\n'
        '**教训（新）**：sign-flip null 具有方向性语义——对'
        '近共线结构它会**增强** rank-1 性（破坏词间对消），'
        'obs 落在 null 最低端；此时"能量集中"判据应改用双侧'
        '或组重标记 null，预注册前必须想清楚 null 的方向。\n\n'
        '**结论（重复三遍）**：h12 的 perp 重定向通道 = 单一'
        '主方向占 86% 能量但跨词不稳定（split-half 不超置换 '
        'null），且不锁定注入轴（lang 对齐 p 0.051 边界）——'
        '重编码不是固定子空间内的旋转，而是词相关的方向重写。'
        '与 2938（同词跨剂量几何保持）对照：跨词/跨条件时残差'
        '位移无固定子空间结构。Ω-B 工作包（2977-2985 七环）'
        '至此完整：竞争亚加法定位（剂量窗-载体-解析-身份）'
        '四层解剖闭环。\n\n'
        '**接续**：下一候选 2986：A（主选）方案 v3 Ω-C 长上'
        '下文调制开题（KV 引力场对 L17 开关阈值/L34 词类签名'
        '的漂移，长度档 {2,16,64,256,1024}）；B 2978 dose law '
        '分层修正复测（离线）；C r_lang 方向的身份（与 2964 '
        'L34 载体读出方向关系）；D h12 增益曲线形状。\n')
    with io.open(MEMO, 'a', encoding='utf-8') as f:
        f.write('\n' + sec)
    out.append('memo appended')

# ---------- 4. workspace log ----------
wl = io.open(WSLOG, encoding='utf-8').read()
if 'Phase 2985' not in wl:
    entry = (
        '\n## Phase 2985（2026-09-20）\n'
        '- perp 通道几何身份：判决 '
        'redirect_concentrated_unstable（run4 权威，锚 10/10，'
        '五重跨产物 bit 级 0.00 + a10 恒等 5.72e-17）；e1 0.859 '
        '但 split-half 0.983 不超 null 0.992、lang 对齐 p '
        '0.051 边界——重编码是词相关方向重写，非固定子空间。\n'
        '- 三次 correction：import 模块名、n17 层输入口径'
        '（2980 教训复现）、I_h 头级 C 矩阵差 + d_bar median '
        '口径。\n'
        '- 新教训：sign-flip null 对近共线集反向（obs 落 null '
        '最低端），集中性判据须组重标记/双侧 null。\n'
        '- Ledger 124 / L14 92。产物 '
        'phase2985/redirect_subspace_identity/。\n')
    with io.open(WSLOG, 'a', encoding='utf-8') as f:
        f.write(entry)
    out.append('wslog appended')

# ---------- 5. MEMORY.md ----------
mem = io.open(MEMO_MEM, encoding='utf-8').read()
old_chain = ('2984 h12 消融归宿=功能严格局部化'
             '（主效应/再分配/带响应全无迁移，'
             '交互专用载体定案）。核心：')
new_chain = ('2984 h12 消融归宿=功能严格局部化；'
             '2985 perp 通道身份=单方向 86% 能量但'
             '词级不稳定+非轴锁定（重写非旋转）。'
             '核心：')
changed = False
if old_chain in mem:
    mem = mem.replace(old_chain, new_chain, 1)
    changed = True
else:
    out.append('WARN: chain anchor not found')
old_next = ('- max=2984，下一个 **2985**（A 主选 perp '
            '通道几何身份 SVD；B Ω-C 长上下文开题；C '
            'dose law 分层复测；D h12 增益曲线形状）。')
new_next = ('- max=2985，下一个 **2986**（A 主选 Ω-C '
            '长上下文调制开题：长度档漂移；B dose law '
            '分层复测；C r_lang 方向身份；D h12 增益'
            '曲线）。')
if old_next in mem:
    mem = mem.replace(old_next, new_next, 1)
    changed = True
else:
    out.append('WARN: MEMORY next-anchor not found')
if changed:
    io.open(MEMO_MEM, 'w', encoding='utf-8').write(mem)
    out.append('memory updated chars=%d' % len(mem))

io.open(OUTLOG, 'w', encoding='utf-8').write(
    '\n'.join(out) + '\n')
print('closeout done')
