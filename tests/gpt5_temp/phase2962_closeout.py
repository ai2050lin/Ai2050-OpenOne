# -*- coding: utf-8 -*-
"""Phase 2962 closeout: seal + Ledger + MEMO + worklog + MEMORY.
Idempotent, placeholder replacement."""
import hashlib
import json
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2962',
                   'word_class_signature_matrix')
LEDGER = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\2026-09-20.md')
WMEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\MEMORY.md')
SCRIPT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
          r'\phase2962_word_class_signature_matrix.py')

res = json.load(open(os.path.join(OUT, 'result.json'),
                     encoding='utf-8'))
verdict = res['final_verdict']
created = json.load(open(os.path.join(OUT, 'execution.json'),
                         encoding='utf-8'))['created']


def sha8(path):
    return hashlib.sha256(
        open(path, 'rb').read()).hexdigest()[:8]


hashes = {
    'execution.json': sha8(os.path.join(OUT, 'execution.json')),
    'result.json': sha8(os.path.join(OUT, 'result.json')),
    'signature_matrix.npz': sha8(
        os.path.join(OUT, 'signature_matrix.npz')),
    'script': sha8(SCRIPT),
}

# ---------------- Ledger ----------------
led = json.load(open(LEDGER, encoding='utf-8'))
meas_id = 'meas2962_word_class_signature_matrix'
if not any(m['meas_id'] == meas_id
           for m in led['measurements']):
    led['measurements'].append({
        'meas_id': meas_id,
        'type': 'word_class_signature_matrix',
        'verdict': verdict,
        'source': ('phase2962/word_class_signature_matrix; 45 '
                   'single forwards (2937 pass1 protocol), 3 '
                   'classes x 15 English single-token words '
                   '(concrete/abstract/function, tokenizer '
                   'pre-check before freeze); S1 routing '
                   'ICC17 0.1163 p 4.58e-02 (boundary band, '
                   'discipline 11, not hard); S2 readout SVD '
                   'coords all-blind (maxT p 0.961, per-k min '
                   '0.65) - 2940 replicated on new list/classes; '
                   'S3 load-band concrete-vs-abstract zero diff '
                   '(obs 0.0048 p 0.976) with function-word '
                   'descriptive gap 1.23 CONFOUNDED by token id '
                   '(rho -0.605); verdict mixed per frozen map'),
    })
    l14 = [lk for lk in led['linkage']
           if lk['link_id'] == 'L14_readout_spectrum_cross_model'][0]
    if meas_id not in l14['connects']:
        l14['connects'].append(meas_id)
led.pop('ledger_sha256_8', None)
new_h = hashlib.sha256(json.dumps(
    led, sort_keys=True, ensure_ascii=False)
    .encode('utf-8')).hexdigest()[:8]
led['ledger_sha256_8'] = new_h
json.dump(led, open(LEDGER, 'w', encoding='utf-8'),
          indent=2, ensure_ascii=False)
chk = new_h

# ---------------- MEMO ----------------
memo = open(MEMO, encoding='utf-8').read()
if '## Phase 2962' not in memo:
    section = '''## Phase 2962: 词类机制签名矩阵预注册——三签名分层检验，阶段二开局 [@STAMP@]

**判决：`@VERDICT@`**（冻结映射的 else 分支如实落位）——方案 v2 阶段二启动：45 个英文单 token 词（具体名词/抽象概念/功能词 × 15，tokenizer 可达性预检先于冻结通过 15/15/15），2937 pass1 协议 verbatim 单前向 ×45，零消融；三签名 = 路由头分布（S1）× 读出 SVD 坐标（S2）× 承重带剖面（S3），全部配标签置换 null 校准（纪律 11），S2 内部族 8 用 maxT（纪律 7）。

**锚 6/6 过**：a1 Vt8 重建 3.04e-08 / a2 决定性 0.0 / a3 正交性 2.11e-15 / a4 单 token 45/45 / a5 头块-直读 rel 3.48e-16 / a6 非退化门（纪律 12）全过（sd17 min 4.0e-02）。

**三签名结果**：
1. **S1 路由头分布**：ICC17 = 0.1163，置换 p = 4.58e-02——**边界带**（0.01 < p ≤ 0.05，纪律 11 口径：不与硬显著混池，单独登记）；ICC16 = 0.1336（描述性）。类信息在路由层只有弱信号，经 null 校准后不足硬显著。
2. **S2 读出 SVD 坐标**：全盲确认——maxT p = 0.961，per-k 未校正 p 全 ≥ 0.65，最大 |med 差| 12.53 也在 null 常规域。**2940 的词属性盲在新词表（n=45）、新类目（具体/抽象/功能）上复现**——读出坐标层对词类关闭，修正版思路一的该半边预测再次成立。
3. **S3 承重带剖面**：concrete-vs-abstract 零差（obs 0.0048，p 0.976）；但 **function 组描述性偏移明显**（B 中位 −1.2899 vs −2.5201/−2.5153，差 ~1.23）——**配对混淆在案**：rho(token_id, B) = −0.605，功能词高频低 tid，该差距可能是频率驱动而非类驱动（2940 混淆检查纪律的直接体现）。此差距按纪律 9 只登记描述性，确认性检验留给新 Phase 预注册。

**结论（重复 3 次）**：**词类签名在读出坐标层确认缺席（2940 复现）、在承重带 concrete/abstract 内缺席、在路由头分布仅边界带弱信号——机制特异性不在"词类"粗粒度上，粗词类的明显差异（function 组承重带 +1.23）被频率混淆支配，须频率受控设计才能检验。**

**硬伤与勘误（run1→run2）**：run1 a5 处 o_proj 输入形状 (1,4096) 未展平致 TypeError——删旧 execution/result（纪律 3）、np.dot(x.reshape(-1)) 修复重跑，run2 权威。工程：Edit 幻影编辑再现（3 处修改 1 处未落盘，含 a5 死代码块），Python 补丁 + 磁盘复核修复；主脚本草稿残留（未定义变量 g）在运行前由 ast 语法检查 + Grep 复核拦截。

**文件+SHA256-8**：execution @HEXE@ / result @HRES@ / signature_matrix.npz @HNpz@ / script @HSCR@。产物 `tests/glm5/result/rdc_query_construction_20260913/phase2962/word_class_signature_matrix/`。Ledger 101 条 / L14 connects 69 / ledger @LEDHASH@。

**接续**：候选 2963：A（主选）**function-vs-content 承重带差距的频率受控预注册复检**——新词表（未观测词）+ 频率匹配设计（按 tid 分层配对）+ token-id 协变量偏相关，纪律 9/10/11 全适用；B S1 边界带功效复检（词表扩容 n≥60，判断路由弱信号是真效应还是噪声）；C 旋转轴功能身份（v_rot vs W_ov/u35 代数关系，一次前向）；D 跨模型卡片差距清单（glm4，为阶段三铺路）。
'''
    stamp = created.replace('T', ' ')[:16]
    reps = [('@STAMP@', stamp), ('@VERDICT@', verdict),
            ('@LEDHASH@', chk),
            ('@HEXE@', hashes['execution.json']),
            ('@HRES@', hashes['result.json']),
            ('@HNpz@', hashes['signature_matrix.npz']),
            ('@HSCR@', hashes['script'])]
    for k, v in reps:
        section = section.replace(k, v)
    memo += section
    open(MEMO, 'w', encoding='utf-8').write(memo)

# ---------------- worklog ----------------
entry = ('- Phase 2962 闭环：signature_mixed_pattern_registered。45 词 3 类签名'
         '矩阵（2937 pass1 协议×45 单前向，零消融）：S1 路由 ICC17 0.1163 '
         'p 0.0458 边界带；S2 读出坐标全盲（maxT 0.961，2940 复现）；S3 '
         'concrete/abstract 零差但 function 描述性 +1.23 被 rho(tokid,B)'
         '=-0.605 频率混淆支配。结论：粗词类机制特异性不存在于读出/承重带'
         '层，路由层仅弱信号；下一步频率受控复检。勘误：run1 a5 形状 bug '
         '(1,4096)→np.dot reshape；Edit 幻影编辑 3/1。Ledger 101 / hash '
         + chk + '。\n')
wl = ''
if os.path.exists(WLOG):
    wl = open(WLOG, encoding='utf-8').read()
if 'Phase 2962' not in wl:
    wl += entry
    open(WLOG, 'w', encoding='utf-8').write(wl)

# ---------------- MEMORY ----------------
mem = open(WMEM, encoding='utf-8').read()
mem = mem.replace(
    '当前 max=**2961**，下一个 **2962**（候选 A 词类机制签名矩阵预注册，'
    '方案 v2 阶段二启动）',
    '当前 max=**2962**，下一个 **2963**（候选 A function-vs-content 承重带'
    '频率受控复检；B S1 边界带功效复检 n≥60；C 旋转轴功能身份；D 跨模型'
    '差距清单）')
old = '→原语卡片压缩：25 卡入册，阶段一收官(2961)。'
if old in mem:
    mem = mem.replace(
        old,
        old[:-1] + '→词类签名矩阵：读出词盲复现+S1 边界带+承重带类内零差(2962)。')
add = ('- 词类分组设计规范（2962）：类分离主张三件套——(i) 语言同质控制'
       '（全英文词表消语言混淆）；(ii) token-id/频率混淆强制在案'
       '（rho(tokid,B)=-0.605 实测：功能词承重带描述性差距 +1.23 可能纯'
       '频率驱动）；(iii) 已观测数据上的组差距按纪律 9 只登记描述性，'
       '确认性检验须新 Phase + 未观测词 + 频率匹配预注册。边界带 p'
       '（0.01<p≤0.05）单独登记不与硬显著混池（纪律 11）。科学结论：'
       '读出 SVD 坐标层对词类关闭（2940 在 n=45 新类目复现），路由头分布'
       '仅边界带弱信号（ICC17 0.1163 p 0.0458），粗词类特异性须到频率'
       '受控设计里找。\n')
if '词类分组设计规范（2962）' not in mem:
    anchor_line = '- 纯文档 Phase 锚规范（2961）：'
    i = mem.find(anchor_line)
    if i >= 0:
        mem = mem[:i] + add + mem[i:]
    else:
        mem = mem.rstrip() + '\n' + add
open(WMEM, 'w', encoding='utf-8').write(mem)

print('closeout done: ledger=%s n_meas=%s' % (
    chk, len(led['measurements'])))
