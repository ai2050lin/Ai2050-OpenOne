# -*- coding: utf-8 -*-
# 3162 closeout 五写: ledger append + MEMO(3162 节 + 3161 决策) + daily + workspace MEMORY + self-check
# 幂等: ledger by phase; MEMO by 节标题; daily by marker; MEMORY by 3162 段头
import io, json, os, shutil, hashlib, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
MDIR_MEM = os.path.join(ROOT, '.workbuddy', 'memory')
PDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                    'phase3162', 'g5a1_atlas_foundation')
NOW = time.strftime('%Y-%m-%d %H:%M')
OUTP = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3162_closeout_out.txt')
out = []

def sha8_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()[:8]

# ---- 产物 sha 现场复核（与 result_summary.json 断言一致） ----
sres = json.load(io.open(os.path.join(PDIR, 'summary', 'result_summary.json'), encoding='utf-8'))
pairs = [('execution.json', sres['execution_sha8']), ('result_audit.json', sres['audit_sha8']),
         ('atlas_census.json', sres['census_sha8']), ('atlas_registry.json', sres['registry_sha8']),
         ('atlas_v0.html', sres['html_sha8'])]
for fn, expect in pairs:
    got = sha8_file(os.path.join(PDIR, fn))
    assert got == expect, ('artifact sha mismatch', fn, got, expect)
out.append('artifact shas re-verified: ' + ' '.join('%s=%s' % (f, e) for f, e in pairs))

# ---- 快照 ----
snap_dir = os.path.join(ROOT, 'tests', 'gpt5_temp', '_snap_3162')
os.makedirs(snap_dir, exist_ok=True)
for src in (LEDGER, MEMO):
    shutil.copy2(src, os.path.join(snap_dir, os.path.basename(src) + '.bak'))
out.append('snapshot: ledger+memo copied to _snap_3162')

# ============ 1. Ledger（幂等 by phase, 弹性并发断言） ============
led = json.loads(io.open(LEDGER, encoding='utf-8').read())
ms = led['measurements']
n0 = len(ms)
if any(m.get('phase') == 3162 for m in ms):
    out.append('ledger: 3162 already present, skip')
else:
    entry = {
        'phase': 3162, 'name': 'g5a1_atlas_foundation', 'line': 'G',
        'date': '2026-10-09', 'model': 'zero-gpu (evidence audit over qwen3-4b+qwen3-14b+glm4-9b artifacts)',
        'verdict': 'g5a1_atlas_registry_built|disk_verified_16/16|fail_0|eread_ok_True|infra_ok_True|g4_True',
        'detail': ('G5-A1 atlas foundation v0 (zero GPU). External review adopted: 4-tier evidence '
                   'levels (E0 candidate / E1 repeatable / E2 predictive / E3 causal-scoped), 8-field '
                   'rule schema (rule_id/scope/metric_definition/replication/heldout_prediction/'
                   'counterexamples/causal_evidence/evidence_level), principles (readable!=steerable; '
                   'cross-model fingerprint!=mechanism; failures are first-class atlas entries), and '
                   'recall of 3 overclaims (F9 nine-laws-all-cross-model; F10 five-model-device -> '
                   'mainline 3 models, pending_replication=[ds7b,glm4,gemma4]; F11 5min/phase budget '
                   '-> real 10.6s-1081s). Audit: 16 nodes x 122 checks all pass disk-verified, incl. '
                   'N11 E_read RESOLVED (q03_result.json 0.331615/0.398601/0.389835, bitwise drift '
                   '0.0 x3, dual-registered in metric_dict v4 global_kpis 0652c008); calibration: N04 '
                   'RoPE single-model (9/9 = position conditions not models), N05 massive 77x '
                   'quantified 4b-only (d1=0/731/2319 presence 3/3 via 3157/3160), N13 C_steer=0 '
                   'qwen3-4b-only scope, N06 T_C 0.561-0.564 = shared+content-specific coexist, N08 '
                   'quotient formalization rejected 0/3 (kept as failure entry). Failure ledger F1-'
                   'F11 recorded. Census 48 rows node x model; gaps ranked: #1 phase3161 head '
                   'attribution (mechanism chain last untested link, prereg unchanged), #2 cross-'
                   'model same-protocol re-measure of C_steer/RoPE/massive, #3 cross-family '
                   'connection (knowledge/reasoning/grammar), #4 out-of-spectrum falsification only. '
                   'Artifacts: execution d40ac23e (design 56b74d8c), audit 2568b3a6, census c6098cbb, '
                   'registry ff46fd86, html 56771169. G4 counting typo (15 vs 16 incl N00; census 45 '
                   'vs 48) fixed post first run, node list and design_sha unchanged.'),
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    ms.append(entry)
    blob = json.dumps(led, ensure_ascii=False, indent=1, sort_keys=False).encode('utf-8')
    led['ledger_sha256_8'] = hashlib.sha256(blob).hexdigest()[:8]
    json.dump(led, io.open(LEDGER, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    n1 = len(json.loads(io.open(LEDGER, encoding='utf-8').read())['measurements'])
    assert n1 in (n0 + 1, n0 + 2), 'concurrent ledger write anomaly'
    out.append('ledger: appended 3162 (n=%d, was %d) chain_sha8=%s' % (n1, n0, led['ledger_sha256_8']))

# ============ 2. MEMO（幂等 by 节标题; 本节全部 CRLF） ============
raw = open(MEMO, 'rb').read()
txt = raw.decode('utf-8')          # BOM 字符保留在串首
sec_marker = '## Phase 3162: 图谱基座 v0'
if sec_marker in txt:
    out.append('memo: 3162 section already present, skip')
else:
    lines = [
        '## Phase 3162: 图谱基座 v0——证据分级审计与 L1 描述性普查（G5-A1）[__NOW__]',
        '',
        '**主判决：`g5a1_atlas_registry_built|disk_verified_16/16|fail_0|eread_ok_True|infra_ok_True|g4_True`**'
        '（零 GPU；design_sha 56b74d8c；16 节点 × 122 项数值检查全部通过；独立产物 sha 见文末）。',
        '',
        '### 外部审查采纳与三项校准（本 Phase 动机）',
        '1. 「九条规律全部跨模型复现」→「若干指定指标在三个已测模型中呈现跨模型一致性」：逐条重述并分级（注册表 N01–N15），massive 77× 塌缩标为 4b 单模型量化、RoPE 9/9 标为位置条件数、E_read 由「待核实」转为已核实。',
        '2. 「结构层稳定/数值层不稳定」二分法 → 「某些归一化结构指标具有跨模型一致性；其适用范围、实现与失效条件逐项确定」：结构不是天然稳定类——KSTAR 指纹不稳（0.244–0.908）即反例（F5）。',
        '3. 「大规模图谱测试已就绪」→「可启动受控的描述性图谱普查，但先冻结指标口径、证据等级与验收规则」：本 Phase 即该冻结（八字段 schema + 证据四层制 E0/E1/E2/E3 + 三原则 + 失败账本入图谱）。',
        '',
        '### 审计结果（16 节点全部 disk_verified）',
        '- **N11 E_read「待核实」解除**：`tests/deepseek/result/q03_result.json` 三模型 0.3316153089205424 / 0.39860084652900696 / 0.389835258324941 与 `metric_dict v4` global_kpis.E_read 双处逐位一致；drift=0.0×3（carrier npz 锚定）；池化 0.37335（deadline_dual_track k1_recompute）。',
        '- 校准要点：N04 RoPE=单模型 qwen3-4b（g1_rope=False，严格内部门未过 1.375e-2>1e-3）；N05 massive 77× 仅 4b 量化（d1=0/731/2319 三模型存在性已由 3157/3160 支持）；**N13 C_steer=0.0000 确认为 qwen3-4b 单模型范围**（441 cells×10 配置，Wilson 上界 1.01%，rand=0，identity 逐位恒等）；N06 T_C 0.561–0.564=「共享分量+内容特异并存」；N08 子空间商结构 0/3 被拒留档；N15 k3-only 泛化（rev3151b 纠错链在盘）。',
        '- 基建 N00：ledger n=312 chain 9417b14f；metric_dict v4 content 0652c008；队列 sealed={Q01,Q02,Q03,Q08,Q09,Q12,Q05,Q06}、device_built={Q04}。',
        '',
        '### 失败与限界账本（F1–F11，first-class 图谱条目）',
        'F1 商结构被拒（3158，quotient absent×3）；F2 C_steer 零结果=读得出≠控得住（Q06，单模型范围）；F3 谱外崩塌（水果类=worst_class_b4，图谱只测谱内）；F4 K1 死线 3/3 否决（池化 0.3734=门 7.5×）且 K2/K3 从未测量；F5 KSTAR 指纹不稳（0.244–0.908，指标所处计算位置决定可复现性）；F6 RoPE 严格门未过；F7 3156 rank-1 轴不可从 npz 复现（轴向量必须随产物持久化）；F8 bf16 batch kernel 路径效应（相对差至 2.2e-2，锚前向 batch=1）；F9「九条全部跨模型复现」表述撤回；F10「五模型装置定型」表述撤回（主线 3 模型，ledger 记 pending_replication=[ds7b,glm4,gemma4]）；F11「约 5min/Phase」GPU 预算口径撤回（实测 10.6s–1081s）。',
        '',
        '### L1 描述性普查（覆盖矩阵 48 行 = 16 节点 × 3 模型）',
        '- 全验证：N01/N02/N03/N06/N07/N08/N09/N10/N11/N12/N14/N15（跨模型或其载体模型）；单模型：N04、N13（qwen3-4b）；partial：N05（14b/9b）。',
        '- **图谱缺口排序**：① 3161 头归因（机制链 3159→3160→3161 唯一未测环节，已预注册）→ 排下一执行；② C_steer / RoPE / massive 的跨模型同口径复测（L2）；③ 跨族连接（知识/推理/语法，G5-A2 草案）；④ 谱外迁移只做证伪实验。',
        '- 图谱定位声明：本图谱是「有证据等级的研究地图」，不是已还原的 LLM 计算原理；图谱普查本身帮助寻找机制（先发现→后确认→再因果干预）。',
        '',
        '### 机械性修正备注',
        'G4 门首跑计数笔误（规律节点 15 vs 节点总数 16 含 N00；census 45 vs 48 行），修正后重跑——节点清单与 design_sha（56b74d8c）未变。',
        '',
        '### Phase 3161 决策（按图谱缺口，非机械推进）',
        '外部审查建议「按图谱不确定性决定后续实验」：注册表显示机制链缺口=头归因，且预注册在案（3159 closeout 冻结，设计不变）：对 big-drop 块（L_mid）逐头置零（hook 在 self_attn 输出按 (b,T,head·dh:(head+1)·dh) 切片，batch 行=头配置压缩前向），4 锚×top64 方向 6×α=0.1，top-4 头集中度三分门 ≥0.5→localized_heads / <0.2→distributed_heads / 之间→weakly_localized；跨模型头层位相对深度对比+消耗曲线指纹（对齐口径）。GPU 预算 ~10min/模型。3161 执行顺位调至 3162 之后（审计先行=外部审查第 5 节采纳），编号不变。',
        '',
        '### 锚',
        'execution **d40ac23e**（design 56b74d8c）；audit **2568b3a6**；census **c6098cbb**；registry **ff46fd86**；atlas_v0.html **56771169**。ledger n=**313**。产物目录 `phase3162\\g5a1_atlas_foundation\\`。',
    ]
    block = '\r\n'.join(lines).replace('__NOW__', NOW)
    txt = txt.rstrip('\r\n') + '\r\n\r\n' + block + '\r\n'
    open(MEMO, 'wb').write(txt.encode('utf-8'))
    out.append('memo: appended 3162 section (CRLF) + 3161 decision')

# ============ 3. Daily（幂等 by marker） ============
daily = os.path.join(MDIR_MEM, '2026-10-09.md')
marker = '3162 图谱基座 v0 闭环'
if os.path.exists(daily) and marker in io.open(daily, encoding='utf-8').read():
    out.append('daily: 3162 already present, skip')
else:
    n_ms = len(json.loads(io.open(LEDGER, encoding='utf-8').read())['measurements'])
    line = ('- 3162 图谱基座 v0 闭环（G5-A1，零 GPU）：外部审查三项校准采纳并落地为机器可读注册表'
            '（八字段 schema + 证据四层制 E0/E1/E2/E3 + 三原则）；16 节点×122 项数值检查全部 disk_verified'
            '（E_read 待核实项解除：q03_result 与 metric_dict v4 双处逐位一致 drift=0）；失败账本 F1–F11 入图谱'
            '（含三条表述撤回：九条全复现/五模型装置/5min 每相）；覆盖矩阵 48 行，缺口排序：3161 头归因 >'
            ' 跨模型同口径复测 > 跨族连接 > 谱外证伪；atlas_v0.html + atlas_registry.json 产出；'
            'ledger n=%d。3161 决策=推进（机制链唯一未测环节）。\n' % n_ms)
    with io.open(daily, 'a', encoding='utf-8') as f:
        f.write(line)
    out.append('daily: appended 3162 line')

# ============ 4. Workspace MEMORY（3160 段后追加 3162 段） ============
mp = os.path.join(MDIR_MEM, 'MEMORY.md')
mtxt = io.open(mp, encoding='utf-8').read()
if '3162 图谱基座 v0 闭环' in mtxt:
    out.append('workspace MEMORY: 3162 already present, skip')
else:
    m10 = '（逐头置零，top-4 集中度三分门）。'
    i10 = mtxt.rfind(m10)
    assert i10 >= 0, '3160 segment tail not found in workspace MEMORY'
    ins_at = i10 + len(m10)
    seg12 = ('**✅ 3162 图谱基座 v0 闭环（2026-10-09）**：外部审查三项校准采纳（九条规律→逐条证据分级；'
             '结构/数值二分→逐项定范围；图谱就绪→受控普查+口径冻结）；16 节点×122 检查全 disk_verified，'
             'E_read 待核实解除（q03 与 metric_dict v4 双处逐位 drift=0）；证据分级制度入库：E0 候选/E1 可重复/'
             'E2 可预测/E3 协议内因果 + 八字段 schema + 失败账本 F1–F11 入图谱 + 三原则（读得出≠控得住/'
             '指纹≠机制/失败必入图谱）；校准：RoPE 单模型、massive 77× 仅 4b 量化、C_steer=0 单模型范围、'
             '商结构 0/3 被拒留档；artifacts：registry ff46fd86/html 56771169/design 56b74d8c/ledger n=313。'
             '3161 决策=**推进**（机制链 3159→3160→3161 唯一未测环节，预注册不变）。')
    mtxt = mtxt[:ins_at] + seg12 + mtxt[ins_at:]
    open(mp, 'wb').write(mtxt.encode('utf-8'))
    out.append('workspace MEMORY: 3162 segment inserted after 3160 tail')

# ============ 5. Self-check ============
ok = []
led2 = json.loads(io.open(LEDGER, encoding='utf-8').read())
ok.append(('ledger_3162', any(m.get('phase') == 3162 for m in led2['measurements'])))
b = open(MEMO, 'rb').read()
t = b.decode('utf-8')
ok.append(('memo_3162', sec_marker in t))
i32 = t.find(sec_marker)
seg = t[i32:]
ok.append(('memo_3162_bare_lf_zero', seg.count('\n') == seg.count('\r\n')))
ok.append(('memo_bom', b[:3] == b'\xef\xbb\xbf'))
ok.append(('memo_3161_decision', 'Phase 3161 决策' in t))
ok.append(('daily_3162', marker in io.open(daily, encoding='utf-8').read()))
mp_txt = io.open(mp, encoding='utf-8').read()
ok.append(('memory_3162', '3162 图谱基座 v0 闭环' in mp_txt))
ok.append(('memory_3160_intact', '3160 消耗机制判别闭环' in mp_txt))
bad = [k for k, v in ok if not v]
out.append('SELF-CHECK: %s' % ('OK' if not bad else 'FAIL %s' % bad))
io.open(OUTP, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('CLOSEOUT DONE')
