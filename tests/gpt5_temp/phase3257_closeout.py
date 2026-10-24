# -*- coding: utf-8 -*-
"""Phase 3257 closeout: five writes (result assertions / ledger #409 / MEMO /
dual wlog / MEMORY.md). Idempotent per write; ledger append chain-checked."""
import hashlib
import json
import os
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RES = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913'
                        r'\phase3257\g5h20_final_audit\result.json')
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
WLOG_WS = os.path.join(ROOT, r'research\gpt5\docs\WORKLOG_gpt5.md')
WLOG_U = os.path.join(ROOT, r'research\deepseek\docs\WORKLOG_gpt5_mirror.md')
MEM = os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md')
TS = time.strftime('%Y-%m-%d %H:%M')
REPORT = []


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


# ---------- 1. result assertions ----------
R = json.load(open(RES, encoding='utf-8'))
assert R['verdict'] == 'final_audit_passed' and R['mode'] == 'formal'
assert R['design_sha8'] == 'c2390c7f' and R['res_sha8'] == '961a9e97'
assert all(R['gates'].values())
assert R['three_account_audit']['anchor_files']['match'] == 28
assert R['three_account_audit']['ledger']['post'] == '5601b3f5'
assert R['manifest_reverify']['bad'] == []
assert R['manifest_reverify']['ledger_snapshot_ok'] is True
SRC = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913')
assert sha8(open(os.path.join(SRC, r'phase3254\g5h17_registry_landing'
                              r'\atlas_registry_v1_34.json'), 'rb').read()) == '3deabf25'
assert sha8(open(os.path.join(SRC, r'phase3254\g5h17_registry_landing'
                              r'\gap_ledger_v1_12.json'), 'rb').read()) == 'aa8cff21'
assert sha8(open(os.path.join(SRC, r'phase3256\g5h19_closure_seal'
                              r'\atlas_registry_v1_35.json'), 'rb').read()) == 'e8335331'
assert os.path.isfile(os.path.join(SRC, r'phase3257\g5h20_final_audit'
                                       r'\apply_preview.json'))
ep = open(os.path.join(ROOT, r'tests\gpt5_temp\p3257_endpoint_out.txt'), encoding='utf-8').read()
ui = open(os.path.join(ROOT, r'tests\gpt5_temp\p3257_ui_out.txt'), encoding='utf-8').read()
assert 'ENDPOINT CHECK OK' in ep and 'FAIL' not in ep
assert 'UI CHECK: 11/11' in ui and 'console errors: none' in ui
REPORT.append('1 assertions: OK (verdict=%s res=%s; anchors 28/28; ledger #408 '
              '5601b3f5; manifest reverified; sealed v1.34/v1.12/v1.35 untouched; '
              'endpoint 15 asserts + UI 11/11 live)' % (R['verdict'], R['res_sha8']))

# ---------- 2. ledger ----------
raw = open(LED, 'rb').read()
pre_file_sha8 = sha8(raw)
led = json.loads(raw.decode('utf-8'))
PREV_POST = '5601b3f5'
already = [e for e in led['measurements'] if e.get('phase') == 3257]
if not already:
    assert len(led['measurements']) == 408 and led['ledger_sha256_8'] == PREV_POST
    entry = {
        "phase": 3257,
        "name": "g5h20_final_audit",
        "line": "G",
        "date": "2026-10-21",
        "model": "zero GPU: G-line final audit (live endpoint + UI + disk "
                 "three-account) + N-line apply rehearsal (read-only)",
        "verdict": ("final_audit_passed|HEADLINE - G-line closure package "
                    "certified end-to-end: registry v1.35 (e8335331) live-"
                    "visible with 35-generation chain, seal annotation in "
                    "client; ALL 28 provenance anchor_files sha8 re-measured "
                    "on disk and matched; gap v1.12 appendix anchors "
                    "re-verified under their per-entry conventions (file-sha "
                    "x4 + summary file-sha + res_sha8); ledger #408 chain "
                    "recompute OK; N-line handoff manifest sha8 re-verified "
                    "and apply_preview.json written with ZERO writes to any "
                    "G-line sealed file"),
        "evidence_level": ("registered (preregistered MEMO 3256 tail (a)+(b); "
                           "live endpoint check 15 asserts PASS + playwright "
                           "UI 11/11 PASS zero console errors incl version "
                           "switch 1.34<->1.35; disk audit: anchor_files "
                           "28/28, gap anchors 6/6 under explicit convention "
                           "table, ledger recompute from prev post "
                           "b411f7ef)"),
        "model_scope": "n/a (audit + rehearsal; zero GPU)",
        "prereg_id": "G5-H20 (MEMO 3256 tail)",
        "superseded_by": None,
        "detail": (
            "G-line final audit + N-line rehearsal. Arm (a): /atlas/registry "
            "live -> version 1.35 default, versions chain 1.35..1.1 (35 "
            "gens), upgrade_log n=14 with closure-seal tail; v1.34 still "
            "resolvable (n_upgrades=13); /atlas/gaps v1.12 regression OK "
            "(appendix 6, APPX-6 anchor p3250/9d0f1825 verbatim); UI: "
            "AtlasLens select 35 options, default v1.35, switch round-trip "
            "1.34->1.35 OK, GAP panel unaffected. Disk: registry v1.35 "
            "provenance.anchor_files 28/28 sha8 re-measured match; gap "
            "appendix anchors re-verified under explicit per-entry "
            "convention table (APPX-1/2 file-sha, APPX-3 summary file-sha, "
            "APPX-4/5 cross-line agate a_gate_closure_v1.json 24c60160 in "
            "deepseek atlas, APPX-6 res_sha8 9d0f1825); ledger #408 "
            "recompute from b411f7ef OK. Arm (b): manifest sha8 re-verified "
            "(0 bad; ledger snapshot 408/5601b3f5 ok) + apply_preview.json "
            "(5 read-only steps: sha8 verify -> closure facts as priors not "
            "evidence -> N-line self-append -> evidence levels non-"
            "transferable). SMOKE device fix: gap anchor resolution replaced "
            "by explicit convention table (mixed historical conventions, "
            "traceable to 3207 recheck + 3254 landing audit). Endpoint "
            "assert fix: seal annotation cites 0.7436 (4-decimal), not "
            "74.36. GPU backlog unchanged: checkpoint training-step "
            "trajectory + APPX-1..5 GPU items. freeze c2390c7f (SMOKE "
            "cc1590c9), res 961a9e97. Pre-append ledger file sha8 "
            + pre_file_sha8 + "."),
    }
    led['measurements'].append(entry)
    led['ledger_sha256_8'] = PREV_POST
    new_post = sha8(json.dumps(led, ensure_ascii=False, indent=1).encode('utf-8'))
    led['ledger_sha256_8'] = new_post
    open(LED + '.bak_3257', 'wb').write(raw)
    with open(LED, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(led, ensure_ascii=False, indent=1))
    REPORT.append('2 ledger: appended #409 post=%s (pre_file=%s)'
                  % (new_post, pre_file_sha8))
else:
    new_post = led['ledger_sha256_8']
    REPORT.append('2 ledger: 3257 entry already present, post=%s (idempotent)'
                  % new_post)
chk = json.load(open(LED, encoding='utf-8'))
assert len(chk['measurements']) == 409
chk['ledger_sha256_8'] = PREV_POST
rec = sha8(json.dumps(chk, ensure_ascii=False, indent=1).encode('utf-8'))
assert rec == new_post == json.load(open(LED, encoding='utf-8'))['ledger_sha256_8'], \
    'ledger post recompute mismatch'
REPORT.append('2 ledger self-check: n=409 post=%s recompute OK (chain prev=%s)'
              % (new_post, PREV_POST))

# ---------- 3. MEMO ----------
mb = open(MEMO, 'rb').read()
pre_memo_sha8 = sha8(mb)
if b'## Phase 3257:' not in mb:
    lines = []
    A_ = lines.append
    A_('')
    A_('## Phase 3257: G5-H20 G 线收官核验（v1.35 客户端可见性实机 + 三账磁盘终审）+ N 线施加预演（read-only，零 GPU） [%s]' % TS)
    A_('')
    A_('**执行模式**：预注册自 3256 tail（(a) 收官核验臂 + (b) N 线预演臂）；freeze **`c2390c7f`**（SMOKE `cc1590c9`）；res **`961a9e97`**；verdict **`final_audit_passed`**。封存 v1.34 `3deabf25` / v1.12 `aa8cff21` / v1.35 `e8335331` 字节不动。')
    A_('')
    A_('### 收官核验臂 (a)：客户端可见性 + 三账终审')
    A_('')
    A_('- **端点实机 15 断言全过**：/atlas/registry 默认 **v1.35**、版本链 1.35..1.1（**35 代**）、upgrade_log n=14 封章注记在尾；v1.34 可回溯（n_upgrades=13）；/atlas/gaps v1.12 回归（appendix 6、APPX-6 anchor p3250/`9d0f1825` 逐字）；三账交叉（ledger #408 tail + registry seal 现役引用 + 计数一致）。断言修正 ×1：seal 注记引用 0.7436（四位小数）而非 74.36。')
    A_('- **UI 实机 11/11 零控制台错误**：AtlasLens select 35 options（链头 v1.1 精确 value 匹配）、默认 v1.35、切换 1.34↔1.35 往返（计数 13/14 正确）、GAP 面板回归无扰。断言笔误修正 ×1（v1.1 判断恒 false）。')
    A_('- **磁盘终审**：registry v1.35 provenance.anchor_files **28/28 sha8 实测匹配**；gap v1.12 appendix 锚按**显式口径表**逐一复核 6/6（APPX-1/2 result.json 文件 sha、APPX-3 summary 文件 sha、APPX-4/5 跨线 agate `a_gate_closure_v1.json` 24c60160（deepseek atlas）、APPX-6 res_sha8 `9d0f1825`——历史口径混合可溯源至 3207 recheck + 3254 落册审计）；ledger #408 自 b411f7ef recompute OK。')
    A_('')
    A_('### N 线施加预演臂 (b)：read-only 五步预演')
    A_('')
    A_('- manifest sha8 独立重验（0 bad；ledger 快照 408/`5601b3f5` ok）→ **`apply_preview.json`**：① N 线自读 manifest（无 G 线路径耦合）② sha8 自验 ③ closure facts 作共享先验**不作 N 线证据** ④ N 线自 append 条目、永不改写 G 线封存 ⑤ 证据级不可跨线转移（E 级从 N 线自己的 E0 起算）。**全程零写入 G 线文件**。')
    A_('')
    A_('**限界**：零 GPU；封存文件不动；GPU 挂账（checkpoint 训练步轨迹 + APPX-1..5 GPU 项）维持。')
    A_('')
    A_('**完整性**：ledger #409（pre 文件 sha8 %s）；产物 result.json + apply_preview.json + design.json + smoke 子目录；独立磁盘复核见 p3257_disk_verify。' % pre_memo_sha8[:8])
    body = (chr(10).join(lines) + chr(10)).encode('utf-8')
    assert mb.endswith(b'\n')
    open(MEMO, 'ab').write(body)
    REPORT.append('3 memo: appended Phase 3257 section (%d bytes)' % len(body))
else:
    REPORT.append('3 memo: Phase 3257 already present (idempotent)')
mb2 = open(MEMO, 'rb').read()
assert mb2.count(b'## Phase 3257:') == 1

# ---------- 4. dual wlog ----------
wl = ('- 3257 G 线收官核验完成（final_audit_passed）：端点实机 15 断言（v1.35 默认、版本链 35 代、v1.34 可回溯、gaps v1.12 回归、三账交叉）+ UI 实机 11/11 零控制台错误（select 35 options、1.34↔1.35 切换往返、GAP 面板回归）+ 磁盘终审（registry v1.35 anchor_files 28/28 sha8 实测匹配；gap appendix 锚按显式口径表 6/6——APPX-1/2 文件 sha、APPX-3 summary 文件 sha、APPX-4/5 跨线 agate 24c60160、APPX-6 res_sha8 9d0f1825；ledger #408 recompute）。'
      'N 线施加预演（read-only）：manifest sha8 重验 0 bad → apply_preview.json 五步（自读/sha8 自验/facts 作先验不作证据/自 append/证据级不跨线），零写入 G 线。'
      'verdict final_audit_passed；freeze c2390c7f（SMOKE cc1590c9）/ res 961a9e97，ledger #409，封存 v1.34/v1.12/v1.35 字节不动。GPU 挂账维持（checkpoint 训练步轨迹 + APPX-1..5 GPU 项）。G 线收线状态：收线包已核验可交付。\n')
for p in (WLOG_WS, WLOG_U):
    try:
        txt = open(p, encoding='utf-8').read()
    except IOError:
        txt = ''
    if '3257 G 线收官核验' in txt:
        REPORT.append('4 wlog %s: already present (idempotent)' % p)
        continue
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'a', encoding='utf-8', newline='') as f:
        f.write(wl)
    REPORT.append('4 wlog %s: appended' % p)

# ---------- 5. MEMORY.md ----------
mem = open(MEM, encoding='utf-8').read()
old_block = ('- **3257 已预注册**：(a) G 线收官核验臂（v1.35 客户端可见性 + 三账终审）；(b) N 线施加预演臂（read-only）；(c) GPU 挂账维持。')
new_block = ('- **3257 终审**：final_audit_passed——端点 15 断言 + UI 11/11（v1.35 链 35 代）+ 磁盘终审（anchors 28/28、gap 锚口径表 6/6、ledger #408 recompute）；N 线预演 apply_preview（read-only 零写入）。G 线收线包可交付。#409 `c2390c7f`/`961a9e97`。'
             '- **挂账**：checkpoint 训练步轨迹；APPX-1..5 GPU 项；N 线正式施加；水果类/K4 删除待确认。')
if old_block in mem:
    mem = mem.replace(old_block, new_block)
else:
    assert '3257 终审' in mem, 'MEMORY tail block not found'
if len(mem) > 3000:
    o = ('- **3256 封章**：3239–3255 收线链 CLOSED（G0 −1.7515/ORC +0.3290/XC_p5_b8 −0.4490 修复 74.36%）；registry v1.35 `e8335331`（upgrade_log+1 注记，revert 逐位）；N 线收线包 manifest。#408 `5061e1e8`/`01393c60`。')
    n = ('- **3256 封章**：3239–3255 收线链 CLOSED（修复 74.36%）；registry v1.35 `e8335331`（seal 注记，revert 逐位）；N 线收线包 manifest。#408。')
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = ('- **3255**：/atlas/gaps 增 appendix + GapLedgerPanel 入 AtlasLens（实机 17/17）；MEMO 综论定谳**双轨道几何**（embedding 秩 1 轨不达 readout / 载轴轨支配崩塌）——**INFORMATION_GAP 词表侧不可补，XC_p5_b8 −0.449 即收线终点**。#407 `1d19fc3d`/`ac6bd1d9`。')
    n = ('- **3255**：/atlas/gaps 增 appendix + GapLedgerPanel 入 AtlasLens（实机 17/17）；MEMO 定谳**双轨道几何**——INFORMATION_GAP 词表侧不可补，XC_p5_b8 −0.449 即收线终点。#407。')
    if o in mem:
        mem = mem.replace(o, n)
if len(mem) > 3000:
    o = ('- **3252**：**embedding 层 top1=1.0000 ×3**（崩塌方向 embedding 即秩 1）；U 形（中层低谷→readout 回升 0.29–0.31）；近邻稳定 0.966/0.971/0.926。#404。')
    n = ('- **3252**：**embedding 层 top1=1.0000 ×3**（崩塌方向 embedding 即秩 1）；U 形轨迹；近邻稳定 ×3。#404。')
    if o in mem:
        mem = mem.replace(o, n)
assert len(mem) <= 3000, len(mem)
with open(MEM, 'w', encoding='utf-8', newline='') as f:
    f.write(mem)
REPORT.append('5 memory: replaced tail block, %d chars' % len(mem))

print('\n'.join(REPORT))
print('CLOSEOUT OK')
