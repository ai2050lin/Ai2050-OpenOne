# -*- coding: utf-8 -*-
"""按 Phase 分目录重组 deepseek 线产物（2026-10-01 登记约定变更 v2）。

约定：每个 Phase 的产物 -> tests/deepseek/Phase{N}/ 与 tests/deepseek_temp/Phase{N}/
非 Phase 轮次（复核/迁移/环境探针）-> _review/ 与 _infra/
记录类文件（research/deepseek/docs/AGI_DEEPSEEK_MEMO.md）不迁移。
"""
import os, io, json, hashlib, sys, time

ROOT = r'D:\AI2050\Ai2050-OpenOne\tests'
DS = os.path.join(ROOT, 'deepseek')
DT = os.path.join(ROOT, 'deepseek_temp')

# ---------------- mapping: relpath -> target subfolder ----------------
SCRIPT_MAP = {}

def put(d, name, folder):
    key = 'deepseek/' + name
    assert key not in SCRIPT_MAP, key
    SCRIPT_MAP[key] = folder

def putt(name, folder):
    key = 'deepseek_temp/' + name
    assert key not in SCRIPT_MAP, key
    SCRIPT_MAP[key] = folder

# --- Phase 1: 设计草案（RDC 破解路线裁决与测试方案 v1）---
for n in ['probe_ds.py', 'probe_p1.py', 'probe_proc.py', 'probe_models2.py', 'enc_check.py']:
    put(DS, n, 'Phase1')
for n in ['memo_append_design.md', 'wlog_append_20260930b.md', 'verify_memo_append.txt',
          'verify_plan.txt', 'verify_wlog.txt', 'enc_check.txt',
          'probe_p1.txt', 'probe_ds.txt', 'probe_proc.txt', 'probe_models2.txt',
          'probe_model.txt', 'probe_assets.txt', 'probe_assets2.txt', 'probe_assets3.txt',
          'probe_3149.txt', 'probe_memo.txt']:
    putt(n, 'Phase1')

# --- Phase 2: E1 词嵌入锚点裁决 ---
put(DS, 'e1_embed_probe_20260930.py', 'Phase2')
for n in ['e1_embed_probe_report.txt', 'memo_append_e1.md', 'wlog_append_e1.md',
          'verify_e1.txt', 'verify_wlog_e1.txt']:
    putt(n, 'Phase2')

# --- Phase 3: 逻辑裁决 + E2 条件化层扫描 ---
for n in ['e2_context_conditioned_probe_20261001.py', 'do_append_e2.py', 'do_wlog_e2.py',
          'hash_e2.py']:
    put(DS, n, 'Phase3')
for n in ['e2_report.txt', 'memo_append_e2.md', 'wlog_append_e2.md',
          'verify_append_e2.txt', 'verify_wlog_e2.txt', 'hash_e2.txt']:
    putt(n, 'Phase3')

# --- Phase 4: N1 主轴三段（含 E3/E3b 词嵌入特征审计、绑定检查）---
for n in ['check_tie.py', 'n1_main_axis_scan.py', 'n1_v2_main_axis_scan.py',
          'n1b_ontology_readout.py', 'n1c_ontology_cloze.py',
          'e3_embed_feature_audit.py', 'e3b_embed_followup.py',
          'do_append_n1.py', 'do_memory_n1.py', 'do_memory_n1b.py']:
    put(DS, n, 'Phase4')
for n in ['N1_design_seal.json', 'check_tie.txt', 'e3_report.txt', 'e3b_report.txt',
          'n1_report_qwen3-4b.txt',
          'n1b_report_glm4-9b-chat-hf.txt', 'n1b_report_qwen2-7b.txt',
          'n1b_report_qwen2.5-3b-instruct.txt', 'n1b_report_qwen3-1.7b.txt',
          'n1b_report_qwen3-4b.txt',
          'n1c_report_qwen2.5-3b-instruct.txt', 'n1c_report_qwen3-4b.txt',
          'n1v2_report_gemma-3-4b-it.txt', 'n1v2_report_glm4-9b-chat-hf.txt',
          'n1v2_report_qwen2-7b.txt', 'n1v2_report_qwen2.5-3b-instruct.txt',
          'n1v2_report_qwen3-1.7b.txt', 'n1v2_report_qwen3-4b.txt',
          'memo_append_n1.md', 'memory_n1_section.md',
          'verify_append_n1.txt', 'verify_memory_n1.txt']:
    putt(n, 'Phase4')

# --- Phase 5: N2 is-a 重建源定位 ---
for n in ['n2_reconstruction_source.py', 'n2b_robustness.py', 'n2c_slot_commitment.py',
          'n2d_attn_write.py', 'n2e_template_robustness.py', 'n2f_topk_diag.py',
          'n2g_critical_layers.py', 'probe_n2.py', 'do_append_n2.py', 'do_memory_n2.py']:
    put(DS, n, 'Phase5')
for n in ['n2_report_qwen3-4b.txt',
          'n2b_report_qwen2.5-3b-instruct.txt', 'n2b_report_qwen3-4b.txt',
          'n2c_report_qwen2.5-3b-instruct.txt', 'n2c_report_qwen3-4b.txt', 'n2c_part2.txt',
          'n2d_report_glm4-9b-chat-hf.txt', 'n2d_report_qwen2.5-3b-instruct.txt',
          'n2d_report_qwen3-4b.txt',
          'n2e_report_qwen2.5-3b-instruct.txt', 'n2e_report_qwen3-4b.txt',
          'n2f_report_qwen3-4b.txt',
          'n2g_report_qwen2.5-3b-instruct.txt', 'n2g_report_qwen3-4b.txt',
          'probe_n2.txt', 'memo_append_n2.md', 'verify_append_n2.txt']:
    putt(n, 'Phase5')

# --- Phase 6: N2-h1 五维类别子空间 ---
for n in ['n2h1_permutation_ablation.py', 'n2h1b_subspace_align.py',
          'n2h1c_cross_material.py', 'probe_n2h1.py', 'do_append_n2h1.py',
          'do_wlog_n2h1.py']:
    put(DS, n, 'Phase6')
for n in ['N2h1_design_seal.json', 'n2h1_report_qwen3-4b.txt',
          'n2h1b_report_glm4-9b-chat-hf.txt', 'n2h1b_report_qwen2.5-3b-instruct.txt',
          'n2h1c_report_glm4-9b-chat-hf.txt', 'n2h1c_report_qwen3-4b.txt',
          'memo_append_n2h1.md', 'wlog_n2h1.md', 'probe_n2h1.txt',
          'verify_append_n2h1.txt', 'verify_wlog_n2h1.txt']:
    putt(n, 'Phase6')

# --- Phase 7: N3 通用性 + 外部方案裁决 ---
for n in ['n3_subspace_generality.py', 'do_append_n3.py', 'do_memory_n3.py']:
    put(DS, n, 'Phase7')
for n in ['N3_design_seal.json', 'n3_report_glm4-9b-chat-hf.txt',
          'n3_report_qwen2.5-3b-instruct.txt', 'n3_report_qwen3-4b.txt',
          'memo_append_n3.md', 'wlog_n3.md',
          'verify_append_n3.txt', 'verify_memory_n3.txt']:
    putt(n, 'Phase7')

# --- 非 Phase：复核轮 R1/R2（含 memo 完整性调查报告目录）---
for n in ['do_append_r2.py', 'do_wlog_r2.py', 'verify_final_r2.py']:
    put(DS, n, '_review')
for n in ['memo_append_r2.md', 'verify_append_r2.txt', 'verify_final_r2.txt',
          'verify_wlog_r2.txt',
          'memo_review_20261001/REVIEW_REPORT.md',
          'memo_review_20261001/append_review_entry.py',
          'memo_review_20261001/artifact_verify_report.txt',
          'memo_review_20261001/extract_structure.py',
          'memo_review_20261001/structure_report.txt',
          'memo_review_20261001/verify_artifacts.py',
          'memo_review_20261001/verify_round2.py',
          'memo_review_20261001/verify_round2.txt']:
    putt(n, '_review')

# --- 非 Phase：基础设施（环境探针 / 迁移 / 清理 / 扫描 / 一轮台账）---
for n in ['cleanup_exec_20260930.py', 'do_move_phase1_7.py', 'do_move_r2.py',
          'do_move_supplement.py', 'dup_verify_2746.py', 'inv_products.py',
          'plan_move.py', 'probe_e4.py', 'probe_memo_integrity.py', 'probe_memo_loss.py',
          'scan_bigfiles.py', 'scan_junctions.py', 'scan_tests_20260930.py',
          'scan_tests2_20260930.py', 'vis_dup_check.py']:
    put(DS, n, '_infra')
for n in ['bigfiles_cleanup_verdict_20260930.md', 'bigfiles_scan.txt',
          'cleanup_exec_20260930.txt', 'cleanup_verify.txt', 'dup_verify_2746.txt',
          'inv_dirs.txt', 'inv_products.txt', 'junction_scan.txt', 'leftover_check.txt',
          'memo_index.txt', 'memo_stat.txt', 'memo_stats.txt', 'memo_stats2.txt',
          'move_manifest_phase1_7.json', 'move_manifest_phase1_7.txt',
          'move_manifest_phase1_7_supplement.txt', 'planned_move.txt',
          'probe_e4.txt', 'probe_memo_integrity.txt', 'probe_memo_loss.txt',
          'tests_cleanup_report_20260930.md', 'tests_scan_report.txt',
          'tests_scan2_report.txt', 'verify.txt', 'verify2.txt', 'verify3.txt',
          'vis_dup_check.txt', 'visdata_hits.txt', 'visprov.txt']:
    putt(n, '_infra')
# 本轮（登记约定变更 v2）自身的台账与证据
for n in ['_inventory_report.json', '_classify_draft.json', '_classify_grouped.txt']:
    putt(n, '_infra')

# 本轮临时草稿（用完即删，不落盘留档）
SCRATCH = ['deepseek_temp/_headings.txt', 'deepseek_temp/_kw_scan.txt',
           'deepseek_temp/_r2_probe.txt', 'deepseek_temp/_amb.txt']

# ---------------- execute ----------------
def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()

def walkfiles(base):
    out = []
    for rt, ds, fs in os.walk(base):
        for f in fs:
            fp = os.path.join(rt, f)
            out.append(os.path.relpath(fp, ROOT).replace('\\', '/'))
    return out

for d in [os.path.join(DS, '_infra'), os.path.join(DT, '_infra')]:
    if not os.path.isdir(d):
        os.makedirs(d)

report = []
report.append('reorg run at %s' % time.strftime('%Y-%m-%d %H:%M:%S'))

present = set(walkfiles(DS)) | set(walkfiles(DT))
# 已在目标子目录内的文件不参与
already = set(k for k in present if '/Phase' in '/' + k.split('/', 1)[1] or
              k.startswith('deepseek/_infra/') or k.startswith('deepseek/_review/') or
              k.startswith('deepseek_temp/_infra/') or k.startswith('deepseek_temp/_review/'))
todo = [k for k in sorted(present) if k not in already and k not in SCRATCH]

unmapped = [k for k in todo if k not in SCRIPT_MAP]
if unmapped:
    report.append('!! UNMAPPED FILES (%d):' % len(unmapped))
    for k in unmapped:
        report.append('   ' + k)
extra = [k for k in SCRIPT_MAP if k not in present and not os.path.exists(
    os.path.join(ROOT, k.split('/', 1)[0], SCRIPT_MAP[k], k.split('/', 1)[1].replace('/', os.sep)))]
plan = {k: SCRIPT_MAP[k] for k in todo}
report.append('planned files %d ; present %d ; todo %d ; unmapped %d ; missing %d'
              % (len(SCRIPT_MAP), len(present), len(todo), len(unmapped), len(extra)))
if unmapped:
    io.open(os.path.join(DT, '_infra', 'do_reorg_ABORT.txt'), 'w', encoding='utf-8').write('\n'.join(report))
    print('ABORT: unmapped files')
    sys.exit(1)

before = {}
for k in plan:
    p = os.path.join(ROOT, k.replace('/', os.sep))
    before[k] = (os.path.getsize(p), sha(p)) if os.path.isfile(p) else (None, None)

moved = []
errs = []
for k, folder in sorted(plan.items()):
    src = os.path.join(ROOT, k.replace('/', os.sep))
    head, name = k.split('/', 1)
    dst_dir = os.path.join(ROOT, head, folder)
    dst = os.path.join(dst_dir, name.replace('/', os.sep))
    try:
        d = os.path.dirname(dst)
        if not os.path.isdir(d):
            os.makedirs(d)
        if os.path.exists(dst):
            errs.append('DEST EXISTS %s' % k)
            continue
        os.replace(src, dst)
        moved.append(k)
    except Exception as e:
        errs.append('%s -> %s : %r' % (k, dst, e))

after = {}
for k in moved:
    head, name = k.split('/', 1)
    folder = SCRIPT_MAP[k]
    dst = os.path.join(ROOT, head, folder, name.replace('/', os.sep))
    if os.path.isfile(dst):
        after[k] = (os.path.getsize(dst), sha(dst))
    else:
        after[k] = (None, None)

mismatch = [k for k in moved if before[k] != after[k]]
deleted = []
for k in SCRATCH:
    p = os.path.join(ROOT, k.replace('/', os.sep))
    if os.path.isfile(p):
        os.remove(p)
        deleted.append(k)

# 源目录根部残留检查（应只剩子目录）
residual = {}
for base, tag in [(DS, 'deepseek'), (DT, 'deepseek_temp')]:
    rf = [f for f in sorted(os.listdir(base)) if os.path.isfile(os.path.join(base, f))]
    rd = [d for d in sorted(os.listdir(base)) if os.path.isdir(os.path.join(base, d))]
    residual[tag] = {'files': rf, 'dirs': rd}

report.append('')
report.append('moved %d ; deletions %d ; errors %d ; sha_mismatch %d'
              % (len(moved), len(deleted), len(errs), len(mismatch)))
for e in errs:
    report.append('  ERR ' + e)
for k in mismatch:
    report.append('  SHA MISMATCH %s' % k)
report.append('deleted scratch: %s' % ', '.join(deleted))
report.append('')
report.append('--- target layout (full tree) ---')
tree = {}
for base, tag in [(DS, 'deepseek'), (DT, 'deepseek_temp')]:
    for rt, ds, fs in os.walk(base):
        rel = os.path.relpath(rt, base).replace('\\', '/')
        if rel == '.':
            continue
        tree[tag + '/' + rel.split('/')[0]] = tree.get(tag + '/' + rel.split('/')[0], 0) + len(fs)
for key in sorted(tree):
    report.append('  %-28s %3d' % (key, tree[key]))
report.append('')
report.append('--- source-root residual ---')
for tag, v in residual.items():
    report.append('  %s root files=%d %s' % (tag, len(v['files']), v['files'][:10]))
    report.append('  %s root dirs =%s' % (tag, v['dirs']))

io.open(os.path.join(DT, '_infra', 'reorg_verify.txt'), 'w', encoding='utf-8').write('\n'.join(report))
mpath = os.path.join(DT, '_infra', 'reorg_manifest.json')
old = {}
if os.path.isfile(mpath):
    try:
        old = json.load(io.open(mpath, encoding='utf-8'))
    except Exception:
        old = {}
manifest = {'run_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'convention': 'Phase 产物分目录 v2',
            'moved': dict(old.get('moved', {}), **{k: plan[k] for k in moved}),
            'sha_before': dict(old.get('sha_before', {}), **{k: before[k][1] for k in moved}),
            'sha_after': dict(old.get('sha_after', {}), **{k: after[k][1] for k in moved}),
            'runs': old.get('runs', []) + [time.strftime('%Y-%m-%d %H:%M:%S')],
            'errors': old.get('errors', []) + errs, 'sha_mismatch': mismatch,
            'deleted': deleted, 'residual': residual, 'tree': tree,
            'total_moved': len(old.get('moved', {})) + len([k for k in moved if k not in old.get('moved', {})])}
io.open(mpath, 'w', encoding='utf-8').write(json.dumps(manifest, ensure_ascii=False, indent=1))
vpath = os.path.join(DT, '_infra', 'reorg_verify.txt')
prev = io.open(vpath, encoding='utf-8').read() if os.path.isfile(vpath) else ''
io.open(vpath, 'w', encoding='utf-8').write(prev + '\n\n' + '=' * 60 + '\n' + '\n'.join(report))
print('\n'.join(report))
