import os, re, hashlib, shutil, json, time

root = r'D:\AI2050\Ai2050-OpenOne'
T = os.path.join(root, 'tests', 'gpt5_temp')
G = os.path.join(root, 'gpt5_temp')
DST_SCRIPT = os.path.join(root, 'tests', 'deepseek')
DST_TEMP = os.path.join(root, 'tests', 'deepseek_temp')

pats_T = [
    r'^e1_embed_probe_20260930\.py$', r'^e1_embed_probe_report\.txt$',
    r'^e2_context_conditioned_probe_20261001\.py$', r'^e2_report\.txt$',
    r'^e3_embed_feature_audit\.py$', r'^e3_report\.txt$',
    r'^e3b_embed_followup\.py$', r'^e3b_report\.txt$',
    r'^N1_design_seal\.json$', r'^N2h1_design_seal\.json$', r'^N3_design_seal\.json$',
    r'^n1[\w\-]*\.(py|txt)$', r'^n2[\w\-]*\.(py|txt)$', r'^n3[\w\-]*\.(py|txt)$',
    r'^probe_n2[\w\-]*\.(py|txt)$', r'^probe_e4\.(py|txt)$',
]
pats_G = [
    r'^do_append_(e2|n1|n2|n2h1|n3)\.py$',
    r'^do_wlog_(e2|n2h1)\.py$',
    r'^do_memory_(n1|n1b|n2|n3)\.py$',
    r'^enc_check\.(py|txt)$', r'^hash_e2\.(py|txt)$',
    r'^memo_append_(design|e1|e2|n1|n2|n2h1|n3)\.md$',
    r'^memory_n1_section\.md$',
    r'^probe_(ds|e4|p1|proc|models2|memo_integrity|memo_loss|model|3149|assets|assets2|assets3|memo)\.(py|txt)$',
    r'^verify\.txt$', r'^verify[0-9]\.txt$',
    r'^verify_append_(e2|n1|n2|n2h1|n3)\.txt$',
    r'^verify_e1\.txt$', r'^verify_memo_append\.txt$', r'^verify_plan\.txt$',
    r'^verify_memory_(n1|n3)\.txt$',
    r'^verify_wlog(_e1|_e2|_n2h1)?\.txt$',
    r'^wlog_append_20260930b\.md$', r'^wlog_append_e1\.md$', r'^wlog_append_e2\.md$',
    r'^wlog_n2h1\.md$', r'^wlog_n3\.md$',
    r'^check_tie\.(py|txt)$',
    r'^memo_stat\.txt$', r'^memo_stats\.txt$', r'^memo_stats2\.txt$', r'^memo_index\.txt$',
]

def collect(d, pats):
    hits = []
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        if os.path.isfile(p) and any(re.search(pat, f) for pat in pats):
            hits.append(f)
    return hits

sel = []
for f in collect(T, pats_T):
    sel.append((os.path.join(T, f), f))
for f in collect(G, pats_G):
    sel.append((os.path.join(G, f), f))

os.makedirs(DST_SCRIPT, exist_ok=True)
os.makedirs(DST_TEMP, exist_ok=True)

def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()

manifest = []
errors = []
for src, name in sel:
    dst_dir = DST_SCRIPT if name.lower().endswith('.py') else DST_TEMP
    dst = os.path.join(dst_dir, name)
    if os.path.exists(dst):
        errors.append('TARGET EXISTS %s' % dst)
        continue
    h0 = sha(src)
    n0 = os.path.getsize(src)
    try:
        shutil.move(src, dst)
    except Exception as e:
        errors.append('MOVE FAIL %s : %r' % (name, e))
        continue
    ok_exists = os.path.exists(dst)
    ok_gone = not os.path.exists(src)
    h1 = sha(dst) if ok_exists else 'NA'
    manifest.append({
        'name': name,
        'from': src.replace(root, ''),
        'to': dst.replace(root, ''),
        'bytes': n0,
        'sha256': h0,
        'hash_ok': (h0 == h1),
        'src_gone': ok_gone,
    })

# move review dir
rv_src = os.path.join(T, 'memo_review_20261001')
rv_dst = os.path.join(DST_TEMP, 'memo_review_20261001')
if os.path.isdir(rv_src):
    mfiles = []
    for dp, dn, fn in os.walk(rv_src):
        for f in fn:
            mfiles.append(os.path.join(dp, f))
    hs = {p: sha(p) for p in mfiles}
    shutil.move(rv_src, rv_dst)
    ok = all(os.path.exists(os.path.join(rv_dst, os.path.basename(p))) and
             sha(os.path.join(rv_dst, os.path.basename(p))) == hs[p] for p in mfiles)
    manifest.append({'name': 'memo_review_20261001/', 'from': rv_src.replace(root, ''),
                     'to': rv_dst.replace(root, ''), 'bytes': sum(os.path.getsize(os.path.join(rv_dst, os.path.basename(p))) for p in mfiles),
                     'sha256': 'DIR(%d files)' % len(mfiles), 'hash_ok': ok, 'src_gone': not os.path.exists(rv_src)})

out = []
out.append('# Phase 1-7 (deepseek/N-line) 产物迁移清单  %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
out.append('# 目标约定：脚本 -> tests/deepseek/ ；报告/文本/校验 -> tests/deepseek_temp/')
out.append('# 共 %d 项 ；hash_ok 全部 = %s' % (len(manifest), all(m['hash_ok'] for m in manifest)))
out.append('')
out.append('%-56s %-28s %8s %10s %s' % ('FILE', 'TO', 'BYTES', 'HASH_OK', 'SRC_GONE'))
for m in manifest:
    out.append('%-56s %-28s %8d %10s %s' % (m['name'], m['to'].replace('\\tests\\', '').rsplit('\\', 1)[0], m['bytes'], m['hash_ok'], m['src_gone']))
out.append('')
out.append('errors: %d' % len(errors))
for e in errors:
    out.append('  ' + e)

open(os.path.join(G, 'move_manifest_phase1_7.txt'), 'w', encoding='utf-8').write('\n'.join(out))
json.dump({'moved': manifest, 'errors': errors, 'at': time.strftime('%Y-%m-%dT%H:%M:%S')},
          open(os.path.join(G, 'move_manifest_phase1_7.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('moved', len(manifest), 'errors', len(errors), 'all_hash_ok', all(m['hash_ok'] for m in manifest))
