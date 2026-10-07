import os, re, hashlib, json

root = r'D:\AI2050\Ai2050-OpenOne'
T = os.path.join(root, 'tests', 'gpt5_temp')
G = os.path.join(root, 'gpt5_temp')

# ---- explicit whitelist: deepseek line (N1-N3 / E1-E3 / R1) ----
pats_T = [
    r'^e1_embed_probe_20260930\.py$', r'^e1_embed_probe_report\.txt$',
    r'^e2_context_conditioned_probe_20261001\.py$', r'^e2_report\.txt$',
    r'^e3_embed_feature_audit\.py$', r'^e3_report\.txt$',
    r'^e3b_embed_followup\.py$', r'^e3b_report\.txt$',
    r'^N1_design_seal\.json$', r'^N2h1_design_seal\.json$', r'^N3_design_seal\.json$',
    r'^n1[a-z0-9_]*\.py$', r'^n1[a-z0-9_]*\.txt$',
    r'^n2[a-z0-9_]*\.py$', r'^n2[a-z0-9_]*\.txt$',
    r'^n3[a-z0-9_]*\.py$', r'^n3[a-z0-9_]*\.txt$',
    r'^probe_n2[a-z0-9_]*\.(py|txt)$',
    r'^probe_e4\.(py|txt)$',
]
pats_G = [
    r'^do_append_(e2|n1|n2|n2h1|n3)\.py$',
    r'^do_wlog_(e2|n2h1)\.py$',
    r'^do_memory_(n1|n1b|n2|n3)\.py$',
    r'^enc_check\.(py|txt)$',
    r'^hash_e2\.(py|txt)$',
    r'^memo_append_(design|e1|e2|n1|n2|n2h1|n3)\.md$',
    r'^memory_n1_section\.md$',
    r'^probe_(ds|e4|p1|proc|models2|memo_integrity|memo_loss|model|3149|assets|assets2|assets3|memo)\.(py|txt)$',
    r'^verify(_append_(e2|n1|n2|n2h1|n3)|_wlog(_e1|_e2|_n2h1)?|_memory_(n1|n3)|_memo_append|_plan|_e1)?\.txt$',
    r'^verify[0-9]?\.txt$',
    r'^wlog(_append_20260930b|_append_e1|_append_e2|_n2h1|_n3)\.md$',
    r'^check_tie\.(py|txt)$',
    r'^memo_stat\.txt$', r'^memo_stats\.txt$', r'^memo_stats2\.txt$',
    r'^memo_index\.txt$',
]

def collect(d, pats):
    hits = []
    if not os.path.isdir(d):
        return hits
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        if not os.path.isfile(p):
            continue
        for pat in pats:
            if re.search(pat, f):
                hits.append(f); break
    return hits

sel_T = collect(T, pats_T)
sel_G = collect(G, pats_G)

out = []
out.append('=== tests/gpt5_temp selected %d ===' % len(sel_T))
for f in sel_T:
    out.append('   %-56s %8d' % (f, os.path.getsize(os.path.join(T, f))))
out.append('=== root gpt5_temp selected %d ===' % len(sel_G))
for f in sel_G:
    out.append('   %-56s %8d' % (f, os.path.getsize(os.path.join(G, f))))

# what is in T that is NOT selected but looks N-line (safety net)
leftover = []
for f in sorted(os.listdir(T)):
    p = os.path.join(T, f)
    if os.path.isfile(p) and not re.match(r'^[_a-z0-9]', f):
        continue
    if os.path.isfile(p) and re.search(r'^(e[123]|n[123]|N[123]|probe_n)', f) and f not in sel_T:
        leftover.append(f)
out.append('=== leftover candidates in tests/gpt5_temp (NOT selected) %d ===' % len(leftover))
for f in leftover:
    out.append('   %s' % f)

# review dir
rv = os.path.join(T, 'memo_review_20261001')
out.append('=== review dir ===')
if os.path.isdir(rv):
    for f in sorted(os.listdir(rv)):
        out.append('   memo_review_20261001/%s  %d' % (f, os.path.getsize(os.path.join(rv, f))))

# git tracking check
out.append('=== git ===')
out.append('  .git exists %s' % os.path.isdir(os.path.join(root, '.git')))
out.append('  .gitignore exists %s' % os.path.exists(os.path.join(root, '.gitignore')))
if os.path.exists(os.path.join(root, '.gitignore')):
    gi = open(os.path.join(root, '.gitignore'), encoding='utf-8', errors='replace').read()
    out.append('  gitignore has gpt5_temp: %s ; deepseek: %s' % ('gpt5_temp' in gi, 'deepseek' in gi))

open(os.path.join(G, 'planned_move.txt'), 'w', encoding='utf-8').write('\n'.join(out))
print('sel_T', len(sel_T), 'sel_G', len(sel_G), 'leftover', len(leftover))
