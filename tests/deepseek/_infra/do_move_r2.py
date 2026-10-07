import os, hashlib, shutil

root = r'D:\AI2050\Ai2050-OpenOne'
G = os.path.join(root, 'gpt5_temp')
DS = os.path.join(root, 'tests', 'deepseek')
DT = os.path.join(root, 'tests', 'deepseek_temp')

print('--- current root gpt5_temp ---')
for f in sorted(os.listdir(G)):
    p = os.path.join(G, f)
    if os.path.isfile(p):
        print('   %-52s %8d' % (f, os.path.getsize(p)))
    else:
        print('   %-52s <DIR %d>' % (f, len(os.listdir(p))))

# R2 migration-round products (this conversation's deepseek-line work)
scripts = ['plan_move.py', 'do_move_phase1_7.py', 'do_move_supplement.py',
           'inv_products.py', 'scan_bigfiles.py', 'scan_junctions.py',
           'scan_tests_20260930.py', 'scan_tests2_20260930.py', 'cleanup_exec_20260930.py',
           'dup_verify_2746.py', 'vis_dup_check.py']
texts = ['move_manifest_phase1_7.txt', 'move_manifest_phase1_7.json',
         'move_manifest_phase1_7_supplement.txt', 'planned_move.txt',
         'inv_products.txt', 'inv_dirs.txt', 'leftover_check.txt',
         'bigfiles_scan.txt', 'bigfiles_cleanup_verdict_20260930.md',
         'cleanup_exec_20260930.txt', 'cleanup_verify.txt', 'junction_scan.txt',
         'tests_cleanup_report_20260930.md', 'tests_scan_report.txt', 'tests_scan2_report.txt',
         'dup_verify_2746.txt', 'vis_dup_check.txt', 'visdata_hits.txt', 'visprov.txt']

def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()

moved = []
err = []
for name, dst_dir in [(n, DS) for n in scripts] + [(n, DT) for n in texts]:
    src = os.path.join(G, name)
    if not os.path.isfile(src):
        continue
    dst = os.path.join(dst_dir, name)
    if os.path.exists(dst):
        err.append('EXISTS ' + name); continue
    h0 = sha(src)
    shutil.move(src, dst)
    moved.append((name, dst_dir.replace(root, ''), sha(dst) == h0))
    print('moved %-48s -> %s  hash_ok=%s' % (name, dst_dir.replace(root, ''), sha(dst) == h0))

print('--- remaining root gpt5_temp ---')
rest = sorted(os.listdir(G))
print(rest if rest else 'EMPTY')
print('moved', len(moved), 'errors', err)
