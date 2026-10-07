# 3064 run1->run2 fixes: cache access + stale cleanup
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3064_omega_p61_ds7b_chain_'
     r'replication.py')
s = io.open(P, encoding='utf-8').read()

# 1) cleanup loop: also remove stale run_log and
#    execution.json (rerun discipline)
old1 = ("for fn in (NAME + '.npz',):\n"
        "    p = os.path.join(OUT, fn)\n"
        "    if os.path.exists(p):\n"
        "        os.remove(p)")
new1 = ("for fn in (NAME + '.npz', 'run_log.txt',\n"
        "           'execution.json', 'result.json',\n"
        "           'seal.json'):\n"
        "    p = os.path.join(OUT, fn)\n"
        "    if os.path.exists(p):\n"
        "        os.remove(p)")
cnt = s.count(old1)
assert cnt == 1, 'anchor1 count=%d' % cnt
s = s.replace(old1, new1)

# 2) DynamicCache compatibility in b5
old2 = ("pk_r = out_r.past_key_values[L_TGT][0][0,\n"
        "    rows5[0]].double().cpu().numpy()")
new2 = ("_kc = getattr(out_r.past_key_values,\n"
        "              'key_cache', None)\n"
        "_kl = _kc[L_TGT] if _kc is not None \\\n"
        "    else out_r.past_key_values[L_TGT][0]\n"
        "pk_r = _kl[0, :, rows5[0]].double() \\\n"
        "    .cpu().numpy()")
cnt = s.count(old2)
assert cnt == 1, 'anchor2 count=%d' % cnt
s = s.replace(old2, new2)

old3 = ("pk_p = out_p.past_key_values[L_TGT][0][0,\n"
        "    off5 + rows5[0]].double().cpu().numpy()")
new3 = ("_kc = getattr(out_p.past_key_values,\n"
        "              'key_cache', None)\n"
        "_kl = _kc[L_TGT] if _kc is not None \\\n"
        "    else out_p.past_key_values[L_TGT][0]\n"
        "pk_p = _kl[0, :, off5 + rows5[0]] \\\n"
        "    .double().cpu().numpy()")
cnt = s.count(old3)
assert cnt == 1, 'anchor3 count=%d' % cnt
s = s.replace(old3, new3)

io.open(P, 'w', encoding='utf-8').write(s)
print('PATCH_OK')
