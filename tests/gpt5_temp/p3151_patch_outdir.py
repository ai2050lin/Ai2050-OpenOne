import io, os, shutil

fp = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\phase3151_g1p1_combo_additive_vs_interaction.py'
s = io.open(fp, encoding='utf-8').read()
o = ("OUT = os.path.join(RDIR, 'phase3151', NAME)\n"
     "os.makedirs(OUT, exist_ok=True)")
w = ("OUT = os.path.join(RDIR, 'phase3151', NAME)\n"
     "if SMOKE:\n"
     "    OUT = os.path.join(OUT, 'smoke')\n"
     "os.makedirs(OUT, exist_ok=True)")
assert s.count(o) == 1, s.count(o)
s = s.replace(o, w)
io.open(fp, 'w', encoding='utf-8', newline='\n').write(s)
print('OUTDIR_SPLIT_OK')

d = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3151'
     r'\g1p1_combo_additive_vs_interaction')
sd = os.path.join(d, 'smoke')
os.makedirs(sd, exist_ok=True)
for f in ['execution.json', 'result.json', 'run_log.txt',
          'collect_smoke.npz']:
    src = os.path.join(d, f)
    if os.path.exists(src):
        shutil.move(src, os.path.join(sd, f))
        print('moved', f)
