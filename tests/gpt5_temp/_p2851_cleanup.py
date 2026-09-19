import os
p = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913/phase2851'
rep = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/_p2851_cleanup.txt'
lines = []
if os.path.isdir(p):
    for f in sorted(os.listdir(p)):
        fp = os.path.join(p, f)
        if os.path.isfile(fp):
            os.remove(fp)
            lines.append('removed ' + f)
    left = sorted(os.listdir(p))
else:
    lines.append('no_dir')
    left = []
lines.append('left=' + repr(left))
with open(rep, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(lines) + '\n')
print('cleanup done')
