# -*- coding: utf-8 -*-
import io, hashlib, os
PS = [r'C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md',
      r'C:\Users\Admin\.workbuddy\skills\rdc-main-axis-probe\SKILL.md',
      r'C:\Users\Admin\.workbuddy\skills\rdc-dual-arm-phase-template\SKILL.md']
out = []
for p in PS:
    b = io.open(p, 'rb').read()
    name = os.path.basename(os.path.dirname(p))
    out.append('%-28s bytes=%-7d CRLF=%-5d bare_LF=%-5d sha8=%s' % (
        name, len(b), b.count(b'\r\n'), b.count(b'\n') - b.count(b'\r\n'),
        hashlib.sha256(b).hexdigest()[:8]))
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\Phase13\_eol_probe.txt', 'w',
        encoding='utf-8').write('\n'.join(out) + '\n')
print('DONE')
