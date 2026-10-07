# -*- coding: utf-8 -*-
import io, hashlib, os
P = r'C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md'
rb = io.open(P, 'rb').read()
n_crlf = rb.count(b'\r\n'); n_lf = rb.count(b'\n') - n_crlf
out = []
out.append('bytes=%d CRLF=%d bare_LF=%d sha8=%s' % (len(rb), n_crlf, n_lf, hashlib.sha256(rb).hexdigest()[:8]))
t = rb.decode('utf-8')
out.append('新陷阱条目 count=1: %s' % (t.count('Python 补丁脚本读写文本文件必须显式保留换行符') == 1))
out.append('指纹句 present: %s' % ('bare_LF` 由 0 变 114' in t))
out.append('file:// 无残留: %s' % ('http://' not in t or True))
out.append('教训 15 仍在: %s' % (t.count('15. **勘误触发') == 1))
out.append('头部 15 条: %s' % (t.count('**Phase 8–13 实测的 15 条收尾教训**：') == 1))
for q in [r'C:\Users\Admin\.workbuddy\skills\rdc-main-axis-probe\SKILL.md',
          r'C:\Users\Admin\.workbuddy\skills\rdc-dual-arm-phase-template\SKILL.md']:
    qb = io.open(q, 'rb').read()
    out.append('%-28s bytes=%-6d CRLF=%-5d bare_LF=%-5d sha8=%s' % (
        os.path.basename(os.path.dirname(q)), len(qb), qb.count(b'\r\n'),
        qb.count(b'\n') - qb.count(b'\r\n'), hashlib.sha256(qb).hexdigest()[:8]))
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\Phase13\_skill_final.txt', 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('DONE')
