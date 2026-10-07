# -*- coding: utf-8 -*-
import io
P = r'C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md'
b = io.open(P, 'rb').read()
out = []
out.append('bytes=%d  count_LF=%d  count_CRLF=%d  ends_with_CRLF=%s' % (
    len(b), b.count(b'\n'), b.count(b'\r\n'), b.endswith(b'\r\n')))
a = b.find('- Edit/Write 报成功后必须 Grep/Read 复核真实磁盘。'.encode('utf-8'))
out.append('ANCHOR byte offset=%d' % a)
seg = b[a - 5: a + 100]
out.append('ANCHOR 邻域 repr: %r' % seg)
out.append('---- 新条目邻域 ----')
c = b.find('Python 补丁脚本读写文本文件必须显式保留换行符'.encode('utf-8'))
out.append('NEW offset=%d' % c)
out.append('NEW 后 160B repr: %r' % b[c + 300: c + 460])
out.append('---- 尾 200B ----')
out.append(repr(b[-200:]))
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\Phase13\_skill_tail_probe.txt', 'w',
        encoding='utf-8').write('\n'.join(out) + '\n')
print('DONE')
