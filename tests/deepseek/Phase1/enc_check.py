import os

root = r'D:\AI2050\Ai2050-OpenOne'
p = os.path.join(root, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
b = open(p, 'rb').read()
out = []
out.append('bytes %d' % len(b))
out.append('has_utf8_bom %s' % b.startswith(b'\xef\xbb\xbf'))

try:
    T = b.decode('utf-8')
    out.append('strict_utf8_ok True  chars %d' % len(T))
    bad = []
except UnicodeDecodeError as e:
    out.append('strict_utf8_ok False  %s' % e)
    T = b.decode('utf-8', errors='replace')
    bad = [i for i, ch in enumerate(T) if ch == '\ufffd']
    out.append('replacement_chars %d at %s' % (len(bad), bad[:20]))

# 逐段诊断：找出非法字节位置
if 'strict_utf8_ok True' not in '\n'.join(out):
    pos = 0
    while pos < len(b):
        try:
            b[pos:].decode('utf-8')
            break
        except UnicodeDecodeError as e:
            out.append('  bad at byte %d: %r (context %r)' % (pos + e.start, b[e.start:e.end], b[max(0,e.start-30):e.end+30]))
            pos = pos + e.end
            if len(out) > 40:
                break

out.append('ends_with_newline %s' % b.endswith(b'\n'))
out.append('last60_bytes %r' % b[-60:])

open(os.path.join(root, 'gpt5_temp', 'enc_check.txt'), 'w', encoding='utf-8').write('\n'.join(out))
print('ok')
