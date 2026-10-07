# -*- coding: utf-8 -*-
"""wlog EOL 一次性修复：2026-10-02.md 由混合 EOL 归一为 CRLF。

背景：`closeout_docs_phase14.py` 与 `wlog_supplement_phase14.py` 均用 `'\\n'` 拼接，
而本项目 14 个 wlog 里 11 个是 CRLF（09-15..09-26/28..09-30），故 09-18 / 09-27 / 10-01 / 10-02
成为混合文件。**只修 2026-10-02.md**（其内容全部由本次 Phase 13/14 链在今日写入），
09-27 / 10-01 / 09-18 保持原样（历史，不动）。

修复方式：按「文本行」重排 EOL —— 断言**行列表逐条相同**（信息零变化），仅换行符由
`\\n` 变 `\\r\\n`。审计证据写入 `wlog_eol_repair_phase14.txt`。
注意：`closeout_docs_phase14.txt` 里记录的 sha256（修复前）会因此过期，报告中显式标注。
"""
import io
import os
import hashlib
import json

W = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-02.md'
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\Phase14\wlog_eol_repair_phase14.txt'

b0 = open(W, 'rb').read()
sha0 = hashlib.sha256(b0).hexdigest()
t0 = b0.decode('utf-8')
lines0 = t0.splitlines()                    # 不含行尾符
# 重排为 CRLF（保留末尾单个换行）
t1 = '\r\n'.join(lines0) + '\r\n'
open(W, 'wb').write(t1.encode('utf-8'))
b1 = open(W, 'rb').read()
sha1 = hashlib.sha256(b1).hexdigest()
lines1 = b1.decode('utf-8').splitlines()

chk = [
    ('行列表逐条相同（信息零变化，%d 行）' % len(lines0), lines0 == lines1),
    ('CRLF 全部覆盖（lf == crlf）', b1.count(b'\n') == b1.count(b'\r\n')),
    ('bare_lf == 0', b1.count(b'\n') - b1.count(b'\r\n') == 0),
    ('无 BOM（wlog 惯例）', b1[:3] != b'\xef\xbb\xbf'),
    ('Phase 14 两节都在', t1.count('## Phase 14 / N2h1-α-7') == 1 and t1.count('## Phase 14 收尾链补充') == 1),
    ('内容长度不变（去掉行尾符后字符数）', len(t0.replace('\r', '').replace('\n', '')) == len(t1.replace('\r', '').replace('\n', ''))),
    ('bytes 只增（每行 +1 CR）', len(b1) - len(b0) == len(lines0)),
]
rep = {
    'file': 'D:/AI2050/Ai2050-OpenOne/.workbuddy/memory/2026-10-02.md',
    'reason': 'closeout_docs / wlog_supplement 用 \\n 拼接；本项目 wlog 主导惯例为 CRLF',
    'scope': 'only 2026-10-02.md (all content written today by the Phase 13/14 chain)',
    'untouched': ['2026-09-18.md', '2026-09-27.md', '2026-10-01.md'],
    'before': {'bytes': len(b0), 'lf': b0.count(b'\n'), 'crlf': b0.count(b'\r\n'), 'sha256': sha0},
    'after': {'bytes': len(b1), 'lf': b1.count(b'\n'), 'crlf': b1.count(b'\r\n'), 'sha256': sha1},
    'stale_hash_note': 'closeout_docs_phase14.txt 里 wlog sha256（%s）为修复前值' % sha0[:12],
    'checks': [{'name': k, 'ok': bool(v)} for k, v in chk],
}
L = ['=== wlog_eol_repair_phase14 ===',
     'bytes %d -> %d ; lf %d->%d ; crlf %d->%d' % (len(b0), len(b1), b0.count(b'\n'), b1.count(b'\n'),
                                                  b0.count(b'\r\n'), b1.count(b'\r\n')),
     'sha256 %s -> %s' % (sha0, sha1), '']
for k, v in chk:
    L.append('  %-46s %s' % (k, 'OK' if v else '**FAIL**'))
L.append('')
L.append('ALL OK' if all(v for _, v in chk) else 'HAS FAIL')
io.open(OUT, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
io.open(OUT.replace('.txt', '.json'), 'w', encoding='utf-8').write(json.dumps(rep, ensure_ascii=False, indent=1))
print('\n'.join(L))
assert all(v for _, v in chk), '有 FAIL'
