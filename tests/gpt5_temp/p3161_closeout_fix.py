# -*- coding: utf-8 -*-
"""Fix closeout seal-consistency asserts: seal was computed over the pre-seal-field
file state, so sha8_file(final) != seal by construction. Verify instead that
verdict embeds the res_sha8 field, for all models + summary."""
import py_compile

P = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3161_closeout.py'
raw = open(P, 'rb').read()
s = raw.decode('utf-8')

OLD1 = """# result 内嵌 sha 与磁盘一致性
for m in MS:
    assert R[m]['seal_sha8'] == sha['res_' + m], ('seal mismatch', m)
assert RS['seal_sha8'] == sha['res_summary']
out.append('seal consistency: 4/4 OK')"""
NEW1 = """# result 内嵌 sha 一致性（seal 哈希的是 pre-seal 文件态, 磁盘最终文件含 seal 字段,
# 字节级复验由独立磁盘复核脚本做; 此处验证 verdict 尾部内嵌 res_sha8 与字段一致）
def _verdict_sha_ok(r):
    return r['verdict'].endswith('|sha8_' + r['res_sha8'])

for m in MS:
    assert _verdict_sha_ok(R[m]), ('verdict sha mismatch', m)
assert _verdict_sha_ok(RS), ('verdict sha mismatch', 'summary')
out.append('seal consistency: verdict-embedded sha OK 4/4')"""

OLD2 = """ok.append(('seal_consistency', all(R[m]['seal_sha8'] == sha['res_' + m] for m in MS) and
           RS['seal_sha8'] == sha['res_summary']))"""
NEW2 = """ok.append(('seal_consistency', all(_verdict_sha_ok(R[m]) for m in MS) and _verdict_sha_ok(RS)))"""

cnt1 = s.count(OLD1)
cnt2 = s.count(OLD2)
assert cnt1 == 1, 'OLD1 count=%d' % cnt1
assert cnt2 == 1, 'OLD2 count=%d' % cnt2
s2 = s.replace(OLD1, NEW1).replace(OLD2, NEW2)
assert '\r' not in s2
open(P, 'wb').write(s2.encode('utf-8'))
s3 = open(P, 'rb').read().decode('utf-8')
assert s3 == s2
py_compile.compile(P, doraise=True)
print('CLOSEOUT PATCH OK')
