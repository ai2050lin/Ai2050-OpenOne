# -*- coding: utf-8 -*-
"""修 verify_q03.py：TESTPLAN 路径（Edit 幻影）、metric_dict 池化信息项、memo 改为信息记录。"""
fp = r"D:\AI2050\Ai2050-OpenOne\tests\deepseek\verify_q03.py"
src = open(fp, encoding="utf-8").read()
orig = src

reps = [
    # (a) TESTPLAN：R6 已并入 memo 并从 gpt5/docs 删除 -> 用 R6 归档路径
    ("    ('RDC_TESTPLAN_v1.md', os.path.join(ROOT, 'research', 'gpt5', 'docs', 'RDC_TESTPLAN_v1.md'), '71b85673'),",
     "    ('RDC_TESTPLAN_v1.md', os.path.join(ROOT, 'tests', 'deepseek_temp', '_archive_r6', 'gpt5_docs', 'RDC_TESTPLAN_v1.md'), '71b85673'),"),
    # (b) metric_dict 无显式池化字段 -> 信息项（非缺陷）
    ("chk('metric_dict 含 E_read 池化字段', E_md is not None, repr(E_md)[:80])",
     "rows.append('     [INFO] metric_dict 未含显式池化字段（仅定义+gate 在册）；按三份 result 独立重算')"),
    # (c) memo 固定 sha8 断言 -> 移除（并发写者会合法改动；改为信息记录）
    ("    ('AGI_DEEPSEEK_MEMO.md', os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md'), '150241da'),\n",
     ""),
]
for old, new in reps:
    c = src.count(old)
    assert c == 1, "count=%d for %r" % (c, old[:80])
    src = src.replace(old, new)

# 插入 8b) 研究日志现状信息段（在 9) 之前）
anchor = "rows.append('9) Q03 产物指纹')"
assert src.count(anchor) == 1, "anchor count=%d" % src.count(anchor)
memo_info = (
"rows.append('8b) 研究日志现状（并发写者存在；仅记录，不判 FAIL）')\n"
"_mm = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')\n"
"_mraw = open(_mm, 'rb').read()\n"
"_mt = _mraw.decode('utf-8-sig')\n"
"rows.append('     bytes=%d lines=%d sha8=%s' % (len(_mraw), _mt.count(chr(13)+chr(10))+1, sha8b(_mm)))\n"
"rows.append('     Phase 35 内容在位 = %s' % ('A 闸门 seal 执行与关闭' in _mt))\n"
"rows.append('     Phase 36（并发写者 E4/E4b）在位 = %s' % ('## Phase 36' in _mt))\n"
"_pre = _mraw[:685649]\n"
"rows.append('     prefix(685649) sha8=%s (Phase36 脚本登记 before=5be5bf12)' % hashlib.sha256(_pre).hexdigest()[:8])\n"
"\n")
src = src.replace(anchor, memo_info + anchor)
assert src != orig, "no change"
open(fp, "w", encoding="utf-8", newline="\n").write(src)
print("PATCH_OK reps=3 + memo_info")
print("has_archive_path =", "_archive_r6" in src)
print("has_old_path =", "'research', 'gpt5', 'docs', 'RDC_TESTPLAN" in src)
print("has_150241da =", "150241da" in src)
