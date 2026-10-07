# -*- coding: utf-8 -*-
"""R6 归位：① 删除 gpt5/docs 下 8 件已并入的 .md；② 迁 6 个 JSON 到 deepseek 目录；
③ 脚本分桶（耐用→tests/deepseek/，临时→tests/deepseek_temp/）；
④ 结果分桶（tests/deepseek_temp/_review/ 全部 → tests/deepseek/result/）。
每步校验哈希 + 目的存在 + 源消失。"""
import os, hashlib, shutil, json

ROOT = r"D:\AI2050\Ai2050-OpenOne"
REPORT = os.path.join(ROOT, r"tests\deepseek_temp\_reorg_r6_report.txt")
MANIFEST = os.path.join(ROOT, r"tests\deepseek\result\reorg_r6_manifest.json")
log = []
def A(s): log.append(s)
def sh8b(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]

moves = []   # (kind, src, dst, sha8, bytes)

def move_file(src, dst):
    assert os.path.exists(src), "SRC MISSING %s" % src
    assert not os.path.exists(dst), "DST EXISTS %s" % dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    h = sh8b(src); n = os.path.getsize(src)
    shutil.move(src, dst)
    assert os.path.exists(dst) and not os.path.exists(src), "move failed %s" % src
    assert sh8b(dst) == h, "hash drift %s" % dst
    moves.append((os.path.basename(src), src, dst, h, n))
    return h

# ---- A) delete the 8 merged .md ----
GDOCS = os.path.join(ROOT, r"research\gpt5\docs")
MD_HASH = {
    "RDC_TESTPLAN_v1.md": "71b85673",
    "LOOP_DIAGNOSIS_AND_EXIT_v1.md": "013d08a9",
    "RDC_RESEARCH_CONSTITUTION_v1.md": "01df6398",
    "META_SINGLE_SOURCE_Q01.md": "95366ca9",
    "METRIC_DICT_Q02.md": "ba518fea",
    "DEADLINE_DUAL_TRACK_Q09.md": "0d8633c7",
    "PROP_CITATION_AUDIT_Q12.md": "32933855",
    "SEAL_REQUEST_A_GATE.md": "7678cbe0",
}
A("== A) delete 8 merged .md from research/gpt5/docs ==")
for fn, exp in MD_HASH.items():
    fp = os.path.join(GDOCS, fn)
    assert os.path.exists(fp), "missing %s" % fn
    g = sh8b(fp)
    assert g == exp, "hash drift %s exp=%s got=%s" % (fn, exp, g)
    bk = os.path.join(ROOT, r"tests\deepseek_temp\_archive_r6\gpt5_docs", fn)
    assert os.path.exists(bk) and sh8b(bk) == exp, "backup missing %s" % fn
    os.remove(fp)
    assert not os.path.exists(fp)
    A("  DEL %-34s %s (backup ok)" % (fn, exp))

# ---- B) move 6 JSON ----
A("")
A("== B) move JSON artifacts ==")
GATL = os.path.join(ROOT, r"research\gpt5\atlas")
JSON_PLAN = [
    ("deadline_dual_track_v1.json", r"tests\deepseek\result\deadline_dual_track_v1.json"),
    ("prop_citation_audit_v1.json", r"tests\deepseek\result\prop_citation_audit_v1.json"),
    ("meta_single_source_v4.json", r"tests\deepseek\result\meta_single_source_v4.json"),
    ("seal_request_v1.json", r"tests\deepseek\result\seal_request_v1.json"),
    ("metric_dict.json", r"research\deepseek\atlas\metric_dict.json"),
    ("phase_queue_v1.json", r"research\deepseek\atlas\phase_queue_v1.json"),
]
for fn, rel in JSON_PLAN:
    h = move_file(os.path.join(GATL, fn), os.path.join(ROOT, rel))
    A("  MOV %-30s -> %-52s %s" % (fn, rel, h))

# ---- C/D) scripts bucket ----
A("")
A("== C/D) scripts bucket ==")
REV = os.path.join(ROOT, r"tests\deepseek\_review")
DUR = os.path.join(ROOT, r"tests\deepseek")
TMP = os.path.join(ROOT, r"tests\deepseek_temp")
DURABLE = {"append_memo_r6.py", "classify_phases_r3.py",
           "disk_verify_continue_r3.py", "disk_verify_q01_r4.py", "disk_verify_q02_r4.py",
           "disk_verify_q09q12_r5.py", "gen_constitution_r3.py", "gen_loop_diagnosis_r3.py",
           "gen_loop_html_r3.py", "gen_q01_r4.py", "gen_q01q02_html_r4.py", "gen_q02_r4.py",
           "gen_q09_r5.py", "gen_q09q12_html_r5.py", "gen_seal_request_r5.py",
           "prop_citation_audit.py", "verify_final_r2.py"}
allpy = sorted(f for f in os.listdir(REV) if f.endswith(".py"))
assert set(allpy) == DURABLE | (set(allpy) - DURABLE), "sanity"
assert DURABLE <= set(allpy), "durable not subset: %s" % (DURABLE - set(allpy))
for fn in allpy:
    tgt = os.path.join(DUR if fn in DURABLE else TMP, fn)
    h = move_file(os.path.join(REV, fn), tgt)
    A("  %-4s %-42s %s" % ("DUR" if fn in DURABLE else "TMP", fn, h))
rest = os.listdir(REV)
A("  _review leftover = %s" % rest)
assert rest == [], "leftover in _review: %s" % rest
os.rmdir(REV)

# ---- E) results bucket ----
A("")
A("== E) results bucket ==")
REV2 = os.path.join(ROOT, r"tests\deepseek_temp\_review")
RES = os.path.join(ROOT, r"tests\deepseek\result")
os.makedirs(RES, exist_ok=True)
ents = sorted(os.listdir(REV2))
nres = 0
for e in ents:
    src = os.path.join(REV2, e); dst = os.path.join(RES, e)
    assert not os.path.exists(dst), "DST EXISTS %s" % dst
    if os.path.isdir(src):
        shutil.move(src, dst)
        assert os.path.isdir(dst) and not os.path.exists(src)
        nres += 1
        A("  DIR  %s/" % e)
    else:
        h = move_file(src, dst)
        nres += 1
A("  moved entries = %d ; leftover = %s" % (nres, os.listdir(REV2)))
assert os.listdir(REV2) == []
os.rmdir(REV2)

# ---- manifest ----
man = {
    "round": "R6",
    "stamp": "2026-10-03T02:35",
    "deleted_merged_md": {k: v for k, v in MD_HASH.items()},
    "moves": [{"name": m[0], "src": m[1], "dst": m[2], "sha8": m[3], "bytes": m[4]} for m in moves],
    "bucket_rules": {
        "tests/deepseek/": "测试脚本（可复跑/复核/生成）",
        "tests/deepseek_temp/": "临时脚本（探针/一次性补丁）",
        "tests/deepseek/result/": "测试结果（txt/json/html/md 报告）",
    },
}
open(MANIFEST, "w", encoding="utf-8").write(json.dumps(man, ensure_ascii=False, indent=1))

# ---- verify ----
A("")
A("== verify ==")
A("  gpt5/docs 中 8 件已删: %s" % (not any(os.path.exists(os.path.join(GDOCS, f)) for f in MD_HASH)))
A("  gpt5/atlas 中 6 件已迁: %s" % (not any(os.path.exists(os.path.join(GATL, f)) for f, _ in JSON_PLAN)))
nd = len([f for f in os.listdir(DUR) if f.endswith(".py") and os.path.isfile(os.path.join(DUR, f))])
nt = len([f for f in os.listdir(TMP) if f.endswith(".py") and os.path.isfile(os.path.join(TMP, f))])
nr = len(os.listdir(RES))
A("  tests/deepseek/*.py = %d (期望 17)  tests/deepseek_temp/*.py = %d (期望 >=27)  tests/deepseek/result/ 条目 = %d" % (nd, nt, nr))
assert os.path.exists(MANIFEST)

txt = "\n".join(log)
open(REPORT, "w", encoding="utf-8").write(txt)
print(txt)
print("REORG_OK")
