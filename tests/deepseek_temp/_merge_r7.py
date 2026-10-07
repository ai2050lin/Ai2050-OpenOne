# -*- coding: utf-8 -*-
"""R7：把已确证的 3 件 deepseek 线文档全文并入 AGI_DEEPSEEK_MEMO.md（Phase 32–34），
并把 metric_dict_v1_backup.json 迁入 research/deepseek/atlas/。
规则（继承 R6 / Phase 23–30 的并入模式）：
  · 标题降一级（H1→H2 … H5→H6），代码围栏内不动
  · 原名先备份到 tests/deepseek_temp/_archive_r7/gpt5_docs/ 再删除
  · memo: BOM + CRLF、bare_lf=0、前缀逐字节不变
  · 并入完整性 = 逐行包含性 100%
"""
import os, hashlib, json, re, shutil

ROOT = r"D:\AI2050\Ai2050-OpenOne"
G5D  = os.path.join(ROOT, "research", "gpt5", "docs")
G5A  = os.path.join(ROOT, "research", "gpt5", "atlas")
MEMO = os.path.join(ROOT, "research", "deepseek", "docs", "AGI_DEEPSEEK_MEMO.md")
ARCH = os.path.join(ROOT, "tests", "deepseek_temp", "_archive_r7", "gpt5_docs")
DSKA = os.path.join(ROOT, "research", "deepseek", "atlas")
REPORT = os.path.join(ROOT, "tests", "deepseek", "result", "_merge_r7_report.txt")

STAMP = "2026-10-03 02:56"
DOCS = [
    ("Phase 32", "MAIN_AXIS_VERDICT_v1.md",
     "并入 `MAIN_AXIS_VERDICT_v1.md` —— 主轴裁决 v1：embedding → layer → unembed 三段分工（N1 全文）"),
    ("Phase 33", "EMBED_ANCHOR_VERDICT_v1.md",
     "并入 `EMBED_ANCHOR_VERDICT_v1.md` —— 词嵌入锚点裁决 v1：E1 全文"),
    ("Phase 34", "MEMO_AUDIT_2750_3148.md",
     "并入 `MEMO_AUDIT_2750_3148.md` —— MEMO 审计报告 Phase 2750–3148（全文）"),
]
MOVE_JSON = ("metric_dict_v1_backup.json", G5A, DSKA)

log = []
def L(s): log.append(s); print(s)

def sha8(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]
def size(p): return os.path.getsize(p)

def demote(txt):
    out, fence = [], False
    for l in txt.split("\n"):
        if l.lstrip().startswith("```"):
            fence = not fence
        if not fence:
            m = re.match(r"^(#{1,5}) ", l)
            if m:
                l = "#" + l
        out.append(l)
    return "\n".join(out)

# ---------- 0. 前置：memo 现状 ----------
raw = open(MEMO, "rb").read()
assert raw.startswith(b"\xef\xbb\xbf"), "memo 无 BOM"
old_txt = raw.decode("utf-8-sig")
base_lf = old_txt.replace("\r\n", "\n").rstrip("\n")
n_before = len(raw)
L("memo before: %d B sha8=%s crlf=%d" % (n_before, hashlib.sha256(raw).hexdigest()[:8], old_txt.count("\r\n")))

os.makedirs(ARCH, exist_ok=True)
os.makedirs(DSKA, exist_ok=True)

# 0b. 快照当前 memo（R7 前基线，供后续 drift 比对；本 memo 未被 git 跟踪）
SNAP = os.path.join(ARCH, "..", "AGI_DEEPSEEK_MEMO.pre_r7.bin")
snap = os.path.abspath(SNAP)
open(snap, "wb").write(raw)
assert sha8(snap) == hashlib.sha256(raw).hexdigest()[:8]
L("snapshot -> %s (%d B sha8=%s)" % (os.path.basename(snap), len(raw), sha8(snap)))

# ---------- 1. 逐件并入 ----------
blocks = []
contain_check = []
for sec, fn, title in DOCS:
    src = os.path.join(G5D, fn)
    assert os.path.exists(src), "缺文件 %s" % src
    rawd = open(src, "rb").read()
    doc = rawd.decode("utf-8-sig", "replace").replace("\r\n", "\n")
    h = hashlib.sha256(rawd).hexdigest()[:8]
    # 备份
    bkp = os.path.join(ARCH, fn)
    shutil.copy2(src, bkp)
    assert sha8(bkp) == h, "备份哈希不符 %s" % fn
    dem = demote(doc).rstrip("\n")
    block = "## %s: %s [%s]" % (sec, title, STAMP) + "\n\n" + dem + "\n"
    blocks.append((fn, h, block))
    contain_check.append((fn, dem))
    L("  built %s (%d B, sha8=%s) -> %s" % (fn, len(rawd), h, sec))

# ---------- 2. 写入 memo ----------
new_lf = base_lf + "\n" + "\n".join(b.rstrip("\n") for _, _, b in blocks)
payload = b"\xef\xbb\xbf" + new_lf.replace("\n", "\r\n").encode("utf-8")
open(MEMO, "wb").write(payload)

# ---------- 3. 回读验证 ----------
raw2 = open(MEMO, "rb").read()
txt2 = raw2.decode("utf-8-sig")
L("memo after : %d B sha8=%s crlf=%d bare_lf=%d bom=%s"
  % (len(raw2), hashlib.sha256(raw2).hexdigest()[:8], txt2.count("\r\n"),
     txt2.count("\n") - txt2.count("\r\n"), raw2.startswith(b"\xef\xbb\xbf")))
assert not (txt2.count("\n") - txt2.count("\r\n")), "出现 bare LF"
assert raw2.startswith(b"\xef\xbb\xbf"), "BOM 丢失"
# 前缀逐字节不变
prefix_bytes = b"\xef\xbb\xbf" + base_lf.replace("\n", "\r\n").encode("utf-8")
assert raw2[:len(prefix_bytes)] == prefix_bytes, "前缀被改动！"
L("prefix byte-identical: True")

# 包含性
lf_memo = txt2.replace("\r\n", "\n")
for fn, dem in contain_check:
    lines = [l for l in dem.split("\n") if l.strip()]
    miss = [l for l in lines if l not in lf_memo]
    L("  containment %s: %d/%d hit, miss=%d" % (fn, len(lines) - len(miss), len(lines), len(miss)))
    assert not miss, "并入不完整：%s 缺 %d 行" % (fn, len(miss))

# Phase 标题
ph = re.findall(r"^## Phase\s*([0-9]+)", lf_memo, re.M)
L("phase headings = %d %s" % (len(ph), ph))

# ---------- 4. 删原件 ----------
for fn, h, _ in blocks:
    src = os.path.join(G5D, fn)
    if os.path.exists(src):
        os.remove(src)
    L("  removed original %s (backup sha8=%s)" % (fn, sha8(os.path.join(ARCH, fn))))

# ---------- 5. 迁移 JSON ----------
jfn, srcd, dstd = MOVE_JSON
sp, dp = os.path.join(srcd, jfn), os.path.join(dstd, jfn)
if os.path.exists(sp):
    hj = sha8(sp)
    shutil.copy2(sp, dp)
    assert sha8(dp) == hj, "迁移哈希不符"
    os.remove(sp)
    L("  moved %s -> research/deepseek/atlas/ (sha8=%s)" % (jfn, hj))
else:
    L("  %s 已在目标或缺失，跳过" % jfn)

open(REPORT, "w", encoding="utf-8").write("\n".join(log) + "\n")
L("REPORT -> %s" % REPORT)
