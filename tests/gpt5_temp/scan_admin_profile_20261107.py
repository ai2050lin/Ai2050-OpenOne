# -*- coding: utf-8 -*-
# 只读扫描 C:\Users\Admin 清理候选。不做任何修改/删除/移动。
import os, time, datetime

PROFILE = "C:/Users/Admin"
OUT = "D:/AI2050/Ai2050-OpenOne/tempdata/admin_cleanup_scan_20261107.txt"
START = time.time()
GLOBAL_DEADLINE = START + 470.0
REPARSE = 0x400
LARGE = 200 * 1024 * 1024

err_count = 0
large = {}

def human(n):
    x = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024.0 or u == "TB":
            return ("%d B" % int(x)) if u == "B" else ("%.2f %s" % (x, u))
        x /= 1024.0

def is_reparse(e):
    try:
        st = e.stat(follow_symlinks=False)
        return bool(getattr(st, "st_file_attributes", 0) & REPARSE)
    except Exception:
        return False

def dir_size(root, deadline):
    global err_count
    total = 0; nf = 0; nd = 0; trunc = False; junc = 0
    stack = [root]
    while stack:
        if time.time() > deadline:
            trunc = True
            break
        d = stack.pop()
        try:
            it = os.scandir(d)
        except Exception:
            err_count += 1
            continue
        with it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        if is_reparse(e):
                            junc += 1
                            continue
                        nd += 1
                        stack.append(e.path)
                    elif e.is_file(follow_symlinks=False):
                        sz = e.stat(follow_symlinks=False).st_size
                        total += sz
                        nf += 1
                        if sz >= LARGE:
                            large[e.path] = sz
                except Exception:
                    err_count += 1
    return {"size": total, "files": nf, "dirs": nd, "trunc": trunc, "junc": junc}

# tag: AUTO=建议清理(可再生缓存/临时) ASK=需用户确认 OFF=勿动
CURATED = [
    ("Windows用户临时目录", "AppData/Local/Temp", "AUTO", "临时文件，关闭运行中程序后可清"),
    ("pip缓存", "AppData/Local/pip/cache", "AUTO", "官方命令 pip cache purge"),
    ("uv缓存", "AppData/Local/uv/cache", "AUTO", "官方命令 uv cache clean"),
    ("uv基座解释器", "AppData/Roaming/uv/python", "OFF", "项目venv依赖的基座解释器，勿删"),
    ("npm缓存", "AppData/Local/npm-cache", "AUTO", "npm cache clean --force"),
    ("npm缓存Roaming", "AppData/Roaming/npm-cache", "AUTO", "npm cache clean --force"),
    ("yarn缓存", "AppData/Local/Yarn", "AUTO", "yarn cache clean"),
    ("pnpm存储", "AppData/Local/pnpm", "ASK", "pnpm store prune"),
    ("HF模型缓存dotcache", ".cache/huggingface", "ASK", "删后需重新下载模型"),
    ("HF模型缓存Local", "AppData/Local/huggingface", "ASK", "删后需重新下载模型"),
    ("TorchHub缓存", ".cache/torch", "ASK", "删后需重新下载"),
    ("Triton编译缓存", "AppData/Local/triton", "AUTO", "编译缓存，可再生"),
    ("conda包缓存", ".conda/pkgs", "ASK", "conda clean -a 官方清理"),
    ("Ollama模型", ".ollama", "ASK", "本地模型文件，删后需重新拉取"),
    ("Chrome缓存", "AppData/Local/Google/Chrome/User Data/Default/Cache", "AUTO", "浏览器缓存，可再生"),
    ("ChromeCodeCache", "AppData/Local/Google/Chrome/User Data/Default/Code Cache", "AUTO", "可再生"),
    ("Edge缓存", "AppData/Local/Microsoft/Edge/User Data/Default/Cache", "AUTO", "可再生"),
    ("EdgeCodeCache", "AppData/Local/Microsoft/Edge/User Data/Default/Code Cache", "AUTO", "可再生"),
    ("系统INetCache", "AppData/Local/Microsoft/Windows/INetCache", "AUTO", "系统网络缓存"),
    ("错误报告WER", "AppData/Local/Microsoft/Windows/WER", "AUTO", "崩溃报告队列"),
    ("崩溃转储CrashDumps", "AppData/Local/CrashDumps", "AUTO", "调试用，可清"),
    ("D3DSCache", "AppData/Local/D3DSCache", "AUTO", "着色器缓存，可再生"),
    ("NVIDIA DXCache", "AppData/Local/NVIDIA/DXCache", "AUTO", "着色器缓存，可再生"),
    ("NVIDIA GLCache", "AppData/Local/NVIDIA/GLCache", "AUTO", "着色器缓存，可再生"),
    ("缩略图缓存Explorer", "AppData/Local/Microsoft/Windows/Explorer", "AUTO", "thumbcache，可再生"),
    ("VSCode Cache", "AppData/Roaming/Code/Cache", "AUTO", "可再生"),
    ("VSCode CachedData", "AppData/Roaming/Code/CachedData", "AUTO", "可再生"),
    ("VSCode CodeCache", "AppData/Roaming/Code/Code Cache", "AUTO", "可再生"),
    ("VSCode GPUCache", "AppData/Roaming/Code/GPUCache", "AUTO", "可再生"),
    ("VSCode logs", "AppData/Roaming/Code/logs", "AUTO", "日志"),
    ("electron缓存", "AppData/Local/electron/Cache", "AUTO", "可再生"),
    ("Playwright浏览器", "AppData/Local/ms-playwright", "ASK", "浏览器内核，做浏览器自动化需保留"),
    ("安装器PackageCache", "AppData/Local/Package Cache", "OFF", "影响卸载/修复，勿手动删"),
    ("已安装程序本体", "AppData/Local/Programs", "OFF", "程序安装目录"),
    ("matplotlib缓存", ".matplotlib", "AUTO", "可再生"),
    ("WorkBuddy数据", ".workbuddy", "OFF", "配置/记忆/插件，勿动"),
    ("通用缓存dotcache", ".cache", "ASK", "汇总行，已含上方HF/Torch子项"),
]

EXCLUDE_FROM_TOTALS = {"通用缓存dotcache"}

def main():
    global err_count
    results = {}
    # ---- Phase A: 精确测量各缓存候选 ----
    a_deadline = min(GLOBAL_DEADLINE, time.time() + 300.0)
    for label, rel, tag, note in CURATED:
        if time.time() > a_deadline:
            results[label] = None
            continue
        p = os.path.join(PROFILE, rel.replace("/", os.sep))
        if not os.path.exists(p):
            results[label] = {"missing": True}
            continue
        r = dir_size(p, a_deadline)
        r["tag"] = tag
        results[label] = r

    # ---- Phase B: 顶层占用 ----
    top = []
    root_files = []
    try:
        entries = list(os.scandir(PROFILE))
    except Exception:
        entries = []
    dirs_first = [e for e in entries if e.is_dir(follow_symlinks=False)]
    files_only = [e for e in entries if not e.is_dir(follow_symlinks=False)]

    pending = []
    for e in sorted(dirs_first, key=lambda x: x.name.lower()):
        if is_reparse(e):
            top.append({"name": e.name, "type": "junction", "size": 0, "files": 0, "dirs": 0, "trunc": False})
            continue
        r = dir_size(e.path, min(GLOBAL_DEADLINE, time.time() + 15.0))
        r["name"] = e.name; r["type"] = "dir"
        top.append(r)
        if r["trunc"]:
            pending.append(e.path)
    if pending and time.time() < GLOBAL_DEADLINE:
        budget = max(30.0, (GLOBAL_DEADLINE - time.time()) / max(1, len(pending)))
        for p in pending:
            if time.time() > GLOBAL_DEADLINE:
                break
            r2 = dir_size(p, min(GLOBAL_DEADLINE, time.time() + budget))
            base = os.path.basename(p)
            for row in top:
                if row["name"] == base:
                    row["size"] = r2["size"]; row["files"] = r2["files"]
                    row["dirs"] = r2["dirs"]; row["trunc"] = r2["trunc"]
    for e in files_only:
        try:
            sz = e.stat(follow_symlinks=False).st_size
            root_files.append({"name": e.name, "size": sz})
            if sz >= LARGE:
                large[e.path] = sz
        except Exception:
            err_count += 1

    # ---- 报告 ----
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = []
    L.append("=" * 78)
    L.append("C:\\Users\\Admin 清理分析报告 —— 只读扫描，未做任何修改/删除/移动")
    L.append("生成时间: %s   总耗时: %.0fs   不可读条目: %d" % (now, time.time() - START, err_count))
    L.append("说明: 标记 partial 的目录因扫描时限只统计了部分文件，实际占用 >= 报告值。")
    L.append("=" * 78)
    L.append("")
    L.append("【一】常见缓存/临时位置（清理候选，精确测量）")
    L.append("%-8s%12s%10s  %-55s %s" % ("分类", "大小", "文件数", "路径", "说明"))
    auto_sum = 0; ask_sum = 0
    for label, rel, tag, note in CURATED:
        r = results.get(label)
        if r is None:
            L.append("%-8s%12s%10s  %-55s %s" % (tag, "--", "--", rel, "(超时未测)"))
            continue
        if r.get("missing"):
            L.append("%-8s%12s%10s  %-55s %s" % (tag, "不存在", "", rel, note))
            continue
        mark = " partial" if r["trunc"] else ""
        L.append("%-8s%12s%10d  %-55s %s%s" % (tag, human(r["size"]), r["files"], rel, note, mark))
        if label not in EXCLUDE_FROM_TOTALS:
            if tag == "AUTO":
                auto_sum += r["size"]
            elif tag == "ASK":
                ask_sum += r["size"]
    L.append("")
    L.append("AUTO(建议清理) 合计: %s" % human(auto_sum))
    L.append("ASK(需确认)   合计: %s  (不含汇总行重复统计)" % human(ask_sum))
    L.append("")
    L.append("【二】顶层目录/文件占用")
    L.append("%-30s%-10s%12s%10s  %s" % ("名称", "类型", "大小", "文件数", "备注"))
    for row in sorted([r for r in top if r["type"] == "dir"], key=lambda r: -r["size"]):
        note = "partial(部分扫描)" if row["trunc"] else ""
        L.append("%-30s%-10s%12s%10d  %s" % (row["name"], "目录", human(row["size"]), row["files"], note))
    for row in [r for r in top if r["type"] == "junction"]:
        L.append("%-30s%-10s%12s" % (row["name"], "junction", "已跳过"))
    for f in sorted(root_files, key=lambda f: -f["size"]):
        L.append("%-30s%-10s%12s" % (f["name"], "文件", human(f["size"])))
    L.append("")
    L.append("【三】大文件 TOP40 (>=200MB, 来自已扫描范围, 不完整)")
    for p, sz in sorted(large.items(), key=lambda kv: -kv[1])[:40]:
        L.append("%10s  %s" % (human(sz), p))
    L.append("")
    L.append("【四】分类说明")
    L.append("AUTO = 可再生缓存/临时文件，清理风险低，建议优先用各自官方命令或 Windows 磁盘清理。")
    L.append("ASK  = 删除会带来重新下载/重装成本或影响功能，需要逐项确认。")
    L.append("OFF  = 程序本体/安装器/配置记忆类，禁止手动清理。")
    L.append("Downloads/Desktop/Documents 属个人文件，本工具只报大小，不给出清理建议。")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("REPORT WRITTEN")

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        try:
            with open(OUT, "a", encoding="utf-8") as f:
                f.write("\n[EXCEPTION] %r\n" % (ex,))
        except Exception:
            pass
        print("EXCEPTION OCCURRED")
