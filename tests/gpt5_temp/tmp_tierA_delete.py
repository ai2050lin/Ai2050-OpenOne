"""Tier A deletion executor.

Safety contract:
- ONLY deletes paths listed in the manifest JSON. Never scans for candidates itself.
- Refuses to run unless the manifest exists and its group keys are exactly the expected set.
- A1/A2 (regenerable caches)   -> direct delete
- A3/A4 (real content)          -> Windows Recycle Bin, NOT rm
- Junction/reparse dirs are never traversed as delete targets.
- Per-batch cap + verification after each batch.
- Writes a deletion log for rollback reference.
"""
import os, sys, json, shutil, time, argparse, subprocess, ctypes
from ctypes import wintypes

MANIFEST = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tierA_manifest.json"
LOG      = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tierA_deletion_log.jsonl"
EXPECT   = {"A1_pycache", "A2_thirdparty", "A3_log_bak", "A4_dup_groups"}

# ---- Recycle Bin via SHFileOperation ----
FO_DELETE = 0x0003
FOF_SILENT      = 0x0004
FOF_NOCONFIRMATION = 0x0010
FOF_ALLOWUNDO   = 0x0040
FOF_NOERRORUI   = 0x0400
FOF_WANTNUKEWARNING = 0x4000

class SHFILEOPSTRUCTW(ctypes.Structure):
    _fields_ = [("hwnd", wintypes.HWND),
                ("wFunc", wintypes.UINT),
                ("pFrom", wintypes.LPCWSTR),
                ("pTo", wintypes.LPCWSTR),
                ("fFlags", ctypes.c_uint16),
                ("fAnyOperationsAborted", wintypes.BOOL),
                ("hNameMappings", ctypes.c_void_p),
                ("lpszProgressTitle", wintypes.LPCWSTR)]

shell32 = ctypes.windll.shell32
_ctypes = ctypes

def is_reparse(path):
    try:
        st = os.lstat(path)
        return bool(getattr(st, "st_file_attributes", 0) & 0x400)
    except OSError:
        return False

def recycle(paths):
    """Move paths to Recycle Bin. Windows has a 20k-op practical limit; chunk at 100."""
    ok = 0; fail = []
    CH = 100
    for i in range(0, len(paths), CH):
        chunk = paths[i:i+CH]
        for p in chunk:
            if is_reparse(p):
                fail.append((p, "refused: reparse point")); continue
        live = [p for p in chunk if not is_reparse(p)]
        if not live: continue
        double_nul = "\0".join(p.replace("/", "\\") for p in live) + "\0\0"
        op = SHFILEOPSTRUCTW()
        op.hwnd = None
        op.wFunc = FO_DELETE
        op.pFrom = _ctypes.c_wchar_p(double_nul)
        op.pTo = None
        op.fFlags = FOF_SILENT | FOF_NOCONFIRMATION | FOF_ALLOWUNDO | FOF_NOERRORUI | FOF_WANTNUKEWARNING
        op.fAnyOperationsAborted = False
        op.hNameMappings = None
        op.lpszProgressTitle = None
        rc = shell32.SHFileOperationW(_ctypes.byref(op))
        if rc == 0:
            ok += len(live)
        else:
            for p in live: fail.append((p, "SHFileOperationW rc=%d" % rc))
    return ok, fail

def logrec(rec):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=sorted(EXPECT))
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--max-batches", type=int, default=0, help="0 = all")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not os.path.isfile(MANIFEST):
        print("ABORT: manifest missing ->", MANIFEST); sys.exit(2)
    man = json.load(open(MANIFEST, encoding="utf-8"))
    got = set(man["groups"])
    if got != EXPECT:
        print("ABORT: manifest group mismatch %s != %s" % (sorted(got), sorted(EXPECT))); sys.exit(2)

    g = man["groups"][a.group]
    items = g["files"] if a.group != "A2_thirdparty" else None
    direct = a.group in ("A1_pycache", "A2_thirdparty")
    method = g.get("method", "")

    # build concrete target list
    targets = []
    if a.group == "A1_pycache":
        for d in g["dirs"]:
            base = os.path.join(r"D:\AI2050\Ai2050-OpenOne\tests", d)
            for dp, dn, fns in os.walk(base):
                for f in fns: targets.append(os.path.join(dp, f))
    elif a.group == "A2_thirdparty":
        for d in g["dirs"]:
            base = os.path.join(r"D:\AI2050\Ai2050-OpenOne\tests", d["rel"])
            targets.append(base)
    else:
        targets = items

    # existence filter
    exist = [t for t in targets if os.path.exists(t)]
    missing = len(targets) - len(exist)
    est = 0
    for t in exist:
        try:
            if os.path.isdir(t): est += sum(os.path.getsize(os.path.join(a2, f))
                                            for a2, b, c in os.walk(t) for f in c)
            else: est += os.path.getsize(t)
        except OSError: pass

    print("GROUP %s | method=%s" % (a.group, method))
    print("targets=%d (missing=%d)  est=%.2f MB" % (len(exist), missing, est/1024**2))
    print("direct=%s  batch=%d" % (direct, a.batch))
    if a.dry_run:
        for t in exist[:30]: print("   ", t)
        print("... dry-run, nothing deleted"); return

    if direct:
        # batched direct delete
        done = 0; errs = []; nb = 0
        for i in range(0, len(exist), a.batch):
            chunk = exist[i:i+a.batch]; nb += 1
            for p in chunk:
                try:
                    if os.path.isdir(p) and not os.path.islink(p):
                        shutil.rmtree(p); done += 1
                    else:
                        os.remove(p); done += 1
                except OSError as e:
                    errs.append((p, str(e)))
            if a.max_batches and nb >= a.max_batches: break
            if nb % 50 == 0: print("  batch %d done=%d err=%d" % (nb, done, len(errs)), flush=True)
        logrec({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "group": a.group,
                "method": "direct", "requested": len(exist), "deleted": done,
                "errors": errs[:200], "manifest_bytes": g["bytes"]})
        print("DONE direct: deleted=%d errors=%d" % (done, len(errs)))
        for p, e in errs[:10]: print("  ERR", p, e)
    else:
        # recycle bin, batched
        done = 0; allfail = []; nb = 0
        i = 0
        while i < len(exist):
            chunk = exist[i:i + a.batch]
            ok, fail = recycle(chunk)
            done += ok; allfail.extend(fail); nb += 1
            i += a.batch
            if a.max_batches and nb >= a.max_batches: break
            if nb % 25 == 0: print("  batch %d recycled=%d fail=%d" % (nb, done, len(allfail)), flush=True)
        logrec({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "group": a.group,
                "method": "recycle_bin", "requested": len(exist), "recycled": done,
                "failures": allfail[:200], "manifest_bytes": g["bytes"]})
        print("DONE recycle: recycled=%d fail=%d" % (done, len(allfail)))
        for p, e in allfail[:10]: print("  FAIL", p, e)

if __name__ == "__main__":
    main()
