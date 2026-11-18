# -*- coding: utf-8 -*-
"""READ-ONLY: resolve the full junction chain and hardlink identity."""
import os
import ctypes
from datetime import datetime

RESULT = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result"
CODEX = r"C:\Users\Admin\.codex\visualizations"
VR = r"D:\ai2050_native\visualization-research"
REPORT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\d_drive_junction_chain_20261117.txt"

REPARSE = 0x400
GFA = ctypes.windll.kernel32.GetFileAttributesW
GFA.restype = ctypes.c_uint32
GFA.argtypes = [ctypes.c_wchar_p]


def is_reparse(p):
    a = GFA(p)
    return a != 0xFFFFFFFF and bool(a & REPARSE)


def link(p):
    try:
        return os.readlink(p).replace("\\\\?\\", "")
    except Exception as e:
        return "ERR:" + repr(e)


def file_id(p):
    """NTFS file index via GetFileInformationByHandle (best effort)."""
    import ctypes.wintypes as wt

    class BHFI(ctypes.Structure):
        _fields_ = [("dwFileAttributes", wt.DWORD),
                    ("ftCreationTime", wt.FILETIME),
                    ("ftLastAccessTime", wt.FILETIME),
                    ("ftLastWriteTime", wt.FILETIME),
                    ("dwVolumeSerialNumber", wt.DWORD),
                    ("nFileSizeHigh", wt.DWORD),
                    ("nFileSizeLow", wt.DWORD),
                    ("nNumberOfLinks", wt.DWORD),
                    ("nFileIndexHigh", wt.DWORD),
                    ("nFileIndexLow", wt.DWORD)]
    k = ctypes.windll.kernel32
    GENERIC_READ = 0x80000000
    FILE_SHARE = 0x1 | 0x2 | 0x4
    OPEN_EXISTING = 3
    FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    h = k.CreateFileW(ctypes.c_wchar_p(p), GENERIC_READ, FILE_SHARE, None,
                      OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, None)
    if h == -1 or h == 0xFFFFFFFFFFFFFFFF:
        return None
    try:
        info = BHFI()
        ok = k.GetFileInformationByHandle(ctypes.c_void_p(h), ctypes.byref(info))
        if not ok:
            return None
        return (info.dwVolumeSerialNumber,
                (info.nFileIndexHigh << 32) | info.nFileIndexLow,
                info.nNumberOfLinks)
    finally:
        k.CloseHandle(ctypes.c_void_p(h))


L = []
w = L.append
w("JUNCTION CHAIN / READ-ONLY  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("")

w("[1] C: CODEX TOP-LEVEL JUNCTIONS AND THEIR TARGETS")
for dp, dns, fns in os.walk(CODEX):
    for d in dns:
        q = os.path.join(dp, d)
        if is_reparse(q):
            t = link(q)
            w("    %-58s" % os.path.relpath(q, CODEX)[:58])
            w("        -> %s  exists=%s" % (t, os.path.exists(t)))
    dns[:] = [d for d in dns if not is_reparse(os.path.join(dp, d))]
    if dp.count(os.sep) - CODEX.count(os.sep) >= 3:
        dns[:] = []
w("")

w("[2] result JUNCTION -> C: TARGET -> FINAL RESOLUTION")
names = ["mlp_puzzle_v1", "mlp_trigger_read100_v1", "mlp_trigger_v1",
         "qwen14_mlp_atlas_v1", "record_selection_v1", "record_selection_v2",
         "record_selection_v3", "source_output_v1", "state_formation_v1"]
for n in names:
    p = os.path.join(RESULT, n)
    t1 = link(p)
    chain = [p, t1]
    cur = t1
    for _ in range(4):
        if not cur.startswith("ERR:") and os.path.exists(cur) and is_reparse(cur):
            cur = link(cur)
            chain.append(cur)
        else:
            break
    w("    %s" % n)
    for i, c in enumerate(chain):
        w("        L%d %s%s" % (i, c, "  [REPARSE]" if not c.startswith("ERR:") and os.path.exists(c) and is_reparse(c) else ""))
    w("        final exists=%s  realpath=%s" %
      (os.path.exists(chain[-1]), os.path.realpath(chain[-1]) if os.path.exists(chain[-1]) else "-"))
w("")

w("[3] FILE IDENTITY: C: junction view vs D: visualization-research")
pairs = [
    ("qwen14_mlp_atlas_v1", "B28-A.npy"),
    ("state_formation_v1", None),
    ("record_selection_v3", None),
]
for dname, fn in pairs:
    cpath = None
    for dp, dns, fns in os.walk(os.path.join(CODEX, "2026", "11")):
        if os.path.basename(dp) == dname and fns:
            cpath = dp
            break
        dns[:] = [d for d in dns if not is_reparse(os.path.join(dp, d))]
    dpath = os.path.join(VR, dname)
    w("    --- %s ---" % dname)
    w("        C path = %s" % cpath)
    w("        D path = %s" % dpath)
    if cpath and os.path.isdir(dpath):
        cf = sorted(os.listdir(cpath))
        df = sorted(os.listdir(dpath))
        w("        names equal = %s (C=%d D=%d)" % (cf == df, len(cf), len(df)))
        sample = [x for x in df if x.endswith((".npy", ".npz", ".json"))][:3]
        for s in sample:
            a = file_id(os.path.join(cpath, s))
            b = file_id(os.path.join(dpath, s))
            w("        %-34s C_id=%s  D_id=%s  SAME=%s" %
              (s[:34], a, b, a is not None and b is not None and a[:2] == b[:2]))
w("")

w("[4] WHERE DOES C: CODE DATA ACTUALLY LIVE (volume serial check)")
samples = []
for dp, dns, fns in os.walk(os.path.join(CODEX, "2026", "11", "03")):
    dns[:] = [d for d in dns if not is_reparse(os.path.join(dp, d))]
    for f in fns[:2]:
        samples.append(os.path.join(dp, f))
    if len(samples) >= 2:
        break
for s in samples:
    w("    %s  id=%s" % (s[-100:], file_id(s)))
w("    C: root id = %s" % (file_id("C:\\"),))
w("    D: root id = %s" % (file_id("D:\\"),))
w("")
w("===== END CHAIN =====")
with open(REPORT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
print("CHAIN OK")
