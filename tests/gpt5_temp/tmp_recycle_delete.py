"""Recycle-Bin deleter via .NET (Microsoft.VisualBasic.FileIO).

More reliable than SHFileOperationW under this environment (which returned rc=2).
Reads an explicit path list file; never scans for candidates.
Usage: tmp_recycle_delete.py <listfile> <chunk> <max_chunks>
"""
import sys, os, json, time
import subprocess

def ps_recycle(paths):
    """PowerShell: Microsoft.VisualBasic.FileIO.FileSystem.DeleteFile with RecycleBin option."""
    lst = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\_rb_tmp.txt"
    with open(lst, "w", encoding="utf-8") as f:
        f.write("\n".join(paths))
    script = (
        "$ErrorActionPreference='Continue';"
        "Add-Type -AssemblyName Microsoft.VisualBasic;"
        "$ok=0;$fail=0;"
        "$failList=@();"
        "Get-Content -LiteralPath '" + lst + "' -Encoding UTF8 | Where-Object { $_.Trim() -ne '' } | ForEach-Object {"
        "  $p=$_;"
        "  if(-not (Test-Path -LiteralPath $p)){ $fail++; $failList += ($p + ' :: missing'); return }"
        "  try{"
        "    [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile("
        "      $p,"
        "      [Microsoft.VisualBasic.FileIO.UIOption]::OnlyErrorDialogs,"
        "      [Microsoft.VisualBasic.FileIO.RecycleOption]::SendToRecycleBin);"
        "    $ok++"
        "  } catch { $fail++; $failList += ($p + ' :: ' + $_.Exception.Message) }"
        "};"
        "Write-Output ('OK=' + $ok + ' FAIL=' + $fail);"
        "if($failList.Count -gt 0){ $failList | Select-Object -First 40 | ForEach-Object { Write-Output ('ERR ' + $_) } }"
    )
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                       capture_output=True, text=True)
    out = r.stdout or ""
    ok = fail = 0
    errs = []
    for line in out.splitlines():
        if line.startswith("OK="):
            parts = line.split()
            ok = int(parts[0].split("=")[1]); fail = int(parts[1].split("=")[1])
        elif line.startswith("ERR "):
            errs.append(line[4:])
    try: os.remove(lst)
    except OSError: pass
    return ok, fail, errs, r.stderr[:300]

def main():
    lf, chunk = sys.argv[1], int(sys.argv[2])
    maxc = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    # CRITICAL: strip CR. Lists written with newline="\n" on Windows may carry \r,
    # and os.path.exists() then silently returns False -> "recycled=0" fake failure.
    paths = []
    for l in open(lf, encoding="utf-8"):
        p = l.strip().strip("\r").strip()
        if p:
            paths.append(p)
    exist = [p for p in paths if os.path.exists(p)]
    print("requested=%d exists=%d chunk=%d" % (len(paths), len(exist), chunk), flush=True)
    tot_ok = 0; tot_err = []
    i = 0; n = 0
    while i < len(exist):
        batch = exist[i:i+chunk]
        # skip reparse points entirely
        safe = []
        for p in batch:
            try:
                st = os.lstat(p)
                if getattr(st, "st_file_attributes", 0) & 0x400:
                    tot_err.append((p, "reparse point, skipped")); continue
                safe.append(p)
            except OSError:
                tot_err.append((p, "lstat failed"))
        if safe:
            ok, fail, errs, se = ps_recycle(safe)
            tot_ok += ok
            tot_err.extend((e.split(" :: ")[0], e.split(" :: ")[1] if " :: " in e else "err") for e in errs)
            if fail or se.strip():
                print("  chunk %d ok=%d fail=%d stderr=%s" % (n, ok, fail, se.strip()[:120]), flush=True)
        i += chunk; n += 1
        if n % 10 == 0:
            print("  progress %d/%d chunks, recycled=%d" % (n, (len(exist)+chunk-1)//chunk, tot_ok), flush=True)
        if maxc and n >= maxc: break
    print("DONE recycled=%d err=%d" % (tot_ok, len(tot_err)))
    for p, e in tot_err[:25]:
        print("  ERR", p, "|", e)
    with open(r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tierA_recycle_errors.json", "w", encoding="utf-8") as f:
        json.dump(tot_err[:2000], f, ensure_ascii=False, indent=1)

main()
