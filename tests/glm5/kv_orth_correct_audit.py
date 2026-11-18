"""Separate audit dictionaries after the frozen computation shared their reference."""

from pathlib import Path

import kv_orth_audit as Original
import kv_orth_run as C


PRIMARY=set(C.ENDPOINTS[3:])
PATHS={"h18","fact","book"}
compute=Original.matched_causal


def separate(rows,z,cfg,stats):
    result=compute(rows,z,cfg,stats)
    shared=result["primary"]
    assert shared is result["cross_coefficients"] and set(shared)==PRIMARY|PATHS
    result["primary"]={k:v for k,v in shared.items() if k in PRIMARY}
    result["cross_coefficients"]={k:v for k,v in shared.items() if k in PATHS}
    return result


def main():
    C.setup();p=C.verify()
    prereg=C.A.load("audit-schema-note.json")
    assert prereg["adapter_sha"]==C.A.digest(Path(__file__))
    assert prereg["freeze"]==C.B.sha("freeze.json")
    assert (C.OUT/"audit-schema-note.json").stat().st_mtime_ns<(C.OUT/"confirm-natural.json").stat().st_mtime_ns
    if (C.OUT/"audit.json").exists() and not (C.OUT/"audit-first.json").exists():
        C.P.check_write(C.OUT/"audit-first.json").write_bytes((C.OUT/"audit.json").read_bytes())
    Original.matched_causal=separate
    Original.audit(p)
    a=C.A.load("audit.json")
    if "causal" in a:
        assert set(a["primary"])==set(C.ENDPOINTS)
        first=C.A.load("audit-first.json")
        assert all(a["primary"][k]==first["primary"][k] for k in C.ENDPOINTS)
        assert all(a["causal"]["cross_coefficients"][k]==first["causal"]["cross_coefficients"][k] for k in PATHS)
    C.A.save("audit-schema-proof.json",dict(note=C.B.sha("audit-schema-note.json"),corrected=C.B.sha("audit.json"),
        previous=C.B.sha("audit-first.json") if (C.OUT/"audit-first.json").exists() else None,statistics_unchanged=True))


if __name__=="__main__":main()
