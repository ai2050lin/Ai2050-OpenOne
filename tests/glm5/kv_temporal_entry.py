"""Run the frozen Phase 2947 with its declared D-drive reservation."""

import sys

import kv_temporal_run as C


def setup():
    C.A.OUT,C.A.TEMP,C.A.RUN=C.OUT,C.TMP,C.RUN
    C.A.R.OUT,C.A.R.TEMP,C.A.R.RUN=C.OUT,C.TMP,C.RUN
    C.P.OUT,C.P.TMP,C.P.RUN=C.OUT,C.TMP,C.RUN
    C.P.BUDGET=240*2**20
    C.P.check_write(C.TMP,C.P.BUDGET,operation=C.RUN)
    C.P.setup()
    C.B.C.OUT=C.OUT


if __name__=="__main__":
    setup()
    cmd=sys.argv[1]
    if cmd=="audit":
        from kv_temporal_audit import main
        main()
    elif cmd in ("preview","close"):
        from kv_temporal_closeout import close
        close(cmd=="preview")
    else:
        C.run(cmd)
