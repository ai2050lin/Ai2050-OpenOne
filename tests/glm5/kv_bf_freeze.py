"""Serialize NumPy discovery counts without editing the frozen selector."""

import kv_bf_study as S


def json_counts(data):
    candidate,counts=FROZEN_SELECTOR(data)
    return candidate,{name:{k:int(v) for k,v in x.items()} for name,x in counts.items()}


FROZEN_SELECTOR=S.frozen_method


if __name__=="__main__":
    S.C.setup()
    S.frozen_method=json_counts
    S.freeze()
