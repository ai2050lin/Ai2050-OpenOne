"""Requalify the unchanged physical-row questions on a larger fresh pilot."""
import os
from pathlib import Path
import sys

import clt_b26_conflict_v4 as V4
import clt_b26_conflict_v3 as D
import clt_b26_conflict as B
from source_binding_design import read,sha

B.OUT=B.HERE/'result/clt_b26_conflict_v5'
B.TMP=B.ROOT/'tests/glm5_temp/clt_b26_conflict_v5'
B.SEED=202611181415
B.SIZES['pilot']=64
B.shared.OUT=B.OUT
os.environ.update(TEMP=str(B.TMP),TMP=str(B.TMP),TMPDIR=str(B.TMP))
PRIOR=(*V4.PRIOR,'clt_b26_conflict_v4')
D.PRIOR=PRIOR


def freeze():
    earlier=B.OUT.parent/'clt_b26_conflict_v4/pilot-collection.json'
    assert not read(earlier)['passed']
    original=B.commit

    def bind(name,data):
        if name=='prereg.json':
            data['code'][Path(__file__).relative_to(B.ROOT).as_posix()]=sha(Path(__file__))
            data['previous_failures']['v4']=sha(earlier)
            data['protocol_change']+=' Unchanged wording in a separately frozen 64-family calibration with no pooling.'
        original(name,data)

    B.commit=bind
    try:V4.freeze()
    finally:B.commit=original


if __name__=='__main__':
    command=sys.argv[1]
    if command in ('pilot','discovery','confirm'):B.collect(command)
    elif command in ('freeze','select','evaluate'):{'freeze':freeze,'select':B.fit_model,'evaluate':B.evaluate}[command]()
    else:
        import clt_b26_conflict_intervene as causal
        if command=='freeze-intervention':causal.freeze()
        else:causal.run(command)
