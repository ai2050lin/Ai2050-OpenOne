"""Requalify the boundary-delimited design for the first displayed data slot only."""
from pathlib import Path
import sys

import clt_b26_three_cues_v2 as V2
import clt_b26_three_cues as B
from source_binding_design import read,sha

V2_FAMILY=V2.family
B.OUT=B.HERE/'result/clt_b26_three_cues_v3'
B.TMP=B.ROOT/'tests/glm5_temp/clt_b26_three_cues_v3'
B.SEED=202611181802
B.configure()


def first_only(tok,split,i,names,words,rng):
    rows=V2_FAMILY(tok,split,4*(i//2)+i%2,names,words,rng)
    if rows is None:return None
    for row in rows:
        assert row['first']
        row['family']=i;row['id']=f'{split}-{i:03}-{row["cell"]:02}'
    return rows


def freeze():
    prior=B.OUT.parent/'clt_b26_three_cues_v2/pilot-collection.json'
    assert not read(prior)['passed']
    original=B.common.commit; V2.family=first_only
    def bind(name,data):
        if name=='prereg.json':
            data['code'][Path(__file__).relative_to(B.ROOT).as_posix()]=sha(Path(__file__))
            data['second_pilot']=sha(prior)
            data['protocol_change']='Scope restricted to the first displayed data line below BEGIN. Outer records still exchange intact and remain distinct from fixed identity and the two independently swapped label records. No old family reused. Same technical, behavior and predictive gates. Conclusions exclude the last slot.'
        original(name,data)
    B.common.commit=bind
    try:V2.freeze()
    finally:B.common.commit=original


if __name__=='__main__':
    stage=sys.argv[1]
    if stage in ('pilot','discovery','confirm'):B.collect(stage)
    elif stage in ('freeze','select','evaluate'):{'freeze':freeze,'select':B.select,'evaluate':B.evaluate}[stage]()
    else:
        import clt_b26_three_cues_intervene as interventions
        interventions.freeze() if stage=='freeze-causal' else interventions.run(stage)
