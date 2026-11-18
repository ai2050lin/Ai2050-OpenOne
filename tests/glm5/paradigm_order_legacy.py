"""Apply the original frozen native paths if old-marker signs recover."""

import argparse
from itertools import batched
from pathlib import Path
import time

import numpy as np

import paradigm_order_field as A
import paradigm_order_marker as K
import paradigm_order_native as N
from deploy.storage_policy import check_write


R,C,S,M,OUT=A.R,A.C,A.S,A.M,A.OUT
PLAN='marker-native-plan.json'
GATE='marker-native-gate.json'


def source():
    return A.sha(Path(__file__).resolve())


def freeze():
    assert not (OUT/PLAN).exists() and not (OUT/'marker-files.json').exists()
    R.write(PLAN,dict(at=C.stamp(),code=source(),paths=N.sha('candidate.json'),
        marker=N.sha(K.FILE),native=N.sha(N.PLAN),previous_native=N.sha('native-files.json'),
        protocol='Before old-marker method output is read, freeze an identical schedule, two paths, distances, dose control, B32 arms and first 64 matching families. Run only if 128 old-marker clean families reproduce the 99 percent same-sign layout 0/2/3 pattern +,-,+. Main new-marker paths and eligible criteria remain unchanged. This is a same-family sensitivity run.'))
    print('Frozen old-marker native replay',N.sha(PLAN),flush=True)


def run():
    import torch
    p=R.read(PLAN);d=R.read('design.json');c=R.read('candidate.json')
    assert p['code']==source() and p['paths']==N.sha('candidate.json') and p['marker']==N.sha(K.FILE)
    assert p['native']==N.sha(N.PLAN) and R.read(N.PLAN)['code']==N.source_id()
    assert p['previous_native']==N.sha('native-files.json')
    data=S.load('marker',N.sha(K.FILE))
    outcomes,_=A.pairs(data)
    signs=[S.X.V.summary(outcomes[:,i]) for i in range(3)]
    passed=bool(signs[0]['ci'][0]>0 and signs[1]['ci'][1]<0 and signs[2]['ci'][0]>0)
    R.write(GATE,dict(at=C.stamp(),plan=N.sha(PLAN),material=N.sha('marker-files.json'),signs=signs,passed=passed))
    print('Old marker native prerequisite',passed,flush=True)
    if not passed:
        return
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    with C.P.GpuLease(),torch.no_grad():
        t=time.monotonic();C.N.progress('legacy-order-load',0,2,t)
        assert d['model']==C.Prior.identity() and d['software']==R.versions()
        model=C.N.Engine();C.N.progress('legacy-order-load',2,2,t)
        try:
            old,_=K.joined(model.tok)
            assert M.digest(old)==R.read(K.FILE)['identity']
            weights=A.X.anchors(model,d)
            rows=old[:M.CAUSAL*6]
            groups=[rows[:48],*batched(rows[48:],N.SLAB)]
            started=time.monotonic()
            names=[]
            for i,group in enumerate(groups):
                path=f'marker-native-{i:02}.npz';receipt=f'marker-native-{i:02}.json'
                if (OUT/receipt).exists():
                    old_receipt=R.read(receipt)
                    assert old_receipt['holdout']==N.sha(PLAN) and old_receipt['sha']==N.sha(path)
                else:
                    check_write(OUT/path,20*2**20,operation='原占位组联合复位')
                    output=N.slab(model,group,c,weights,i,len(groups)*40,started)
                    R.write(receipt,dict(sha=S.artifact(path,output),design=N.sha('design.json'),
                        holdout=N.sha(PLAN),at=C.stamp(),views=len(group)))
                names.append(path)
            S.bind_label('marker-native',names,N.sha(PLAN))
        finally:
            model.close()


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','run'))
    args=cli.parse_args()
    A.setup()
    (freeze if args.action=='freeze' else run)()
