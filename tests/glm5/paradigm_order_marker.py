"""Same-family marker bridge, frozen before exposing old-marker model outputs."""

import argparse
from itertools import product
from pathlib import Path
import time

import paradigm_order_field as A
import paradigm_order_native as N


M,S,R,C,OUT=A.M,A.S,A.R,A.C,A.OUT
FILE='marker-plan.json'
OLD_A='| No entry was kept.'
OLD_R='No form is given.'


def source():
    return A.sha(Path(__file__).resolve())


def joined(tokenizer):
    new=M.build(tokenizer,'confirm')[0]
    old=[]
    with M.scope():
        M.P.EMPTY_A,M.P.EMPTY_R=OLD_A,OLD_R
        for family in M.P.rows_for(tokenizer,'confirm'):
            rows=[M.P.compile_case(tokenizer,family,15,l,a) for l,a in product(range(4),range(2))]
            M.P.verify_group(rows)
            old.extend(r for r in rows if r['layout'] in M.LAYOUTS)
    for left,right in zip(old,new,strict=True):
        assert (left['id'],left['spans'],left['query'],left['sites'])==(right['id'],right['spans'],right['query'],right['sites'])
        layout=left['layout']
        a,r=M.P.LAYOUTS[layout]
        arc=set(left['spans'][f'a{1-a}'])
        req=set(left['spans'][f'r{1-r}'])
        changes={i for i,(x,y) in enumerate(zip(left['token']['ids'],right['token']['ids'],strict=True)) if x!=y}
        assert changes<=arc|req and changes&arc and changes&req
    return old,new


def freeze():
    from transformers import AutoTokenizer

    assert not (OUT/FILE).exists()
    tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    old,new=joined(tok)
    d=R.read('design.json')
    assert M.digest((new,[]))==d['materials']['confirm']['identity']
    R.write(FILE,dict(at=C.stamp(),code=source(),design=N.sha('design.json'),
        holdout=N.sha(N.PLAN),already_exposed_confirm=N.sha('confirm-files.json'),
        snapshot=d['model'],identity=M.digest(old),old_markers=[OLD_A,OLD_R],views=len(old),
        protocol='After seeing the frozen new-marker clean cohort, replay all 128 same families at both prior equal-token placeholder texts only. No new trial selection. The old-marker outcomes for these families were unopened at this freeze. Both placeholders vary as a package; report matched layout means, old/new edge contrasts, and counts of families with the two sign transitions. This is not an independent replication of the exposed new-marker responses.'))
    print('Frozen same-family marker bridge',N.sha(FILE),flush=True)


def run():
    import torch

    plan=R.read(FILE)
    assert plan['code']==source() and plan['design']==N.sha('design.json') and plan['holdout']==N.sha(N.PLAN)
    assert plan['already_exposed_confirm']==N.sha('confirm-files.json')
    design=R.read('design.json')
    assert design['software']==R.versions() and plan['snapshot']==design['model']
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    with C.P.GpuLease(),torch.no_grad():
        tic=time.monotonic();C.N.progress('marker-load',0,2,tic)
        assert C.Prior.identity()==plan['snapshot']
        model=C.N.Engine();C.N.progress('marker-load',2,2,tic)
        try:
            rows,_=joined(model.tok)
            assert M.digest(rows)==plan['identity'] and len(rows)==plan['views']
            S.collect(model,rows,'marker',A.X.anchors(model,design),binding=N.sha(FILE))
        finally:
            model.close()


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','run'))
    args=cli.parse_args()
    A.setup()
    (freeze if args.action=='freeze' else run)()
