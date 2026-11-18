"""Retest the unpassed pilot with one explicit, previously validated answer instruction."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import paradigm_window_run as W
from paradigm_four_material import GUIDE

M=W.M
ORIGINAL=W.OUT
ORIGINAL_SOURCES=W.sources
RUN='PARADIGM-WINDOW-20261118-v2'
SEED=29782027
NOUNS=dict(M.NOUNS,pilot='sleighbells uddercups geodes snowshoes buckeyes broadswords beanbags tabletopper',
           discovery='windsocks snowglobes bookends doormats lunchboxes napkins mandolins saladbowls shawls viaducts balusters pantographs')


@contextmanager
def configured():
    assert not json.loads((ORIGINAL/'behavior.json').read_bytes())['passed']
    for split in ('pilot','discovery'):
        assert not set(M.NOUNS[split].split()).intersection(NOUNS[split].split())
    M.RUN,M.SEED,M.NOUNS=RUN,SEED,NOUNS
    M.FRAMES=tuple((*frame[:-1],' brackets.\n'+GUIDE) for frame in M.FRAMES)
    W.OUT=ORIGINAL.with_name('paradigm_window_20261118_v2')
    W.TEMP=W.TEMP.with_name(W.OUT.name)
    W.sources=lambda:ORIGINAL_SOURCES()|{
        f'tests/glm5/{Path(__file__).name}':W.sha(Path(__file__)),
        'tests/glm5/result/paradigm_window_20261118_v1/behavior.json':W.sha(ORIGINAL/'behavior.json'),
        'tests/glm5/result/paradigm_window_20261118_v1/design.json':W.sha(ORIGINAL/'design.json')}
    W.setup()
    yield


def smoke():
    import time
    import paradigm_window_native as native
    import paradigm_window_generate as gen
    d=W.R.read('design.json')
    assert W.R.read('behavior.json')['passed']
    with W.engine(d,'window-retest-smoke') as (model,weights):
        views,*_=W.checked_material(model,d,'discovery')
        rows=native.material(views,2,(2,3))[:8]
        started=time.monotonic()
        result,_=native.rollout(model,rows,weights,native.ARMS,
            lambda n:W.C.N.progress('window-native-smoke',n,40,started))
        started=time.monotonic()
        both,panels=native.rollout(model,rows,weights,native.GEN_ARMS,
            lambda n:W.C.N.progress('window-generation-smoke-prefill',n,40,started),cache=True)
        for arm in native.GEN_ARMS:
            assert (both['top'][:,native.GEN_ARMS.index(arm)]==result['top'][:,native.ARMS.index(arm)]).all()
        started=time.monotonic()
        generated=gen.decode(model,panels,lambda step:W.C.N.progress('window-generation-smoke-decode',step,gen.CAP,started))
        W.R.write('smoke.json',dict(at=W.C.stamp(),source='discovery-only',cases=len(generated),
                                   code={Path(m.__file__).name:W.sha(Path(m.__file__)) for m in (native,gen)},
                                   cache_checks=[x['check'] for x in generated]))
        print('Technical discovery-only smoke passed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','pilot','smoke','candidate','confirm','native','generate'))
    action=parser.parse_args().action
    with configured():
        if action in ('confirm','native','generate'):
            W.confirm(action)
        else:
            {'freeze':W.design,'pilot':W.pilot,'smoke':smoke,'candidate':W.freeze_candidate}[action]()
