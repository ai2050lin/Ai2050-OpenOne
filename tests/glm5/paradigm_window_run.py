"""Pilot and independent forecasts for a frozen residual localization experiment."""

import argparse
from contextlib import contextmanager
from itertools import batched
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from deploy.storage_policy import check_write
import paradigm_marker_field as parent
import paradigm_window_material as M
from paradigm_four_report import behavior
from tools.power_analysis import check_phase

R, C, S, F, X, P = parent.R, parent.C, parent.S, parent.F, parent.X, parent.P
OUT = R.ROOT/'tests/glm5/result/paradigm_window_20261118_v1'
PREVIOUS = parent.OUT
TEMP = R.ROOT/'tests/glm5_temp'/OUT.name
PEAK = 165*2**20
SLAB = 192
MIN_BEHAVIOR = .75
WIDTHS = ('first40', 'last40', 'last20')
FEATURES = ('b0','b8','b20','all','first40','last40','last20')
SCHEDULE = ((0, 'proximal', ()), (8, 'windows', (9,6,4,5)),
            (20, 'windows', (7,8,4,9)), (31, 'suffix', ()))


def setup():
    S.OUT,S.TMP = OUT,TEMP
    R.OUT,R.TEMP = OUT,TEMP
    R.M.RUN,R.M.SEED,R.M.WIDTH = M.RUN,M.SEED,M.WIDTH
    R.BATCH,R.SLAB = 8,SLAB
    R.configure(PEAK)


def sha(path):
    return parent.sha(path)


def sources():
    return parent.source_id() | {f'tests/glm5/{p.name}':sha(p) for p in (Path(__file__), Path(M.__file__))}


def design():
    from transformers import AutoTokenizer
    if (OUT/'design.json').exists():
        raise FileExistsError('Design is immutable')
    t = AutoTokenizer.from_pretrained(R.MODEL, local_files_only=True)
    previous = {f:sha(PREVIOUS/f) for f in ('analysis.json','candidate.json','holdout-plan.json','manifest.json')}
    copy = __import__('json').loads((PREVIOUS/'candidate.json').read_bytes())['stages']
    assert [(s['block'],s['kind'],tuple(s.get('sites',()))) for s in copy] == list(SCHEDULE)
    power = check_phase(.14,M.SIZES['confirm'],.20,.01)
    assert power['pass']
    R.write('design.json',dict(at=C.stamp(),run=M.RUN,model=C.Prior.identity(),software=R.versions(),
        provenance=R.metadata(),code=sources(),previous=previous,materials=M.inventory(t),power=power,
        anchors=F.prior_material.alphabet(t),schedule=SCHEDULE,features=FEATURES,
        method=dict(behavior_minimum=MIN_BEHAVIOR,marker_banks=M.MARKERS,archive_banks=M.ARCHIVES,
            pilot='24 families, frame A/B. Four axes paired by all 8 nuisances; academic/Python relative format margin by frame, marker, layout, archive style. Minimum .75. Reversal and natural argmax are outcomes.',
            discovery='32 families frame A/B to fit ridge. Confirm uses only independent noun/count families in previously unrun frames C/D; 128 clean, first 64 causal and first 16 for intervention generation. Frame E and 128 lockbox families only hashed.',
            changes='1111: fixed late archive with request movement 2→3 is primary; fixed early request with archive movement 0→2 is a separate falsifier. Archive code/prose pair differs only at the same slot. Same-code alternate archive payload is the natural nuisance direction.',
            window='At post-B31, analyze full common suffix, first 40, last 40, and last 20 valid tokens; end at E−1. Candidate primary compact late20. Never edit source or onset. B8 and B20 omissions evaluated factorially with B0/B31 fixed. Also compare full B31-only arm.',
            prediction='Discovery intercept by marker and ridge on actual pre-E projections of code minus prose features, then differences of layouts. B0, B31 last20, and B0/B8/B20/B31 full compared with the discovery intercept. All target logits excluded from predictors. Family bootstrap at 99%; B31 last20 can replace intercept only with positive 99% MSE gain in the primary edge.',
            intervention='Native original scale donor at identical position, permuted-direction and same-code alternate natural payload controls, exact sham and donor-to-original reinstall, plus delayed post-B31 reinstatement of clean late20 on the full joint branch. Require 98% of rows to match every token and whole stage within 3%, cosine ≥.9, edit after cast; controls may be scaled when explicitly reported. Full and late20 primary, other sizes descriptive. On both markers require at least 10% natural contrast attenuation, 99% lower bound above zero vs both controls, and retained ratio >=.70 for late20 localization, with bootstrap contrast late20−.70×full lower bound >0. Exact restoration is a separate software gate.',
            generation='Free unrestricted greedy, fixed 24 token cap or EOS, paired format/language/count/brackets with observed cap counted as unfinished. Compare clean code/prose and native full/late20 on 16 confirm families (balanced both frames, banks and markers). Small exploratory sample; intervention fork must have exact same prefill top1 as causal readout.',
            resources='Native BF16, RTX 5080 serial blocks, 8×160. Keep family/sample margins, projection and dose sharded; no full vocabulary logits on disk. Storage preflight project D only.')))
    print('Frozen design',sha(OUT/'design.json'),flush=True)


def checked_material(model, manifest, split):
    rows = M.material(model.tok,split)
    assert M.digest(rows) == manifest['materials'][split]['sha256']
    return rows


@contextmanager
def engine(manifest, stage):
    import torch
    assert manifest['code']==sources() and manifest['software']==R.versions()
    for name, expected in manifest['previous'].items():
        assert sha(PREVIOUS/name)==expected
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    with C.P.GpuLease(),torch.no_grad():
        started=time.monotonic();C.N.progress(stage+'-load',0,2,started)
        assert C.Prior.identity()==manifest['model']
        model=C.N.Engine();C.N.progress(stage+'-load',2,2,started)
        try:
            yield model, X.anchors(model,manifest)
        finally:
            model.close()


def positions(p, block):
    t, q = p['torch'], p['q'][:,0]
    _, kind, sites = next(s for s in SCHEDULE if s[0]==block)
    if kind != 'suffix':
        base = parent.positions(t,p,dict(kind=kind,sites=list(sites)))[0]
        return {FEATURES[(0,8,20).index(block)]:base}
    offset = t.arange(M.WIDTH,device='cuda')[None,:]
    start=t.tensor([r['sites'][4] for r in p['rows']],device='cuda')
    result={}
    for name,a,b in (('all',start,q),('first40',start,start+40),('last40',q-40,q),('last20',q-20,q)):
        take=(offset>=a[:,None])&(offset<b[:,None])
        assert (a>=start).all() and (b<=q).all() and (a<b).all()
        result[name]=t.where(take,offset,-1).sort(1,descending=True).values[:,:int((b-a).max())].flip(1)
    return result


def clean_pass(model,rows,weights,label,binding=None):
    t=model.t
    proj=S.contrast(model,weights)
    files=[];started=time.monotonic()
    total=(len(rows)+SLAB-1)//SLAB
    for part,group in enumerate(batched(rows,SLAB)):
        name=f'{label}-{part:02}.npz'; log=name.replace('.npz','.json')
        if (OUT/log).exists():
            done=R.read(log)
            assert done['design']==sha(OUT/'design.json') and done['holdout']==binding and done['sha']==sha(OUT/name)
        else:
            check_write(OUT/name,8*2**20)
            panels=X.panels(model,group,0)
            for p in panels:
                p['torch']=t;p['features']={}
            for layer in range(40):
                with model.block(layer) as block:
                    for p in panels:
                        h=p['x'].cuda()
                        q,k,v=model.qkv(block,h,[a.cuda() for a in p['angles']])
                        h=model.residual(block,h,model.attention(block,q,k,v,p['mask'].cuda()))
                        if layer in (0,8,20,31):
                            j=t.arange(len(h),device='cuda')[:,None]
                            for key,where in positions(p,layer).items():
                                good=where>=0
                                v=(h[j,where.clamp_min(0)].float()@proj)*good
                                p['features'][key]=(v.sum(1)/good.sum(1)).cpu().numpy()
                        p['x']=h.cpu()
                if layer%5==0 or layer==39:
                    C.N.progress(label,part*40+layer+1,total*40,started)
            batch=[]
            for p in panels:
                z=P.full_logits(model,p,p['x'].cuda())
                row=S.meta(p)
                row.update(margin=F.probe(model,p,z,weights),top=z.argmax(-1).cpu().numpy(),
                           marker=[x['marker'] for x in p['rows']],frame=[x['frame'] for x in p['rows']],
                           feature=np.stack([p['features'][k] for k in FEATURES],1))
                batch.append(row)
            data={key:np.concatenate([r[key] for r in batch]) for key in batch[0]}
            R.write(log,dict(at=C.stamp(),sha=S.artifact(name,data),design=sha(OUT/'design.json'),holdout=binding,views=len(group)))
        files.append(name)
    S.bind_label(label,files,binding)


def pilot():
    d=R.read('design.json')
    with engine(d,'window-pilot') as (model,weights):
        sweep,factors,style=checked_material(model,d,'pilot')
        natural=[row for row in sweep if row['archive']!=2]
        clean_pass(model,[*natural,*style],weights,'pilot-style')
        clean_pass(model,factors,weights,'pilot-factors')
        raw=S.load('pilot-style')
        lengths=len(natural)
        assert list(raw['ids'])==[r['id'] for r in [*natural,*style]]
        diff=(raw['margin'][:lengths,1]-raw['margin'][lengths:,1]).reshape(-1,2,3,2)
        frame=np.array([x['frame'] for x in natural[::12]])
        rates=np.stack([(diff[frame==j]>0).mean(0) for j in (0,1)])
        other=behavior(S.load('pilot-factors'),factors,model.tok)
        passed=bool((rates>=MIN_BEHAVIOR).all() and all(x['passed'] for x in other.values()))
        R.write('behavior.json',dict(at=C.stamp(),passed=passed,rates=rates.tolist(),factors=other))
        print('Behavior gate',passed,'minimum',float(rates.min()),flush=True)
        if passed:
            discovery,*_=checked_material(model,d,'discovery')
            clean_pass(model,[r for r in discovery if r['archive']!=2],weights,'discovery')


def edges(raw):
    assert (raw['layout'].reshape(-1,2,3,2)==np.array(M.LAYOUTS)[None,None,:,None]).all()
    assert (raw['archive'].reshape(-1,2,3,2)==[0,1]).all()
    df=np.diff(raw['margin'][:,1].reshape(-1,2,3,2),axis=3)[...,0]
    xf=np.diff(raw['feature'].reshape(-1,2,3,2,len(FEATURES)),axis=3)[:,:,:,0]
    return df,np.diff(df,axis=2),np.diff(xf,axis=2)


def freeze_candidate():
    import paradigm_window_native as native
    import paradigm_window_generate as gen
    assert R.read('behavior.json')['passed'] and not (OUT/'candidate.json').exists()
    _,y,x=edges(S.load('discovery'))
    indexes=dict(b0=[0],late20=[6],wide=[0,1,2,3])
    fits={name:{route:parent.fit_pack(x[:,:,edge,columns],y[:,:,edge]) for edge,route in enumerate(('archive','request'))}
          for name,columns in indexes.items()}
    R.write('candidate.json',dict(at=C.stamp(),design=sha(OUT/'design.json'),behavior=sha(OUT/'behavior.json'),
        discovery=sha(OUT/'discovery-files.json'),execution={Path(m.__file__).name:sha(Path(m.__file__)) for m in (native,gen)},
        schedule=SCHEDULE,windows=WIDTHS,arms=native.ARMS,feature_index=indexes,forecasts=fits,
        partitions=dict(causal=M.CAUSAL,generate=M.GENERATE,confirm=M.SIZES['confirm'])))
    print('Frozen candidate',sha(OUT/'candidate.json'),flush=True)


def confirm(mode):
    import paradigm_window_native as native
    import paradigm_window_generate as gen
    d=R.read('design.json');c=R.read('candidate.json');binding=sha(OUT/'candidate.json')
    assert c['design']==sha(OUT/'design.json') and c['behavior']==sha(OUT/'behavior.json')
    assert c['discovery']==sha(OUT/'discovery-files.json') and R.read('behavior.json')['passed']
    assert all(c['execution'][Path(m.__file__).name]==sha(Path(m.__file__)) for m in (native,gen))
    with engine(d,mode) as (model,weights):
        views,*_=checked_material(model,d,'confirm')
        if mode=='confirm':
            clean_pass(model,[r for r in views if r['archive']!=2],weights,'confirm',binding)
        elif mode=='native':
            assert (OUT/'confirm-files.json').exists()
            native.run(model,views,weights,binding)
        else:
            assert (OUT/'native-files.json').exists()
            gen.run(model,views,weights,binding)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','pilot','candidate','confirm','native','generate'))
    action=cli.parse_args().action
    setup()
    if action in ('confirm','native','generate'):
        confirm(action)
    else:
        {'freeze':design,'pilot':pilot,'candidate':freeze_candidate}[action]()
