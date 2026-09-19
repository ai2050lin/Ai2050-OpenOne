"""Reserved new-material confirmation after selection, with lossless duplicate aliasing."""
from rdc_relation_inference import *
from rdc_relation_estimators import KINDS,errors
from phase2716_rdc_relation_dynamics import TEMPORAL,temporal_dots,temporal_as_bank
from phase2716_rdc_relation_probability import Readout,references


def main():
    frozen=check_frozen();out=BASE/'confirmation';out.mkdir(parents=True,exist_ok=True)
    if not (out/'protocol.json').exists():save(out/'protocol.json',{'timestamp':stamp(),'code':snapshot(Path(__file__)),'frozen_sha':sha(BASE/'frozen.json'),'new_sources':128,'selection_after_fresh':'None',
      'identity':'Source/text/numeric-template/whole-skeleton exclusions fixed before training; official treebank test material.',
      'inference':'Full native lower-layer state conditional prediction and known-input partial-state temporal update, not autonomous generation.',
      'duplicate_policy':'Zero-mixture routes with identical model arrays alias exact current predictions; no lossy compression or omission of coordinates.'})
    trainraw,trainmeta=data(rows());raw,meta=data(rows(True),True);save(out/'rows.json',meta);train=np.array([i for i,m in enumerate(trainmeta) if m['split']=='train']);reports={};candidates=[];aliases={}
    baseline=CurrentRule('current');bp=baseline(raw['h12']);base_model=baseline.model
    for kind in KINDS:
        model=load_model(BASE/f'rules/{kind}');info=read(BASE/f'rules/{kind}/result.json')
        if kind!='full_quadratic' and info['mix']==0 and all(np.array_equal(model[k],base_model[k]) for k in model):p=bp;canonical='current'
        else:p=CurrentRule(kind)(raw['h12']);canonical=kind
        aliases[kind]=canonical
        if not (out/f'current_{canonical}.npz').exists():npz(out/f'current_{canonical}.npz',prediction=p)
        report={}
        for name,a,b in [('h23',0,2560),('h36',2560,5120)]:
            y=raw[name];tr=trainraw[name][train];e,ea=errors(y,p[:,a:b],tr,meta);report[name]=e
            if canonical==kind:npz(out/f'current_{kind}_{name}_errors.npz',**ea)
        reports['current_'+kind]={'alias':canonical,'metrics':report};candidates.append(('current',kind,p[:,2560:]))
    for kind in TEMPORAL:
        model=load_model(BASE/f'dynamics/{kind}');info=read(BASE/f'dynamics/{kind}/result.json');scales=read(BASE/'dynamics/temporal_scales.json');other={k:trainraw[k][train] for k in scales};dd,_=temporal_dots(raw,scales=scales,other=other);dots,kk=temporal_as_bank(dd,kind);p=predict(model,Bank.gram(dots,kk,info['mix']))
        npz(out/f'temporal_{kind}.npz',prediction=p);e,ea=errors(raw['next_h36'],p,trainraw['next_h36'][train],meta);npz(out/f'temporal_{kind}_errors.npz',**ea);reports['temporal_'+kind]={'metrics':e};candidates.append(('temporal',kind,p))
    # The native layer roll uses a separately frozen actual-H23-trained full cross-coordinate map.
    stage=load_model(BASE/'dynamics/layer23_36');sx=read(BASE/'dynamics/layer23_36/scale.json')['current']
    for kind in ('current','full_quadratic'):
        with np.load(out/f'current_{kind}.npz') as z:ph=z['prediction'][:,:2560]
        p=predict(stage,1+ph.astype(float)@trainraw['h23'][train].astype(float).T/sx);npz(out/f'layer_roll_{kind}.npz',prediction=p);e,ea=errors(raw['h36'],p,trainraw['h36'][train],meta);reports['layer_roll_'+kind]={'metrics':e};candidates.append(('current','layer_roll_'+kind,p))
    save(out/'state_result.json',{'timestamp':stamp(),'reports':reports,'duplicate_aliases':aliases,'choices_unchanged':frozen['choices']})
    rd=Readout();prob=[]
    try:
        refs={scope:references(meta,scope,True) for scope in ('current','temporal')}
        for scope,kind,p in candidates:
            actual,post,observed,capture=refs[scope];prob.append(rd.evaluate(p,post,observed,meta,out/'probability',kind,scope,'fresh',capture))
        for scope in refs:
            actual,post,observed,capture=refs[scope];prob.append(rd.evaluate(actual,post,observed,meta,out/'probability','actual_H36_FP32_oracle',scope,'fresh',capture))
    finally:rd.close()
    save(out/'result.json',{'timestamp':stamp(),'states':reports,'probability':prob,'frozen_sha':sha(BASE/'frozen.json'),'selected_before_confirmation':frozen['choices']});guard(32*1024**2);print('RELATION_CONFIRMATION_COMPLETE',usage(),flush=True)


if __name__=='__main__':main()
