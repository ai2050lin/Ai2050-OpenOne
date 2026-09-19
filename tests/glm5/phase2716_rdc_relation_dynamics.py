"""Known incoming embedding updates and real full-coordinate two-stage state transfer."""
from rdc_relation_estimators import Bank,select,predict,load_model,paired_gain,splits,errors
from rdc_relation_common import *
from phase2715_rdc_prefix_relations import PrefixRelations
TEMPORAL=('previous_only','previous_embedding','previous_embedding_bilinear','previous_embedding_relation')


def embeddings(ids):
    from safetensors import safe_open
    import torch
    path=ROOT/'models/hf/qwen3-4b';key='model.embed_tokens.weight';index=read(path/'model.safetensors.index.json')['weight_map'];result={}
    with safe_open(str(path/index[key]),framework='pt',device='cpu') as f:
        for tid in sorted(set(ids)):result[tid]=f.get_slice(key)[tid:tid+1,:].float().numpy()[0].copy()
    return result


def data(material,fresh=False):
    parser=PrefixRelations();ids=[r['prompt_ids'][p+1] for r in material for p in r['anchors']];embed=embeddings(ids)
    x12=[];x23=[];x36=[];next36=[];next_embed=[];descriptor=[];meta=[]
    for r in material:
        z=load_field(r,fresh)
        for j,p in enumerate(r['anchors']):
            x12.append(unbits(z['h12'][p]));x23.append(unbits(z['h23'][p]));x36.append(unbits(z['h36'][j*3]));next36.append(unbits(z['h36'][j*3+1]));next_embed.append(embed[r['prompt_ids'][p+1]])
            descriptor.append(parser.descriptor(r['prompt_ids'][:p+2],r['language']))
            meta.append({k:r[k] for k in ('sample_id','source_group','language','genre','split')}|{'anchor':j,'position':p,'known_next_token_id':r['prompt_ids'][p+1]})
    return {'h12':np.stack(x12),'h23':np.stack(x23),'h36':np.stack(x36),'next_h36':np.stack(next36),'embedding':np.stack(next_embed),'descriptor':np.stack(descriptor)},meta


def temporal_dots(raw,train=None,scales=None,other=None):
    names=('h36','embedding','descriptor');scales={k:float(np.mean(np.sum(raw[k][train].astype(float)**2,axis=1))) for k in names} if scales is None else scales
    other=raw if other is None else other
    dots={k:raw[k].astype(float)@other[k].astype(float).T/max(scales[k],1e-15) for k in names}
    return dots,scales


def temporal_as_bank(dots,kind):
    x=dots['h36'];e=dots['embedding'];g=dots['descriptor']
    # Reuse the same validated ridge selector; each route's non-current feature is an exact kernel.
    if kind=='previous_only':return {'current':x},'current'
    extra=e if kind=='previous_embedding' else e+x*e if kind=='previous_embedding_bilinear' else e+g+e*g
    return {'current':x,'history_mean':extra},'history_mean'


def affine(x,y,train):
    mx=x[train].mean(0,dtype=np.float64);my=y[train].mean(0,dtype=np.float64);xc=x[train]-mx;yc=y[train]-my
    a=np.sum(xc*yc,axis=0)/(np.sum(xc*xc,axis=0)+1e-8);b=my-a*mx
    return {'a':a.astype(np.float32),'b':b.astype(np.float32)}


def main():
    out=BASE/'dynamics';out.mkdir(parents=True,exist_ok=True)
    if not (out/'protocol.json').exists():save(out/'protocol.json',{'timestamp':stamp(),'code':snapshot(Path(__file__)),
      'temporal_candidates':TEMPORAL,'temporal_inputs':'Previous actual raw H36 and known incoming token embedding; candidate prefix descriptor includes that known token but no later tokens or next HiddenState.',
      'temporal_boundary':'A partial-state empirical predictor. Previous H36 alone is not assumed to contain all past K/V. Self-fed generation is a separate test.',
      'layer_transfer':'Full current H23 -> H36 linear kernel fit only on train; compare teacher actual H23 with rolled H12->predictedH23, using both current and preselected quadratic first-stage routes.',
      'coordinate_control':'Independent affine map for every same-index coordinate, fit on these same320 training sources, then compose two fitted maps. No cross-layer functional equivalence assumed.',
      'selection':'Validation normalized MSE, mixing0,.1,.25,.5,1, same ridge grid. Full-vocabulary KL route selection follows before fresh model capture.'})
    raw,meta=data(rows());train,val,test=splits(meta);save(out/'rows.json',meta);dots,scales=temporal_dots(raw,train);save(out/'temporal_scales.json',scales);reports={};preds={}
    for kind in TEMPORAL:
        folder=out/kind;dd,kk=temporal_as_bank(dots,kind)
        if (folder/'result.json').exists():model=load_model(folder);best=read(folder/'result.json');grid=best['validation_grid']
        else:model,best,grid=select(dd,kk,raw['next_h36'],train,val,[(0,2560)]);npz(folder/'model.npz',**model)
        p=predict(model,Bank.gram(dd,kk,best['mix'])[:,train]);preds[kind]=p[test];report,ea=errors(raw['next_h36'][test],p[test],raw['next_h36'][train],[meta[i] for i in test])
        reports[kind]={**best,'validation_grid':grid,'test':report};save(folder/'result.json',reports[kind]);npz(folder/'predictions.npz',validation=p[val],test=p[test],validation_indices=val,test_indices=test);npz(folder/'errors.npz',**ea)
        print('TEMPORAL_RELATION_FIT',kind,best['mix'],report['relative_mse'],flush=True)
    persistence,pa=errors(raw['next_h36'][test],raw['h36'][test],raw['next_h36'][train],[meta[i] for i in test]);npz(out/'persistence_errors.npz',**pa)
    sx=float(np.mean(np.sum(raw['h23'][train].astype(float)**2,axis=1)));dd={'current':raw['h23'].astype(float)@raw['h23'].astype(float).T/sx}
    model,best,grid=select(dd,'current',raw['h36'],train,val,[(0,2560)]);npz(out/'layer23_36/model.npz',**model);save(out/'layer23_36/scale.json',{'current':sx})
    teacher=predict(model,Bank.gram(dd,'current')[:,train]);layer={'teacher':{},'rolled':{},'coordinatewise':{}}
    for label,idx in [('validation',val),('test',test)]:
        layer['teacher'][label]={'MSE':float(np.mean((teacher[idx]-raw['h36'][idx])**2))}
    npz(out/'layer23_36/teacher_predictions.npz',validation=teacher[val],test=teacher[test]);save(out/'layer23_36/result.json',{**best,'validation_grid':grid})
    for route in ('current','full_quadratic'):
        with np.load(BASE/f'rules/{route}/predictions.npz') as z:vp=z['validation'][:,:2560];tp=z['test'][:,:2560]
        vroll=predict(model,1+vp.astype(float)@raw['h23'][train].astype(float).T/sx);troll=predict(model,1+tp.astype(float)@raw['h23'][train].astype(float).T/sx)
        report,ea=errors(raw['h36'][test],troll,raw['h36'][train],[meta[i] for i in test]);layer['rolled'][route]=report
        npz(out/f'layer23_36/{route}_rolled.npz',validation=vroll,test=troll);npz(out/f'layer23_36/{route}_rolled_errors.npz',**ea)
    f1=affine(raw['h12'],raw['h23'],train);f2=affine(raw['h23'],raw['h36'],train);npz(out/'coordinatewise_affine.npz',a12_23=f1['a'],b12_23=f1['b'],a23_36=f2['a'],b23_36=f2['b'])
    p1=raw['h12']*f1['a']+f1['b'];cp=raw['h23']*f2['a']+f2['b'];cr=p1*f2['a']+f2['b']
    layer['coordinatewise']={'H12_H23_MSE':float(np.mean((p1[test]-raw['h23'][test])**2)),'H23_H36_teacher_MSE':float(np.mean((cp[test]-raw['h36'][test])**2)),'H12_H23_H36_roll_MSE':float(np.mean((cr[test]-raw['h36'][test])**2))}
    npz(out/'coordinatewise_predictions.npz',validation=cr[val],test=cr[test],teacher_test=cp[test]);save(out/'layer_result.json',layer)
    winner=min(TEMPORAL,key=lambda k:(reports[k]['validation_normalized_MSE'],TEMPORAL.index(k)))
    save(out/'result.json',{'timestamp':stamp(),'temporal_candidates':reports,'validation_MSE_winner':winner,'persistence_baseline':persistence,'layer':layer,
      'paired_temporal_test_gains':{k:paired_gain((preds['previous_only']-raw['next_h36'][test])**2,(p-raw['next_h36'][test])**2,[meta[i] for i in test]) for k,p in preds.items() if k!='previous_only'},
      'embedding_identity':'Rows directly read from unchanged actual model.embed_tokens.weight, no fitting to target or future native state.'})
    guard(20*1024**2);print('RELATION_DYNAMICS_COMPLETE',winner,usage(),flush=True)


if __name__=='__main__':main()
