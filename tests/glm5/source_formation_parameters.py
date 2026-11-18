"""Original Q/K scalar-weight lineage and partial native input-coordinate sums."""
import argparse,gc,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_formation_design as D
from source_formation_behavior import Runtime
from source_formation_analysis import metric,close
from source_binding_native_v2 import unbits

def prereg():
    assert not (D.OUT/'qwen14/confirm-native-manifest.json').exists()
    assert not (D.OUT/'qwen14/lockbox_frame-native-manifest.json').exists()
    D.save('native-parameter-addendum.json',{'code_sha256':D.sha(Path(__file__)),
        'timing':'New branch after discovery QK-forecast results; before any heldout internal collection/results. Does not change primary forecast gate, candidates or GLM admission.',
        'hypothesis':'128 current-layer normalized input coordinates selected by variance times original Q/K row-block column energy predict native Q/K better than128 random coordinates.',
        'selection':'For each frozen head and Q/K branch, discovery var(normalized_input_i)*sum_j(original W_ji^2); largest128. Q final query; K five stored nonfinal header/query-cue positions pooled. Freeze before heldout.',
        'prediction':'Original Q/K rows, optional original bias, retained selected input variation plus discovery mean of other input contributions; actual original QK norm when present. No fitted regression weights.',
        'baseline':'Same128 fixed random coordinates and discovery per-position Q/K mean. Full-coordinate original projection is numerical lineage calibration only.',
        'input_boundary':'Actual same-layer input RMSNorm computed with all native hidden coordinates. This is a local original-parameter decomposition, not earlier-state forecasting or a128-input computation shortcut.',
        'report':'Q/K full-vector errors and family-bootstrap intervals, original row and input-coordinate indices, scalar-weight counts. No independent scalar-parameter intervention or semantic label claims.',
        'tolerance':.02,'evidence_status':'discovery-informed branch frozen before heldout internals; descriptive validation, no extra pass threshold'})

def verify():
    from source_formation_native import verify as v
    v();assert D.read(D.OUT/'native-parameter-addendum.json')['code_sha256']==D.sha(Path(__file__))

def fields(rt,rows,hidden,keys,queries,manifest,bi,c):
    import torch
    b=c['block'];x=torch.from_numpy(np.array(hidden[:,b])).view(torch.bfloat16)
    norm=rt.Norm(rt.dim,eps=rt.cfg.rms_norm_eps).bfloat16()
    norm.weight=torch.nn.Parameter(rt.weight(f'model.layers.{b}.input_layernorm.weight','cpu'),requires_grad=False)
    with torch.inference_mode():x=norm(x).float().numpy()
    qtarget=unbits(queries[:,bi,c['head']])[:,None]
    ktarget=[]
    for i,r in enumerate(rows):
        ix=np.array(r['positions'][:5])+manifest['length']-len(r['tokens'])
        ktarget.append(unbits(keys[i,bi,c['kv_head'],ix]))
    return {'q':(x[:,-1:],qtarget),'k':(x[:,:5],np.stack(ktarget))}

def weights(rt,c,kind):
    import torch
    b=c['block'];head=c['head'] if kind=='q' else c['kv_head'];prefix=f'model.layers.{b}.self_attn.'
    w=rt.weight(prefix+kind+'_proj.weight','cpu')[head*rt.hd:(head+1)*rt.hd].float().numpy().copy()
    key=prefix+kind+'_proj.bias'
    bias=rt.weight(key,'cpu')[head*rt.hd:(head+1)*rt.hd].float().numpy().copy() if key in rt.index else np.zeros(rt.hd,np.float32)
    normkey=prefix+kind+'_norm.weight'
    gamma=rt.weight(normkey,'cpu').float().numpy().copy() if normkey in rt.index else None
    return w,bias,gamma

def postnorm(rt,x,gamma):
    import torch
    a=torch.from_numpy(np.ascontiguousarray(x)).bfloat16()
    if gamma is not None:
        norm=rt.Norm(rt.hd,eps=rt.cfg.rms_norm_eps).bfloat16()
        norm.weight=torch.nn.Parameter(torch.from_numpy(gamma).bfloat16(),requires_grad=False)
        with torch.inference_mode():a=norm(a)
    return a.float().numpy()

def execute(model,split,fit=False):
    verify();D.admission(model,split);root=D.OUT/model;meta=D.read(root/'formation-fit.json');rows=D.read(D.OUT/f'{model}-materials.json')['splits'][split]
    manifest=D.read(root/f'{split}-native-manifest.json');rt=Runtime(model);timer=time.monotonic()
    h=np.load(root/f'{split}-native/hidden.npy',mmap_mode='r');k=np.load(root/f'{split}-native/kpre.npy',mmap_mode='r');q=np.load(root/f'{split}-native/qpre.npy',mmap_mode='r')
    params={} if fit else np.load(root/'native-parameter-fit.npz');results=[];lineage=[]
    if not fit:assert D.read(root/'native-parameter-fit.json')['fit_sha256']==D.sha(root/'native-parameter-fit.npz')
    for bi,c in enumerate(meta['heads']):
        data=fields(rt,rows,h,k,q,manifest,bi,c)
        for kind,(x,target) in data.items():
            tag=f'B{c["block"]}H{c["head"]}_{kind}';w,bias,gamma=weights(rt,c,kind);shape=x.shape
            if fit:
                flat=x.reshape(-1,rt.dim);score=flat.var(0)*np.square(w).sum(0)
                params[tag+'_selected']=np.sort(np.argsort(score,kind='stable')[-128:])
                params[tag+'_random']=np.sort(D.rng(model+'/'+tag+'/parameter_random').choice(rt.dim,128,replace=False))
                params[tag+'_input_mean']=flat.mean(0);params[tag+'_target_mean']=target.mean(0)
            baseline=params[tag+'_target_mean'].reshape(1,-1);entry={'id':tag,'methods':{}}
            mean=params[tag+'_input_mean'];meanraw=mean@w.T+bias
            for name in ['selected','random','full']:
                if name=='full':raw=x@w.T+bias
                else:
                    idx=params[tag+'_'+name];raw=meanraw+(x[...,idx]-mean[idx])@w[:,idx].T
                pred=postnorm(rt,raw,gamma)
                m=metric(rows,target.reshape(len(rows),-1),pred.reshape(len(rows),-1),baseline,f'{model}/{split}/{tag}/{name}')
                per=np.linalg.norm((pred-target).reshape(len(rows),-1),axis=1)/np.maximum(np.linalg.norm(target.reshape(len(rows),-1),axis=1),1e-8)
                m['max_relative_error']=float(per.max());entry['methods'][name]=m
            entry['calibration_passed']=entry['methods']['full']['max_relative_error']<.02
            head=c['head'] if kind=='q' else c['kv_head']
            lineage.append({'id':tag,'weight':f'model.layers.{c["block"]}.self_attn.{kind}_proj.weight','output_rows':[head*rt.hd,(head+1)*rt.hd],
                'selected_input_coordinates':params[tag+'_selected'].tolist(),'selected_scalar_weights':128*rt.hd,
                'source_mean_input_coordinates':rt.dim,'normalization_uses_full_input':True,'actual_QK_norm':gamma is not None})
            results.append(entry)
        D.log(model+'/'+split+'/native-parameters',bi+1,4,timer)
    close(h,k,q)
    if fit:
        path=root/'native-parameter-fit.npz';assert not path.exists();np.savez(path,**params)
        D.save(f'{model}/native-parameter-fit.json',{'fit_sha256':D.sha(path),'addendum_sha256':D.sha(D.OUT/'native-parameter-addendum.json'),
            'lineage':lineage,'selection_scope':'discovery only'},model)
    else:params.close()
    D.save(f'{model}/{split}-native-parameters.json',{'results':results,'calibration_passed':all(x['calibration_passed'] for x in results),
        'lineage':lineage,'scope':'Original scalar weights and current normalized hidden input; not a forecast from hs20.',
        'addendum_sha256':D.sha(D.OUT/'native-parameter-addendum.json')},model)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prereg','fit','evaluate']);ap.add_argument('--model');ap.add_argument('--split');a=ap.parse_args()
    if a.stage=='prereg':prereg()
    elif a.stage=='fit':execute(a.model,'discovery',True)
    else:execute(a.model,a.split)
