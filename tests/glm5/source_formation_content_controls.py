"""Discovery-fit routing baselines with the same true V content as local forecasts."""
import argparse,gc,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_formation_design as D
from source_formation_analysis import metric,bootstrap,close
from source_formation_behavior import Runtime
from source_binding_native_v2 import unbits

def prereg():
    assert not (D.OUT/'qwen14/confirm-formation.json').exists()
    assert not (D.OUT/'qwen14/lockbox_frame-formation.json').exists()
    D.save('content-control-addendum.json',{'code_sha256':D.sha(Path(__file__)),
        'timing':'After discovery formation results; before opening heldout mechanism results. Confirm native capture may already be running. Secondary diagnostic, does not change primary gate or GLM admission.',
        'methods':{'uniform':'Equal probability on every valid prefix token.',
            'position32':'Discovery-only attention probability density by32 normalized token-position bins; renormalized on each new prefix.',
            'equal_records':'Equal mass1/6 per record and uniform within each record token span, zero outside. Uses externally supplied record segmentation, not queried source/entity or answer.'},
        'common_inputs':'All methods use actual same-layer V and original WO, just like the primary local write forecast. Consequently these are conditional write baselines, not autonomous model surrogates.',
        'comparison':'All output coordinates; native forecast vs each baseline squared-error differences with family bootstrap. No threshold search or candidate change.'})
def verify():
    from source_formation_native import verify as v
    v();assert D.read(D.OUT/'content-control-addendum.json')['code_sha256']==D.sha(Path(__file__))
def fit(model):
    verify();D.admission(model,'discovery');root=D.OUT/model;meta=D.read(root/'formation-fit.json')
    rows=D.read(D.OUT/f'{model}-materials.json')['splits']['discovery'];m=D.read(root/'discovery-native-manifest.json')
    attention=np.load(root/'discovery-native/attention.npy',mmap_mode='r');saved={}
    for bi,c in enumerate(meta['heads']):
        mass=np.zeros(32);counts=np.zeros(32)
        for i,r in enumerate(rows):
            L=len(r['tokens']);bins=np.minimum(np.arange(L)*32//L,31)
            mass+=np.bincount(bins,weights=attention[i,bi,c['head'],-L:],minlength=32)
            counts+=np.bincount(bins,minlength=32)
        saved[f'B{c["block"]}_density']=mass/np.maximum(counts,1)
    close(attention);p=root/'content-control-fit.npz';assert not p.exists();np.savez(p,**saved)
    D.save(f'{model}/content-control-fit.json',{'fit_sha256':D.sha(p),'formation_fit_sha256':meta['fit_sha256'],
        'addendum_sha256':D.sha(D.OUT/'content-control-addendum.json'),'selection_scope':'discovery only'},model)

def evaluate(model,split):
    verify();D.admission(model,split);root=D.OUT/model;meta=D.read(root/'formation-fit.json');rows=D.read(D.OUT/f'{model}-materials.json')['splits'][split]
    params=np.load(root/'content-control-fit.npz');forecast=np.load(root/f'{split}-formation-predictions.npz');fitparams=np.load(root/'formation-fit.npz')
    assert D.read(root/'content-control-fit.json')['fit_sha256']==D.sha(root/'content-control-fit.npz')
    v=np.load(root/f'{split}-native/vproj.npy',mmap_mode='r');L=v.shape[-2];n=len(rows);rt=Runtime(model);results=[];timer=time.monotonic()
    for bi,c in enumerate(meta['heads']):
        tag=f'B{c["block"]}H{c["head"]}';b=c['block'];head=c['head']
        w=rt.weight(f'model.layers.{b}.self_attn.o_proj.weight','cpu')[:,head*rt.hd:(head+1)*rt.hd].float().numpy().copy()
        truth=forecast[tag+'_truth'];native=np.square(truth-forecast[tag+'_native']).sum(1);mean=fitparams[f'B{b}_mean_context']@w.T;entry={'id':tag,'methods':{}}
        for method in ['uniform','position32','equal_records']:
            p=np.zeros((n,L),np.float32)
            for i,r in enumerate(rows):
                size=len(r['tokens'])
                if method=='uniform':p[i,-size:]=1/size
                elif method=='position32':
                    bins=np.minimum(np.arange(size)*32//size,31);z=params[f'B{b}_density'][bins];p[i,-size:]=z/z.sum()
                else:
                    for rec in range(6):
                        ix=np.array(r['records'][str(rec)]['span'])+L-size;p[i,ix]=1/(6*len(ix))
            contexts=[]
            for lo in range(0,n,64):
                values=unbits(v[lo:lo+64,bi,c['kv_head']]);contexts.append(np.einsum('nt,ntd->nd',p[lo:lo+64],values,optimize=True))
            pred=np.concatenate(contexts)@w.T;m=metric(rows,truth,pred,mean,f'{model}/{split}/{tag}/{method}')
            gain=np.square(truth-pred).sum(1)-native
            m.update(native_MSE_gain=float(gain.mean()),native_gain_ci_family=bootstrap(rows,gain,f'{model}/{split}/{tag}/{method}/gain'))
            entry['methods'][method]=m
        results.append(entry);D.log(model+'/'+split+'/content-controls',bi+1,4,timer)
    close(v);params.close();forecast.close();fitparams.close()
    D.save(f'{model}/{split}-content-controls.json',{'results':results,'secondary_diagnostic':True,
        'scope':'Same actual V and original WO; separates routing forecast benefit from privileged content retention. Equal-record uses external segmentation.',
        'addendum_sha256':D.sha(D.OUT/'content-control-addendum.json')},model)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prereg','fit','evaluate']);ap.add_argument('--model');ap.add_argument('--split');a=ap.parse_args()
    if a.stage=='prereg':prereg()
    elif a.stage=='fit':fit(a.model)
    else:evaluate(a.model,a.split)
