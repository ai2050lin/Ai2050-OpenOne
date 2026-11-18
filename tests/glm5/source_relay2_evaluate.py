"""Locked forecasts on complete panels; explicit future-content information accounting."""
import argparse,gc,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_relay2_design as D
import source_relay2_math as M
from source_relay2_fit import load,close
from source_binding_native_v2 import unbits
from source_relay2_behavior import Runtime
from source_formation_analysis import rotary

METHODS=['native','random','pca','embedding','aware_native','aware_pca']
MEMORY_GROUPS=['records','question','instruction','assistant_tail','other']

def metrics(rows,truth,pred,mean,G,actualp,p,key):
    se=M.error_rows(truth,pred,G);base=M.error_rows(truth,mean,G)
    kl=(actualp*(np.log(np.maximum(actualp,1e-12))-np.log(np.maximum(p,1e-12)))).sum(-1)
    masses=np.zeros((len(rows),6));source=np.zeros((len(rows),3));entity=np.zeros((len(rows),2))
    L=p.shape[-1]
    for i,r in enumerate(rows):
        for j in range(6):masses[i,j]=p[i,np.array(r['records'][str(j)]['span'])+L-len(r['tokens'])].sum()
        source[i]=masses[i].reshape(3,2).sum(1);entity[i]=masses[i].reshape(3,2).sum(0)
    return {'write_R':float(1-se.sum()/max(base.sum(),1e-12)),'write_MSE':float(se.mean()),
        'attention_KL':float(kl.mean()),'record_accuracy':float(np.mean(masses.argmax(1)==[r['target_record'] for r in rows])),
        'source_accuracy':float(np.mean(source.argmax(1)==[r['source'] for r in rows])),
        'entity_accuracy':float(np.mean(entity.argmax(1)==[r['entity'] for r in rows])),
        'write_gain_family':M.paired(rows,base-se,key)},se,kl,masses

def evaluate(split):
    import torch
    D.verify_internal();D.admission(split);root=D.OUT/D.MODEL;meta=D.read(root/'fit.json');assert D.sha(root/'fit.npz')==meta['fit_sha256']
    assert not (root/f'{split}-forecast.json').exists();fit=np.load(root/'fit.npz');manifest=D.read(root/f'{split}-native-manifest.json');assert manifest['replay']['passed']
    rows=D.read(D.OUT/'qwen14-materials.json')['splits'][split];n=len(rows);L=manifest['length'];rt=Runtime(D.MODEL);timer=time.monotonic()
    transforms={(axis,kind):M.unpack(fit,f'{axis}_transform_{kind}') for axis in ['q','k'] for kind in ['native','random','pca','embedding']}
    embed=rt.weight('model.embed_tokens.weight','cpu');tables={axis:embed[:,torch.tensor(transforms[axis,'embedding']['idx'])].float().numpy().copy() for axis in ['q','k']};del embed;gc.collect()
    h=load(split,'early');q=load(split,'qpre');k=load(split,'kpre');v=load(split,'vproj');pa=load(split,'attention');hc=load(split,'head_context')
    truth=unbits(hc);actualp=np.array(pa);close(hc,pa)
    arrays={f'{method}_{kind}':np.zeros((n,2,128),np.float32) for method in METHODS for kind in ['trueV','predV','past_trueV','past_predV']}
    arrays.update({f'{method}_p':np.zeros((n,2,L),np.float32) for method in METHODS})
    arrays.update({f'aware_native_hybrid_{group}':np.zeros((n,2,128),np.float32) for group in MEMORY_GROUPS})
    arrays['aware_native_meanV']=np.zeros((n,2,128),np.float32)
    arrays['truth']=truth;arrays['actual_p']=actualp;arrays['full_replay']=np.zeros_like(truth)
    qerr={m:np.zeros((n,2)) for m in METHODS};kerr={m:np.zeros((n,2)) for m in METHODS};verr={m:np.zeros((n,2)) for m in METHODS}
    for lo in range(0,n,32):
        hi=min(lo+32,n);rs=rows[lo:hi];x=unbits(h[lo:hi]);aq=unbits(q[lo:hi]);ak=unbits(k[lo:hi]);av=unbits(v[lo:hi]);cos,sin=rotary(rt,rs,L);cos=cos.float();sin=sin.float()
        valid=np.arange(L)[None]>=L-np.array([len(r['tokens']) for r in rs])[:,None];validt=torch.tensor(valid);ids=np.zeros((len(rs),L),np.int64)
        for i,r in enumerate(rs):ids[i,-len(r['tokens']):]=r['tokens']
        features={}
        for kind in ['native','random','pca','embedding']:
            if kind=='embedding':
                for axis,index in [('q',ids[:,-1]),('k',ids)]:
                    tr=transforms[axis,kind];features[axis,kind]=(tables[axis][index]-tr['center'][tr['idx']])/tr['scale'][tr['idx']]
            else:
                features['q',kind]=M.transform(x[:,-1],transforms['q',kind]);features['k',kind]=M.transform(x,transforms['k',kind])
        for bi in range(2):
            pp,_=M.distribution_torch(torch.tensor(aq[:,bi]),torch.tensor(ak[:,bi]),cos,sin,validt)
            arrays['full_replay'][lo:hi,bi]=np.einsum('nt,ntd->nd',pp.numpy(),av[:,bi])
        for method in METHODS:
            kind=method.removeprefix('aware_');pq=M.predict(features['q',kind],fit['q_'+method]);pk=M.predict(features['k',kind],fit['k_'+method]);pv=M.predict(features['k',kind],fit['v_'+kind])
            for bi in range(2):
                a,b=bi*128,(bi+1)*128;qhat=pq[:,a:b];khat=pk[:,:,a:b];vhat=pv[:,:,a:b]
                p,_=M.distribution_torch(torch.tensor(qhat),torch.tensor(khat),cos,sin,validt);p=p.numpy();past=p.copy();past[:,-1]=0;past/=past.sum(-1,keepdims=True)
                arrays[method+'_p'][lo:hi,bi]=p
                for label,pr,vs in [('trueV',p,av[:,bi]),('predV',p,vhat),('past_trueV',past,av[:,bi]),('past_predV',past,vhat)]:
                    arrays[method+'_'+label][lo:hi,bi]=np.einsum('nt,ntd->nd',pr,vs,optimize=True)
                qerr[method][lo:hi,bi]=np.square(aq[:,bi]-qhat).sum(-1)
                kerr[method][lo:hi,bi]=(np.square(ak[:,bi]-khat).sum(-1)*valid).sum(-1)/valid.sum(-1)
                verr[method][lo:hi,bi]=(np.square(av[:,bi]-vhat).sum(-1)*valid).sum(-1)/valid.sum(-1)
                if method=='aware_native':
                    arrays['aware_native_meanV'][lo:hi,bi]=fit['v_native'][0,bi*128:(bi+1)*128]
                    for group in MEMORY_GROUPS:
                        hybrid=vhat.copy()
                        for i,r in enumerate(rs):
                            ix=np.array(r['groups'][group],dtype=int)+L-len(r['tokens']);hybrid[i,ix]=av[i,bi,ix]
                        arrays['aware_native_hybrid_'+group][lo:hi,bi]=np.einsum('nt,ntd->nd',p,hybrid,optimize=True)
        del x,features,aq,ak,av,pq,pk,pv;gc.collect();D.log(split+'/forecast',hi,n,timer)
    close(h,q,k,v);reports=[]
    for bi,c in enumerate(D.HEADS):
        entry={**c,'methods':{},'memory_hybrids':{}};G=fit[f'G_{bi}'];mean=fit[f'mean_context_{bi}']
        rel=float(np.sqrt(M.error_rows(truth[:,bi],arrays['full_replay'][:,bi],G).sum()/max(np.einsum('ni,ij,nj->',truth[:,bi],G,truth[:,bi]),1e-12)))
        assert rel<.02,('Native attention replay',rel);entry['full_replay_relative_error']=rel
        for method in METHODS:
            p=arrays[method+'_p'][:,bi];m,se,kl,masses=metrics(rows,truth[:,bi],arrays[method+'_trueV'][:,bi],mean,G,actualp[:,bi],p,f'{split}/{bi}/{method}')
            arrays[f'{method}_SE_{bi}']=se;arrays[f'{method}_KL_{bi}']=kl;arrays[f'{method}_record_mass_{bi}']=masses
            m.update(Q_MSE=float(qerr[method][:,bi].mean()),K_MSE_per_valid_token=float(kerr[method][:,bi].mean()),V_MSE_per_valid_token=float(verr[method][:,bi].mean()))
            m['content_variants']={}
            base=M.error_rows(truth[:,bi],mean,G)
            for kind in ['predV','past_trueV','past_predV']:
                err=M.error_rows(truth[:,bi],arrays[method+'_'+kind][:,bi],G)
                m['content_variants'][kind]={'write_R':float(1-err.sum()/base.sum()),'write_MSE':float(err.mean()),'gain_vs_trueV':M.paired(rows,se-err,f'{split}/{bi}/{method}/{kind}')}
            entry['methods'][method]=m
        pred_err=M.error_rows(truth[:,bi],arrays['aware_native_predV'][:,bi],G)
        for group in MEMORY_GROUPS:
            err=M.error_rows(truth[:,bi],arrays['aware_native_hybrid_'+group][:,bi],G)
            entry['memory_hybrids'][group]={'write_MSE':float(err.mean()),'write_R':float(1-err.sum()/M.error_rows(truth[:,bi],mean,G).sum()),'gain_vs_all_predicted':M.paired(rows,pred_err-err,f'{split}/{bi}/hybrid/{group}')}
        a=entry['methods']['aware_native'];b=entry['methods']['native'];gain=arrays[f'native_SE_{bi}']-arrays[f'aware_native_SE_{bi}']
        primary=M.paired(rows,gain,f'{split}/{bi}/primary');primary.update(relative_write_gain=1-a['write_MSE']/b['write_MSE'],relative_KL_gain=1-a['attention_KL']/b['attention_KL'])
        entry['primary']=primary;reports.append(entry)
    adjusted=M.holm([e['primary']['permutation_p_one_sided'] for e in reports])
    for e,p in zip(reports,adjusted):
        e['primary']['holm_p']=p;e['passes']=bool(e['primary']['relative_write_gain']>=.05 and e['primary']['relative_KL_gain']>=.05 and e['primary']['ci95_family'][0]>0 and p<.05)
    path=root/f'{split}-forecast.npz';assert not path.exists();np.savez(path,**arrays)
    D.save(f'{D.MODEL}/{split}-forecast.json',{'heads':reports,'fit_sha256':meta['fit_sha256'],'native_manifest_sha256':D.sha(root/f'{split}-native-manifest.json'),
        'arrays_sha256':D.sha(path),'elapsed_seconds':time.monotonic()-timer,
        'scope':'Local forecasts from observed hs20. trueV uses original future content; predV uses only learned hs20 prediction. Hybrid replaces one stated past-position group with true future V, always predicts final V. No result means end-to-end closure.'})
    fit.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--split',required=True);a=ap.parse_args();evaluate(a.split)
