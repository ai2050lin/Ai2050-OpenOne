"""Freeze full-width relational and native-gate candidate predictors, then confirm."""
import argparse
from collections import Counter
from rdc_update_common import *
from rdc_update_graph import pack,kernels,save_pack,KERNELS

def reports(rows,err,den):
    out=[]
    for split,cohort in sorted({(r['split'],r['cohort']) for r in rows}):
        ix=[i for i,r in enumerate(rows) if (r['split'],r['cohort'])==(split,cohort)]
        out.append({'split':split,'cohort':cohort,'rows':len(ix),'relative_mse':float(err[ix].sum()/den[ix].sum()),
          'cluster':clustered(err[ix]/np.maximum(den[ix],1e-12),[rows[i]['source_group'] for i in ix])})
    return out

def decoders(pred,weights):
    import torch.nn.functional as F
    x=pred[:,2560:5120]
    return {'direct':pred[:,:2560],'native_x':F.linear(F.silu(F.linear(x,weights['g']))*F.linear(x,weights['u']),weights['d']),
      'native_joint':F.linear(pred[:,5120:],weights['d'])}

def freeze():
    import torch
    from rdc_law_predict import ridge_lambda
    from rdc_law_native import parameter
    out=BASE/'graph';start=time.monotonic()
    if (out/'frozen.json').exists():return
    assert read(out/'pilot.json')['passed'];guard(1500*1024**2)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    rows=gzread(PRIOR/'natural_discovery.json.gz')
    with np.load(out/'head_mapping.npz') as z:coef=torch.tensor(z['coefficients'],device='cuda',dtype=torch.float32)
    pk,targets,audit=pack(rows,coef);save_pack(out/'discovery',rows,pk,audit);ks=kernels(pk)
    npz(out/'all_discovery_kernels.npz',**{k:v.cpu().numpy() for k,v in ks.items()});del pk
    train=np.array([i for i,r in enumerate(rows) if r['split']=='train']);val=np.array([i for i,r in enumerate(rows) if r['split']=='validation'])
    ti=torch.tensor(train,device='cuda');groups=Counter(rows[i]['source_group'] for i in train)
    weight=torch.tensor([1/groups[rows[i]['source_group']] for i in train],device='cuda');weight/=weight.sum();sw=weight.sqrt()
    records=[];selected={};eig_audit={}
    for b,y in targets.items():
        yy=torch.tensor(y,device='cuda');center=weight@yy[ti];yc=yy[ti]-center;den=(yy[:,:2560]-center[:2560]).square().sum(-1)
        w={k:parameter(f'model.layers.{b}.mlp.{name}_proj.weight') for k,name in [('g','gate'),('u','up'),('d','down')]}
        candidates=[]
        for name,k in ks.items():
            scale=k[ti,ti].mean();kn=k/scale;kt=(kn[ti][:,ti]*sw[:,None]*sw[None,:]).double();eig,vec=torch.linalg.eigh((kt+kt.T)*.5)
            eig_audit[name]={'minimum':float(eig.min()),'maximum':float(eig.max())}
            assert eig.min()>-1e-6*eig.max()
            for df in (32,128):
                lam,actual_df=ridge_lambda(eig,df)
                coefficients=sw[:,None]*(((vec/(eig+lam)[None,:])@vec.T).float()@(sw[:,None]*yc))
                pred=kn[:,ti]@coefficients+center
                npz(out/'banks'/f'b{b}_{name}_{df}.npz',coefficients=coefficients.cpu().numpy(),center=center.cpu().numpy(),scale=np.array(float(scale)),train_indices=train)
                for decoder,p in decoders(pred,w).items():
                    error=(p-yy[:,:2560]).square().sum(-1).cpu().numpy();baseline=den.cpu().numpy()
                    r={'block':b,'kernel':name,'df':df,'actual_df':actual_df,'lambda':lam,'decoder':decoder,
                      'validation_relative_mse':float(error[val].sum()/baseline[val].sum()),'reports':reports(rows,error,baseline)}
                    candidates.append(r);npz(out/'errors'/f'b{b}_{name}_{df}_{decoder}.npz',squared_error=error,baseline_squared_error=baseline)
        winner=min(candidates,key=lambda r:r['validation_relative_mse']);selected[str(b)]={k:winner[k] for k in ('block','kernel','df','decoder','validation_relative_mse')}
        records.extend(candidates);print('DIRECTED_FROZEN_BLOCK',selected[str(b)],flush=True);del yy,w
    result={'timestamp':stamp(),'source':snapshot(__file__),'graph_source':snapshot(Path(__file__).with_name('rdc_update_graph.py')),
      'selected':selected,'records':records,'kernel_eigenvalue_audit':eig_audit,'train_rows':len(train),'validation_rows':len(val),'old_test_rows':128,
      'material_sha256':sha(BASE/'natural_material.json.gz'),'fit_material_sha256':sha(PRIOR/'natural_discovery.json.gz'),
      'head_mapping_sha256':sha(out/'head_mapping.npz'),'banks_sha256':{p.name:sha(p) for p in (out/'banks').glob('*.npz')},
      'online_inputs':'All visible prefix H12 coordinates and original amplitudes, current H12/token embedding, fixed source positions, frozen H12 head readout. No gold graph, target state or future token.',
      'decoder_scope':'Predict MLP directly or predict full x/activation then compile with unchanged actual native matrices. Kernel selection uses64 validation rows, never new outputs.',
      'seconds':time.monotonic()-start}
    save(out/'frozen.json',result);ledger('directed_predictor_freeze',result['seconds'])

def confirm():
    import torch
    from rdc_law_native import parameter
    out=BASE/'graph';start=time.monotonic();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    if (out/'confirmation.json').exists():return
    frozen=read(out/'frozen.json');assert sha(BASE/'natural_material.json.gz')==frozen['material_sha256']
    for name,digest in frozen['banks_sha256'].items():assert sha(out/'banks'/name)==digest
    old=gzread(PRIOR/'natural_discovery.json.gz');train=[r for r in old if r['split']=='train']
    natural=gzread(BASE/'natural_material.json.gz');rows=[]
    for r in natural:
        for a,p in enumerate(r['anchors']):rows.append(dict(r,anchors=[p],field_anchor=a,record_id=r['sample_id']+f':a{a}'))
    with np.load(out/'head_mapping.npz') as z:coef=torch.tensor(z['coefficients'],device='cuda',dtype=torch.float32)
    left,targets,audit=pack(rows,coef);right,_,_=pack(train,coef);save_pack(out/'confirmation',rows,left,audit)
    kk=kernels(left,right);npz(out/'all_confirmation_kernels.npz',**{k:v.cpu().numpy() for k,v in kk.items()});del left,right
    records=[];paired=[];selection_error={};all_errors={}
    for b,target in targets.items():
        actual=torch.tensor(target[:,:2560],device='cuda');w={k:parameter(f'model.layers.{b}.mlp.{name}_proj.weight') for k,name in [('g','gate'),('u','up'),('d','down')]}
        # All prespecified candidates reported; only old validation winner is deployed.
        for name,k in kk.items():
          for df in (32,128):
            with np.load(out/'banks'/f'b{b}_{name}_{df}.npz') as z:
                center=torch.tensor(z['center'],device='cuda');pred=k/float(z['scale'])@torch.tensor(z['coefficients'],device='cuda')+center
            den=(actual-center[:2560]).square().sum(-1).cpu().numpy()
            for decoder,p in decoders(pred,w).items():
                error=(p-actual).square().sum(-1).cpu().numpy();all_errors[b,name,df,decoder]=error
                selected=all(frozen['selected'][str(b)][key]==v for key,v in [('kernel',name),('df',df),('decoder',decoder)])
                records.append({'block':b,'kernel':name,'df':df,'decoder':decoder,'validation_selected':selected,'reports':reports(rows,error,den)})
                npz(out/'confirmation_errors'/f'b{b}_{name}_{df}_{decoder}.npz',squared_error=error,baseline_squared_error=den)
                if selected:
                    npz(out/'selected_predictions'/f'b{b}.npz',prediction=p.cpu().numpy(),actual=actual.cpu().numpy(),squared_error=error,baseline_squared_error=den)
                    selection_error[b]=error
        for left_name,right_name in [('directed_raw','shuffled_heads_raw'),('directed_query_raw','reversed_query_raw'),('directed_raw','square_raw'),('directed_raw','mean_raw'),('directed_raw','directed_rms')]:
            delta=all_errors[b,left_name,128,'direct']-all_errors[b,right_name,128,'direct']
            paired.append({'block':b,'left':left_name,'right':right_name,'df':128,'decoder':'direct','reports':reports(rows,delta,den)})
        del actual,w
    compressed(out/'confirmation_rows.json.gz',[{k:r[k] for k in ('sample_id','record_id','source_group','cohort','split','anchors','field_anchor')} for r in rows])
    result={'timestamp':stamp(),'source':snapshot(__file__),'frozen_sha256':sha(out/'frozen.json'),'windows':128,'content_boundaries':len(rows),'records':records,'paired':paired,
      'scope':'New disjoint source sentences, three nonpunctuation content boundaries per window; inference inputs never include gold dependency edges. Equal-window/document and boundary-weighted results distinguished.',
      'seconds':time.monotonic()-start}
    save(out/'confirmation.json',result);ledger('directed_new_natural_confirmation',result['seconds']);print('DIRECTED_CONFIRMATION_DONE',len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--confirm',action='store_true');args=p.parse_args();confirm() if args.confirm else freeze()
