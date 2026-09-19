"""Format-constrained gradients, full weak-direction retention, and effect forecasts.

All gradient objectives are supervised. A separate H12-only predictor forecasts
their local effects without receiving heldout target labels as input.
"""
import argparse
import gc
from rdc_update_common import *

PARTS=('full','content','format')

def material_arrays(rows):
    xx=[];rr=[];qq=[];ee=[];means=[]
    for row in rows:
        with np.load(native_path(row)) as z:
            xx.append(unbits(z['last_x'])[-1]);rr.append(unbits(z['last_residual'])[-1]);h=unbits(z['H12_sources'])
            qq.append(h[-1]);means.append(h.mean(0));ee.append(unbits(z['embedding'])[-1])
    return np.array(xx),np.array(rr),np.array([r['target_ids'][0] for r in rows]),{'q':np.array(qq),'e':np.array(ee),'mean':np.array(means)}

def solve_ridge(gram,cross,relative):
    # Every declared direction participates; ridge continuously attenuates rather
    # than hard-thresholding eigenvectors. This is not exact orthogonal projection.
    import torch
    scale=gram.diag().mean();lam=relative*scale
    eig=torch.linalg.eigvalsh((gram+gram.T)*.5)
    assert eig.min()>-1e-8*eig.max()
    coef=torch.linalg.solve(gram+lam*torch.eye(len(gram),device=gram.device,dtype=gram.dtype),cross)
    residual=gram@coef-cross
    return coef,{'samples':len(gram),'relative_ridge':relative,'lambda':float(lam),'smallest_eigenvalue':float(eig.min()),
      'largest_eigenvalue':float(eig.max()),'linear_residual_relative':float(residual.norm()/cross.norm().clamp_min(1e-30)),
      'retained_sample_factors':len(gram),'hard_eigenvalue_cutoff':False,
      'scope':'Regularized all-factor solve, not a discovered semantic rank or guaranteed pure-content subspace.'}

def dot_factor_direction(f,direction):
    # Exact all-scalar inner product against each rank-factor gradient.
    return ((f['bg']@direction['g'])*f['x']).sum(-1)+((f['bu']@direction['u'])*f['x']).sum(-1)+((f['s']@direction['d'])*f['a']).sum(-1)

def effect_forecast(rows,features,derivatives,out):
    import torch
    from rdc_law_predict import ridge_lambda
    train=np.array([i for i,r in enumerate(rows) if r['split']=='train']);val=np.array([i for i,r in enumerate(rows) if r['split']=='validation'])
    features={k:torch.tensor(v,device='cuda') for k,v in features.items()}
    for k,v in features.items():features[k]=v/v.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-8)
    q=(features['q']@features['q'].T+features['e']@features['e'].T)/(2*2560);h=features['mean']@features['mean'].T/2560
    kernels={'query':1+q,'query_source_mean':1+q+h+q*h}
    y=torch.tensor(derivatives,device='cuda',dtype=torch.float64);center=y[train].mean(0);records=[];candidates={}
    for name,k in kernels.items():
        k=k.double();scale=k[train][:,train].diag().mean();k/=scale;gram=k[train][:,train]/len(train);eig,vec=torch.linalg.eigh((gram+gram.T)*.5)
        for df in (32,128):
            lam,actual=ridge_lambda(eig,df);coef=((vec/(eig+lam)[None,:])@vec.T)@(y[train]-center)/len(train)
            pred=k[:,train]@coef+center;loss=float((pred[val]-y[val]).square().mean())
            records.append({'kernel':name,'df':df,'lambda':lam,'validation_mse':loss});candidates[name,df]=pred
            npz(out/f'{name}_{df}.npz',coefficients=coef.cpu().numpy(),center=center.cpu().numpy(),scale=np.array(float(scale)),predictions=pred.cpu().numpy())
    selected=min(records,key=lambda r:r['validation_mse']);prediction=candidates[selected['kernel'],selected['df']]
    meanpred=center[None].expand_as(y)
    per=[]
    for split,rep in sorted({(r['split'],r['representation']) for r in rows}):
        ix=[i for i,r in enumerate(rows) if (r['split'],r['representation'])==(split,rep)]
        error=(prediction[ix]-y[ix]).square().sum(-1);baseline=(meanpred[ix]-y[ix]).square().sum(-1)
        per.append({'split':split,'representation':rep,'rows':len(ix),'relative_derivative_mse':float(error.sum()/baseline.sum().clamp_min(1e-30)),
          'target_dimension':'Every declared direction × full/content/format derivative, joint vector evaluation'})
    save(out/'frozen.json',{'timestamp':stamp(),'selection':selected,'candidates':records,'reports':per,
      'online_inputs':'Current native H12, current token embedding, mean of all visible H12 sources; no heldout answer label or target gradient input.',
      'supervision':'Training derivatives computed with training target labels. Validation target derivatives select finite predictor; heldout gradients only score forecast error after prediction.',
      'role':'Forecasts local loss effects of a fixed train-only update, not future tokens or actual parameter changes themselves.'})
    return prediction.cpu().numpy()

def freeze():
    import torch
    from transformers import AutoTokenizer
    from rdc_law_native import Tail,factor_gram,parameter_norm
    from rdc_binding_gradients import combination,normalized
    from phase2735_rdc_binding_decomposition import collect_parts,describe
    out=BASE/'learning';start=time.monotonic()
    if (out/'frozen.json').exists():return
    assert (BASE/'capture/result.json').exists();guard(2200*1024**2)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    rows=gzread(BASE/'program_material.json.gz');x,r,tt,features=material_arrays(rows)
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True);digits=[tok(str(i),add_special_tokens=False)['input_ids'][0] for i in range(1,9)]
    immutable(out/'protocol.json',{'train_EN_cases':96,'train_Python_cases':96,'train_split':'mixed-program training families only',
      'format_constraint':'All192 normalized English+Python training format gradients, ridge1e-6 or1e-3; no hard rank cutoff.',
      'directions':['mean_EN_content','content_format_constrained_1e-6','content_format_constrained_1e-3','EN_content_projected_code_ridge','mean_EN_format','mean_EN_full','rademacher'],
      'step_norms':[.02,.10],'reverse_control':'Reverse the selected1e-6 constrained direction at .02; no outcome selection.',
      'precision':'Native-valued final MLP FP32, direct conditional loss and full probability normalization FP64. Actual native BF16 norms calibrated separately before own-history deployment.',
      'forecast':'H12-only train/validation predictor frozen before any finite parameter update. Oracle Taylor derivatives kept as a separate supervised upper reference.',
      'native_norm_relative_tolerance':.005,'native_calibration':'Only weights and unit direction determine calibration, never heldout outcomes.'})
    tail=Tail();xx=torch.tensor(x,device='cuda');rr=torch.tensor(r,device='cuda');targets=torch.tensor(tt,device='cuda')
    factors,stats=collect_parts(tail,xx,rr,targets,digits);npz(out/'initial_scores.npz',**stats)
    grams={k:factor_gram(v)['total'] for k,v in factors.items()}
    cross=factor_gram(factors['content'],factors['format'])['total']
    error=float((grams['content']+grams['format']+cross+cross.T-grams['full']).abs().max()/grams['full'].abs().max())
    assert error<1e-6
    npz(out/'gram.npz',**{k:v.cpu().numpy() for k,v in grams.items()},content_format=cross.cpu().numpy())
    for part,f in factors.items():npz(out/f'{part}_factors.npz',**{k:v.float().cpu().numpy() for k,v in f.items()})
    en=np.array([i for i,r in enumerate(rows) if r['split']=='train' and r['representation']=='en']);code=np.array([i for i,r in enumerate(rows) if r['split']=='train' and r['representation']=='python'])
    fi=np.r_[en,code];mean=torch.zeros(len(rows),device='cuda',dtype=torch.float64);mean[en]=1/len(en)
    fnorm=grams['format'].diag()[fi].sqrt();cnorm=grams['content'].diag()[code].sqrt();assert torch.all(fnorm>0) and torch.all(cnorm>0)
    gf=grams['format'][fi][:,fi]/fnorm[:,None]/fnorm[None,:];fc=(cross.T@mean)[fi]/fnorm
    cc=grams['content'][code][:,code]/cnorm[:,None]/cnorm[None,:];cy=(grams['content']@mean)[code]/cnorm
    coeff=[];info=[]
    def add(name,content,format_):coeff.append((name,content,format_))
    zero=torch.zeros_like(mean);add('mean_EN_content',mean,zero)
    for ridge,name in [(1e-6,'content_format_constrained_1e-6'),(1e-3,'content_format_constrained_1e-3')]:
        c,report=solve_ridge(gf,fc,ridge);fcoef=zero.clone();fcoef[fi]=-c/fnorm;add(name,mean,fcoef);info.append({'direction':name,**report})
    c,report=solve_ridge(cc,cy,1e-6);co=zero.clone();co[code]=c/cnorm;add('EN_content_projected_code_ridge',co,zero);info.append({'direction':'EN_content_projected_code_ridge',**report})
    add('mean_EN_format',zero,mean)
    add('mean_EN_full',mean,mean)
    npz(out/'all_training_constraint_matrices.npz',normalized_format_gram=gf.cpu().numpy(),normalized_code_content_gram=cc.cpu().numpy(),
      format_norms=fnorm.cpu().numpy(),content_norms=cnorm.cpu().numpy(),format_indices=fi,content_indices=code,
      **{name+'_content_coefficients':c.cpu().numpy() for name,c,f in coeff},**{name+'_format_coefficients':f.cpu().numpy() for name,c,f in coeff})
    derivative=[];direction_meta=[]
    for name,co,fo in coeff+[('rademacher',None,None)]:
        if co is None:
            gen=torch.Generator(device='cuda').manual_seed(2737)
            dense={k:torch.randint(0,2,v.shape,generator=gen,device='cuda',dtype=torch.int8).float().mul_(2).sub_(1) for k,v in tail.w.items()}
        else:
            dense=combination(factors['content'],co)
            if torch.count_nonzero(fo):
                extra=combination(factors['format'],fo)
                for k in dense:dense[k]+=extra[k]
                del extra
            dense={k:v.float() for k,v in dense.items()}
        unit,norm=normalized(dense);del dense
        values={part:dot_factor_direction({k:v.float() for k,v in f.items()},unit).cpu().numpy() for part,f in factors.items()}
        derivative.append(np.stack([values[k] for k in PARTS],-1));npz(out/'directions'/f'{name}.npz',**{k:v.cpu().numpy() for k,v in unit.items()})
        direction_meta.append({'name':name,'unnormalized_norm':norm,'normalized_parameter_norm':float(parameter_norm(unit)),
          'mean_EN_training_content_inner_product':float(values['content'][en].mean()),
          'mean_squared_normalized_training_format_response':float(np.mean((values['format'][fi]/fnorm.cpu().numpy())**2)),
          'scope':'Fixed train-only direction in all74711040 last-MLP scalar parameters; content/format names refer to the declared1..8 digit task only.'})
        del unit;print('LEARNING_DIRECTION_FROZEN',name,flush=True)
    derivative=np.stack(derivative,1);npz(out/'oracle_derivatives.npz',derivatives=derivative)
    forecast=effect_forecast(rows,features,derivative.reshape(len(rows),-1),out/'effect_forecast')
    npz(out/'frozen_forecast.npz',derivatives=forecast.reshape(derivative.shape))
    result={'timestamp':stamp(),'source':snapshot(__file__),'rows':len(rows),'initial':describe(rows,stats),'directions':direction_meta,'regularized_solves':info,
      'gram_decomposition_error':error,'direction_order':[r['name'] for r in direction_meta],'parts':PARTS,
      'direction_sha256':{p.name:sha(p) for p in (out/'directions').glob('*.npz')},'material_sha256':sha(BASE/'program_material.json.gz'),
      'forecast_sha256':sha(out/'frozen_forecast.npz'),'status':'Frozen before finite updates and own-history outcomes','seconds':time.monotonic()-start}
    save(out/'frozen.json',result);ledger('format_constrained_directions_and_effect_forecast',result['seconds']);print('LEARNING_FREEZE_DONE',result['seconds'],flush=True)
    del tail,factors,grams;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':freeze()
