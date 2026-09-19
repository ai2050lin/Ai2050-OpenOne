"""Weighted native reconstruction and genuinely earlier full-coordinate factor prediction."""
import gc
from rdc_conditional_common import *
from rdc_conditional_estimators import FullKernel,group_ids
SRC=CAMPAIGN/'i_factorial';OUT=CAMPAIGN/'j_predictive_gates'


def sigmoid(g):return .5*(1+np.tanh(np.asarray(g,np.float64)*.5))


def gate_fit(g,u,a,train,groups,weighted):
    b=g*u;sig=sigmoid(g);coef={};zero={}
    global_sig=sig[train].mean(0)
    for key in np.unique(groups):
        ix=train[groups[train]==key];assert len(ix)>0
        denominator=np.square(b[ix]).sum(0);mask=denominator>0
        c=global_sig.copy()
        if weighted:c[mask]=(b[ix]*a[ix]).sum(0)[mask]/denominator[mask]
        else:c=sig[ix].mean(0)
        coef[int(key)]=c;zero[int(key)]=int((~mask).sum())
    prediction=b*np.stack([coef[int(k)] for k in groups])
    return prediction,np.stack([coef[int(k)] for k in np.unique(groups)]),zero


def errors(y,p,train_energy):
    err=np.mean((np.asarray(p,np.float64)-y)**2,0);energy=np.mean(np.square(np.asarray(y,np.float64)),0)
    good=energy>0;relative=np.divide(err,energy,out=np.zeros_like(err),where=good)
    cut=np.quantile(train_energy,np.linspace(0,1,11));bins=np.searchsorted(cut[1:-1],train_energy,side='right')
    return {'mse':float(err.mean()),'total_error_energy_ratio':float(err.sum()/max(energy.sum(),1e-30)),
      'n':len(y),'units':len(err),'zero_test_energy':int((~good).sum()),
      'relative_mse_quantiles':dict(zip(('q0','q50','q90','q99','q100'),np.quantile(relative[good],[0,.5,.9,.99,1]).tolist())),
      'train_energy_deciles':[{'bin':i,'units':int((bins==i).sum()),'mse':float(err[bins==i].mean()),'error_energy_ratio':float(err[bins==i].sum()/max(energy[bins==i].sum(),1e-30))} for i in range(10) if (bins==i).any()]},dict(coordinate_mse=err,energy=energy,relative_mse=relative,nonzero_energy=good,train_energy_bin=bins)


def native(rows,layer):
    path=OUT/f'features/L{layer}.npz'
    if path.exists():
        with np.load(path) as z:return {k:z[k] for k in z.files}
    data={k:[] for k in ('gate','up','a','down','mlp_x','attention_x')}
    for i,r in enumerate(rows):
        with np.load(SRC/f'fields/{r["sample_id"]}.npz') as z:
            for k in data:data[k].append(z[f'L{layer}_{k}'])
    data={k:np.stack(v) for k,v in data.items()};npz(path,**data)
    return data


def main():
    import torch
    torch.set_num_threads(2)
    rows=read(SRC/'material.json');assert len(list((SRC/'commits').glob('*.json')))==4096
    tr,va,te=splits(rows)
    immutable(OUT/'protocol.json',{'phase':2704,'source_sha':sha(Path(__file__)),'estimator_sha':sha(ROOT/'tests/glm5/rdc_conditional_estimators.py'),'material_sha':sha(SRC/'material.json'),
      'reconstruction':'Observed same-block g/u. Ordinary mean sigmoid versus least-squares constant c=sum(B*a)/sum(B^2),B=g*u; groups global/language/familylanguage16/hash16/layoutquery16; zero denominator falls back to global mean sigmoid. Unconstrained finite-precision LS may lie outside[0,1], counts reported. No truth/answer label in gate fit.',
      'forecast':'All4096 grouped as2048/1024/1024. H12 C or UVC fullcoordinates predicts L23 gate/up/a/down, H24 C. Linear and quadratic identical input kernels. No measured L23 g/u, H24 or future norm as inputs.',
      'capacity':'g+up two heads2J, direct a headJ; therefore not equal output parameter count. Additional a+up two-head control matches2J outputbudget. Head train-RMS normalization and shared validation ridge across two heads declared, auxiliaryup ignored for a evaluation. No claim of equal information in supervision.',
      'factor_write':'Predicted g/up -> SiLU(g_hat)*up_hat -> real Wdown; compared with direct a -> Wdown and direct down prediction. Genuine forecasts use only earlier states and fixed weights, never observed future factors.',
      'error':'Every unit/coordinate retained; train-energy deciles and allunit relative errors; final predicted writes use FP64 fixed checkpoint product, compare observedBF16 and report arithmetic oracle separately.',
      'limits':['Kernel regressors are external approximations, not extracted complete native dynamics.','Two-factor prediction can amplify extrapolation error.','Query C/UVC omits full historical KV.','Equal group count is not identical effective capacity; effective degreesoffreedom reported for learned kernels.']})
    recon=[]
    for l in (11,23,35):
        raw=native(rows,l);g,u,a=[unbits(raw[k]).astype(np.float64) for k in ('gate','up','a')]
        train_energy=np.square(a[tr]).mean(0);stored={};baseline=None
        for weighted in (False,True):
          for mode in ('global','language','family_language','hash16','layout_query'):
            name=('weighted_' if weighted else 'ordinary_')+mode;group=group_ids(rows,mode)
            p,c,zero=gate_fit(g,u,a,tr,group,weighted);m,arrays=errors(a[te],p[te],train_energy)
            recon.append({'layer':l,'model':name,**m,'zero_training_denominators':zero,'coefficient_outside_unit_interval':int(((c<0)|(c>1)).sum())})
            npz(OUT/f'unit_errors/L{l}_{name}.npz',**arrays,coefficient=c,group_ids=np.unique(group),test=te)
            if name=='weighted_global':baseline=arrays['coordinate_mse']
            if name=='weighted_family_language':stored['weighted_family_mse']=arrays['coordinate_mse']
        ideal=g*sigmoid(g)*u;m,arrays=errors(a[te],ideal[te],train_energy)
        recon.append({'layer':l,'model':'observed_native_arithmetic_oracle',**m})
        npz(OUT/f'unit_errors/L{l}_arithmetic.npz',**arrays,test=te)
        stored['weighted_global_mse']=baseline;stored['family_improves']=stored['weighted_family_mse']<baseline
        npz(OUT/f'unit_errors/L{l}_comparison.npz',**stored)
        print('WEIGHTED_GATES',l,int(stored['family_improves'].sum()),flush=True)
        del raw,g,u,a,p,c,ideal;gc.collect()
    save(OUT/'reconstruction_result.json',{'phase':2704,'timestamp':stamp(),'results':recon})
    # Native prediction branch; all targets from the same frozen samples.
    raw=native(rows,23);g,u,a,down=[unbits(raw[k]) for k in ('gate','up','a','down')]
    with np.load(SRC/'features/state.npz') as z:uvc=z['uvc'];h24=unbits(z['h_c'][:,24])
    wd=checkpoint('model.layers.23.mlp.down_proj.weight').float().numpy().astype(np.float64)
    forecasts=[];units=a.shape[1];aenergy=np.square(a[tr].astype(np.float64)).mean(0);denergy=np.square(down[tr].astype(np.float64)).mean(0)
    target_blocks={'g_up':[g,u],'a_up':[a,u],'a':[a],'down':[down],'H24':[h24]}
    for input_mode in ('C','UVC'):
      x=uvc[:,0,2] if input_mode=='C' else uvc[:,0].reshape(len(rows),-1)
      for algorithm in ('linear','quadratic'):
        f=FullKernel(x,tr,va,te,algorithm);predictions={}
        for target,blocks in target_blocks.items():
            scales=[max(float(np.sqrt(np.mean(np.square(b[tr].astype(np.float64))))),1e-12) for b in blocks]
            y=np.concatenate([b/s for b,s in zip(blocks,scales)],1)
            mid=f'H12_{input_mode}_{algorithm}_{target}';p,m=f.fit(y,OUT/f'models/{mid}.npz')
            offset=0
            for b,s in zip(blocks,scales):p[:,offset:offset+b.shape[1]]*=s;offset+=b.shape[1]
            predictions[target]=p
            rawy=np.concatenate(blocks,1);trainenergy=np.square(rawy[tr].astype(np.float64)).mean(0)
            report,arr=errors(rawy[te],p,trainenergy)
            forecasts.append({'input':'H12_'+input_mode,'algorithm':algorithm,'target':target,'kind':'direct','head_scales':scales,**m,**report})
            npz(OUT/f'predictions/{mid}.npz',prediction=p,target=rawy[te],test=te,**arr)
            print('GATE_FORECAST',mid,report['mse'],flush=True)
        derived=predictions['g_up'][:,:units].astype(np.float64)*sigmoid(predictions['g_up'][:,:units])*predictions['g_up'][:,units:]
        for name,p in [('derived_from_g_up',derived),('equal_head_budget_direct_a',predictions['a_up'][:,:units])]:
            report,arr=errors(a[te],p,aenergy);forecasts.append({'input':'H12_'+input_mode,'algorithm':algorithm,'target':'a','kind':name,**report})
            npz(OUT/f'predictions/H12_{input_mode}_{algorithm}_{name}.npz',prediction=p.astype(np.float32),target=a[te],test=te,**arr)
        for name,p in [('factor_write',derived@wd.T),('direct_a_write',predictions['a'].astype(np.float64)@wd.T),('equal_budget_a_write',predictions['a_up'][:,:units].astype(np.float64)@wd.T)]:
            report,arr=errors(down[te],p,denergy);forecasts.append({'input':'H12_'+input_mode,'algorithm':algorithm,'target':'down','kind':name,**report})
            npz(OUT/f'predictions/H12_{input_mode}_{algorithm}_{name}.npz',prediction=p.astype(np.float32),target=down[te],test=te,**arr)
        del f,predictions,derived;gc.collect()
    for target,y in [('a',a),('down',down),('H24',h24)]:
        p=np.repeat(y[tr].mean(0)[None],len(te),0);report,arr=errors(y[te],p,np.square(y[tr].astype(np.float64)).mean(0))
        forecasts.append({'input':'training_only','algorithm':'mean','target':target,'kind':'baseline',**report})
    report,arr=errors(down[te],a[te].astype(np.float64)@wd.T,denergy)
    forecasts.append({'input':'observed_L23_a','algorithm':'real_Wdown','target':'down','kind':'arithmetic_oracle_not_forecast',**report})
    save(OUT/'result.json',{'phase':2704,'timestamp':stamp(),'reconstruction':recon,'forecasts':forecasts,'limits':read(OUT/'protocol.json')['limits']})
    announce('j_predictive_gates',state='analysis_complete',completed=4096,total=4096)


if __name__=='__main__':main()
