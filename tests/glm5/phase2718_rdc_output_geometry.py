"""Exact full-vocabulary probability-loss geometry, all native normalized coordinates."""
from rdc_relation_common import *
from phase2716_rdc_relation_probability import Readout,references


def normalized_history_readout(rd):
    """Freeze-selected RMS-mean candidates: complete vocabulary, no extra state archive."""
    import hashlib
    from rdc_relation_estimators import Bank,select,predict
    out=BASE/'output_geometry/normalized_history';fits=read(BASE/'source_normalization/reconstructible_fits.json');scales=read(BASE/'source_normalization/scales.json');meta=[];x=[];message=[];target=[]
    for fresh in (False,True):
      for r in rows(fresh):
        z=load_field(r,fresh);h=unbits(z['h12']);h23=unbits(z['h23']);h36=unbits(z['h36'])
        for j,p in enumerate(r['anchors']):
            past=h[:p].copy();rms=np.sqrt(np.mean(past.astype(float)**2,axis=1));past[:]=past/np.maximum(rms[:,None],1e-8);x.append(h[p]);message.append(past.mean(0));target.append(np.concatenate([h23[p],h36[3*j]]))
            meta.append({k:r[k] for k in ('sample_id','source_group','language','genre')}|{'split':'fresh' if fresh else r['split'],'anchor':j,'position':p})
    data={'current':np.stack(x),'history_mean':np.stack(message)};y=np.stack(target);bank=Bank(data,scales={k:scales[k] for k in data});dd=bank.dots();tr=np.array([i for i,m in enumerate(meta) if m['split']=='train']);va=np.array([i for i,m in enumerate(meta) if m['split']=='validation']);reports=[]
    for name in ('source_RMS_mean','source_RMS_mean_matched_df'):
        info=fits[name];model,choice,grid=select(dd,'history_mean',y,tr,va,[(0,2560),(2560,5120)],fixed_df=info['fixed_df'],fixed_mix=info['mix']);actualsha={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in model.items()};assert actualsha==info['array_sha'],('Normalized mean reconstruction mismatch',name)
        gram=Bank.gram(dd,'history_mean',info['mix'])
        for split in ('test','fresh'):
            idx=np.array([i for i,m in enumerate(meta) if m['split']==split]);mm=[meta[i] for i in idx];p=predict(model,gram[np.ix_(idx,tr)])[:,-2560:];_,post,observed,_=references(mm,'current',split=='fresh');rr=[]
            for begin in range(0,len(p),16):
                end=min(begin+16,len(p));lp=rd.logprob(p[begin:end]);lq=rd.native(post[begin:end]);kl=(lq.exp()*(lq-lp)).sum(-1)
                for j,i in enumerate(range(begin,end)):rr.append({'sample_id':mm[i]['sample_id'],'anchor':mm[i]['anchor'],'language':mm[i]['language'],'KL':float(kl[j]),'argmax_agreement':bool(lp[j].argmax()==lq[j].argmax()),'observed_token_NLL':-float(lp[j,observed[i]])})
            report={'route':name,'split':split,'anchors':len(rr),'KL':float(np.mean([r['KL'] for r in rr])),'argmax_agreement':float(np.mean([r['argmax_agreement'] for r in rr])),'observed_token_NLL':float(np.mean([r['observed_token_NLL'] for r in rr]))};save(out/f'{name}_{split}.json',{'summary':report,'rows':rr});reports.append(report);print('NORMALIZED_HISTORY_READOUT',report,flush=True)
    save(out/'result.json',{'timestamp':stamp(),'reports':reports,'frozen_fit_sha':sha(BASE/'source_normalization/reconstructible_fits.json'),'all_coefficient_hashes_recomputed_equal':True,
      'scope':'Posthoc normalized source candidates frozen by state validation before this probability evaluation. Full vocabulary scored; predicted states and probabilities exactly reconstructible from retained fields and verified fit recipe, not vocabulary Top-K.'});return reports


def main():
    import torch
    out=BASE/'output_geometry';guard(7*1024**2)
    if (out/'result.json').exists():return
    save(out/'protocol.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'phase':2718,
      'question':'How do full-coordinate state errors compile into complete-vocabulary probability loss, and when is local Fisher curvature inadequate?',
      'objects':'Actual and estimated raw H36, each normalized with its own RMS and actual gamma. q and p use identical FP32 actual readout with TF32 off. This is a diagnostic using actual target state, never a new prediction input.',
      'coverage':'256 fresh current-quadratic anchors,256 fresh temporal-bilinear anchors, all1024 saved self-fed generation steps with native reference on exactly the same self prefix.',
      'identities':'KL(q||p)=log E_q exp(dz)-E_q dz = integral_0^1 (1-t) Var_softmax(z0+t dz)(dz) dt. A_j=dn_j [W^T integral_0^1(p_t-q)dt]_j; sum_j A_j=KL.',
      'numerics':'Full151936 vocabulary. Gauss-Legendre16,32,64 then128 if needed; errors checked against directly computed KL. All2560 coordinate attribution terms retained for fresh anchors; generation all-step profiles and four complete fixtures.',
      'additional_frozen_comparison':'Also score both already selected source-RMS-mean candidates and their matched-df control against native BF16 reference on test/fresh complete vocabulary; no readout-driven retuning.',
      'limits':'Postnorm native-coordinate straight-line path attribution, not unique causal decomposition or independently transferable semantic coordinates. Fisher expression is local approximation; exact integral is standard calculus, not a new theorem about intelligence.'})
    rd=Readout();meta=read(BASE/'confirmation/rows.json');inputs=[]
    for scope,route in [('current','full_quadratic'),('temporal','previous_embedding_bilinear')]:
        actual,_,_,_=references(meta,scope,True)
        with np.load(BASE/f'confirmation/{scope}_{route}.npz') as z:p=z['prediction'][:,-2560:]
        inputs.append((scope,p,actual,[{k:m[k] for k in ('sample_id','source_group','language','anchor')} for m in meta]))
    gp=[];gq=[];gm=[]
    for path in sorted((BASE/'generation/commits').glob('*.json')):
        r=read(path)
        with np.load(BASE/f'generation/fields/{r["sample_id"]}.npz') as z:
            gp.extend(z['self_predicted_h36']);gq.extend(unbits(z['native_h36_on_self_prefix']))
        gm.extend({'sample_id':r['sample_id'],'language':r['language'],'step':j} for j in range(len(r['self_tokens'])))
    inputs.append(('self_generation',np.stack(gp),np.stack(gq),gm));reports={}
    try:
      with torch.inference_mode():
       for scope,pred,actual,rr in inputs:
        result=[];batch_checks=[];attributions=[];step_sum=np.zeros((16,2560),float);step_abs=np.zeros_like(step_sum);step_counts=np.zeros(16,int);fixtures={};profile={k:np.zeros(2560,float) for k in ('signed','absolute','actual_raw_energy','raw_error_energy')}
        for begin in range(0,len(rr),8):
            end=min(begin+8,len(rr));h=torch.tensor(actual[begin:end],device='cuda');hp=torch.tensor(pred[begin:end],device='cuda')
            normalize=lambda x:x*torch.rsqrt(x.square().mean(-1,keepdim=True)+rd.eps)*rd.gamma
            n=normalize(h);pn=normalize(hp);dn=pn-n;z=n@rd.w.T;dz=dn@rd.w.T;lq=z.log_softmax(-1);q=lq.exp();lp=(z+dz).log_softmax(-1);direct=(q*(lq-lp)).sum(-1)
            centered=dz-(q*dz).sum(-1,keepdim=True);fisher=.5*(q*centered.square()).sum(-1);closed=torch.logsumexp(lq+centered,-1)
            assert torch.max(torch.abs(closed-direct))<2e-4
            convergence=[];attr=None
            for nodes in (16,32,64,128):
                xx,ww=np.polynomial.legendre.leggauss(nodes);integral=torch.zeros_like(q);curvature=torch.zeros(len(h),device='cuda')
                for t,w in zip((xx+1)/2,ww/2):
                    pt=(z+float(t)*centered).softmax(-1);mean=(pt*centered).sum(-1);variance=(pt*centered.square()).sum(-1)-mean.square()
                    integral+=float(w)*(pt-q);curvature+=float(w*(1-t))*variance
                aa=dn*(integral@rd.w);sumerr=torch.abs(aa.sum(-1)-direct);interr=torch.abs(curvature-direct)
                convergence.append({'nodes':nodes,'attribution_sum_max_abs_error':float(sumerr.max()),'curvature_integral_max_abs_error':float(interr.max())})
                attr=aa.float().cpu().numpy()
                if nodes>=32 and max(float(sumerr.max()),float(interr.max()))<2e-4:break
            assert max(float(sumerr.max()),float(interr.max()))<1e-3,('Quadrature failed',scope,begin,convergence)
            batch_checks.append({'begin':begin,'end':end,'convergence':convergence})
            for j,i in enumerate(range(begin,end)):
                r=rr[i];result.append({**r,'raw_H36_MSE':float((hp[j]-h[j]).square().mean()),'normalized_MSE':float(dn[j].square().mean()),'KL_same_FP32':float(direct[j]),'local_Fisher':float(fisher[j]),'exact_curvature_integral':float(curvature[j]),
                  'coordinate_attribution_sum':float(attr[j].sum(dtype=float)),'absolute_attribution_sum':float(np.abs(attr[j]).sum(dtype=float)),'quadrature_nodes':nodes,'quadrature_check_batch':len(batch_checks)-1})
                if scope=='self_generation':
                    s=r['step'];step_sum[s]+=attr[j];step_abs[s]+=np.abs(attr[j]);step_counts[s]+=1
                    if r['sample_id'] in {x['sample_id'] for x in gm[:64]}:fixtures.setdefault(r['sample_id'],[]).append(attr[j])
                else:attributions.append(attr[j])
            profile['signed']+=attr.sum(0);profile['absolute']+=np.abs(attr).sum(0);profile['actual_raw_energy']+=np.sum(actual[begin:end].astype(float)**2,axis=0);profile['raw_error_energy']+=np.sum((pred[begin:end]-actual[begin:end]).astype(float)**2,axis=0)
            if begin%128==0:print('OUTPUT_GEOMETRY',scope,begin,len(rr),flush=True)
        if attributions:npz(out/f'{scope}_all_coordinate_attributions.npz',attribution=np.stack(attributions))
        else:
            npz(out/'generation_all_coordinate_attributions.npz',signed_sum=step_sum,absolute_sum=step_abs,counts=step_counts)
            for sid,v in fixtures.items():npz(out/f'generation_fixtures/{sid}.npz',attribution=np.stack(v))
        npz(out/f'{scope}_coordinate_profiles.npz',**{k:v/len(rr) for k,v in profile.items()});save(out/f'{scope}_rows.json',result);save(out/f'{scope}_quadrature_checks.json',batch_checks)
        vals=lambda key:np.array([r[key] for r in result]);kl=vals('KL_same_FP32');fi=vals('local_Fisher');mse=vals('raw_H36_MSE')
        reports[scope]={'observations':len(rr),'mean_KL_same_FP32':float(kl.mean()),'mean_local_Fisher':float(fi.mean()),'Fisher_mean_absolute_error':float(np.mean(np.abs(kl-fi))),
          'Fisher_relative_error_median':float(np.median(np.abs(kl-fi)/np.maximum(kl,1e-8))),'raw_MSE_KL_Pearson':float(np.corrcoef(mse,kl)[0,1]),'normalized_MSE_KL_Pearson':float(np.corrcoef(vals('normalized_MSE'),kl)[0,1]),
          'max_attribution_conservation_error':float(np.max(np.abs(vals('coordinate_attribution_sum')-kl))),'max_exact_integral_error':float(np.max(np.abs(vals('exact_curvature_integral')-kl))),
          'all_coordinate_cancellation_ratio':float(np.mean(vals('absolute_attribution_sum')/np.maximum(kl,1e-8)))}
        print('OUTPUT_GEOMETRY_RESULT',scope,reports[scope],flush=True)
       normalized=normalized_history_readout(rd)
    finally:rd.close()
    save(out/'result.json',{'timestamp':stamp(),'reports':reports,'normalized_history_readout':normalized,'reference_scope':'Geometry uses identical FP32 readout on actual/predicted own-normalized states. Additional normalized-history comparison retains the previous BF16-reference KL protocol; the two references are explicitly different.',
      'interpretation':'Full-coordinate probability-error accounting and measured local-approximation limits, not extraction of a sufficient hidden state or a new mathematical law.'});guard(1024**2)


if __name__=='__main__':main()
