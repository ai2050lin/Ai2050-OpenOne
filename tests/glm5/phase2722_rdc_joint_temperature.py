"""Full-vocabulary confidence controls and full-coordinate KL-versus-MSE geometry."""
import gc
from rdc_joint_common import *
from rdc_joint_capture import ledger
from rdc_joint_kernels import gram_for
from rdc_relation_estimators import load_model,predict
from phase2716_rdc_relation_probability import Readout

OUT=BASE/'extension/temperature'
ROUTES=(('current','current_linear'),('temporal','embedding_bilinear'),('temporal','history_trilinear'))


def logits(rd,h,post=None):
    t=rd.torch;pred=[];native=[]
    for begin in range(0,len(h),16):
        pred.append(rd.logprob(h[begin:begin+16]))
        if post is not None:native.append(rd.native(post[begin:begin+16]))
    return t.cat(pred),t.cat(native) if post is not None else None


def newton(rd,z,q,initial=1.):
    t=rd.torch;prob=q.exp();constant=(prob*q).sum(-1).mean();qz=(prob*z).sum(-1).mean();beta=initial;path=[]
    for step in range(13):
        p=(beta*z).softmax(-1);mean=(p*z).sum(-1);gradient=mean.mean()-qz
        curvature=((p*z.square()).sum(-1)-mean.square()).mean().clamp_min(1e-12)
        loss=constant+(beta*z).logsumexp(-1).mean()-beta*qz
        path.append({'step':step,'inverse_temperature':float(beta),'KL':float(loss),'derivative':float(gradient),'curvature':float(curvature)})
        proposal=max(.05,min(5.,float(beta-gradient/curvature)))
        if abs(proposal-beta)<1e-7:break
        # Convex one-parameter objective; safe backtracking on training/validation set being fit.
        for _ in range(15):
            nextloss=constant+(proposal*z).logsumexp(-1).mean()-proposal*qz
            if nextloss<=loss+1e-6:break
            proposal=(beta+proposal)/2
        beta=proposal
    return path


def main():
    import torch
    if (OUT/'result.json').exists():return
    start=time.monotonic();guard(28*1024**2)
    immutable(OUT/'protocol.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'routes':ROUTES,
        'primary':'Train-only positive scalar inverse temperature with Newton convex optimization; validation selects among train iterates including1.',
        'stronger_simple_control':'Also fit one scalar directly to validation KL. This is explicitly validation-calibrated, not described as train-fitted or equal coefficient-selection degrees of freedom. Both comparisons evaluated without test/fresh tuning.',
        'precision':'Same full151936 native BF16 reference and FP32 predicted norm/head as2720; batch16, no sampled softmax. logprob acts as logits up to irrelevant row constant.',
        'geometry':'Raw H36 KL solution minus MSE solution decomposed into projection along MSE state and orthogonal remainder, all2560 axes. Report exact normalized change and full-vocabulary centered-logit proportionality diagnostics.',
        'limits':'Temperature cannot change argmax. Euclidean radial component is geometrical, not a recovered semantic gear. Posthoc mechanism discrimination on original held-out sets, not new blind confirmation.'})
    indices=read(BASE/'rules/indices.json');tr=np.array(indices['train']);va=np.array(indices['validation']);te=np.array(indices['test'])
    meta=read(BASE/'features/main/rows.json');freshmeta=read(BASE/'features/fresh/rows.json')
    with np.load(BASE/'features/main/targets.npz') as z:y,post=z['value'],z['postnorm']
    with np.load(BASE/'features/fresh/targets.npz') as z:fy,fp=z['value'],z['postnorm']
    with np.load(BASE/'rules/all_full_feature_grams.npz') as z:dd={k:z[k] for k in ('current_current','temporal_previous','temporal_embedding','temporal_context')}
    rd=Readout();result={}
    try:
      with torch.inference_mode():
        for scope,name in ROUTES:
            key=scope+'_'+name;offset=2560 if scope=='current' else 5120;slot=0 if scope=='current' else 1
            model=load_model(BASE/'rules'/scope/name);info=read(BASE/'rules'/scope/name/'result.json');gram=gram_for(dd,scope,name,info['mix']);allp=predict(model,gram[:,tr])[:,-2560:]
            z,q=logits(rd,allp[tr],post[tr,slot]);path=newton(rd,z,q);del z,q
            with np.load(BASE/'rules'/scope/name/'predictions.npz') as zz:vp=zz['validation'][:,-2560:];tp=zz['test'][:,-2560:]
            assert np.allclose(vp,allp[va],rtol=1e-5,atol=1e-5)
            z,q=logits(rd,vp,post[va,slot]);qprob=q.exp()
            for item in path:item['validation_KL']=float((qprob*(q-(item['inverse_temperature']*z).log_softmax(-1))).sum(-1).mean())
            train_beta=min(path,key=lambda x:x['validation_KL'])['inverse_temperature'];vpath=newton(rd,z,q);val_beta=min(vpath,key=lambda x:x['KL'])['inverse_temperature']
            save(OUT/f'{key}_temperature_fit.json',{'timestamp':stamp(),'train_path':path,'validation_only_path':vpath,'selected_train_inverse_temperature':train_beta,'selected_validation_inverse_temperature':val_beta})
            del z,q,qprob,allp,gram,model
            splitresults={}
            for split,base,prefix,ii,target,pp,mm in [('test',tp,'main',te,y[te,offset:offset+2560],post[te,slot],[meta[i] for i in te]),
                                                  ('confirmation',None,'fresh',None,fy[:,offset:offset+2560],fp[:,slot],freshmeta)]:
                if base is None:
                    with np.load(BASE/'confirmation/predictions'/f'{scope}_{name}.npz') as zz:base=zz['prediction'][:,-2560:]
                    with np.load(BASE/'confirmation/predictions'/f'{scope}_KL_{name}.npz') as zz:optimized=zz['prediction']
                else:
                    with np.load(BASE/'probability_training'/scope/name/'predictions.npz') as zz:optimized=zz['test']
                zz,qq=logits(rd,base,pp);zk,_=logits(rd,optimized);qp=qq.exp();groups=[m['source_group'] for m in mm]
                b=base.astype(float);o=optimized.astype(float);rho=(b*o).sum(1)/np.maximum((b*b).sum(1),1e-20);radial=(rho-1)[:,None]*b;tangent=o-rho[:,None]*b;delta=o-b
                assert np.allclose(np.sum(delta*delta,1),np.sum(radial*radial+tangent*tangent,1),rtol=1e-10,atol=1e-7)
                radlog,_=logits(rd,(b+radial).astype(np.float32));tanlog,_=logits(rd,(b+tangent).astype(np.float32))
                methods={'MSE':zz,'MSE_train_temperature':(train_beta*zz).log_softmax(-1),'MSE_validation_temperature':(val_beta*zz).log_softmax(-1),
                         'KL_fit':zk,'radial_only':radlog,'tangent_only':tanlog};per=[];table={}
                for method,lp in methods.items():
                    terms=qp*(qq-lp);kl=terms.sum(-1).cpu().numpy();agreement=(lp.argmax(-1)==qq.argmax(-1)).float().cpu().numpy()
                    nll=[]
                    for j,m in enumerate(mm):nll.append(-float(lp[j,m['observed_current_output_token' if scope=='current' else 'observed_temporal_output_token']]))
                    table[method]={'KL':float(kl.mean()),'group_KL':paired_summary(kl,groups),'argmax_agreement':float(agreement.mean()),'observed_token_NLL':float(np.mean(nll))}
                    if method.startswith('MSE_'):assert torch.equal(lp.argmax(-1),zz.argmax(-1))
                    npz(OUT/f'{key}_{split}_{method}_full_vocabulary.npz',mean_KL_contribution=terms.mean(0).cpu().numpy())
                    for j,m in enumerate(mm):per.append({k:m[k] for k in ('sample_id','source_group','anchor')}|{'method':method,'KL':float(kl[j]),'argmax_agreement':bool(agreement[j]),'observed_NLL':nll[j]})
                z0=zz-zz.mean(-1,keepdim=True);z1=zk-zk.mean(-1,keepdim=True);beta=(z0*z1).sum(-1)/(z0*z0).sum(-1).clamp_min(1e-20)
                residual=z1-beta[:,None]*z0;fraction=residual.square().sum(-1)/z1.square().sum(-1).clamp_min(1e-20)
                klbase=(qp*(qq-zz)).sum(-1).cpu().numpy();kltemp=(qp*(qq-methods['MSE_validation_temperature'])).sum(-1).cpu().numpy();klfit=(qp*(qq-zk)).sum(-1).cpu().numpy()
                table['paired']={'MSE_minus_KLfit':paired_summary(klbase-klfit,groups),'validation_temperature_minus_KLfit':paired_summary(kltemp-klfit,groups)}
                table['geometry']={'raw_change_radial_squared_fraction':float(np.sum(radial**2)/np.sum(delta**2)),
                    'radial_scale_nonpositive':int(np.sum(rho<=0)),'raw_scale_quantiles':np.quantile(rho,[0,.25,.5,.75,1]).tolist(),
                    'cosine_mean':float(np.mean(np.sum(b*o,1)/np.maximum(np.linalg.norm(b,axis=1)*np.linalg.norm(o,axis=1),1e-20))),
                    'per_example_best_centered_logit_proportional_residual_fraction_mean':float(fraction.mean()),
                    'scope':'Best per-example centered-logit proportional fit is explanatory only, NOT a fitted deployable temperature or an oracle performance bound.'}
                npz(BASE/'extension'/f'{key}_{split}_all_coordinate_KL_change.npz',radial_mean_square=np.mean(radial**2,0).astype(np.float32),tangent_mean_square=np.mean(tangent**2,0).astype(np.float32),
                    raw_delta_mean_square=np.mean(delta**2,0).astype(np.float32),base_MSE=np.mean((b-target)**2,0).astype(np.float32),KL_fit_MSE=np.mean((o-target)**2,0).astype(np.float32))
                compressed_json(OUT/f'{key}_{split}_rows.json.gz',per);splitresults[split]=table
                del zz,qq,zk,qp,radlog,tanlog,methods,z0,z1,beta,residual,fraction
            result[key]={'train_inverse_temperature':train_beta,'validation_inverse_temperature':val_beta,'splits':splitresults}
            print('TEMPERATURE_RESULT',key,result[key],flush=True)
    finally:rd.close()
    save(OUT/'result.json',{'timestamp':stamp(),'routes':result,'vocabulary':151936,'all_original_fits_unchanged':True});ledger('joint_full_vocabulary_temperature_and_geometry',time.monotonic()-start);guard()


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
