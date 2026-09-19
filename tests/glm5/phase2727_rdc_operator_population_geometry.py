"""Complete output-coordinate geometry across source-held-out natural/QA cohorts.

Only the original BF16 head is loaded on CUDA, after the full-model run has ended.
The predicted object is a local metric applied to an already observed postnorm
error, not a future hidden state, token, semantic answer or extracted whole LLM.
"""
import argparse
import hashlib
from rdc_operator_common import *
from phase2727_rdc_operator_metric import protocol as metric_protocol


NAMES=('joint_global','joint_selected','output_selected_hybrid')
SCOPES=('natural_reanalysis','QA_query_transfer')
METHODS=('global_full_mean_G','condition_full_mean_G','condition_diagonal_mean_G','condition_isotropic_mean_G','condition_mixture_G')


def full_geometry(w,mean_p,mu):
    import torch
    base=torch.zeros((w.shape[1],w.shape[1]),device=w.device,dtype=w.dtype)
    for at in range(0,len(w),4096):
        ww=w[at:at+4096];base.addmm_(ww.T,mean_p[at:at+4096,None]*ww)
    mean_mu=mu.mean(0)
    average=base-mu.T@mu/len(mu)
    mixture=base-mean_mu[:,None]*mean_mu[None,:]
    covariance=(mu-mean_mu).T@(mu-mean_mu)/len(mu)
    return average,mixture,covariance


def math_test():
    import torch
    rng=torch.Generator(device='cpu').manual_seed(272727)
    w=torch.randn((37,7),generator=rng,dtype=torch.float64)
    p=torch.randn((11,37),generator=rng,dtype=torch.float64).softmax(-1);mu=p@w
    average,mixture,covariance=full_geometry(w,p.mean(0),mu)
    direct=torch.stack([w.T@(torch.diag(q)-q[:,None]*q[None,:])@w for q in p]).mean(0)
    error=float((average-direct).abs().max());assert error<1e-12
    total_error=float((mixture-average-covariance).abs().max());assert total_error<1e-12
    minima={key:float(torch.linalg.eigvalsh(value).min()) for key,value in [('average',average),('mixture',mixture),('between',covariance)]}
    assert min(minima.values())>=-1e-12
    dh=torch.randn(7,generator=rng,dtype=torch.float64);delta=w@dh
    direct_var=float(torch.stack([(q*(delta-(q*delta).sum()).square()).sum() for q in p]).mean())
    matrix_var=float(dh@average@dh);assert abs(direct_var-matrix_var)<1e-12
    save(BASE/'metric_followup/population_geometry/math_check.json',{'timestamp':stamp(),'passed':True,'source':snapshot(Path(__file__)),
        'CPU_only':True,'synthetic_shape':{'vocabulary':37,'coordinates':7,'queries':11},'seed':272727,
        'complete_individual_matrix_mean_error':error,'total_covariance_error':total_error,'full_eigenvalue_minima':minima,
        'mean_direct_variance':direct_var,'mean_matrix_variance':matrix_var,'scope':'Production full_geometry function checked against explicit individual synthetic matrices; not native LLM or language-task evidence.'})
    print('POPULATION_GEOMETRY_CPU_MATH_PASS',error,total_error,flush=True)


def protocol():
    out=BASE/'metric_followup/population_geometry';_,_,natural,qa=metric_protocol()
    jobs=[{'id':r['sample_id']+f'_a{a}','sample_id':r['sample_id'],'source_group':r['source_group'],
        'scope':'natural_reanalysis','language':r['language']} for r in natural for a in range(2)]
    jobs += [{'id':r['question_id'],'sample_id':r['sample_id'],'source_group':r['source_group'],
        'scope':'QA_query_transfer','language':r['language']} for r in qa]
    groups=sorted({r['source_group'] for r in jobs},key=lambda s:hashlib.sha256(('2727/output_metric_clusters/'+s).encode()).hexdigest())
    train=set(groups[:len(groups)//2])
    for r in jobs:r['metric_fit_split']='train' if r['source_group'] in train else 'heldout'
    p={'timestamp':stamp(),'source':snapshot(Path(__file__)),'queries':len(jobs),'query_order':jobs,
        'source_groups':len(groups),'train_source_groups':len(train),'heldout_source_groups':len(groups)-len(train),
        'split':'Global source-group SHA order salted2727/output_metric_clusters/; first half fit, second half heldout, shared by natural and QA. No article can enter both sides.',
        'fits':['all train queries pooled','natural train queries','QA train queries'],
        'formula':'G_average= W^T diag(mean(p_q)) W - mean(mu_q mu_q^T), mu_q=W^T p_q. G_mixture= W^T diag(mean(p_q)) W - mean(mu_q) mean(mu_q)^T. Their difference is the covariance of mu_q.',
        'evaluation':'Five fixed methods predict Var_p(W*dh) from each observed complete postnorm error dh. Full conditional/pooled matrices, diagonal, isotropic trace/D, and probability-mixture geometry. No method selected on heldout results.',
        'replay':'Native and three approximate postnorm vectors are passed through the original BF16head with the original single-vector execution shape. Every full-vocabulary KL and native entropy must match prior commits; real-valued W*dh variance is separately audited from actual BF16logit delta variance.',
        'full_domain':'All151936vocabulary rows and2560native coordinates. Three pairs of complete2560x2560matrices plus allquery full-coordinate dh/mu retained; no PCA/Top-K.',
        'resource':'CUDA head only, no full second model loaded, after metric_followup/result.json. Additional450MiB guard within original6GiB/6hour envelope.',
        'limits':'New heldout split only for this metric-estimation experiment. All native behaviors and the parent comparison materials were previously observed; not a new semantic/source test of the original operator. Predicting a metric given observed native/approximate states is not forecasting language. Conditional/full models have different capacities. Known Fisher/total-covariance identities are not new mathematics.'}
    if not (out/'protocol.json').exists():immutable(out/'protocol.json',p)
    original=read(out/'protocol.json');assert original['query_order']==jobs
    return out,jobs


def main():
    import torch
    from safetensors import safe_open
    out,jobs=protocol()
    if (out/'result.json').exists():return
    assert (BASE/'metric_followup/result.json').is_file()
    start=time.monotonic();guard(450*1024**2);torch.set_num_threads(2)
    free,_=torch.cuda.mem_get_info();assert free>6*1024**3
    checkpoint=ROOT/'models/hf/qwen3-4b';index=read(checkpoint/'model.safetensors.index.json')['weight_map']
    config=read(checkpoint/'config.json')
    head_key='model.embed_tokens.weight' if config['tie_word_embeddings'] else 'lm_head.weight'
    assert head_key in index
    save(out/'head_parameter_mapping.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),
        'tie_word_embeddings':config['tie_word_embeddings'],'native_head_parameter_key':head_key,
        'checkpoint_shard':index[head_key],'config_sha256':sha(checkpoint/'config.json'),
        'reason':'Follow native tied-input/output weight configuration; full-model model.lm_head results remain the replay reference.',
        'initial_failure_receipt':'initial_head_load_failure.json'})
    with safe_open(str(checkpoint/index[head_key]),framework='pt',device='cpu',backend='pread') as f:
        wb=f.get_tensor(head_key).to('cuda:0')
    assert wb.dtype==torch.bfloat16 and tuple(wb.shape)==(151936,2560)
    w=wb.double();device=w.device;vocab,width=w.shape
    sums={g:torch.zeros(vocab,device=device,dtype=torch.float64) for g in ('pooled',*SCOPES)}
    members={g:[] for g in sums};mus=[];deltas=[];targets=[];replays=[];probability_ids=[]
    with torch.inference_mode():
        for i,job in enumerate(jobs):
            commit=read(BASE/'metric_followup/commits'/f'{job["id"]}.json')
            with np.load(BASE/'metric_followup/fields'/f'{job["id"]}.npz') as z:
                native=unbits(z['native_postnorm']).astype(float)
                approximate=[unbits(z[name+'_postnorm']).astype(float) for name in NAMES]
            logits=torch.nn.functional.linear(torch.as_tensor(native,device=device,dtype=torch.bfloat16),wb).float()
            lp=logits.double().log_softmax(-1);p=lp.exp();mu=w.T@p;mus.append(mu)
            entropy=float(-(p*lp).sum());packet=[];local=[]
            for name,a in zip(NAMES,approximate):
                zz=torch.nn.functional.linear(torch.as_tensor(a,device=device,dtype=torch.bfloat16),wb).float()
                kl=float((p*(lp-zz.double().log_softmax(-1))).sum())
                old=next(r for r in commit['rows'] if r['name']==name)
                e=max(abs(kl-old['exact_KL']),abs(entropy-old['native_entropy']));assert e<1e-9,(job['id'],name,e)
                replays.append(e);dh=torch.as_tensor(a-native,device=device,dtype=torch.float64)
                d=w@dh;center=(p*d).sum();variance=float((p*(d-center).square()).sum())
                local.append({'name':name,'real_readout_variance':variance,
                    'observed_BF16_logit_variance':2*old['endpoint_Fisher_half_variance'],'actual_finite_KL':old['exact_KL']})
                packet.append(dh)
            targets.append(local);deltas.append(torch.stack(packet))
            probability_ids.append(identity(p.cpu().numpy()))
            if job['metric_fit_split']=='train':
                for group in ('pooled',job['scope']):sums[group]+=p;members[group].append(i)
            if i<2 or (i+1)%128==0:print('POPULATION_GEOMETRY_REPLAY',i+1,640,'seconds',round(time.monotonic()-start,1),flush=True)
            assert time.monotonic()-start<1800
        mu_all=torch.stack(mus);dh_all=torch.stack(deltas);geometry={};reports=[]
        for group in sums:
            ii=members[group];assert ii
            mean_p=sums[group]/len(ii);mm=mu_all[ii];mean_mu=mm.mean(0)
            average,mixture,covariance=full_geometry(w,mean_p,mm)
            identity_error=float((mixture-average-covariance).abs().max())
            assert identity_error<1e-9 and float((average-average.T).abs().max())<1e-9
            assert torch.isfinite(average).all() and torch.isfinite(mixture).all()
            assert abs(float(mean_p.sum())-1)<1e-12
            probe=dh_all[ii[0],0];pd=w@probe
            direct=float((mean_p*pd.square()).sum()-(mm@probe).square().mean())
            by_matrix=float(probe@average@probe)
            assert abs(direct-by_matrix)/max(abs(direct),1e-12)<1e-8
            geometry[group]={'average':average,'mixture':mixture}
            npz(out/f'{group}_complete_geometry.npz',G_average_full_native=average.cpu().numpy(),
                G_mixture_full_native=mixture.cpu().numpy(),mean_probability_full_vocabulary=mean_p.cpu().numpy(),
                mean_readout_vector=mean_mu.cpu().numpy())
            reports.append({'group':group,'train_queries':len(ii),'train_source_groups':len({jobs[i]['source_group'] for i in ii}),
                'trace_average_G':float(average.trace()),'trace_mixture_G':float(mixture.trace()),
                'trace_between_query_covariance':float(covariance.trace()),'total_covariance_max_absolute_error':identity_error,
                'fixed_first_train_probe_mean_direct_variance':direct,'fixed_probe_mean_matrix_variance':by_matrix})
            del covariance,mm,pd
        evaluated=[];summaries=[]
        for scope in SCOPES:
            ii=[i for i,r in enumerate(jobs) if r['scope']==scope];g=geometry[scope]['average']
            for config,name in enumerate(NAMES):
                dd=dh_all[ii,config]
                estimates={
                    'global_full_mean_G':((dd@geometry['pooled']['average'])*dd).sum(1),
                    'condition_full_mean_G':((dd@g)*dd).sum(1),
                    'condition_diagonal_mean_G':(dd.square()*g.diagonal()).sum(1),
                    'condition_isotropic_mean_G':dd.square().sum(1)*g.trace()/width,
                    'condition_mixture_G':((dd@geometry[scope]['mixture'])*dd).sum(1)}
                for method,pred in estimates.items():
                    for i,value in zip(ii,pred.cpu().numpy()):
                        target=targets[i][config]['real_readout_variance']
                        evaluated.append({**jobs[i],'configuration':name,'method':method,'real_readout_variance':target,
                            'estimated_variance':float(value),'absolute_error':float(abs(value-target)),
                            'relative_absolute_error':float(abs(value-target)/max(target,1e-12))})
        for scope in SCOPES:
            for split in ('train','heldout'):
                for name in NAMES:
                    for method in METHODS:
                        rr=[r for r in evaluated if r['scope']==scope and r['metric_fit_split']==split and r['configuration']==name and r['method']==method]
                        assert rr
                        actual=np.array([r['real_readout_variance'] for r in rr]);pred=np.array([r['estimated_variance'] for r in rr])
                        rel=[r['relative_absolute_error'] for r in rr]
                        summaries.append({'scope':scope,'metric_fit_split':split,'configuration':name,'method':method,'queries':len(rr),
                            'source_groups':len({r['source_group'] for r in rr}),'mean_real_variance':float(actual.mean()),'mean_estimated_variance':float(pred.mean()),
                            'relative_SSE':float(np.sum((pred-actual)**2)/max(np.sum(actual**2),1e-20)),
                            'relative_absolute_error_quantiles':dict(zip(('median','p90','max'),np.quantile(rel,[.5,.9,1]).tolist()))})
        pairs=[]
        for scope in SCOPES:
            for name in NAMES:
                group=[r for r in evaluated if r['scope']==scope and r['metric_fit_split']=='heldout' and r['configuration']==name]
                lookup={(r['id'],r['method']):r for r in group};ids=sorted({r['id'] for r in group})
                for baseline in ('global_full_mean_G','condition_diagonal_mean_G','condition_isotropic_mean_G','condition_mixture_G'):
                    gains=[lookup[(q,baseline)]['absolute_error']-lookup[(q,'condition_full_mean_G')]['absolute_error'] for q in ids]
                    pairs.append({'scope':scope,'configuration':name,'baseline':baseline,'candidate':'condition_full_mean_G',
                        'metric':'absolute variance-prediction error gain','anchor_mean':float(np.mean(gains)),
                        'article_cluster':clustered(gains,[lookup[(q,baseline)]['source_group'] for q in ids])})
        npz(out/'all_query_native_readout_vectors.npz',native_mu=mu_all.cpu().numpy(),observed_postnorm_deltas=dh_all.cpu().numpy())
        compressed(out/'all_metric_predictions.json.gz',evaluated)
        save(out/'result.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'queries':640,
            'query_order':jobs,'configurations':list(NAMES),'full_vocab_probability_identities':probability_ids,
            'geometry':reports,'summaries':summaries,'paired_heldout':pairs,'replay_comparisons':len(replays),
            'maximum_prior_KL_or_entropy_replay_error':max(replays),'real_vs_observed_targets':targets,
            'stored_axes':{'native_mu':'query,all2560coordinate','observed_postnorm_deltas':'query,three_configurations,all2560coordinate'},
            'native_head_only':True,'quantized':False,'limits':read(out/'protocol.json')['limits']})
    del geometry,w,wb,mu_all,dh_all,mus,deltas;torch.cuda.empty_cache()
    ledger('full_population_output_coordinate_geometry',time.monotonic()-start);guard()
    print('POPULATION_GEOMETRY_COMPLETE',len(replays),reports,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan-only',action='store_true');parser.add_argument('--math-test',action='store_true');args=parser.parse_args()
    if args.math_test:math_test()
    elif args.plan_only:
        folder,jobs=protocol();print('POPULATION_GEOMETRY_PLAN_FROZEN',len(jobs),folder,flush=True)
    else:main()
