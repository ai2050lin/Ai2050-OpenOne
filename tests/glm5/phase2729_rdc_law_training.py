"""Paired coherent/order-control native-MLP training trajectories and checkpoints."""
import gc
from collections import defaultdict
from rdc_law_common import *
from rdc_law_native import Tail, dense_gradient, parameter_norm, factor_gram
from phase2728_rdc_law_formation import protocol, panel_arrays


def training_arrays(views,controlled=False):
    arrays={'x':[],'residual':[],'targets':[]};meta=[]
    for row in views:
        if row['split']!='train':continue
        sid=row['sample_id'];pos=row['training_positions']
        path=BASE/('formation/controlled_capture/fields' if controlled else 'capture/main/sources')/f'{sid}.npz'
        with np.load(path) as z:
            for k in ('x','residual'):arrays[k].append(unbits(z[k] if controlled else z[k][pos]))
        arrays['targets'].extend(row['target_ids'])
        meta.extend({'sample_id':sid,'source_group':row['source_group'],'cohort':row['cohort'],'position':p,'target_id':target}
                    for p,target in zip(pos,row['target_ids']))
    return {k:np.asarray(v,dtype=np.int64) if k=='targets' else np.concatenate(v) for k,v in arrays.items()},meta


def evaluate(tail,x,residual,targets,reference_lp,cohorts):
    import torch
    result=defaultdict(list);factors=defaultdict(list)
    with torch.no_grad():
        for at in range(0,len(x),16):
            z=tail.forward(x[at:at+16],residual[at:at+16],targets[at:at+16],True)
            for k in ('g','u','activation','m','postnorm','loss','entropy','argmax'):result[k].append(z[k].cpu().numpy())
            kl=(reference_lp[at:at+16].exp()*(reference_lp[at:at+16]-z['logprobs'])).sum(-1)
            result['KL_initial_FP32'].append(kl.cpu().numpy())
            for k,v in z['factors'].items():factors[k].append(v.double())
            del z
        f={k:torch.cat(v) for k,v in factors.items()}
        grams=factor_gram(f)
        packet={k:np.concatenate(v) for k,v in result.items()}
        packet.update({'gradient_gram_'+k:v.cpu().numpy() for k,v in grams.items()})
    phi=packet['g']/(1+np.exp(-np.clip(packet['g'],-80,80)))
    moments=[]
    for c in ('gum','ewt','cmrc'):
        ix=np.array([i for i,cohort in enumerate(cohorts) if cohort==c]);u=packet['u'][ix].astype(float);a=phi[ix].astype(float)
        moments.append(np.stack([a.mean(0),u.mean(0),(a*u).mean(0),((a-a.mean(0))*(u-u.mean(0))).mean(0)]))
    packet['condition_unit_moments']=np.stack(moments)
    return packet


def main():
    import torch
    p=protocol();out=BASE/'formation/trajectories'
    if (out/'result.json').exists():return
    assert (BASE/'formation/initial_stable/result.json').exists() and (BASE/'formation/controlled_capture/result.json').exists()
    start=time.monotonic();guard(1900*1024**2)
    coherent,meta=training_arrays(p['controlled_views']);control,controlmeta=training_arrays(p['controlled_views'],True)
    assert meta==controlmeta and np.array_equal(coherent['targets'],control['targets']) and len(meta)==1536
    xa,ra,_,_=panel_arrays(p['panel']);tail=Tail();w=tail.w;device=tail.device
    x=torch.as_tensor(xa,device=device);residual=torch.as_tensor(ra,device=device)
    targets=torch.tensor([r['target_id'] for r in p['panel']],device=device)
    cohorts=[r['cohort'] for r in p['panel']]
    datasets={name:{k:torch.as_tensor(v,device=device) for k,v in data.items()} for name,data in [('coherent',coherent),('prefix_order_control',control)]}
    original={k:v.clone() for k,v in w.items()};initial_norm=float(parameter_norm(w))
    with torch.no_grad():
        reference_lp=torch.cat([tail.forward(x[at:at+16],residual[at:at+16])['logprobs'] for at in range(0,len(x),16)])
        initial_packet=evaluate(tail,x,residual,targets,reference_lp,cohorts)
        npz(out/'initial_panel.npz',**initial_packet)
    reports=[];run_manifest=[]
    for seed in p['multistep_seeds']:
      rng=np.random.default_rng(seed)
      draws=rng.integers(len(meta),size=(p['multistep_steps'],p['multistep_batch_size']))
      npz(out/f'seed{seed}_frozen_draws.npz',indices=draws)
      for condition in ('coherent','prefix_order_control'):
        name=f'{condition}_seed{seed}';folder=out/name
        if (folder/'result.json').exists():
            run_manifest.append(read(folder/'result.json'));continue
        for k in w:
            with torch.no_grad():w[k].copy_(original[k])
        data=datasets[condition];trace=[];checkpoints=[]
        t0=time.monotonic()
        with torch.no_grad():
          for step,ix in enumerate(draws,1):
            z=tail.forward(data['x'][ix],data['residual'][ix],data['targets'][ix],True)
            grad=dense_gradient(z['factors']);gnorm=float(parameter_norm(grad))
            clip=min(1.,p['multistep_global_gradient_clip']/max(gnorm,1e-30))
            step_loss=float(z['loss'].mean());del z
            for k in w:w[k].add_(grad[k],alpha=-p['multistep_learning_rate']*clip)
            dnorm=float(torch.stack([(w[k]-original[k]).square().sum() for k in w]).sum().sqrt())
            trace.append({'step':step,'batch_loss_before':step_loss,'raw_gradient_norm':gnorm,'clip_factor':clip,
                'learning_rate':p['multistep_learning_rate'],'actual_parameter_displacement_norm':dnorm,
                'relative_parameter_displacement':dnorm/initial_norm})
            del grad
            if step in p['multistep_checkpoints']:
                packet=evaluate(tail,x,residual,targets,reference_lp,cohorts)
                path=folder/f'checkpoint{step:03d}_panel.npz';npz(path,**packet)
                summary={'seed':seed,'condition':condition,'step':step,'panel_file':str(path.relative_to(BASE)),'field_sha':sha(path),
                    'relative_parameter_displacement':dnorm/initial_norm,'strata':[]}
                for split in ('train','validation','test'):
                  for cohort in ('gum','ewt','cmrc'):
                    at=np.array([i for i,r in enumerate(p['panel']) if r['split']==split and r['cohort']==cohort])
                    delta=packet['loss'][at]-initial_packet['loss'][at]
                    summary['strata'].append({'split':split,'cohort':cohort,'queries':len(at),
                        'initial_loss':float(initial_packet['loss'][at].mean()),'current_loss':float(packet['loss'][at].mean()),
                        'mean_loss_delta':float(delta.mean()),'KL_initial_FP32':float(packet['KL_initial_FP32'][at].mean()),
                        'loss_delta_source_cluster':clustered(delta,[p['panel'][i]['source_group'] for i in at]),
                        'mean_relative_MLP_change':float(np.mean(np.mean((packet['m'][at]-initial_packet['m'][at])**2,axis=-1)/np.maximum(np.mean(initial_packet['m'][at]**2,axis=-1),1e-15)))})
                checkpoints.append(summary);reports.append(summary)
                save(folder/'progress.json',{'timestamp':stamp(),'completed_step':step,'trace':trace,'checkpoints':checkpoints})
                print('LAW_TRAINING_CHECKPOINT',name,step,'displacement',round(dnorm/initial_norm,6),'elapsed',round(time.monotonic()-t0,1),flush=True)
                del packet
                guard(400*1024**2)
        delta_path=folder/'final_native_parameter_deltas.npz'
        delta_ids=tail.save_delta(delta_path)
        run={'timestamp':stamp(),'name':name,'seed':seed,'condition':condition,'steps':len(trace),'training_examples_per_epoch_pool':len(meta),
            'draws_sha':sha(out/f'seed{seed}_frozen_draws.npz'),'parameters':p['full_parameter_count'],
            'delta_path':str(delta_path.relative_to(BASE)),'delta_sha':sha(delta_path),'delta_arrays':delta_ids,
            'trace':trace,'checkpoints':checkpoints,'seconds':time.monotonic()-t0,
            'scope':'Only original final native MLP matrices updated in FP32 with actual next-token CE and fixed upstream inputs. Not a probe fit, not reconstruction of original pretraining.'}
        save(folder/'result.json',run);run_manifest.append(run)
        ledger('native_training_trajectory_'+name,run['seconds'],steps=len(trace),parameters=p['full_parameter_count'])
    # Paired conditions use the same held-out queries and the exact same training draws per seed.
    paired=[]
    for seed in p['multistep_seeds']:
      for step in (1,8,32,64):
        with np.load(out/f'coherent_seed{seed}/checkpoint{step:03d}_panel.npz') as z:coherent_loss=z['loss']
        with np.load(out/f'prefix_order_control_seed{seed}/checkpoint{step:03d}_panel.npz') as z:controlled_loss=z['loss']
        for split in ('validation','test'):
            ix=np.array([i for i,r in enumerate(p['panel']) if r['split']==split]);gain=controlled_loss[ix]-coherent_loss[ix]
            paired.append({'seed':seed,'step':step,'split':split,'queries':len(ix),'coherent_loss_advantage':float(gain.mean()),
                'source_cluster':clustered(gain,[p['panel'][i]['source_group'] for i in ix])})
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'native_source':snapshot(ROOT/'tests/glm5/rdc_law_native.py'),
        'runs':[{k:r[k] for k in ('name','seed','condition','steps','parameters','delta_path','delta_sha','seconds')} for r in run_manifest],
        'training_pool_positions':len(meta),'training_source_groups':len({r['source_group'] for r in meta}),
        'panel_queries':len(p['panel']),'paired_coherent_advantages':paired,
        'all_target_draws_matched':True,'confirmation_used':False,
        'seconds_total_wall':time.monotonic()-start,
        'limits':['Two sample-order seeds do not characterize all training trajectories.','Only final MLP changes, not formation throughout all layers.',
            'Order control is unnatural and changes difficulty; it controls target identity/frequency/length, not every semantic or statistical factor.',
            'Loss reduction is corpus next-token prediction, not a knowledge/logic correctness score.','Learned effects are measured in FP32; native BF16 deployment needs separate testing.']}
    save(out/'result.json',result)
    for k in w:
        with torch.no_grad():w[k].copy_(original[k])
    del tail,w,original,datasets,reference_lp;gc.collect();torch.cuda.empty_cache()
    print('LAW_TRAINING_TRAJECTORIES_COMPLETE',len(run_manifest),paired[-2:],flush=True)


if __name__=='__main__':main()
