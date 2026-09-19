"""Training-formation evidence: full native-parameter gradient factors and actual updates.

This is restricted controlled continuation, not a reconstruction of undocumented
pretraining. All original checkpoints are read-only and all updates are in memory.
"""
import argparse
import gc
from collections import Counter, defaultdict
from rdc_law_common import *
from rdc_law_native import Tail, tail_forward, factor_gram, dense_gradient, parameter_norm


def protocol():
    path=BASE/'formation/protocol.json'
    if path.exists():return read(path)
    rows=gzread(BASE/'material.json.gz')
    panel=[];views=[]
    for cohort in ('gum','ewt','cmrc'):
        for split,n,vn in [('train',48,32),('validation',16,8),('test',32,16)]:
            candidates=sorted([r for r in rows if r['cohort']==cohort and r['split']==split],key=lambda r:rank('formation:'+r['source_id']))
            assert len(candidates)>=n
            for r in candidates[:n]:
                ai=int(rank('anchor:'+r['sample_id'])[:8],16)%len(r['anchors'])
                p=r['anchors'][ai]
                panel.append({'sample_id':r['sample_id'],'source_group':r['source_group'],'cohort':cohort,'language':r['language'],
                    'split':split,'anchor_index':ai,'position':p,'target_id':r['prompt_ids'][p+1],
                    'current_token_id':r['prompt_ids'][p],'prefix_tokens':p+1,'relation_types':r.get('relation_types',[])})
            for r in candidates[:vn]:
                n=len(r['prompt_ids']);boundary=n-17
                order=np.arange(n)
                seed=int(rank('order-control:'+r['source_id'])[:8],16)
                rng=np.random.default_rng(seed)
                order[1:boundary]=rng.permutation(order[1:boundary])
                controlled=np.asarray(r['prompt_ids'])[order].tolist()
                assert controlled[boundary:]==r['prompt_ids'][boundary:]
                assert Counter(controlled)==Counter(r['prompt_ids'])
                positions=list(range(n-17,n-1))
                views.append({'sample_id':r['sample_id'],'source_group':r['source_group'],'cohort':cohort,'language':r['language'],'split':split,
                    'original_ids':r['prompt_ids'],'controlled_ids':controlled,'order':order.tolist(),'seed':seed,'training_positions':positions,
                    'target_ids':[r['prompt_ids'][p+1] for p in positions],
                    'control_scope':'Only token order at positions1..n-18 permuted; first token, complete final17 input IDs, all target IDs, token frequency and sequence length preserved. Unnatural control, not a natural language example.'})
    donors=[i for cohort in ('gum','ewt','cmrc') for i in [i for i,r in enumerate(panel) if r['cohort']==cohort and r['split']=='train'][:4]]
    p={'timestamp':stamp(),'source':snapshot(Path(__file__)),'panel':panel,'controlled_views':views,'donor_panel_indices':donors,
        'full_parameter_count':3*2560*9728,'all_parameter_matrices':['model.layers.35.mlp.gate_proj.weight','model.layers.35.mlp.up_proj.weight','model.layers.35.mlp.down_proj.weight'],
        'exact_factors':'grad_Wd=s a^T; grad_Wu=b_u x^T; grad_Wg=b_g x^T. Pairwise parameter dot products use all factor coordinates, no truncation.',
        'initial_update_relative_norms':[1e-5,1e-4,1e-3],'initial_update_rule':'Single-example SGD, eta chosen only to realize stated full-parameter relative step norm; predict deltaL_b=-eta<g_b,g_a> BEFORE updated forward.',
        'multistep_seeds':[2728,2729],'multistep_steps':64,'multistep_checkpoints':[0,1,8,32,64],
        'multistep_batch_size':16,'multistep_learning_rate':0.01,'multistep_global_gradient_clip':10.0,
        'multistep_pairing':'Same source/target-index draws per seed for coherent and controlled-prefix conditions. Only the prefix order differs. Train split only, no confirmation used.',
        'precision':'Original BF16 checkpoint values interpreted in FP32 for smooth derivative/update audit; native BF16 floor and final BF16 deployment tested separately. FP32 training is not claimed bit-identical BF16 inference.',
        'native_class':'Actual Transformers Qwen3MLP registered with all original g/u/d parameters; exact analytic gradient independently tested against autograd.',
        'formation_boundary':'Restricted new continuation can test whether these updates cause cooperation/conflict and conditional change. It cannot establish how the released checkpoint originally learned language.'}
    immutable(path,p)
    print('LAW_FORMATION_PROTOCOL_FROZEN',len(panel),len(views),len(donors),flush=True)
    return p


def panel_arrays(panel,mode='main'):
    x=[];r=[];m=[];post=[]
    for item in panel:
        sid=item['sample_id'];p=item['position'];ai=item['anchor_index']
        with np.load(BASE/'capture'/mode/'sources'/f'{sid}.npz') as z:
            x.append(unbits(z['x'][p]));r.append(unbits(z['residual'][p]))
        with np.load(BASE/'capture'/mode/'fields'/f'{sid}.npz') as z:
            m.append(unbits(z['L35_mlp'][ai]));post.append(unbits(z['postnorm'][ai]))
    return np.stack(x),np.stack(r),np.stack(m),np.stack(post)


def pair_controls(panel,gram):
    """Independent-query summaries, with explicit same-target and nuisance conditioning."""
    norms=np.sqrt(np.maximum(np.diag(gram),1e-30));cos=gram/(norms[:,None]*norms[None,:])
    groups=defaultdict(list)
    for i,a in enumerate(panel):
        for j,b in enumerate(panel):
            if i==j or a['source_group']==b['source_group']:continue
            # Output identity must be separated from relation/genre overlap.
            key=(a['target_id']==b['target_id'],a['cohort']==b['cohort'],a['current_token_id']==b['current_token_id'],
                 abs(a['prefix_tokens']-b['prefix_tokens'])<=16)
            groups[key].append((i,j))
    reports=[]
    for key,pairs in sorted(groups.items()):
        v=np.array([cos[i,j] for i,j in pairs])
        reports.append({'same_target_id':key[0],'same_cohort':key[1],'same_current_id':key[2],'length_within16':key[3],
            'ordered_pairs':len(pairs),'cosine_mean':float(v.mean()),'positive_fraction':float(np.mean(v>0)),
            'scope':'Dependent pair summaries, not independent-pair confidence intervals or semantic causal effects.'})
    # Matched target + language + rough position bins. Compare target-controlled relation overlap.
    contrasts=[]
    for i,a in enumerate(panel):
        if not a['relation_types']:continue
        same=[];different=[]
        aset=set(a['relation_types'])-{'punct','root','det','case'}
        for j,b in enumerate(panel):
            if i==j or a['source_group']==b['source_group'] or a['target_id']!=b['target_id'] or a['language']!=b['language']:continue
            if a['prefix_tokens']//32!=b['prefix_tokens']//32 or not b['relation_types']:continue
            bset=set(b['relation_types'])-{'punct','root','det','case'}
            score=len(aset&bset)/max(len(aset|bset),1)
            (same if score>=.5 else different).append(cos[i,j])
        if same and different:
            contrasts.append({'sample_id':a['sample_id'],'source_group':a['source_group'],
                'same_target_position_relation_overlap_cosine_gain':float(np.mean(same)-np.mean(different)),
                'high_overlap_candidates':len(same),'low_overlap_candidates':len(different)})
    summary=clustered([r['same_target_position_relation_overlap_cosine_gain'] for r in contrasts],[r['source_group'] for r in contrasts])
    return reports,contrasts,summary


def initial():
    import torch
    p=protocol();out=BASE/'formation/initial_stable'
    if (out/'result.json').exists():return
    assert read(BASE/'capture/main/result.json')['rows']==960
    assert read(BASE/'verification/math.json')['passed']
    start=time.monotonic();guard(160*1024**2)
    arrays=panel_arrays(p['panel']);tail=Tail();w=tail.w
    x,residual,native_m,native_n=[torch.as_tensor(a,device=tail.device) for a in arrays]
    targets=torch.tensor([r['target_id'] for r in p['panel']],device=tail.device)
    factors={k:[] for k in ('x','a','s','bg','bu')};losses=[];floors=[];states=[]
    with torch.no_grad():
        for at in range(0,len(x),16):
            o=tail.forward(x[at:at+16],residual[at:at+16],targets[at:at+16],True)
            assert torch.equal(tail.module(x[at:at+16]),o['m']),'Actual native FP32 MLP mismatch'
            for k in factors:factors[k].append(o['factors'][k].double())
            losses.append(o['loss'].double());states.append(o['m'].float())
            native_lp=(native_n[at:at+16]@tail.head.T).log_softmax(-1)
            floors.append(torch.stack([((o['m']-native_m[at:at+16]).square().mean(-1)/native_m[at:at+16].square().mean(-1).clamp_min(1e-15)),
                (native_lp.exp()*(native_lp-o['logprobs'])).sum(-1)],-1).cpu().numpy())
        f={k:torch.cat(v) for k,v in factors.items()};baseline=torch.cat(losses);states=torch.cat(states)
        grams=factor_gram(f)
        # Full dense native-parameter autograd cross-check, not merely a synthetic test.
    with torch.no_grad():
        same_shape=tail.forward(x[:1],residual[:1],targets[:1],True)
        dg=dense_gradient(same_shape['factors'])
        batch_gradient=dense_gradient({k:v[:1].float() for k,v in f.items()})
        shape_errors={k:float((dg[k]-batch_gradient[k]).abs().max()/dg[k].abs().max().clamp_min(1e-12)) for k in w}
        del same_shape,batch_gradient
    for a in w.values():a.requires_grad_(True)
    o=tail.forward(x[:1],residual[:1],targets[:1],False);o['loss'].sum().backward()
    errors={k:float((w[k].grad-dg[k]).abs().max()/w[k].grad.abs().max().clamp_min(1e-12)) for k in w}
    assert max(errors.values())<2e-5,errors
    for a in w.values():a.grad=None;a.requires_grad_(False)
    del o,dg
    initial_weights={k:v.clone() for k,v in w.items()}
    norm=float(parameter_norm(w));updates=[];prediction_rows=[]
    with torch.no_grad():
        for donor in p['donor_panel_indices']:
            df={k:v[donor:donor+1].float() for k,v in f.items()}
            grad=dense_gradient(df);gradnorm=float(parameter_norm(grad))
            for relative in p['initial_update_relative_norms']:
                eta=relative*norm/gradnorm
                predicted=(-eta*grams['total'][:,donor]).cpu().numpy()
                # Save prospective prediction before applying the training update.
                uid=f'donor{donor:03d}_relative{relative:.0e}'
                npz(out/'predicted_before_update'/f'{uid}.npz',predicted_loss_delta=predicted)
                for key in w:w[key].copy_(initial_weights[key]-eta*grad[key])
                actual_norm=float(torch.stack([(w[k]-initial_weights[k]).square().sum() for k in w]).sum().sqrt())
                new=[]
                for at in range(0,len(x),16):
                    z=tail.forward(x[at:at+16],residual[at:at+16],targets[at:at+16])
                    new.append(z['loss'].double());del z
                measured=(torch.cat(new)-baseline).cpu().numpy()
                rr={'id':uid,'donor':donor,'relative_step_requested':relative,'eta':eta,'actual_parameter_delta_norm':actual_norm,
                    'initial_parameter_norm':norm,'actual_relative_step':actual_norm/norm,
                    'max_abs_linearization_error':float(np.max(abs(measured-predicted))),
                    'self_loss_delta':float(measured[donor]),'self_predicted_delta':float(predicted[donor])}
                for split in ('train','validation','test'):
                    ix=np.array([i for i,r in enumerate(p['panel']) if r['split']==split and i!=donor])
                    delta=measured[ix];pred=predicted[ix];scale=np.maximum(np.mean(delta**2),1e-30)
                    rr[split]={'queries':len(ix),'measured_mean_delta':float(delta.mean()),'predicted_mean_delta':float(pred.mean()),
                        'linearization_relative_SSE':float(np.mean((delta-pred)**2)/scale),
                        'sign_agreement_above1e-5':float(np.mean((delta[np.abs(delta)>1e-5]>0)==(pred[np.abs(delta)>1e-5]>0))) if np.any(abs(delta)>1e-5) else None,
                        'loss_change_cluster':clustered(delta,[p['panel'][i]['source_group'] for i in ix])}
                updates.append(rr)
                npz(out/'actual_updates'/f'{uid}.npz',measured_loss_delta=measured,predicted_loss_delta=predicted)
                prediction_rows.append({'id':uid,'prediction_file':str((out/'predicted_before_update'/f'{uid}.npz').relative_to(BASE)),
                    'actual_file':str((out/'actual_updates'/f'{uid}.npz').relative_to(BASE))})
            del grad
            print('LAW_FORMATION_INITIAL_DONOR',donor,'updates',len(updates),'elapsed',round(time.monotonic()-start,1),flush=True)
        for key in w:w[key].copy_(initial_weights[key])
        # Recompute baseline after reset, confirming temporary updates do not leak into later trials.
        replay=tail.forward(x[:16],residual[:16],targets[:16])['loss'].double()
        assert torch.equal(replay,baseline[:16]),'Native parameter restoration failed'
        npz(out/'complete_gradient_factors.npz',**{k:v.float().cpu().numpy() for k,v in f.items()},
            loss=baseline.cpu().numpy(),mlp_fp32=states.cpu().numpy(),**{'gram_'+k:v.cpu().numpy() for k,v in grams.items()})
    gram=grams['total'].cpu().numpy();controls,contrasts,contrast_summary=pair_controls(p['panel'],gram)
    compressed(out/'pair_controls.json.gz',{'strata':controls,'matched_relation_contrasts':contrasts,'matched_summary':contrast_summary})
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'native_source':snapshot(ROOT/'tests/glm5/rdc_law_native.py'),
        'panel_queries':len(x),'independent_sources':len({r['source_group'] for r in p['panel']}),'ordered_gradient_pairs':len(x)**2,
        'parameters':p['full_parameter_count'],'actual_native_class_FP32_forward_exact':True,'full_dense_autograd_relative_errors':errors,
        'batch16_vs_batch1_gradient_relative_errors':shape_errors,
        'numerical_refinement':'Subtract one-hot target from probability before head matrix product; avoids subtracting near-equal readout means. Autograd verification now uses identical single-query shape. Earlier initial/ results preserved as initial numerical version; new outputs never overwrite them.',
        'FP32_vs_observed_BF16_floor':{'mean_relative_MLP_MSE':float(np.concatenate(floors)[:,0].mean()),'mean_full_vocab_KL':float(np.concatenate(floors)[:,1].mean())},
        'gradient_pair_controls':controls,'target_position_matched_relation_contrast':contrast_summary,
        'actual_updates':updates,'prediction_manifest':prediction_rows,'parameter_restoration_exact':True,
        'seconds':time.monotonic()-start,'formation_status':'Actual controlled native-parameter updates with measured full-vocabulary CE effects. Restricted continuation evidence, not original pretraining formation history.',
        'scope':'Only final MLP matrices change; initial x/residual/norm/head fixed. Current gradients do not determine undocumented training history. Derivative predictions describe small local training influence, not language answers.'}
    save(out/'result.json',result);ledger('native_parameter_initial_training_updates',time.monotonic()-start,updates=len(updates))
    del tail,initial_weights,w,x,residual,f,grams;gc.collect();torch.cuda.empty_cache()
    print('LAW_FORMATION_INITIAL_COMPLETE',len(updates),contrast_summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    protocol() if args.freeze else initial()
