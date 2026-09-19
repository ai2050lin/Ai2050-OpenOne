"""Frozen-input/full-coordinate competition: earlier state, history and native factor forecasts."""
import argparse
from collections import Counter,defaultdict
from rdc_law_common import *
from rdc_law_predict import KERNELS,DECODERS,DF_GRID,normalized_features,kernel_torch,ridge_lambda,widths,decode,save_bank


def protocol():
    path=BASE/'prediction/protocol.json'
    if path.exists():return read(path)
    p={'timestamp':stamp(),'source':snapshot(Path(__file__)),'blocks':[16,35],'early_boundary':12,
        'kernels':KERNELS,'decoders':DECODERS,'effective_df_grid':DF_GRID,
        'inputs':['Actual H12 query','Actual current embedding','Every causal H12 source via complete-coordinate mean','Block11 actual attention-weighted complete-coordinate H12 source mean','Known position/language/natural-vs-QA task'],
        'all_source_scope':'All H12 source vectors retained and query means include ALL prefix positions. Means are declared candidate summaries, not claimed sufficient historical state.',
        'target_channels':['native normalized x','SiLU(gate)','up','native activation product','native MLP writeback'],
        'train_weighting':'Equal total weight to each of6corpus/task cohorts, equal source within cohort, equal anchors within source; identical weights for every route.',
        'normalization':'Every coordinate retained. q/embedding per-vector RMS then training-only vector means/global scalar RMS for each feature family. No PCA/whitening/Top-K.',
        'capacity':'Ridge full-eigensystem effective df matched per output. No eigendirection truncation. x/direct-m have identical output dimensionality; other decoders have explicitly different target dimensions.',
        'selection':'Each kernel/decoder chooses df by validation cohort-equal mean relative MLP MSE. Best deployable route selected by same criterion for block16 and validation full-vocabulary KL for block35; shuffled-history controls cannot be deployment winners.',
        'joint_linear_identity':'For a shared linear smoother S and fixed Wd, S(A)Wd^T = S(AWd^T). Direct writeback and predicted joint product should largely coincide apart from native floating arithmetic. These are a calibration pair, NOT independent mechanism discoveries.',
        'nonlinear_test':'Product of separately predicted phi/u versus predicted native product tests missing joint structure, while predicted-x through all real Wg/Wu/Wd tests a genuine future-input compiler.',
        'output_boundary':'For final block35 only, actual residual and norm/head compile the predicted writeback; residual is evaluation context, never early predictor input. Other downstream blocks remain outside this local offline score.',
        'holdout':'No confirmation data read until frozen.json is written. All selection uses validation, not main test or confirmation.',
        'negative_controls':'Three fixed within-language/task history shuffles. Evaluation sham histories drawn only from training-bank histories, not target states.'}
    immutable(path,p);print('LAW_PREDICTION_PROTOCOL_FROZEN',flush=True);return p


def load_arrays(mode='main'):
    # Keep API explicit: no hidden truncation of target matrices.
    import torch
    rows=gzread(BASE/('confirmation_material.json.gz' if mode=='confirmation' else 'material.json.gz'))
    meta=[];raw=defaultdict(list);target={b:defaultdict(list) for b in (16,35)};residual=[];post=[]
    for r in rows:
        with np.load(BASE/'capture'/mode/'fields'/f'{r["sample_id"]}.npz') as z:
            H=unbits(z['H']);raw['q'].append(H[12]);raw['embedding'].append(H[0]);raw['history'].append(z['history_mean_H12_RMS']);raw['routed'].append(z['attention_weighted_H12_RMS'])
            for ai,pos in enumerate(r['anchors']):
                cls=(0 if r['language']=='en' else 2)+int(r['kind']=='QA')
                raw['position'].append(pos);raw['class'].append(cls)
                meta.append({'id':r['sample_id']+f'_a{ai}','sample_id':r['sample_id'],'source_group':r['source_group'],'split':r['split'],
                    'cohort':r['cohort'],'kind':r['kind'],'language':r['language'],'class':cls,'anchor_index':ai,'position':pos,
                    'current_token_id':r['prompt_ids'][pos],'target_id':r['prompt_ids'][pos+1] if pos+1<len(r['prompt_ids']) else None,
                    'novelty':r['novelty'],'held_relation_combinations':r.get('held_relation_combinations',[])})
            for b in target:
                for key in ('x','up','activation','mlp'):target[b][key].append(unbits(z[f'L{b}_{key}']))
                g=unbits(z[f'L{b}_gate']).astype(float)
                target[b]['phi'].append((g*np.exp(-np.logaddexp(0,-g))).astype(np.float32))
            rr=unbits(z['L35_input'])+unbits(z['L35_attention'])
            residual.append(torch.as_tensor(rr).to(torch.bfloat16).float().numpy());post.append(unbits(z['postnorm']))
    raw={k:np.asarray(v) if k in ('position','class') else np.concatenate(v) for k,v in raw.items()}
    target={b:{k:np.concatenate(v) for k,v in d.items()} for b,d in target.items()}
    return meta,raw,target,np.concatenate(residual),np.concatenate(post)


def training_weights(meta,ix):
    anchors=Counter((meta[i]['cohort'],meta[i]['source_group']) for i in ix)
    sources={c:len({meta[i]['source_group'] for i in ix if meta[i]['cohort']==c}) for c in sorted({meta[i]['cohort'] for i in ix})}
    return np.array([len(ix)/(len(sources)*sources[meta[i]['cohort']]*anchors[(meta[i]['cohort'],meta[i]['source_group'])]) for i in ix])


def add_shuffles(features,meta,train_indices):
    f=dict(features);byclass={c:np.array([i for i in train_indices if meta[i]['class']==c]) for c in range(4)}
    maps={}
    for seed in range(3):
        donor=np.empty(len(meta),dtype=np.int64)
        for cls,ii in byclass.items():
            assert len(ii)
            rng=np.random.default_rng(272900+seed*10+cls)
            donor[ii]=rng.permutation(ii)
        trainset=set(train_indices.tolist())
        for i,r in enumerate(meta):
            if i not in trainset:
                options=byclass[r['class']]
                donor[i]=options[int(rank('sham'+str(seed)+r['id'])[:12],16)%len(options)]
        f['shuffled_history_'+str(seed)]=features['history'][donor]
        maps['shuffled_history_'+str(seed)]=donor
    return f,maps


def main():
    import torch
    from rdc_law_native import parameter
    p=protocol();out=BASE/'prediction'
    if (out/'result.json').exists():return
    assert read(BASE/'verification/prediction_math.json')['passed']
    assert (BASE/'capture/main/result.json').exists()
    start=time.monotonic();guard(900*1024**2);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    meta,raw,target,residual,post=load_arrays()
    train=np.array([i for i,r in enumerate(meta) if r['split']=='train']);val=np.array([i for i,r in enumerate(meta) if r['split']=='validation']);test=np.array([i for i,r in enumerate(meta) if r['split']=='test'])
    weight=training_weights(meta,train);features,rulers=normalized_features(raw,train,weight)
    features,maps=add_shuffles(features,meta,train)
    npz(out/'feature_rulers.npz',**{key+'_'+k:np.asarray(v) for key,d in rulers.items() for k,v in d.items()})
    npz(out/'training_features.npz',**{k:v[train] for k,v in features.items()},training_row_indices=train,weights=weight)
    npz(out/'shuffle_assignments.npz',**maps)
    compressed(out/'query_catalog.json.gz',meta)
    device='cuda:0';ft={k:torch.as_tensor(v,device=device,dtype=torch.int64 if k=='class' else torch.float64) for k,v in features.items()}
    tt={k:v[train] for k,v in ft.items()};sqrtw=torch.as_tensor(np.sqrt(weight),device=device,dtype=torch.float64)
    sw=torch.as_tensor(weight/weight.sum(),device=device,dtype=torch.float64)
    head=parameter('model.embed_tokens.weight',device);norm=parameter('model.norm.weight',device)
    eps=read(ROOT/'models/hf/qwen3-4b/config.json')['rms_norm_eps']
    rr=torch.as_tensor(residual,device=device);posttensor=torch.as_tensor(post,device=device)
    reference_lp=[]
    with torch.no_grad():
        for at in range(0,len(meta),16):reference_lp.append((posttensor[at:at+16]@head.T).log_softmax(-1))
    reference_lp=torch.cat(reference_lp)
    decisions=[];all_reports=[];winners={};fixed={};calibration=[]
    # Raw targets remain native; no target-dependent transformation enters any kernel.
    for b in p['blocks']:
        w={k:parameter(f'model.layers.{b}.mlp.{name}_proj.weight',device) for k,name in [('g','gate'),('u','up'),('d','down')]}
        yy=np.concatenate([target[b][k] for k in ('x','phi','up','activation','mlp')],axis=1)
        y=torch.as_tensor(yy,device=device);native=torch.as_tensor(target[b]['mlp'],device=device)
        ym=(y[train].double()*sw[:,None]).sum(0);centered=(y[train].double()-ym)*sqrtw[:,None]
        winner_score=float('inf');winning=None;best_by_decoder={d:(float('inf'),None) for d in DECODERS}
        fixed_early=None
        for kernel_name in KERNELS:
            K=kernel_torch(kernel_name,tt,tt);weighted=K*sqrtw[:,None]*sqrtw[None,:]
            eigen,V=torch.linalg.eigh((weighted+weighted.T)/2)
            assert float(eigen.min())>-1e-6*float(eigen.max()),('Kernel PSD',kernel_name,float(eigen.min()))
            eigen=eigen.clamp_min(0)
            projected=V.T@centered
            cross=kernel_torch(kernel_name,ft,tt)
            grid=[];local_records=[]
            for df in DF_GRID:
                lam,actual_df=ridge_lambda(eigen,df)
                coeff=(V@(projected/(eigen[:,None]+lam)))*sqrtw[:,None]
                pred=cross.float()@coeff.float()+ym.float()
                direct=decode(pred,'direct_mlp',w);joint=decode(pred,'predicted_joint_product',w)
                mismatch=float(torch.mean((direct-joint).square())/torch.mean(native.square()).clamp_min(1e-20))
                calibration.append({'block':b,'kernel':kernel_name,'df':df,'joint_vs_direct_relative_MSE':mismatch,
                    'interpretation':'Expected near-commutation of linear smoother with Wd, limited by stored native BF16 writeback and FP32 arithmetic.'})
                for decoder in DECODERS:
                    m=direct if decoder=='direct_mlp' else joint if decoder=='predicted_joint_product' else decode(pred,decoder,w)
                    relative=((m-native).square().mean(-1)/native.square().mean(-1).clamp_min(1e-20)).cpu().numpy()
                    rawmse=(m-native).square().mean(-1).cpu().numpy()
                    kl=np.full(len(meta),np.nan);argmatch=np.full(len(meta),np.nan)
                    if b==35:
                        with torch.no_grad():
                            for at in range(0,len(meta),16):
                                r=rr[at:at+16]+m[at:at+16]
                                n=norm*r*torch.rsqrt(r.square().mean(-1,keepdim=True)+eps)
                                logits=n@head.T;lp=logits.log_softmax(-1)
                                ref=reference_lp[at:at+16]
                                kl[at:at+len(r)]=(ref.exp()*(ref-lp)).sum(-1).cpu().numpy()
                                argmatch[at:at+len(r)]=(logits.argmax(-1)==ref.argmax(-1)).float().cpu().numpy()
                    cohorts=sorted({r['cohort'] for r in meta})
                    val_score=float(np.mean([relative[[i for i in val if meta[i]['cohort']==c]].mean() for c in cohorts]))
                    output_score=float(np.mean([kl[[i for i in val if meta[i]['cohort']==c]].mean() for c in cohorts])) if b==35 else val_score
                    packet={'block':b,'kernel':kernel_name,'decoder':decoder,'df_target':df,'effective_df':actual_df,'lambda':lam,
                        'learned_output_dimensions':len(__import__('rdc_law_predict').decoder_target_indices(decoder)),
                        'validation_cohort_equal_relative_MSE':val_score,'validation_cohort_equal_output_KL':output_score if b==35 else None,
                        'metrics':[]}
                    for split,ix in [('train',train),('validation',val),('test',test)]:
                      for c in cohorts:
                        at=np.array([i for i in ix if meta[i]['cohort']==c])
                        packet['metrics'].append({'split':split,'cohort':c,'anchors':len(at),'relative_MSE':float(relative[at].mean()),
                            'raw_MSE':float(rawmse[at].mean()),'KL':float(kl[at].mean()) if b==35 else None,
                            'argmax_agreement':float(argmatch[at].mean()) if b==35 else None})
                    all_reports.append(packet);local_records.append(packet)
                    # Per decoder best deployable validation choice, with exact frozen coefficients retained.
                    deployment=not kernel_name.startswith('shuffled')
                    score=output_score if b==35 else val_score
                    info={'block':b,'kernel':kernel_name,'df':df,'lambda':lam,'effective_df':actual_df,'source':snapshot(Path(__file__)),
                        'prediction_inputs':'Only current H12/embedding and causal H12 history, position/language/task. No target x/g/u/m, no answer/gold graph.',
                        'training_rows':len(train),'fit_source_weights':'training_features.npz','feature_rulers':'feature_rulers.npz',
                        'selection_metric':'validation cohort-equal full-vocab KL' if b==35 else 'validation cohort-equal relativeMLP MSE'}
                    if deployment and score<best_by_decoder[decoder][0]:
                        path=out/'banks'/f'L{b}_{decoder}'
                        save_bank(path,info,coeff.cpu().numpy(),ym.cpu().numpy(),decoder)
                        npz(out/'predictions'/f'L{b}_{decoder}.npz',mlp=m.cpu().numpy(),relative_MSE=relative,
                            raw_MSE=rawmse,**({'KL':kl,'argmax_agreement':argmatch} if b==35 else {}))
                        best_by_decoder[decoder]=(score,dict(info,decoder=decoder,bank=str(path.relative_to(BASE))))
                    if kernel_name=='early_linear' and decoder=='direct_mlp' and (fixed_early is None or score<fixed_early['score']):
                        path=out/'banks'/f'L{b}_fixed_early'
                        save_bank(path,info,coeff.cpu().numpy(),ym.cpu().numpy(),decoder)
                        npz(out/'predictions'/f'L{b}_fixed_early.npz',mlp=m.cpu().numpy(),relative_MSE=relative,
                            raw_MSE=rawmse,**({'KL':kl,'argmax_agreement':argmatch} if b==35 else {}))
                        fixed_early=dict(info,decoder=decoder,bank=str(path.relative_to(BASE)),score=score)
                del coeff,pred,direct,joint
            for decoder in DECODERS:
                rec=min([r for r in local_records if r['decoder']==decoder],key=lambda r:r['validation_cohort_equal_relative_MSE'])
                decisions.append({k:v for k,v in rec.items()})
            print('LAW_PREDICT_KERNEL',b,kernel_name,'elapsed',round(time.monotonic()-start,1),flush=True)
            del K,weighted,V,eigen,projected,cross
        winning=min([v[1] for v in best_by_decoder.values()],key=lambda v:best_by_decoder[v['decoder']][0])
        winners[str(b)]=winning;fixed[str(b)]=fixed_early
        save(out/f'L{b}_decoder_choices.json',{'timestamp':stamp(),'by_decoder':{k:v[1] for k,v in best_by_decoder.items()},'winner':winning,'fixed_early':fixed_early})
        del y,yy,centered,ym,w,native
    save(out/'all_validation_grid_and_test_metrics.json',{'timestamp':stamp(),'rows':all_reports,'main_test_not_used_for_selection':True})
    save(out/'linear_commutation_audit.json',{'timestamp':stamp(),'comparisons':calibration,'identity':'S(A)Wd = S(AWd), if same linear smoother and exact arithmetic; no new mechanism gain can be inferred merely by renaming this target.'})
    frozen={'timestamp':stamp(),'source':snapshot(Path(__file__)),'protocol_sha':sha(out/'protocol.json'),'winners':winners,'fixed_early':fixed,
        'all_banks_sha':{str(p.relative_to(out)):sha(p) for p in (out/'banks').glob('*')},
        'confirmation_capture_exists':(BASE/'capture/confirmation/result.json').exists(),'material_sha':sha(BASE/'material.json.gz'),
        'selection_used_test_or_confirmation':False}
    assert not frozen['confirmation_capture_exists']
    immutable(out/'frozen.json',frozen)
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'anchors':len(meta),'training':len(train),'validation':len(val),'test':len(test),
        'source_weight_sum':float(weight.sum()),'kernel_count':len(KERNELS),'decoder_count':len(DECODERS),'grid_candidates':len(all_reports),
        'per_kernel_MSE_selected_records':decisions,'frozen_winners':winners,'fixed_early':fixed,
        'linear_commutation_max_relative_MSE':max(r['joint_vs_direct_relative_MSE'] for r in calibration),
        'seconds':time.monotonic()-start,'scope':'Learned early-state partial-module forecasting, not a complete extracted language model. Capacity matched per output; output channel dimensions differ across decoders. Direct-M and joint-product routes are mostly a linear-commutation calibration pair.'}
    save(out/'result.json',result);ledger('early_history_native_factor_prediction',result['seconds'])
    print('LAW_PREDICTION_FROZEN_COMPLETE',winners,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-protocol',action='store_true');args=parser.parse_args()
    protocol() if args.freeze_protocol else main()
