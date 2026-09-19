"""Untouched-source confirmation of fixed local operators and count-matched conditional controls."""
import gc
from collections import defaultdict
from rdc_operator_common import *
from rdc_native_conditional_operator import weights, metadata, load_block, load_bank, apply_operator, label, silu_np


def main():
    import torch
    out=BASE/'confirmation'
    if (out/'result.json').exists():
        return
    frozen=read(BASE/'operators/frozen.json')
    for name,digest in frozen['bank_files'].items():
        assert sha(BASE/'operators'/name)==digest
    assert (BASE/'capture/confirmation/result.json').exists()
    start=time.monotonic();torch.set_num_threads(2)
    selected=[r for r in rows() if r['split']=='confirmation']
    meta=metadata(selected,'confirmation')
    trainrows=[r for r in rows() if r['split']=='train']
    trainmeta=metadata(trainrows)
    seen_tokens={r['token_id'] for r in trainmeta}
    seen_combos={(r['language'],r['piece'],r['cue'],r['position_bin']) for r in trainmeta}
    for r in meta:
        r['unseen_token_identity']=r['token_id'] not in seen_tokens
        r['unseen_condition_tuple']=(r['language'],r['piece'],r['cue'],r['position_bin']) not in seen_combos
    compressed(out/'rows.json.gz',meta)
    immutable(out/'protocol.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'original_frozen_sha':sha(BASE/'operators/frozen.json'),
        'sources':len(selected),'anchors':len(meta),'names':['frozen_choice','frozen_gate_global','FP32_native_oracle'],
        'controls':'At blocks16/34, eight training-label permutations within(language,32position-bin), preserving exact condition counts and learned-vector count; applied to unchanged true heldout condition labels. These post-fit audit controls do not alter the original selected operator.',
        'shuffle_seeds':list(range(272500,272508)),
        'holdout_scope':'New source/article groups. Unseen token identity and unseen(piece,cue,position)tuple are separate strata; neither is guaranteed a new semantic concept or logical combination.'})
    reports,control_results,checks=[],[],[]
    for b in (6,16,34):
        data=load_block(selected,b,'confirmation');bank=load_bank(BASE/'operators'/f'L{b}_bank');w=weights(b)
        choice=frozen['choices'][str(b)]
        pbank={};preds={}
        names=['frozen_choice','frozen_gate_global','FP32_native_oracle']
        if b in (16,34):
            group='cue' if b==16 else 'piece'
            train=load_block(trainrows,b)
            ph=silu_np(train['gate']).astype(np.float64)
            true_labels=np.array([label(r,group) for r in trainmeta])
            partitions=defaultdict(list)
            for i,r in enumerate(trainmeta):
                partitions[r['language'],r['position_bin']].append(i)
            for seed in range(272500,272508):
                rng=np.random.default_rng(seed);newlabels=true_labels.copy()
                for ii in partitions.values():
                    newlabels[ii]=rng.permutation(newlabels[ii])
                control={g:{key:dict(value) for key,value in centers.items()} if isinstance(centers,dict) and g!='diagonal' else centers for g,centers in bank.items()}
                for key,center in control[group].items():
                    ix=np.flatnonzero(newlabels==key)
                    assert len(ix)==center['count']
                    center['phi']=ph[ix].mean(0).astype(np.float32)
                name=f'count_matched_shuffle_{seed}';pbank[name]=control;names.append(name)
            del train,ph
        pieces={name:[] for name in names}
        for at in range(0,len(meta),64):
            mm=meta[at:at+64];x=torch.as_tensor(data['x'][at:at+64],device='cuda:0')
            with torch.inference_mode():
                for name in names:
                    if name=='FP32_native_oracle':
                        p=(torch.nn.functional.silu(x@w['g'].T)*(x@w['u'].T))@w['d'].T
                    else:
                        actual_name=choice if name=='frozen_choice' or name.startswith('count_matched') else name
                        p=apply_operator(actual_name,x,mm,pbank.get(name,bank),w)
                    pieces[name].append(p.cpu().numpy())
            del x,p
        preds={name:np.concatenate(values) for name,values in pieces.items()}
        npz(out/f'L{b}_predictions.npz',**{k:v for k,v in preds.items() if not k.startswith('count_matched')},native=data['mlp'])
        energies=np.mean(data['mlp'].astype(float)**2,1)
        losses={name:np.mean((p.astype(float)-data['mlp'])**2,1)/np.maximum(energies,1e-20) for name,p in preds.items()}
        for name,loss in losses.items():
            for stratum in ('all','ordinary','event','unseen_token_identity','unseen_condition_tuple'):
                ii=[i for i,r in enumerate(meta) if stratum=='all' or stratum=='ordinary' and not r['event'] or stratum=='event' and r['event'] or stratum in ('unseen_token_identity','unseen_condition_tuple') and r[stratum]]
                if not ii:
                    continue
                record={'block':b,'name':name,'actual_selected_operator':choice if name=='frozen_choice' else None,'stratum':stratum,'rows':len(ii),
                    'relative_MSE':float(loss[ii].mean()),'raw_MSE':float(np.mean(loss[ii]*energies[ii])),
                    'relative_MSE_cluster':clustered(loss[ii],[meta[i]['source_group'] for i in ii]),
                    'paired_gain_over_global_cluster':clustered((losses['frozen_gate_global']-loss)[ii],[meta[i]['source_group'] for i in ii])}
                (control_results if name.startswith('count_matched') else reports).append(record)
        checks.append({'block':b,'all_prediction_coordinates_finite':all(np.isfinite(p).all() for p in preds.values()),'frozen_bank_unchanged':True})
        assert checks[-1]['all_prediction_coordinates_finite']
        print('OPERATOR_CONFIRM_BLOCK',b,[(r['name'],r['relative_MSE']) for r in reports if r['block']==b and r['stratum']=='all'],flush=True)
        del data,bank,w,pbank,preds,pieces
        gc.collect();torch.cuda.empty_cache()
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'sources':len(selected),'anchors':len(meta),
        'new_token_identity_anchors':sum(r['unseen_token_identity'] for r in meta),'new_condition_tuple_anchors':sum(r['unseen_condition_tuple'] for r in meta),
        'reports':reports,'count_matched_condition_controls':control_results,'checks':checks,
        'limits':'Observed prefix cue masks/piece properties are not complete semantic or syntax labels. Controls preserve model capacity and coarse language/position strata, not every lexical, norm or history confound. Choice remains original validation choice; no re-selection on test/confirmation.'}
    save(out/'result.json',result);ledger('frozen_native_operator_confirmation',time.monotonic()-start,anchors=len(meta));guard()
    print('OPERATOR_CONFIRMATION_COMPLETE',result['sources'],result['new_token_identity_anchors'],result['new_condition_tuple_anchors'],flush=True)


if __name__=='__main__':
    main()
