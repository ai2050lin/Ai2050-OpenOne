"""Natural amplification/persistence/cancellation paths, all coordinates and all MLP units."""
from collections import Counter,defaultdict
import gc
from rdc_joint_common import *
from rdc_joint_capture import ledger,array_identity
from phase2721_rdc_joint_native_factors import FactorTrace
from phase2722_rdc_joint_event_trace import Layers
from phase2722_rdc_joint_tail_confirmation import ThreeLayers

OUT=BASE/'extension/native_regimes'


def load_gzip(path):return json.loads(gzip.decompress(path.read_bytes()))


def main():
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    if (OUT/'result.json').exists():return
    start=time.monotonic();guard(70*1024**2)
    old_trace=load_gzip(BASE/'extension/event_trace/rows.json.gz');train_events=[r for r in old_trace if r['split']=='train' and r['role']=='event']
    onset=int(read(BASE/'extension/event_trace/selected_block.json')['block_zero_index'])
    # Late cancellation block is selected on TRAIN only, from ALL later layer boundaries.
    dlog=np.mean([np.diff(np.log(np.maximum(r['energy_by_layer'],1e-20))) for r in train_events],0);cancellation=int(np.argmin(dlog[23:])+23);blocks=[6,onset,cancellation]
    main_material={r['sample_id']:r for r in rows()+rows(True)};tail_material={r['sample_id']:r for r in load_gzip(BASE/'extension/tail_confirmation/material.json.gz')}
    allmeta=load_gzip(BASE/'extension/all_token_amplitudes.json.gz');tailmeta=load_gzip(BASE/'extension/tail_confirmation/all_token_results.json.gz')
    training_pairs=[p for p in read(BASE/'extension/event_trace_selection.json')['pairs'] if p['event']['split']=='train']
    new_events=[r for r in tailmeta if r['event']];confirmation_pairs=[]
    for event in new_events:
        candidates=[r for r in tailmeta if r['position']>0 and not r['event'] and r['token_id']==event['token_id'] and r['source_key']==event['source_key'] and r['source_group']!=event['source_group']]
        control=min(candidates,key=lambda r:(abs(r['position']-event['position']),r['sample_id'],r['position'])) if candidates else None
        confirmation_pairs.append({'event':event,'same_token_non_event':control,'absolute_position_difference':abs(control['position']-event['position']) if control else None})
    protocol={'timestamp':stamp(),'source':snapshot(Path(__file__)),'blocks':blocks,'training_events':len(train_events),'new_events':len(new_events),
        'question':'Do first-position and rare internal amplification use similar full-coordinate output shapes with different native parameter paths; does a later block cancel them, and does that organization reproduce on the19 new numerical events?',
        'selection':'Onset block from old TRAIN mean full-coordinate increment; late block from minimum TRAIN mean log-energy change among blocks23..35; no new-layer outputs used. ALL19 new events retained; same token ID/same corpus/different source-group controls minimize absolute position difference then lexicographic ID.',
        'primary':'All37 boundaries at first/event/control positions; original natural full-token factorization at6/onset/cancellation blocks; all9728 gate/up/activation/down contributions and all2560 coordinates. Native full-vocabulary entropy and actual next-token NLL at sampled positions.',
        'algorithm':'Training conditional mean full-vector MLP signatures, per-unit signed projection and squared norm contributions from actual down-weight columns, and exact residual energy/cross-term accounting. Means are descriptive condition-specific predictions; event label is retrospective, not an available forecast input.',
        'limits':'New extension motivated after Phase2722 numerical event labels and H23/H36 outcomes. New onset/cancellation layer values not yet observed, but NOT a globally blind fresh semantic confirmation. No scalar ablation/patch, no Top-K defining structure, no semantic necessity claim.',
        'resources':'At most7 training events plus19 new events and deduplicated controls/first positions;70MiB bounded allowance within explicit3.125GiB result ceiling; single native BF16 Q4, no other CUDA model. Original archives untouched.',
        'training_pairs':training_pairs,'new_pairs':confirmation_pairs,'train_mean_log_energy_increment':dlog.tolist()}
    if not (OUT/'protocol.json').exists():immutable(OUT/'protocol.json',protocol)
    model,tok=load_native('qwen4');device=model.get_input_embeddings().weight.device;layers=Layers(model);trace=FactorTrace(model,blocks);fullcheck=ThreeLayers(model)
    tail_identities={r['sample_id']:r['arrays'] for r in load_gzip(BASE/'extension/tail_confirmation/array_identities.json.gz')}
    weights={b:model.model.layers[b].mlp.down_proj.weight.detach().float().cpu().numpy().astype(float) for b in blocks}
    norms={b:np.sum(w*w,axis=0)/2560 for b,w in weights.items()};signatures={};unit_profiles=defaultdict(lambda:np.zeros((8,9728),float));unit_counts=Counter();coordinate_profiles=defaultdict(lambda:np.zeros((4,2560),float));coordinate_counts=Counter();records=[];identity=[]
    with np.load(BASE/'native_factors/train_L6_all_unit_and_coordinate_profiles.npz') as z:
        # Despite the historical _sum suffix, the producer saves conditional MEANS.
        first_activation=z['activation_sum'][0].astype(float);signatures['L6_first']=weights[6]@first_activation
    def collect(stage,pairs,material):
        requests={}
        for pair in pairs:
            for role,key in [('event','event'),('same_ID_non_event','same_token_non_event')]:
                r=pair[key]
                if r:requests[r['sample_id'],r['position']]={'sample_id':r['sample_id'],'position':r['position'],'source_group':r['source_group'],'token_id':r['token_id'],'token':r['token'],'role':role,'stage':stage}
        grouped=defaultdict(list)
        for r in requests.values():grouped[r['sample_id']].append(r)
        output=[]
        for sid,probes in grouped.items():
            source=material[sid];positions=sorted({0,*[r['position'] for r in probes]});layers.positions=positions;layers.data={};trace.data={};fullcheck.data={}
            ids=torch.tensor([source['prompt_ids']],device=device);post=model.model(input_ids=ids,use_cache=False).last_hidden_state
            full=np.stack([layers.data[i] for i in range(37)]);lp=model.lm_head(post[0,positions]).float().log_softmax(-1)
            if stage=='train':
                original=field(source,False)
                for l in (12,23,36):assert np.array_equal(full[l],original[f'h{l}'][positions])
            else:
                assert {k:array_identity(v) for k,v in fullcheck.data.items()}==tail_identities[sid],sid
            packet={'layers':full,'positions':np.array(positions),'postnorm':bits(post[0,positions])};npz(OUT/'fields'/f'{sid}.npz',**packet)
            factor={};energy_records={}
            for b in blocks:
                d=trace.data[b];block=model.model.layers[b];a=block.mlp.act_fn(d['gate'])*d['up'];assert torch.equal(a,d['activation'])
                assert torch.equal(block.mlp.down_proj(a),d['mlp'])
                residual=d['input']+d['attention'];assert torch.equal(block.post_attention_layernorm(residual),d['mlp_input']) and torch.equal(residual+d['mlp'],d['output'])
                for key in ('input','attention','mlp_input','gate','up','activation','mlp','output'):factor[f'L{b}_{key}']=bits(d[key][0,positions])
                factor[f'L{b}_attention_probability']=bits(d['attention_probability'][0,:,positions])
            factor['positions']=np.array(positions);npz(OUT/'factors'/f'{sid}.npz',**factor)
            identity.append({'sample_id':sid,'stage':stage,'fields':{k:array_identity(v) for k,v in packet.items()},'factors':{k:array_identity(v) for k,v in factor.items()}})
            first={'sample_id':sid,'position':0,'source_group':source['source_group'],'token_id':source['prompt_ids'][0],'token':source['tokens'][0],'role':'first','stage':stage}
            for probe in [first]+probes:
                j=positions.index(probe['position']);h=unbits(full[:,j]).astype(float);ee=np.mean(h*h,1);increments=np.mean(np.diff(h,axis=0)**2,1)
                rr={**probe,'language':source['language'],'source_key':source['source_key'],'energy_by_layer':ee.tolist(),'maximum_H12_H23_increment_block':int(np.argmax(increments[12:23])+12),
                    'late_maximum_log_energy_decline_block':int(np.argmin(np.diff(np.log(np.maximum(ee,1e-20))))),
                    'native_output_entropy':float(-(lp[j].exp()*lp[j]).sum()),'native_argmax_token_id':int(lp[j].argmax()),
                    'observed_next_token_NLL':-float(lp[j,source['prompt_ids'][probe['position']+1]]) if probe['position']+1<len(source['prompt_ids']) else None,'blocks':{}}
                for b in blocks:
                    g,u,a=[unbits(factor[f'L{b}_{key}'][j]).astype(float) for key in ('gate','up','activation')]
                    inp,att,m,outv=[unbits(factor[f'L{b}_{key}'][j]).astype(float) for key in ('input','attention','mlp','output')]
                    key=f'{stage}_{probe["role"]}_L{b}';unit_profiles[key][:6]+=np.stack([g,u,a,g*g,u*u,a*a]);unit_profiles[key][6]+=a*a*norms[b];unit_counts[key]+=1
                    coordinate_profiles[key]+=np.stack([m,m*m,inp,inp*inp]);coordinate_counts[key]+=1
                    native_e=float(np.mean(outv*outv));computed_e=float(np.mean((inp+att+m)**2))
                    item={'input_energy':float(np.mean(inp*inp)),'attention_energy':float(np.mean(att*att)),'MLP_energy':float(np.mean(m*m)),
                        'output_energy':native_e,'attention_MLP_cross_term':float(2*np.mean(att*m)),
                        'pre_MLP_MLP_cross_term':float(2*np.mean((inp+att)*m)),'FP64_residual_energy_relative_discrepancy':abs(computed_e-native_e)/max(native_e,1e-20),
                        'sum_unit_diagonal_energies':float(np.sum(a*a*norms[b])),'first_source_attention_mass':float(unbits(factor[f'L{b}_attention_probability'][:,j,0]).mean())}
                    sigkey='L6_first' if b==6 else f'L{b}_event'
                    if sigkey in signatures:
                        sig=signatures[sigkey];projection=weights[b].T@sig/2560;terms=a*projection;unit_profiles[key][7]+=terms
                        item.update(signature_cosine=float(np.dot(m,sig)/max(np.linalg.norm(m)*np.linalg.norm(sig),1e-20)),
                            signature_constant_MSE=float(np.mean((m-sig)**2)),all_unit_signature_projection_sum=float(terms.sum()),native_MLP_signature_projection=float(np.mean(m*sig)))
                    rr['blocks'][str(b)]=item
                output.append(rr)
            del ids,post,lp,packet,factor,full,trace.data
            trace.data={}
        return output
    try:
      with torch.inference_mode():
        records+=collect('train',training_pairs,main_material)
        for b in (onset,cancellation):signatures[f'L{b}_event']=coordinate_profiles[f'train_event_L{b}'][0]/coordinate_counts[f'train_event_L{b}']
        # Add TRAIN projection profiles after signatures are fixed; no held-out values contribute.
        for b in blocks:
            sig=signatures['L6_first' if b==6 else f'L{b}_event'];projection=weights[b].T@sig/2560
            for key,n in list(unit_counts.items()):
                if key.startswith('train_') and key.endswith(f'_L{b}'):unit_profiles[key][7]=unit_profiles[key][2]*projection
        npz(OUT/'frozen_training_signatures.npz',**signatures)
        if not (OUT/'frozen.json').exists():immutable(OUT/'frozen.json',{'timestamp':stamp(),'signatures_sha':sha(OUT/'frozen_training_signatures.npz'),'blocks':blocks,'new_layer_values_seen':False,
            'first_signature':'All320 old training first-position mean native activation times original BF16 down weight, FP64 reconstruction; others actual7 training event mean MLP outputs.',
            'prototype_scope':'Conditioned descriptive means only; event label is not an online prediction feature. Original H12 classifier unchanged.'})
        # Permit selected control sources without an event-row archive; verify all their full-array H12/H23/H36 hashes.
        records+=collect('new',confirmation_pairs,tail_material)
    finally:
        layers.close();trace.close();fullcheck.close();del model,layers,trace,fullcheck;gc.collect();torch.cuda.empty_cache()
    for b,w in weights.items():
        sig=signatures['L6_first' if b==6 else f'L{b}_event'];col_norm=np.sqrt(np.sum(w*w,0));cos=w.T@sig/np.maximum(col_norm*np.linalg.norm(sig),1e-20)
        npz(OUT/f'L{b}_all_parameter_column_profiles.npz',column_squared_norm_over_D=norms[b],cosine_to_frozen_signature=cos,projection_to_frozen_signature=w.T@sig/2560)
    npz(OUT/'all_native_unit_conditional_profiles.npz',**{k:(v/unit_counts[k]).astype(np.float32) for k,v in unit_profiles.items()})
    npz(OUT/'all_native_coordinate_conditional_profiles.npz',**{k:(v/coordinate_counts[k]).astype(np.float32) for k,v in coordinate_profiles.items()})
    compressed_json(OUT/'rows.json.gz',records);compressed_json(OUT/'array_identities.json.gz',identity)
    summaries={}
    for stage in ('train','new'):
      for role in ('first','event','same_ID_non_event'):
        rr=[r for r in records if r['stage']==stage and r['role']==role];summary={'tokens':len(rr),'source_groups':len({r['source_group'] for r in rr})}
        summary['blocks']={str(b):{key:float(np.mean([r['blocks'][str(b)][key] for r in rr])) for key in rr[0]['blocks'][str(b)]} for b in blocks}
        summary['native_output_entropy']=float(np.mean([r['native_output_entropy'] for r in rr]));summary['native_observed_next_NLL']=float(np.mean([r['observed_next_token_NLL'] for r in rr if r['observed_next_token_NLL'] is not None]))
        summaries[stage+'_'+role]=summary
    cosine={a+'__'+b:float(np.dot(x,y)/max(np.linalg.norm(x)*np.linalg.norm(y),1e-20)) for a,x in signatures.items() for b,y in signatures.items() if a<b}
    result={'timestamp':stamp(),'blocks':blocks,'all_native_identities_bitwise':True,'sources':len(identity),'probes':len(records),'condition_summaries':summaries,
        'frozen_signature_cosines':cosine,'new_event_onset_counts':dict(Counter(str(r['maximum_H12_H23_increment_block']) for r in records if r['stage']=='new' and r['role']=='event')),
        'new_event_decline_counts':dict(Counter(str(r['late_maximum_log_energy_decline_block']) for r in records if r['stage']=='new' and r['role']=='event')),
        'unit_profile_row_meaning':['gate_mean','up_mean','activation_mean','gate_mean_square','up_mean_square','activation_mean_square','unit_down_column_diagonal_energy','signed_projection_to_frozen_signature'],
        'coordinate_profile_row_meaning':['MLP_mean','MLP_mean_square','input_mean','input_mean_square'],'profile_counts':dict(unit_counts),
        'matched_controls':{'pairs':len(confirmation_pairs),'with_control':sum(p['same_token_non_event'] is not None for p in confirmation_pairs),'position_distance_counts':dict(Counter(str(p['absolute_position_difference']) for p in confirmation_pairs))},
        'limits':protocol['limits']+' Exact parameter sums are known computational identities, not alone an extracted language mechanism; conditional cross-layer signature consistency is the empirical candidate. Native output statistics are observational, not effects of removing the amplification.'}
    save(OUT/'result.json',result);ledger('joint_native_regime_paths_and_all_unit_geometry',time.monotonic()-start,sources=len(identity));guard()
    print('NATIVE_REGIMES_COMPLETE',result,flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
