"""All-layer natural replays for rare internal events and same-token non-event controls."""
from collections import defaultdict,Counter
import gc
from rdc_joint_common import *
from rdc_joint_capture import ledger,array_identity
from phase2721_rdc_joint_native_factors import FactorTrace

OUT=BASE/'extension/event_trace'


class Layers:
    def __init__(self,model):
        self.data={};self.handles=[];self.positions=[]
        self.handles.append(model.get_input_embeddings().register_forward_hook(lambda m,a,o:self.data.update({0:bits(o[0,self.positions])})))
        for i,block in enumerate(model.model.layers,1):
            def hook(m,a,o,i=i):self.data[i]=bits(o[0,self.positions])
            self.handles.append(block.register_forward_hook(hook))
    def close(self):
        for h in self.handles:h.remove()


def main():
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    if (OUT/'result.json').exists():return
    start=time.monotonic();guard(20*1024**2)
    pairs=read(BASE/'extension/event_trace_selection.json')['pairs'];index={r['sample_id']:r for r in rows()+rows(True)}
    requests={}
    for pair in pairs:
        for role,key in [('event','event'),('same_ID_non_event','same_token_non_event')]:
            m=pair[key]
            if m:requests[(m['sample_id'],m['position'])]={'sample_id':m['sample_id'],'position':m['position'],'role':role,'split':m['split'],'source_group':m['source_group'],'token_id':m['token_id'],'token':m['token']}
    grouped=defaultdict(list)
    for r in requests.values():grouped[r['sample_id']].append(r)
    # Entire layer pass first: no selected block influences the event layer inventory.
    model,tok=load_native('qwen4');device=model.get_input_embeddings().weight.device;observer=Layers(model);record=[];allfields={}
    try:
      with torch.inference_mode():
        for sid,rr in grouped.items():
            material=index[sid];positions=sorted({0,*[r['position'] for r in rr]});observer.positions=positions;observer.data={}
            ids=torch.tensor([material['prompt_ids']],device=device);post=model.model(input_ids=ids,use_cache=False).last_hidden_state
            layers=np.stack([observer.data[i] for i in range(37)]);original=field(material,material['split']=='confirmation')
            for li in (12,23,36):assert np.array_equal(layers[li],original[f'h{li}'][positions])
            npz(OUT/'fields'/f'{sid}.npz',layers=layers,positions=np.array(positions),postnorm=bits(post[0,positions]))
            allfields[sid]=(layers,positions)
            for row in rr:
                h=unbits(layers[:,positions.index(row['position'])]).astype(float);e=(h*h).mean(-1);increment=np.mean(np.diff(h,axis=0)**2,1)
                record.append({**row,'energy_by_layer':e.tolist(),'increment_squared_energy':increment.tolist(),
                    'max_log_energy_growth_block':int(np.argmax(np.diff(np.log(np.maximum(e,1e-20))))),
                    'max_squared_increment_block':int(np.argmax(increment)),
                    'H12_to_H23_max_increment_block':int(np.argmax(increment[12:23])+12)})
            del ids,post,original
        train=[r for r in record if r['split']=='train' and r['role']=='event'];weights=np.mean([r['increment_squared_energy'] for r in train],0);selected=int(np.argmax(weights[12:23])+12)
        immutable(OUT/'selected_block.json',{'timestamp':stamp(),'training_events':len(train),'block_zero_index':selected,'all36_block_training_event_increment_energy':weights.tolist(),
            'selection':'Largest training event mean squared full-coordinate residual increment within target H12-to-H23 blocks12..22; later decay is separately retained. Only determines local native-factor audit, not the primary structure or a semantic necessity claim.'})
        observer.close();trace=FactorTrace(model,[selected]);factor_records=[];sums=defaultdict(lambda:np.zeros((3,9728),float));count=Counter()
        try:
          for sid,rr in grouped.items():
            material=index[sid];positions=allfields[sid][1];trace.data={};ids=torch.tensor([material['prompt_ids']],device=device)
            model.model(input_ids=ids,use_cache=False);d=trace.data[selected];block=model.model.layers[selected]
            a=block.mlp.act_fn(d['gate'])*d['up'];assert torch.equal(a,d['activation'])
            assert torch.equal(block.mlp.down_proj(a),d['mlp'])
            residual=d['input']+d['attention'];assert torch.equal(block.post_attention_layernorm(residual),d['mlp_input'])
            assert torch.equal(residual+d['mlp'],d['output'])
            packet={k:bits(d[k][0,positions]) for k in ('input','attention','mlp_input','gate','up','activation','mlp','output')}
            packet['attention_probability']=bits(d['attention_probability'][0,:,positions]);packet['positions']=np.array(positions)
            npz(OUT/'factors'/f'{sid}.npz',**packet)
            for row in rr:
                p=row['position'];j=positions.index(p);key=row['split']+'_'+row['role']
                v=np.stack([unbits(packet[k][j]).astype(float) for k in ('gate','up','activation')]);sums[key]+=v*v;count[key]+=1
                inp=d['input'][0,p].float();att=d['attention'][0,p].float();mlp=d['mlp'][0,p].float();update=(d['output'][0,p].float()-inp)
                fp= d['activation'][0,p].float()@block.mlp.down_proj.weight.float().T
                factor_records.append({**row,'block':selected,'input_energy':float(inp.square().mean()),'attention_energy':float(att.square().mean()),'mlp_energy':float(mlp.square().mean()),
                    'output_energy':float(d['output'][0,p].float().square().mean()),'increment_energy':float(update.square().mean()),
                    'attention_MLP_cross_term':float(2*(att*mlp).mean()),'arithmetic_sum_update_residual_MSE':float((update-att-mlp).square().mean()),
                    'all_unit_FP32_down_relative_MSE':float((fp-mlp).square().mean()/mlp.square().mean().clamp_min(1e-20)),
                    'attention_first_source_mass':float(d['attention_probability'][0,:,p,0].float().mean())})
            del ids,packet,d,a,residual
          npz(BASE/'extension/event_native_all_unit_squared_profiles.npz',**{k:(v/count[k]).astype(np.float32) for k,v in sums.items()})
        finally:trace.close()
    finally:
        observer.close();del observer,model;gc.collect();torch.cuda.empty_cache()
    compressed_json(OUT/'rows.json.gz',record);compressed_json(OUT/'factor_rows.json.gz',factor_records)
    groups={}
    for split in ('train','confirmation'):
      for role in ('event','same_ID_non_event'):
        rr=[r for r in factor_records if r['split']==split and r['role']==role]
        groups[split+'_'+role]={'tokens':len(rr),'groups':len({r['source_group'] for r in rr}),
            'means':{k:float(np.mean([r[k] for r in rr])) for k in ('input_energy','attention_energy','mlp_energy','output_energy','increment_energy','attention_first_source_mass')} if rr else {}}
    result={'timestamp':stamp(),'sources':len(grouped),'unique_token_probes':len(record),'selected_native_block':selected,'all_native_factor_identities':True,
        'layer_origin_counts':{s:dict(Counter(str(r['H12_to_H23_max_increment_block']) for r in record if r['split']==s and r['role']=='event')) for s in ('train','confirmation')},
        'groups':groups,'same_ID_pair_count':sum(p['same_token_non_event'] is not None for p in pairs),
        'limits':'Conditional all-coordinate observations and exact original parameter factorization; sparse event count and reused controls do not establish independent universal gates. Same token ID controls identity, not all syntax, position or history. All source states are natural, not patched.'}
    save(OUT/'result.json',result);ledger('joint_all_layer_internal_event_native_trace',time.monotonic()-start,sources=len(grouped));guard()
    print('EVENT_TRACE_COMPLETE',result,flush=True)


if __name__=='__main__':main()
