"""Trace actual all-unit factors at observed amplification and later comparison blocks."""
import gc
from rdc_joint_common import *
from rdc_joint_capture import ledger,array_identity
from phase2721_rdc_joint_confirmation import frozen_check


class FactorTrace:
    def __init__(self,model,blocks):
        self.model,self.blocks=model,blocks
        self.data={};self.handles=[]
        for li in blocks:
            block=model.model.layers[li]
            def before(m,a,li=li):self.data.setdefault(li,{})['input']=a[0]
            self.handles.append(block.register_forward_pre_hook(before))
            for key,module in [('attention',block.self_attn),('mlp_input',block.post_attention_layernorm),
                               ('gate',block.mlp.gate_proj),('up',block.mlp.up_proj),('mlp',block.mlp),('output',block)]:
                def hook(m,a,o,li=li,key=key):
                    if key=='attention':
                        self.data[li]['attention_probability']=o[1]
                        o=o[0]
                    self.data[li][key]=o
                self.handles.append(module.register_forward_hook(hook))
            def down_pre(m,a,li=li):self.data[li]['activation']=a[0]
            self.handles.append(block.mlp.down_proj.register_forward_pre_hook(down_pre))
    def close(self):
        for h in self.handles:h.remove()


def main():
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    frozen_check();out=BASE/'native_factors'
    if (out/'result.json').exists():return
    start=time.monotonic();selected=read(BASE/'layer_atlas/result.json')['training_selected_amplification_block_zero_index']
    blocks=sorted(set([selected,12,23]));train=[r for r in rows() if r['split']=='train'];fresh=rows(True)
    with np.load(BASE/'layer_atlas/full_coordinate_moments.npz') as z:common=z['train_common_increment']
    targets={li:[int(np.argmax(np.abs(common[role,li]))) for role in (0,2)] for li in blocks}
    immutable(out/'protocol.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'blocks_zero_index':blocks,
        'training_sources':len(train),'fresh_sources':len(fresh),'full_native_positions':'first,second,and two anchors; all2560 residual coordinates,all9728 MLP units,all32 attention heads and every earlier source.',
        'selected_block':'From Phase2719 training-only amplification criterion; blocks12/23 are predeclared comparisons.',
        'example_target_coordinates_from_all_training_layer_means':targets,
        'unit_selection':'Only after full training all-unit accumulation, highest absolute mean contribution to each already-chosen diagnostic coordinate. This is a labeled illustrative path, never the primary mechanism definition or Top-K replacement.',
        'raw_retention':'All four-position full factors for every fresh source, first two training sources per language; all training sources have complete array identities and all-unit moments/replay recipe.',
        'native_checks':'Same full-token shape SiLU(gate)*up, down projection and both residual additions bitwise; FP32 all-coordinate/all-unit sums versus native BF16 logged separately.',
        'scope':'Natural unchanged model forwards, no donor patches or unit deletion; contribution identities establish original computational links, not semantic necessity/sufficiency.'})
    model,tok=load_native('qwen4');device=model.get_input_embeddings().weight.device
    trace=FactorTrace(model,blocks);moments={};records=[];scalar_examples={};fixtures=set()
    for lang in ('en','zh'):fixtures.update(r['sample_id'] for r in [r for r in train if r['language']==lang][:2])
    native_checks=[]
    try:
      with torch.inference_mode():
        for si,material in enumerate((train,fresh)):
          phase='train' if si==0 else 'fresh'
          if si==1:
            for li in blocks:
                av=moments[('train',li)]['target_contribution_sum']/moments[('train',li)]['count'][:,None,None]
                scalar_examples[li]=[int(np.argmax(np.abs(av[role,role]))) for role in range(2)]
            save(out/'training_selected_unit_examples.json',{'timestamp':stamp(),'target_coordinates':targets,'units':scalar_examples,
                'selected_before_first_fresh_factor_forward':True,'scope':'Illustrative strongest mean signed contribution, all units kept and analyzed.'})
          for i,r in enumerate(material):
            positions=[0,1,*r['anchors']]
            ids=torch.tensor([r['prompt_ids']],device=device)
            trace.data={};post=model.model(input_ids=ids,use_cache=False).last_hidden_state
            packet={};rr={'sample_id':r['sample_id'],'source_group':r['source_group'],'language':r['language'],'scope':phase,'blocks':{}}
            for li in blocks:
                d=trace.data[li];block=model.model.layers[li]
                act=block.mlp.act_fn(d['gate'])*d['up']
                assert torch.equal(act,d['activation'])
                down=block.mlp.down_proj(act)
                assert torch.equal(down,d['mlp'])
                residual=d['input']+d['attention']
                assert torch.equal(block.post_attention_layernorm(residual),d['mlp_input'])
                assert torch.equal(residual+d['mlp'],d['output'])
                if len(native_checks)<12:native_checks.append({'source':r['sample_id'],'block':li,'activation_down_norm_both_residuals_bitwise':True})
                for key in ('input','attention','mlp_input','gate','up','activation','mlp','output'):
                    packet[f'L{li}_{key}']=bits(d[key][0,positions])
                probability=d['attention_probability'][0,:,positions]
                packet[f'L{li}_attention_probability']=bits(probability)
                gate=d['gate'][0,positions].float();up=d['up'][0,positions].float();aa=d['activation'][0,positions].float()
                xx=d['mlp_input'][0,positions].float()
                fp_gate=xx@block.mlp.gate_proj.weight.float().T
                fp_up=xx@block.mlp.up_proj.weight.float().T
                fp_down=aa@block.mlp.down_proj.weight.float().T
                row={'gate_FP32_relative_MSE':float((fp_gate-gate).square().mean()/gate.square().mean().clamp_min(1e-20)),
                    'up_FP32_relative_MSE':float((fp_up-up).square().mean()/up.square().mean().clamp_min(1e-20)),
                    'down_FP32_relative_MSE':float((fp_down-d['mlp'][0,positions].float()).square().mean()/d['mlp'][0,positions].float().square().mean().clamp_min(1e-20)),
                    'attention_first_source_mass_by_query':probability[:,:,0].float().mean(0).cpu().tolist(),
                    'attention_MSE_energy_by_query':d['attention'][0,positions].float().square().mean(-1).cpu().tolist(),
                    'MLP_MSE_energy_by_query':d['mlp'][0,positions].float().square().mean(-1).cpu().tolist(),
                    'input_energy_by_query':d['input'][0,positions].float().square().mean(-1).cpu().tolist(),
                    'output_energy_by_query':d['output'][0,positions].float().square().mean(-1).cpu().tolist()}
                rr['blocks'][str(li)]=row
                key=(phase,li)
                if key not in moments:
                    moments[key]={'count':np.zeros(2,int),'gate_sum':np.zeros((2,9728)), 'gate_square':np.zeros((2,9728)),
                        'up_sum':np.zeros((2,9728)),'up_square':np.zeros((2,9728)), 'activation_sum':np.zeros((2,9728)),
                        'activation_square':np.zeros((2,9728)), 'target_contribution_sum':np.zeros((2,2,9728)),
                        'input_square':np.zeros((2,2560)), 'attention_energy':np.zeros((2,2560)), 'mlp_energy':np.zeros((2,2560))}
                mm=moments[key]
                contribution=aa[:,None,:]*block.mlp.down_proj.weight[targets[li]].float()[None]
                for role,ix in enumerate(([0],[2,3])):
                    mm['count'][role]+=len(ix)
                    for name,value in [('gate',gate),('up',up),('activation',aa)]:
                        vv=value[ix].cpu().numpy().astype(float)
                        mm[name+'_sum'][role]+=vv.sum(0)
                        mm[name+'_square'][role]+=(vv*vv).sum(0)
                    mm['target_contribution_sum'][role]+=contribution[ix].cpu().numpy().astype(float).sum(0)
                    mm['input_square'][role]+=xx[ix].square().cpu().numpy().sum(0)
                    mm['attention_energy'][role]+=d['attention'][0,[positions[j] for j in ix]].float().square().cpu().numpy().sum(0)
                    mm['mlp_energy'][role]+=d['mlp'][0,[positions[j] for j in ix]].float().square().cpu().numpy().sum(0)
                if si==1:
                    examples=[]
                    for role,unit in enumerate(scalar_examples[li]):
                        coordinate=targets[li][role]
                        gweight=block.mlp.gate_proj.weight[unit].float()
                        uweight=block.mlp.up_proj.weight[unit].float()
                        x_index=int(np.argmax(np.sqrt(moments[('train',li)]['input_square'][role]/moments[('train',li)]['count'][role])*gweight.abs().cpu().numpy()))
                        at=0 if role==0 else 2
                        examples.append({'role':role,'target_coordinate':coordinate,'MLP_unit':unit,'illustrative_input_coordinate':x_index,
                            'gate_scalar_weight':float(gweight[x_index]),'up_scalar_weight':float(uweight[x_index]),
                            'down_scalar_weight':float(block.mlp.down_proj.weight[coordinate,unit]),
                            'input_value':float(xx[at,x_index]),'gate_one_input_product':float(xx[at,x_index]*gweight[x_index]),
                            'up_one_input_product':float(xx[at,x_index]*uweight[x_index]),'native_full_gate_value':float(gate[at,unit]),
                            'native_full_up_value':float(up[at,unit]),'native_product_activation':float(aa[at,unit]),
                            'one_unit_down_contribution':float(contribution[at,role,unit]),'sum_all_unit_FP32_output_coordinate':float(fp_down[at,coordinate]),
                            'native_MLP_output_coordinate':float(d['mlp'][0,positions[at],coordinate])})
                    row['illustrative_scalar_paths']=examples
            identities={k:array_identity(a) for k,a in packet.items()}
            rr['array_identities']=identities
            if si==1 or r['sample_id'] in fixtures:
                path=out/'fields'/f'{r["sample_id"]}.npz'
                guard(sum(v.nbytes for v in packet.values())+1024**2)
                npz(path,**packet,positions=np.array(positions))
                rr['field_sha']=sha(path)
            records.append(rr)
            compressed_json(out/'commits'/f'{r["sample_id"]}.json.gz',rr)
            del ids,post,packet,d,act,down,gate,up,aa,xx,fp_gate,fp_up,fp_down,contribution,probability,residual
            trace.data={}
            if (i+1)%64==0:print('JOINT_NATIVE_FACTORS',phase,i+1,len(material),flush=True)
        for key,mm in moments.items():
            normalized={}
            for name,v in mm.items():
                normalized[name]=v if name=='count' else (v/mm['count'].reshape((2,)+(1,)*(v.ndim-1))).astype(np.float32)
            npz(out/f'{key[0]}_L{key[1]}_all_unit_and_coordinate_profiles.npz',**normalized)
        for li,units in scalar_examples.items():
            block=model.model.layers[li]
            npz(out/f'L{li}_illustrative_parameter_rows.npz',gate_rows=bits(block.mlp.gate_proj.weight[units]),up_rows=bits(block.mlp.up_proj.weight[units]),
                down_columns=bits(block.mlp.down_proj.weight[:,units]),units=np.array(units),target_coordinates=np.array(targets[li]))
    finally:
        trace.close();del trace,model;gc.collect();torch.cuda.empty_cache()
    aggregate={}
    for phase in ('train','fresh'):
        rr=[r for r in records if r['scope']==phase]
        aggregate[phase]={}
        for li in blocks:
            keys=['gate_FP32_relative_MSE','up_FP32_relative_MSE','down_FP32_relative_MSE','attention_first_source_mass_by_query',
                  'attention_MSE_energy_by_query','MLP_MSE_energy_by_query','input_energy_by_query','output_energy_by_query']
            aggregate[phase][str(li)]={k:np.mean([r['blocks'][str(li)][k] for r in rr],0).tolist() for k in keys}
    report={'timestamp':stamp(),'blocks':blocks,'training_sources':len(train),'fresh_sources':len(fresh),'all_factor_identities_bitwise':True,
        'same_shape_checks':native_checks,'summaries':aggregate,'illustrative_units':scalar_examples,'target_coordinates':targets,
        'limits':'Known computational factorization plus observed distributions; largest mean contribution paths are illustrative, not a sparse mechanism, semantic label, unique cause, necessity or sufficiency. Attention first-source mass is measured per layer/query and is distinct from H norm.',
        'full_coordinate_and_unit_retention':'All fresh four-position native factors persisted; all train complete-factor array identities plus all-unit/coordinate moments and exact replay recipe;4 training fixtures persisted.'}
    save(out/'result.json',report);ledger('joint_native_all_unit_factors',time.monotonic()-start,sources=len(train)+len(fresh));guard()
    print('JOINT_NATIVE_FACTORS_COMPLETE',aggregate,usage(),flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
