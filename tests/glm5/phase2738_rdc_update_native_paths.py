"""Observed attention sources through every coordinate and every MLP unit.

The source allocation fixes observed attention and RMS denominator. It is an
accounting convention, not a counterfactual, unique attribution, or early forecast.
"""
import gc
from rdc_update_common import *

def material():
    language=gzread(BASE/'language_material.json.gz');program=gzread(BASE/'program_material.json.gz');rows=[]
    for family in sorted({r['family'] for r in language}):
        for split in ('language_train','language_test'):
            group=sorted({r['source_group'] for r in language if r['family']==family and r['split']==split})[0]
            rows.extend(r for r in language if r['source_group']==group and r['answer_style']=='direct')
    group=sorted({r['source_group'] for r in program if r['split']=='mixed_holdout'})[0]
    rows.extend(r for r in program if r['source_group']==group);assert len(rows)==24;return rows

def main():
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    out=BASE/'native_paths';start=time.monotonic()
    if (out/'result.json').exists():return
    rows=material();immutable(out/'protocol.json',{'timestamp':stamp(),'source':snapshot(__file__),'samples':[r['sample_id'] for r in rows],
      'blocks':[16,35],'positions':'All predeclared body/prompt anchors of24 fixed samples; all current source positions,32heads,8KVheads,2560coordinates,9728units.',
      'allocation':'Observed attention weights and values contracted with original Wo. Observed RMS denominator fixed; source gate/up reads and symmetric bilinear activation allocation. Native rounding remainders explicit.',
      'prediction_warning':'This is native computational provenance and accounting; observed attention, gate and downstream states are not inputs to the separate H12 forecast.',
      'selection':'First train and first held semantic group in each of5language families, EN/ZH direct; first mixed-held group all4expressions. Not selected by model correctness.'})
    compressed(out/'material.json.gz',rows);guard(800*1024**2);model,tok=load_native('qwen4');model.eval();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;device=model.get_input_embeddings().weight.device;data={};handles=[];positions=[]
    def put(key,value):data[key]=value.detach()
    for b in (16,35):
        layer=model.model.layers[b]
        def pre(m,a,b=b):put(f'{b}_residual',a[0][0,positions])
        handles.append(layer.register_forward_pre_hook(pre))
        handles.append(layer.self_attn.v_proj.register_forward_hook(lambda m,a,o,b=b:put(f'{b}_v',o[0])))
        def attn(m,a,o,b=b):
            put(f'{b}_attention',o[0][0,positions]);put(f'{b}_A',o[1][0,:,positions,:])
        handles.append(layer.self_attn.register_forward_hook(attn))
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(lambda m,a,b=b:put(f'{b}_rms_input',a[0][0,positions])))
        handles.append(layer.post_attention_layernorm.register_forward_hook(lambda m,a,o,b=b:put(f'{b}_x',o[0,positions])))
        for name in ('gate_proj','up_proj','down_proj'):
            handles.append(getattr(layer.mlp,name).register_forward_hook(lambda m,a,o,b=b,name=name:put(f'{b}_{name}',o[0,positions])))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(lambda m,a,b=b:put(f'{b}_activation',a[0][0,positions])))
    reports=[]
    try:
      with torch.inference_mode():
        for i,row in enumerate(rows):
            positions=row['anchors'];data.clear();ids=torch.tensor([row['prompt_ids']],device=device)
            post=model.model(input_ids=ids,use_cache=False).last_hidden_state;frozen_post=post.clone();arrays={};stats=[]
            for b in (16,35):
                layer=model.model.layers[b];a=data[f'{b}_A'].float();v=data[f'{b}_v'].float().reshape(len(row['prompt_ids']),8,128)
                wo=layer.self_attn.o_proj.weight.float().reshape(2560,32,128)
                vv=v.repeat_interleave(4,dim=1)
                # Every head and source: C[a,s,d] = sum_hf A[h,a,s] V[s,h,f] Wo[d,h,f].
                c=torch.einsum('has,shf,dhf->asd',a,vv,wo)
                residual=data[f'{b}_residual'].float();r=data[f'{b}_rms_input'].float();x=data[f'{b}_x'].float()
                den=(r.square().mean(-1,keepdim=True)+layer.post_attention_layernorm.variance_epsilon).sqrt()
                gamma=layer.post_attention_layernorm.weight.float();xs=c*gamma[None,None,:]/den[:,None,:]
                wg=layer.mlp.gate_proj.weight.float();wu=layer.mlp.up_proj.weight.float();wd=layer.mlp.down_proj.weight.float()
                gs=xs@wg.T;us=xs@wu.T;g=data[f'{b}_gate_proj'].float();u=data[f'{b}_up_proj'].float();act=data[f'{b}_activation'].float()
                otherg=g-gs.sum(1);otheru=u-us.sum(1)
                alloc=.5*torch.sigmoid(g)[:,None,:]*(gs*u[:,None,:]+us*g[:,None,:]);other=.5*torch.sigmoid(g)*(otherg*u+otheru*g)
                act_remainder=act-alloc.sum(1)-other
                written=alloc@wd.T;native=data[f'{b}_down_proj'].float();otherwritten=(other+act_remainder)@wd.T
                readback=written.sum(1)+otherwritten;rounding=native-readback
                arrays.update({f'L{b}_'+k:v0.cpu().numpy() for k,v0 in {
                  'source_attention_write':c,'source_gate_read':gs,'source_up_read':us,'source_MLP_write':written,
                  'native_A':a,'native_V':v,'residual':residual,'rms_input':r,'x':x,'gate':g,'up':u,'activation':act,'mlp':native,
                  'rms_denominator':den,'norm_weight':gamma,'other_gate_read':otherg,'other_up_read':otheru,
                  'activation_rounding_remainder':act_remainder,'MLP_rounding_remainder':rounding,
                  'attention_rounding_remainder':data[f'{b}_attention'].float()-c.sum(1),
                  'residual_rounding_remainder':r-residual-data[f'{b}_attention'].float(),
                  'norm_allocation_remainder':x-r*gamma/den}.items()})
                stats.append({'block':b,'attention_relative_RMS_error':float((c.sum(1)-data[f'{b}_attention'].float()).norm()/data[f'{b}_attention'].float().norm()),
                  'MLP_FP32_vs_native_relative_RMS':float(rounding.norm()/native.norm()),
                  'activation_allocation_identity_relative_RMS':float((alloc.sum(1)+other-torch.sigmoid(g)*g*u).norm()/(torch.sigmoid(g)*g*u).norm()),
                  'unit_allocation_scope':'All9728units and all2560output coordinates. Other includes base residual, normalization and gate/up rounding; not a missing-source estimate.'})
                assert stats[-1]['activation_allocation_identity_relative_RMS']<1e-5
                del a,v,wo,vv,c,xs,wg,wu,wd,gs,us,alloc,written
            # All observers must leave the same native forward unchanged.
            data.clear();again=model.model(input_ids=ids,use_cache=False).last_hidden_state;assert torch.equal(frozen_post,again)
            arrays.update(token_ids=np.array(row['prompt_ids'],np.int32),positions=np.array(positions))
            path=out/'fields'/f'{row["sample_id"]}.npz';npz(path,**arrays)
            rec={'sample_id':row['sample_id'],'source_group':row['source_group'],'split':row['split'],'cohort':row['cohort'],
              'tokens':len(row['prompt_ids']),'positions':positions,'audits':stats,'same_shape_repeat_bitwise':True,'archive_sha256':sha(path)}
            save(out/'commits'/f'{row["sample_id"]}.json',rec);reports.append(rec);data.clear();del arrays,post,frozen_post,again,ids
            guard(35*1024**2);print('SOURCE_NATIVE_PATH',i+1,len(rows),round(time.monotonic()-start,1),flush=True)
        result={'timestamp':stamp(),'source':snapshot(__file__),'rows':len(rows),'visible_source_anchor_block_paths':sum(sum(p+1 for p in r['positions'])*2 for r in reports),
          'stored_source_slots_including_causally_masked_future_zeros':sum(len(r['positions'])*r['tokens']*2 for r in reports),
          'reports':reports,'seconds':time.monotonic()-start,'scope':'Exact stored-factor contraction in stated FP32 arithmetic plus native remainders. This narrows computational provenance, not an identification of one language concept with one neuron.'}
        save(out/'result.json',result);ledger('source_to_all_coordinates_units_writes',result['seconds'])
    except Exception as exc:failure(out,start,exc);raise
    finally:
        for h in handles:h.remove()
        del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
