"""Full native-tail JVP/VJP with FP32 smooth and BF16 finite-change controls."""
import argparse
from rdc_construction_common import *
from rdc_construction_storage import FIELD_STORE,verify_storage
from rdc_native_tail import loaded_block,checkpoint_tensor,block_call,final_norm,cuda_singleton,CUDA_TASKS,config
from phase2746_rdc_tail_fixtures import OUT,freeze,STARTS


def metrics(predicted,actual):
    import torch
    p,a=predicted.double().flatten(),actual.double().flatten();den=float(a.square().sum())
    return {'relative_L2':float((p-a).norm())/max(den**.5,1e-30),
        'cosine':float(p@a)/max(float(p.norm()*a.norm()),1e-30),
        'predicted_L2':float(p.norm()),'actual_L2':den**.5}


def tensor(a,dtype=None):
    import torch
    decoded=unbits(a) if a.dtype==np.uint16 else a
    return torch.from_numpy(np.ascontiguousarray(decoded)).to(device='cuda',dtype=dtype or torch.float32)


def initial_state(row,arrays,begin):
    import torch
    from rdc_relation_native_parameters import parameter,decode
    x=tensor(arrays['hidden'][begin])[None,None]
    index=STARTS.index(begin)
    seed=int(rank('tail-direction/'+row['fixture_id'])[:16],16)
    rng=np.random.default_rng(seed)
    random=tensor(rng.choice(np.array([-1.,1.],np.float32),size=2560))
    previous=tensor(arrays['native_previous_MLP_writes'][index])
    unit=int(rank('tail-fixed-unit/'+str(begin))[:16],16)%9728
    down=parameter(ROOT,f'model.layers.{begin-1}.mlp.down_proj.weight',MODELS['qwen4'])
    column=tensor(decode(down[:,unit]))
    directions=torch.stack([random,previous,column])[:,None]
    scale=x.square().mean().sqrt()
    directions=directions/directions.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-20)*scale
    eps=torch.tensor([.001,.01,.1],device='cuda')
    plus=x[None]+eps[:,None,None,None]*directions[None]
    minus=x[None]-eps[:,None,None,None]*directions[None]
    assert plus.shape==(3,3,1,2560)
    return {'x':x,'tangent':directions,'plus':plus.reshape(9,1,2560),'minus':minus.reshape(9,1,2560),
        'native':x.bfloat16(),'native_plus':plus.reshape(9,1,2560).bfloat16(),'native_minus':minus.reshape(9,1,2560).bfloat16(),
        'initial':directions.clone(),'unit':unit,'seed':seed,
        'primal_trace':[x[0,0].cpu().numpy().copy()],
        'tangent_trace':[directions[:,0].cpu().numpy().copy()],
        'shape_float32_relative_max':0.0}


def fixture_inputs(arrays,block,dtype):
    key=tensor(arrays['prefix_keys'][block],dtype)[None]
    value=tensor(arrays['prefix_values'][block],dtype)[None]
    return key,value,tensor(arrays['cos'],dtype),tensor(arrays['sin'],dtype)


def forward_block(layer,block,state,arrays):
    import torch
    key,value,cos,sin=fixture_inputs(arrays,block,torch.float32)
    def fn(x):
        n=x.shape[0]
        return block_call(layer,x,key.expand(n,-1,-1,-1),value.expand(n,-1,-1,-1),cos.expand(n,-1,-1),sin.expand(n,-1,-1),block)
    x=state['x'];directions=state['tangent']
    primal,tangent=torch.func.jvp(fn,(x.expand(3,-1,-1).clone(),),(directions,))
    with torch.no_grad():
        separate=fn(x)
        difference=float((primal[0:1]-separate).norm()/separate.norm().clamp_min(1e-30))
        state['shape_float32_relative_max']=max(state['shape_float32_relative_max'],difference)
        state['x']=separate.detach();state['tangent']=tangent.detach()
        state['plus']=fn(state['plus']);state['minus']=fn(state['minus'])
        state['primal_trace'].append(state['x'][0,0].cpu().numpy().copy())
        state['tangent_trace'].append(state['tangent'][:,0].cpu().numpy().copy())


def native_forward_block(layer,block,state,arrays):
    import torch
    key,value,cos,sin=fixture_inputs(arrays,block,torch.bfloat16)
    def fn(x):return block_call(layer,x,key,value,cos,sin,block)
    with torch.no_grad():
        state['native']=fn(state['native'])
        # Preserve B1 throughout the BF16 branch. A changed batch shape must
        # not be mistaken for a finite-change effect of the chosen direction.
        for name in ['native_plus','native_minus']:
            state[name]=torch.cat([fn(row[None]) for row in state[name]],0)


def evaluate_endpoints(states,fixtures,begin,folder):
    import torch
    eps=[.001,.01,.1];c=config()
    weight=checkpoint_tensor('model.norm.weight',torch.float32)
    # Q4 natively ties lm_head to the input embedding; its checkpoint index
    # therefore need not store a second lm_head tensor.
    readout_name='model.embed_tokens.weight' if c.tie_word_embeddings else 'lm_head.weight'
    wu=checkpoint_tensor(readout_name,torch.float32)
    output=[];posts=[];jposts=[];native_posts=[];native_finite=[];cotangents=[]
    for state,(row,a) in zip(states,fixtures):
        fn=lambda x:final_norm(x,weight,c.rms_norm_eps)
        post,jpost=torch.func.jvp(fn,(state['x'].expand(3,-1,-1).clone(),),(state['tangent'],))
        post=post[0,0];jpost=jpost[:,0]
        pplus=fn(state['plus'])[:,0];pminus=fn(state['minus'])[:,0]
        npost=final_norm(state['native'],weight.bfloat16(),c.rms_norm_eps)[0,0]
        assert np.array_equal(bits(npost),a['native_BF16_tail_postnorm'][STARTS.index(begin)]),('BF16 same-shape original tail replay differs',row['fixture_id'],begin)
        nplus=final_norm(state['native_plus'],weight.bfloat16(),c.rms_norm_eps)[:,0]
        nminus=final_norm(state['native_minus'],weight.bfloat16(),c.rms_norm_eps)[:,0]
        logits=wu@post;jlogits=jpost@wu.T
        probability=logits.double().softmax(-1)
        jprobability=probability[None]*(jlogits.double()-(jlogits.double()*probability).sum(-1,keepdim=True))
        rng=np.random.default_rng(int(rank('tail-output-cotangent/'+row['fixture_id'])[:16],16))
        cot=tensor(rng.choice(np.array([-1.,1.],np.float32),size=wu.shape[0]))/wu.shape[0]**.5
        grad_post=wu.T@cot
        # Reverse through finalnorm now. Remaining blocks are replayed below.
        h=state['x'].detach().requires_grad_(True)
        reverse=torch.autograd.grad(final_norm(h,weight,c.rms_norm_eps),h,grad_post[None,None])[0]
        state['reverse']=reverse.detach();state['cotangent_post']=grad_post.detach()
        state['output_direction_dots']=(jlogits.double()@cot.double()).cpu().numpy()
        state['normed_adjoint_denominator']=float(jlogits.double().norm()*cot.double().norm())
        # Bare readout under identity residual transport is explicitly only a
        # control, evaluated at the same final normalization point.
        _,bare=torch.func.jvp(fn,(state['x'].expand(3,-1,-1).clone(),),(state['initial'],))
        bare_logits=bare[:,0]@wu.T
        comparisons=[]
        for e,epsilon in enumerate(eps):
            for d in range(3):
                index=e*3+d
                symmetric=(pplus[index]-pminus[index])/(2*epsilon)
                finite=(pplus[index]-post)/epsilon
                sym_logits=wu@symmetric;finite_logits=wu@finite
                lp=wu@pplus[index];lm=wu@pminus[index]
                finite_probability=(lp.double().softmax(-1)-lm.double().softmax(-1))/(2*epsilon)
                native_difference=(nplus[index].float()-nminus[index].float())/(2*epsilon)
                comparisons.append({'epsilon':epsilon,'direction':d,
                    'postnorm_central':metrics(jpost[d],symmetric),
                    'postnorm_one_sided':metrics(jpost[d],finite),
                    'full_vocab_logits_central':metrics(jlogits[d],sym_logits),
                    'full_vocab_logits_one_sided':metrics(jlogits[d],finite_logits),
                    'full_vocab_probability_central':metrics(jprobability[d],finite_probability),
                    'BF16_postnorm_central':metrics(jpost[d],native_difference),
                    'bare_readout_control':metrics(bare_logits[d],sym_logits),
                    'native_changed_coordinate_fraction':float((nplus[index]!=nminus[index]).float().mean())})
        record={k:row[k] for k in ['fixture_id','sample_id','source_group','family','kind','split','step']}
        record.update(start=begin,fixed_previous_unit=state['unit'],comparisons=comparisons,
            native_tail_replay_bit_equal=True,
            smooth_vs_native_base=metrics(post,npost.float()),
            maximum_FP32_B3_vs_B1_relative_difference=state['shape_float32_relative_max'],
            cross_layer_direction_gain=(state['tangent'].double().flatten(1).norm(dim=1)/state['initial'].double().flatten(1).norm(dim=1)).cpu().tolist())
        output.append(record);posts.append(post.detach().cpu().numpy());jposts.append(jpost.detach().cpu().numpy())
        native_posts.append(bits(npost));native_finite.append(np.stack([bits(nplus),bits(nminus)]))
        cotangents.append(grad_post.detach().cpu().numpy())
    del wu,weight
    arrays={'postnorm':np.stack(posts),'postnorm_JVP':np.stack(jposts),'native_BF16_postnorm':np.stack(native_posts),
        'native_BF16_finite_endpoints':np.stack(native_finite),'postnorm_cotangent':np.stack(cotangents)}
    return output,arrays


def reverse_block(layer,block,state,arrays,begin):
    import torch
    key,value,cos,sin=fixture_inputs(arrays,block,torch.float32)
    x=tensor(state['primal_trace'][block-begin])[None,None].requires_grad_(True)
    out=block_call(layer,x,key,value,cos,sin,block)
    state['reverse']=torch.autograd.grad(out,x,state['reverse'])[0].detach()


def main(pilot):
    import torch,psutil
    cuda_singleton(CUDA_TASKS);verify_storage(1024**3)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    start=time.monotonic();protocol,material=freeze()
    selected=[r for r in material if r['fixture_id'] in protocol['pilot_fixture_ids']] if pilot else material
    folder=OUT/('pilot' if pilot else 'main');fixtures=[]
    execution={'source':snapshot(__file__),'helper':snapshot(Path(__file__).with_name('rdc_native_tail.py')),
        'protocol_sha256':sha(OUT/'protocol.json'),'pilot':pilot}
    if not pilot:assert read(OUT/'pilot/result.json')['all_passed']
    try:
        for row in selected:
            rec=read(OUT/'commits'/(row['fixture_id']+'.json'));file=BASE/rec['field_path']
            assert sha(file)==rec['field_sha256']
            assert psutil.virtual_memory().available>2*1024**3,'CPU fixture-memory safety reserve'
            with np.load(file) as z:arrays={k:z[k] for k in z.files}
            fixtures.append((rec,arrays))
        all_records=[];commits=[]
        for begin in STARTS:
            receipt=folder/('start_'+str(begin)+'.json')
            if receipt.exists():
                previous=read(receipt)
                assert previous['execution']==execution and sha(BASE/previous['field_path'])==previous['field_sha256']
                all_records+=previous['records'];commits.append(previous);continue
            states=[initial_state(r,a,begin) for r,a in fixtures]
            for block in range(begin,36):
                layer=loaded_block(block,torch.float32)
                for state,(row,a) in zip(states,fixtures):forward_block(layer,block,state,a)
                layer.to(dtype=torch.bfloat16)
                for state,(row,a) in zip(states,fixtures):native_forward_block(layer,block,state,a)
                del layer;gc.collect();torch.cuda.empty_cache()
                if block%3==2 or block==35:
                    print('TAIL_FORWARD',begin,block,len(fixtures),round(time.monotonic()-start,1),flush=True)
            output,arrays=evaluate_endpoints(states,fixtures,begin,folder)
            for block in reversed(range(begin,36)):
                layer=loaded_block(block,torch.float32)
                for state,(row,a) in zip(states,fixtures):reverse_block(layer,block,state,a,begin)
                del layer;gc.collect();torch.cuda.empty_cache()
                if block%6==0:
                    print('TAIL_REVERSE',begin,block,len(fixtures),round(time.monotonic()-start,1),flush=True)
            for record,state in zip(output,states):
                left=state['output_direction_dots']
                right=(state['initial'].double()*state['reverse'].double()).sum((1,2)).cpu().numpy()
                scale=(state['initial'].double().flatten(1).norm(dim=1)*state['reverse'].double().norm()).cpu().numpy()
                error=abs(left-right)/np.maximum(scale,1e-30)
                assert np.max(error)<5e-5,('Complete-tail JVP/VJP adjoint mismatch',record['fixture_id'],error.tolist())
                record.update(full_vocab_JVP_dot_cotangent=left.tolist(),
                    initial_direction_dot_complete_VJP=right.tolist(),adjoint_normalized_errors=error.tolist())
            arrays.update(primal_fields=np.stack([np.stack(s['primal_trace']) for s in states]),
                JVP_fields=np.stack([np.stack(s['tangent_trace']) for s in states]),
                complete_VJP=np.stack([s['reverse'][0,0].cpu().numpy() for s in states]),
                initial_directions=np.stack([s['initial'][:,0].cpu().numpy() for s in states]),
                layer_boundaries=np.arange(begin,37))
            assert all(np.isfinite(unbits(a) if a.dtype==np.uint16 else a).all() for a in arrays.values())
            verify_storage(sum(a.nbytes for a in arrays.values()))
            path=FIELD_STORE/'differential_derivatives'/((('pilot' if pilot else 'main')+'_start'+str(begin))+'.npz')
            npz(path,**arrays)
            value={'timestamp':stamp(),'all_passed':True,'execution':execution,'start':begin,
                'field_path':path.relative_to(BASE).as_posix(),'field_sha256':sha(path),'field_bytes':path.stat().st_size,
                'records':output,'row_ids':[r['fixture_id'] for r,a in fixtures],
                'field_scope':'All residual coordinates and all3direction fields at every remaining layer. Full vocabulary contracted for all scores; logits recomputable from saved postnorm/JVP and exact original W_U, not stored as a second full vocabulary copy.'}
            save(receipt,value);commits.append(value);all_records+=output
            print('TAIL_START_COMMITTED',begin,len(output),round(time.monotonic()-start,1),flush=True)
            del states,arrays;gc.collect();torch.cuda.empty_cache()
        compressed(folder/'records.json.gz',all_records)
        result={'timestamp':stamp(),'all_passed':True,'execution':execution,'fixture_count':len(fixtures),
            'native_tails':len(all_records),'direction_epsilon_comparisons':sum(len(r['comparisons']) for r in all_records),
            'maximum_adjoint_normalized_error':max(max(r['adjoint_normalized_errors']) for r in all_records),
            'all_original_BF16_tail_endpoints_bit_equal':all(r['native_tail_replay_bit_equal'] for r in all_records),
            'maximum_FP32_execution_shape_relative_difference':max(r['maximum_FP32_B3_vs_B1_relative_difference'] for r in all_records),
            'commits':[{'start':v['start'],'field_path':v['field_path'],'field_sha256':v['field_sha256']} for v in commits],
            'seconds':time.monotonic()-start,'peak_CUDA_allocated_bytes':torch.cuda.max_memory_allocated(),
            'scope':'Numerically verified native-conditioned differential and finite-change diagnosis, not a learned language rule. Smooth FP32 reference and exact original BF16 finite changes remain distinct.'}
        save(folder/'result.json',result);ledger('phase2746_tail_differential',result['seconds'],pilot=pilot)
        print('TAIL_DIFFERENTIAL_COMPLETE',len(all_records),result['maximum_adjoint_normalized_error'],flush=True)
    except Exception as exc:
        failure(folder,start,exc);raise
    finally:
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();main(a.pilot)
