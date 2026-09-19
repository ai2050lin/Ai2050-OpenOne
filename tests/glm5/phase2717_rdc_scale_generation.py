"""Small-batch natural generation amortizes nonquantized disk offload; shape drift is audited."""
from rdc_relation_common import *
from rdc_relation_estimators import Bank,predict


def run(model,tok,observer,scales,models,choices,selected,trainraw,collected,out,key):
    import torch
    from phase2717_rdc_relation_scale import dots,kernel,logprob
    observer.close();state={};handles=[]
    for name,layer in [('early',observer.early),('late',observer.depth)]:
        def hook(m,a,o,name=name):state[name]=bits(o[:,-1])
        handles.append(model.model.layers[layer-1].register_forward_hook(hook))
    protocol=out/'generation_batch_protocol.json'
    if not protocol.exists():save(protocol,{'timestamp':stamp(),'code':snapshot(Path(__file__)),'batch_size':4,'padding':'left, actual attention mask, explicit real-token position_ids=cumsum(mask)-1; pad positions clamped0',
      'generation':'64 fixed fresh source units,8 native greedy steps or EOS; no KV cache. Native-generated branch only. Inputs and evaluation still full native width/vocabulary.',
      'reason':'Qwen14 disk-shard reads dominate batch1 runtime. Amortize same unchanged weights over4 cases instead of reducing sample count, steps, precision or input coordinates.',
      'shape_control':'Every first-step final state compared with its saved natural batch1 full-source anchor state. Report drift separately; reject relative MSE above0.01 as a gross shape/masking error, not as scientific mechanism failure.',
      'scope_limit':'Generation arithmetic has a different execution shape from source capture. This is a batched native continuation check, not claimed bitwise-equivalent to batch1 greedy continuation. Same policy used for matched model scale comparisons.',
      'state_save':'All-coordinate per-step error sums and four predeclared full-field fixtures; remaining individual generated states not archived, exact input/model/shape recipe retained.'})
    width=model.config.hidden_size;total_errors={'current':np.zeros((8,width),float),'temporal':np.zeros((8,width),float)};counts=np.zeros(8,int);completed=[];shape=[]
    if (out/'generation_progress.json').exists():
        progress=read(out/'generation_progress.json');completed=progress['completed'];shape=progress['shape_controls']
        with np.load(out/'generation_all_coordinate_errors.npz') as z:
            counts=z['counts'];total_errors={k:z[k+'_squared_error_sum'].astype(float) for k in total_errors}
    device=model.get_input_embeddings().weight.device;eos=model.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos or [tok.eos_token_id]);pad=tok.pad_token_id if tok.pad_token_id is not None else next(iter(eos))
    try:
      with torch.inference_mode():
       for begin in range(0,len(collected),4):
        group=collected[begin:begin+4]
        if all(m['sample_id'] in completed for a,m in group):continue
        assert not any(m['sample_id'] in completed for a,m in group),'Resume only whole committed batches'
        prefixes=[m['prompt_ids'][:m['positions'][0]+1] for a,m in group];ids=[list(x) for x in prefixes];tokens=[[] for _ in group];per=[[] for _ in group];past=[None for _ in group];stopped=[False for _ in group];fixtures=[[] for _ in group]
        for step in range(8):
            active=[i for i in range(len(group)) if not stopped[i]]
            if not active:break
            length=max(len(ids[i]) for i in active);x=torch.full((len(active),length),pad,dtype=torch.long,device=device);mask=torch.zeros_like(x)
            for j,i in enumerate(active):x[j,-len(ids[i]):]=torch.tensor(ids[i],device=device);mask[j,-len(ids[i]):]=1
            positions=(mask.cumsum(-1)-1).clamp_min(0);state.clear();post=model.model(input_ids=x,attention_mask=mask,position_ids=positions,use_cache=False).last_hidden_state[:,-1];lq=model.lm_head(post).float().log_softmax(-1);h={k:unbits(v) for k,v in state.items()}
            emb=model.get_input_embeddings()(torch.tensor([ids[i][-1] for i in active],device=device)).float().cpu().numpy()
            inp={'early':h['early'],'late':np.stack([h['late'][j] if past[i] is None else past[i] for j,i in enumerate(active)]),'new_embedding':emb};dd=dots(inp,trainraw,scales);estimates={}
            for scope in ('current','temporal'):
                name=selected[scope];kd,kn=kernel(dd,name);pred=predict(models[name],Bank.gram(kd,kn,choices[name]['mix']));lp=logprob(model,pred);estimates[scope]=(pred,lp)
            for j,i in enumerate(active):
                a,m=group[i];record={'step':step};counts[step]+=1
                if step==0:
                    drift=np.mean((h['late'][j]-a['late'][0])**2);rel=drift/max(np.mean(a['late'][0]**2),1e-30);assert rel<.01,('Batched position/mask gross mismatch',m['sample_id'],rel)
                    shape.append({'sample_id':m['sample_id'],'batch1_to_batch4_Hfinal_MSE':float(drift),'relative_MSE':float(rel),'max_abs_difference':float(np.max(np.abs(h['late'][j]-a['late'][0])))})
                for scope,(prediction,lp) in estimates.items():
                    if scope=='temporal' and step==0:continue
                    err=(prediction[j]-h['late'][j])**2;total_errors[scope][step]+=err;record[scope]={'MSE':float(err.mean()),'KL':float((lq[j].exp()*(lq[j]-lp[j])).sum()),'argmax_agreement':bool(lq[j].argmax()==lp[j].argmax())}
                v=int(lq[j].argmax());tokens[i].append(v);ids[i].append(v);per[i].append(record);past[i]=h['late'][j];stopped[i]=v in eos
                if begin+i<4:fixtures[i].append({k:state[k][j].copy() for k in state})
        for i,(a,m) in enumerate(group):
            save(out/f'generation/{m["sample_id"]}.json',{'sample_id':m['sample_id'],'language':m['language'],'prefix_ids':prefixes[i],'prefix_text':tok.decode(prefixes[i]),'generated_ids':tokens[i],
              'generated_text':tok.decode(tokens[i]),'stop':'EOS' if stopped[i] else '8_token_budget','steps':per[i],'execution':'batch4 left-padding, explicit real positions, no KV cache'})
            if begin+i<4:npz(out/f'generation_fixtures/{m["sample_id"]}.npz',**{k:np.stack([f[k] for f in fixtures[i]]) for k in ('early','late')})
            completed.append(m['sample_id'])
        npz(out/'generation_all_coordinate_errors.npz',counts=counts,**{k+'_squared_error_sum':v.astype(np.float32) for k,v in total_errors.items()});save(out/'generation_progress.json',{'timestamp':stamp(),'completed':completed,'shape_controls':shape})
        print('SCALE_BATCH_GENERATION',key,len(completed),64,flush=True);guard(4*1024**2)
    finally:
        for h in handles:h.remove()
    steps=[s for p in sorted((out/'generation').glob('*.json')) for s in read(p)['steps']]
    return {'by_scope':{scope:{'steps':len(ss),'mean_KL':float(np.mean([s['KL'] for s in ss])),'argmax_agreement':float(np.mean([s['argmax_agreement'] for s in ss]))} for scope in ('current','temporal') if (ss:=[s[scope] for s in steps if scope in s])},
      'shape_control_sources':len(shape),'shape_control_max_relative_MSE':max(s['relative_MSE'] for s in shape),'batch_size':4,'no_KV_cache':True}
