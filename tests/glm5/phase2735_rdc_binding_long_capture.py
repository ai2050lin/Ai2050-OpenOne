"""Prospective native fields and longer unmodified own-history generation."""
import gc
import re
from rdc_binding_common import *

def score(text,target,ids,stop,cap):
    # Conservative observable suffix, not an LLM judge or arbitrary last number.
    plain=text.strip();patterns=[r'^\s*([1-8])\s*[。.!]?\s*$',
      r'(?:answer(?:\s+is)?|final(?:\s+answer)?(?:\s+is)?|答案(?:是|为)?|结果(?:是|为)?)\s*[:：=]?\s*\*{0,2}([1-8])\*{0,2}[。.!]?\s*$',
      r'\\boxed\{([1-8])\}[。.!]?\s*$']
    parsed=None
    for pattern in patterns:
        m=re.search(pattern,plain,re.I)
        if m:parsed=m.group(1);break
    return {'exact_correct':plain==target,'conservative_final_digit':parsed,
      'conservative_final_correct':None if parsed is None else parsed==target,
      'eos':any(t in stop for t in ids),'token_limit_censored':not any(t in stop for t in ids) and len(ids)>=cap}

def main():
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    from phase2728_rdc_law_capture import Observer
    out=BASE/'format_content';protocol=read(out/'protocol.json')
    if (out/'native_capture_result.json').exists():return
    future=gzread(out/'prospective_material.json.gz');old=gzread(BASE/'program_material.json.gz')
    rows=[r for r in old if r['sample_id'] in protocol['long_baseline_old_ids']]+future
    start=time.monotonic();guard(350*1024**2);model,tok=load_native('qwen4');model.eval();torch.set_num_threads(2)
    observer=Observer(model);device=model.get_input_embeddings().weight.device
    runtime={'model':'qwen3-4b','dtype':str(model.dtype),'quantized':bool(getattr(model,'is_quantized',False)),
      'device_map':getattr(model,'hf_device_map',{}),'batch':1,'padding':False,
      'capture_cache':False,'generation_cache':True,'attention':'native eager','do_sample':False}
    eos=model.generation_config.eos_token_id or tok.eos_token_id;stop=set(eos if isinstance(eos,list) else [eos]);records=[]
    try:
      with torch.inference_mode():
        for i,row in enumerate(rows):
            sid=row['sample_id'];commit=out/'native_commits'/f'{sid}.json'
            if commit.exists():records.append(read(commit));continue
            ids=torch.tensor([row['prompt_ids']],device=device)
            field_identities=None
            if row['split']=='prospective_depth6':
                observer.reset(row['anchors'],False);observer.enabled=True
                post=model.model(input_ids=ids,use_cache=False).last_hidden_state;observer.enabled=False
                p=row['anchors'];logits=model.lm_head(post[0,p]).float()
                fields={'H':np.stack([observer.H[k] for k in range(37)]),'H12_sources':observer.source_H12,
                  'embedding':observer.H[0],'positions':np.array(p),'token_ids':np.array(row['prompt_ids'],np.int32),
                  'postnorm':bits(post[0,p]),'alltoken_layer_moments':np.stack([observer.moments[k] for k in range(37)]),
                  **observer.factors,'last_residual':observer.training['residual'][p],'last_x':observer.training['x'][p],
                  'target_nll':(-logits.log_softmax(-1)[:,row['target_ids'][0]]).cpu().numpy()}
                field_identities=observer.H_hashes
                npz(out/'prospective_fields'/f'{sid}.npz',**fields);observer.reset([],False);del fields,post,logits
            generated=model.generate(input_ids=ids,use_cache=True,do_sample=False,max_new_tokens=protocol['generation_limit'],
              eos_token_id=eos,pad_token_id=tok.pad_token_id)
            seq=generated[0,len(row['prompt_ids']):].tolist();text=tok.decode(seq,skip_special_tokens=True)
            record={k:row[k] for k in ('sample_id','source_group','family','representation','split','depth','target')}
            record.update(generated=text,generated_ids=seq,cap=protocol['generation_limit'],**score(text,row['target'],seq,stop,protocol['generation_limit']))
            if row['split']=='prospective_depth6':
                record['array_sha']=sha(out/'prospective_fields'/f'{sid}.npz')
                record['all_token_layer_identities']=field_identities
            if row['split']!='prospective_depth6':
                prior=read(BASE/'capture/commits'/f'{sid}.json')
                assert seq[:len(prior['generated_ids'])]==prior['generated_ids'];record['original8token_prefix_exact']=True
            save(commit,record);records.append(record);del ids,generated
            if (i+1)%8==0:print('LONG_NATIVE_CAPTURE',i+1,len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
            guard(40*1024**2);assert time.monotonic()-start<7200
    finally:
        observer.close();del observer,model;gc.collect();torch.cuda.empty_cache()
    save(out/'native_capture_result.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'rows':len(records),'future_rows':len(future),
      'runtime':runtime,
      'seconds':time.monotonic()-start,'scope':'Unmodified greedy native own-history,128token cap; conservative parsed final answer reports coverage and abstains on unmarked text.'})
    ledger('format_content_native_capture',time.monotonic()-start)

if __name__=='__main__':main()
