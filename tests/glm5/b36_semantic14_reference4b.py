"""Small reference-model diagnostic, separate from the fixed14B group evidence."""
import gc,time,importlib.metadata
import b36_semantic14 as S

def run():
    S.configure(60*2**20);dest=S.OUT/'reference4b-check.json';assert not dest.exists();start=time.monotonic();path=S.ROOT/'models/hf/qwen3-4b'
    from shared.dist_protocol.natural_register_runner import snapshot,PATTERNS
    identity=[dict(path=p.name,bytes=p.stat().st_size,sha256=S.file_sha(p)) for p in sorted({p for pat in PATTERNS for p in path.glob(pat) if p.is_file()})]
    snap=snapshot(path)
    from transformers import AutoTokenizer,AutoModelForCausalLM
    import torch
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    questions=['What is 1 + 1? Reply with only the digit.','What is 2 + 3? Reply with only the digit.',
        'Evaluate (((1 + 2) - 2) + 3). Output exactly one character: the result digit. No spaces, punctuation or explanation.',
        'Start with the number 1. Add 2 to it. Subtract 2 from it. Add 3 to it. Output exactly one character: the result digit. No spaces, punctuation or explanation.',
        'Execute this Python code mentally. What digit is printed?\n```python\nx = 1\nx += 2\nx -= 2\nx += 3\nprint(x)\n```\nReply with the digit only.']
    samples=[]
    for i,q in enumerate(questions):samples.append(dict(id=f'sanity{i}',user=q,expected=['2','5','4','4','4'][i]))
    for name in ['pilot-plan.json','order-pilot-plan.json']:
        rr=S.read(S.OUT/name)['rows'];selected=[r for r in rr if r['depth']==3 and r['format'] in ['direct','word_plain']][:2]
        for r in selected:
            samples.append(dict(id=r['id'],user=r['prompt'].split('<|im_start|>user\n',1)[1].split('<|im_end|>',1)[0],expected=r['expected_answer']))
    results=[]
    with S.GpuLease():
      with torch.inference_mode():
        model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval()
        for thinking in [False,True]:
            for i,s in enumerate(samples):
                prompt=tok.apply_chat_template([dict(role='user',content=s['user'])],tokenize=False,add_generation_prompt=True,enable_thinking=thinking)
                ids=tok(prompt,return_tensors='pt',add_special_tokens=False).to('cuda')
                output=model.generate(**ids,max_new_tokens=256 if thinking else 32,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
                new=output[0,ids['input_ids'].shape[1]:].cpu().tolist();text=tok.decode(new,skip_special_tokens=False)
                final=text.rsplit('</think>',1)[-1].replace('<|im_end|>','').strip()
                results.append(dict(id=s['id'],thinking=thinking,expected=s['expected'],tokens=new,text=text,parsed_final=final,exact_final=final==s['expected'],truncated=len(new)==(256 if thinking else 32)))
                S.log('reference4B_diagnostic',len(results),2*len(samples),start)
        del model;gc.collect();torch.cuda.empty_cache()
    from tools import provenance
    provenance.RESULTS=S.OUT/'provenance'
    helper=provenance.write_result('reference4b_diagnostic',dict(results=results),model_path=path,seed=S.SEED,tool='b36_semantic14_reference4b')
    S.save_json(dest,dict(kind='real_reference_model_only',model_path=str(path),model_snapshot=snap,files=identity,results=results,samples=samples,
        source_sha256=S.file_sha(__file__),seconds=time.monotonic()-start,provenance=S.read(helper)['provenance'],provenance_helper=str(helper.relative_to(S.ROOT)),
        limit='A different model. No transfer of14B coordinates, group function or statistical evidence. Thinking truncated at256 generated tokens; not counted as successful final answering.'))
    print('reference4b',[(r['id'],r['thinking'],r['parsed_final'],r['truncated']) for r in results],flush=True)

if __name__=='__main__':run()
