"""Prospective length/line-count control added before opening table responses."""
import sys,time,gc,re
import b36_transition14 as M
S=M.S

def inserted(prompt,filler):
    needle='\n```\nWrite';assert prompt.count(needle)==1
    return prompt.replace(needle,'\n```\nNotes:\n'+filler+'\nWrite',1)

def prepare():
    S.configure(500*2**20);assert not (S.OUT/'length-plan.json').exists()
    from transformers import AutoTokenizer
    from tools.power_analysis import check_phase
    tok=AutoTokenizer.from_pretrained(S.MODEL,local_files_only=True);p=S.read(S.OUT/'plan.json')
    base=[r for r in p['rows'] if r['arm']=='no_table'];table={r['family']:r for r in p['rows'] if r['arm']=='table'}
    prose='The room is quiet.\nThe wall is white.\nThe book is closed.\nThe door is open.'
    wanted={len(table[r['family']]['tokens'])-len(r['tokens']) for r in base};cache={};ref=base[0]
    for na in range(26):
        for nb in range(26):
            filler=prose+' It is quiet.'*na+' The air is still.'*nb
            delta=len(tok.encode(inserted(ref['prompt'],filler),add_special_tokens=False))-len(ref['tokens'])
            if delta in wanted and delta not in cache:cache[delta]=filler
    assert wanted<=set(cache),(wanted,set(cache))
    rows=[]
    for row in base:
        target=table[row['family']];filler=cache[len(target['tokens'])-len(row['tokens'])];prompt=inserted(row['prompt'],filler);ids=tok.encode(prompt,add_special_tokens=False)
        assert len(ids)==len(target['tokens']) and prompt.count('\n')==target['prompt'].count('\n')
        assert len(filler.splitlines())==4
        rows.append(dict(row,id=row['family']+':length_control',arm='length_control',prompt=prompt,tokens=ids,table_token_length=len(target['tokens']),neutral_prose=filler))
    power=check_phase(.2,96,p_d=.3,alpha=.05/7);assert power['pass']
    S.save('length-plan.json',dict(rows=rows,main_plan_sha256=S.file_sha(S.OUT/'plan.json'),source_sha256=S.file_sha(__file__),main_source_hashes=p['source_hashes'],model_files=p['model_files'],power=power,test=dict(left='length_control',right='table',metric='step_count_correct'),correction_amendment='Conservative Holm over all7: the6 original contrasts plus this length control. No change to original endpoints, behavior gates, prompts or scores.',exposure='Only no_table responses had been inspected. table generation was complete but its response contents had not been opened before this supplement was specified. Other main-arm responses not inspected.',table_result_existed=(S.OUT/'table.json').exists(),table_result_sha256_at_registration=S.file_sha(S.OUT/'table.json') if (S.OUT/'table.json').exists() else None,matching='Same original no_table prompt and output instructions; insert4 unrelated prose lines under Notes at the same location as Options. Exact per-row total tokenizer length and newline count match table. Natural prose may repeat whole sentences; not asserted semantically inert.',generation=dict(batch=96,cap=48,do_sample=False,dtype='originalBF16',thinking=False),budget=dict(new_peak_bytes=40*2**20,expected_minutes=10),scope='Additional prospective response arm on the same96 families, not96 new independent families. Length and line count are matched; all other content/format differences cannot be eliminated.'))
    print('Length control frozen:96 prompts, exact token length and newline count; Holm7 power',power,flush=True)

def run():
    S.configure(500*2**20);p=S.read(S.OUT/'length-plan.json');assert p['source_sha256']==S.file_sha(__file__) and p['main_plan_sha256']==S.file_sha(S.OUT/'plan.json');start=time.monotonic();lease=None
    for attempt in range(121):
        try:lease=S.GpuLease();lease.acquire();break
        except RuntimeError as e:
            if 'leased by another' not in str(e):raise
            S.log('length_admission_wait',attempt,120,start,status='another process owns GPU lease')
            if attempt==120:raise RuntimeError('Length control admission timed out after120 minutes')
            time.sleep(60)
    try:
        assert lease.handle is not None
        for n,h in p['main_source_hashes'].items():assert S.file_sha(S.ROOT/'tests/glm5'/n)==h,n
        for i,f in enumerate(p['model_files']):assert S.file_sha(S.MODEL/f['path'])==f['sha256'];S.log('length_identity',i+1,len(p['model_files']),start)
        import torch
        from transformers import AutoTokenizer,AutoConfig,AutoModelForCausalLM
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
        tok=AutoTokenizer.from_pretrained(S.MODEL,local_files_only=True);tok.padding_side='left';cfg=AutoConfig.from_pretrained(S.MODEL,local_files_only=True);R=S.bind();runstart=time.monotonic();step=0
        with torch.inference_mode():
            with torch.device('meta'):model=AutoModelForCausalLM.from_config(cfg,dtype=torch.bfloat16,attn_implementation='sdpa').eval()
            for obj,key in [(model.model.embed_tokens,'model.embed_tokens.weight'),(model.model.norm,'model.norm.weight'),(model.lm_head,'lm_head.weight')]:obj.weight=torch.nn.Parameter(R.C.weight(key).cuda(),requires_grad=False)
            model.model.rotary_emb=Qwen3RotaryEmbedding(cfg,device='cpu').cuda();hooks=[]
            for b,layer in enumerate(model.model.layers):
                def load(m,a,b=b):m.load_state_dict(R.C.load_block(b,cfg).state_dict(),assign=True,strict=True)
                def release(m,a,y,b=b):
                    nonlocal step
                    m.to_empty(device='meta')
                    if b==39:step+=1;S.log('length_control',step,48,runstart)
                hooks.extend([layer.register_forward_pre_hook(load),layer.register_forward_hook(release)])
            rows=p['rows'];inputs=tok.pad(dict(input_ids=[r['tokens'] for r in rows]),padding=True,return_tensors='pt').to('cuda');out=model.generate(**inputs,max_new_tokens=48,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id);results=[]
            for row,seq in zip(rows,out):
                ids=seq[inputs['input_ids'].shape[1]:].tolist();eos=tok.eos_token_id in ids
                if eos:ids=ids[:ids.index(tok.eos_token_id)+1]
                text=tok.decode(ids,skip_special_tokens=False);results.append(dict(id=row['id'],tokens=ids,tail_text=text,text=text,parsed=M.evaluate(text,row,eos)))
            for h in hooks:h.remove()
            assert p['source_sha256']==S.file_sha(__file__)
            S.save('length_control.json',dict(results=results,plan_sha256=S.file_sha(S.OUT/'plan.json'),supplement_plan_sha256=S.file_sha(S.OUT/'length-plan.json'),arm='length_control',seconds=time.monotonic()-runstart,admission_and_total_seconds=time.monotonic()-start))
            print('Length control complete',flush=True)
    finally:
        if lease is not None:lease.release()

if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
