"""Prospective self-generated intermediate-state calibration, original BF16."""
import sys,time,gc,re,json,operator,hashlib,math
from pathlib import Path
import b36_semantic14 as S
OLD=S.OUT
S.OUT=S.ROOT/'tests/glm5/result/b36_generated14_20261113_v1'
S.TEMP=S.ROOT/'tests/glm5_temp/b36_generated14_20261113_v1'
S.RUN='B36-GENERATED14-20261113-v1'
S.SEED=202611132120

def parse(text,truth,answers,expected,eos):
    clean=text.replace('<|im_end|>','').strip()
    sm=re.search(r'Steps\s*:\s*([0-9 ,]+)',clean,re.I)
    steps=[int(x) for x in re.findall(r'\d+',sm[1])] if sm else None
    fm=re.search(r'Final\s*:\s*([A-D]|\d+)\b',clean,re.I);final=fm[1].upper() if fm else None
    format_ok=bool(re.fullmatch(r'Steps:\s*\d+(?:\s*,\s*\d+)*\s*\nFinal:\s*(?:[A-D]|\d+)',clean))
    steps_parse=steps is not None and len(steps)==len(truth)
    steps_ok=steps_parse and steps==truth
    final_ok=final==expected
    return dict(steps=steps,final=final,steps_parseable=steps_parse,steps_correct=steps_ok,final_parseable=final is not None,final_correct=final_ok,format_correct=format_ok,eos=eos,joint=bool(steps_ok and final_ok and format_ok and eos),calculation_error=bool(steps_parse and not steps_ok),correct_steps_wrong_final=bool(steps_ok and final is not None and not final_ok))

def prepare():
    S.configure(300*2**20);assert not (S.OUT/'plan.json').exists()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(S.MODEL,local_files_only=True)
    fs=S.read(OLD/'arithmetic-families.json')['families'];seen=set(S.read(OLD/'arithmetic-families.json')['diagnostic_family_ids'])|{g['id'] for g in S.read(OLD/'expanded-plan.json')['families']}
    candidates=sorted([g for g in fs if g['type']=='affine' and g['id'] not in seen],key=lambda g:S.digest(['generated',g['id']]))
    chosen=[];prefixes=set()
    for g in candidates:
        start=10+int(g['id'][1:5],16)%20;x=start;truth=[]
        for kind,k in g['ops']:x=dict(add=operator.add,sub=operator.sub,mul=operator.mul)[kind](x,k);truth.append(x)
        key=S.digest([start,g['ops'][:1]])
        if min(truth)<0 or max(truth)>99 or key in prefixes:continue
        prefixes.add(key);chosen.append(dict(id=g['id'],ops=g['ops'],start=start,truth=truth))
        if len(chosen)==108:break
    assert len(chosen)==108
    rows=[]
    for g in chosen[:12]:
      for depth in [2,3]:
        truth=g['truth'][:depth];choices=list(dict.fromkeys([truth[-1],g['start']]+truth[:-1]+[truth[-1]+1,truth[-1]+2]))[:4];assert len(choices)==4
        shift=int(g['id'][2:4],16)%4;choices=choices[shift:]+choices[:shift]
        for fmt in ['direct','code0','code2']:
            values=choices if fmt!='code2' else choices[2:]+choices[:2]
            expected=str(truth[-1]) if fmt=='direct' else 'ABCD'[values.index(truth[-1])]
            code=[f'x = {g["start"]}']+[f'x {dict(add="+=",sub="-=",mul="*=")[kind]} {k}' for kind,k in g['ops'][:depth]]
            book='\nAnswer choices:\n'+'\n'.join(f'{c}) {v}' for c,v in zip('ABCD',values)) if fmt!='direct' else ''
            user='Execute the following Python assignments in order.\n```python\n'+'\n'.join(code)+'\n```'+book+'\nWrite exactly two lines. First write "Steps: " followed by the value of x AFTER EACH update (exclude the initial assignment), separated by commas. Then write "Final: " followed by '+('the last value of x.' if fmt=='direct' else 'the letter of the correct answer choice.')+' Stop after the second line. Do not include explanation or code fences.'
            prompt=tok.apply_chat_template([dict(role='user',content=user)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            rows.append(dict(id=f'{g["id"]}:{depth}:{fmt}',family=g['id'],depth=depth,format=fmt,prompt=prompt,tokens=tok.encode(prompt,add_special_tokens=False),truth=truth,expected=expected,values=values))
    identity=S.read(OLD/'model-identity.json')
    S.save('plan.json',dict(rows=rows,families=chosen,pilot_families=12,unopened_confirm_families=96,source_sha256=S.file_sha(__file__),prior_manifest_sha256=S.file_sha(OLD/'manifest.json'),model_files=identity['files'],generation=dict(max_new_tokens=64,do_sample=False,batch=12,use_cache=True,thinking=False),gate='Each of6 depth/format cells requires>=11/12 joint success (all intermediate states,final,format,EOS). Every cell required. Failure stops joint mechanism stage. Success requires larger frozen confirmation before mechanism claims.',exposure='New complete operation programs and initial states10..29; no shared semantic prefixes with old programs whose initial states were0..9.12 discovery families;96 reserved candidates never sent to model.',truth='Only prompt syntax given. No correct intermediate values or output examples supplied. Unrestricted greedy generation, no forced answer tokens or teacher-forcing.',budget=dict(peak_disk_bytes=300*2**20,reserve_bytes=20*2**30,max_generation_tokens=64,pilot_rows=72),limit='A discovery screen, not powered causal intervention. Digit range changes versus previous stage, so cross-stage differences are not attributable solely to intermediate generation.'))
    print('Frozen72 rows/12 families,96 reserved',flush=True)

def run():
    S.configure(300*2**20);p=S.read(S.OUT/'plan.json');assert p['source_sha256']==S.file_sha(__file__);start=time.monotonic()
    for i,f in enumerate(p['model_files']):assert S.file_sha(S.MODEL/f['path'])==f['sha256'];S.log('identity',i+1,len(p['model_files']),start)
    import torch
    from transformers import AutoTokenizer,AutoConfig,AutoModelForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    R=S.bind();tok=AutoTokenizer.from_pretrained(S.MODEL,local_files_only=True);tok.padding_side='left';cfg=AutoConfig.from_pretrained(S.MODEL,local_files_only=True)
    with S.GpuLease(),torch.inference_mode():
        with torch.device('meta'):model=AutoModelForCausalLM.from_config(cfg,dtype=torch.bfloat16,attn_implementation='sdpa').eval()
        for obj,key in [(model.model.embed_tokens,'model.embed_tokens.weight'),(model.model.norm,'model.norm.weight'),(model.lm_head,'lm_head.weight')]:obj.weight=torch.nn.Parameter(R.C.weight(key).cuda(),requires_grad=False)
        model.model.rotary_emb=Qwen3RotaryEmbedding(cfg,device='cpu').cuda();calls=[0];hooks=[]
        for b,layer in enumerate(model.model.layers):
            def load(m,a,b=b):m.load_state_dict(R.C.load_block(b,cfg).state_dict(),assign=True,strict=True)
            def release(m,a,y,b=b):
                m.to_empty(device='meta')
                if b==39:
                    calls[0]+=1
                    if calls[0]%8==0:S.log('generated_token_forward',calls[0],384,start)
            hooks.extend([layer.register_forward_pre_hook(load),layer.register_forward_hook(release)])
        for lo in range(0,len(p['rows']),12):
            path=S.OUT/f'chunk-{lo:03d}.json'
            if path.exists():continue
            rr=p['rows'][lo:lo+12];S.log('generation_batch',lo,len(p['rows']),start)
            inputs=tok([r['prompt'] for r in rr],return_tensors='pt',padding=True,add_special_tokens=False).to('cuda')
            output=model.generate(**inputs,max_new_tokens=64,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
            results=[]
            for r,seq in zip(rr,output):
                new=seq[inputs['input_ids'].shape[1]:].tolist();eos=tok.eos_token_id in new
                if eos:new=new[:new.index(tok.eos_token_id)+1]
                text=tok.decode(new,skip_special_tokens=False)
                results.append(dict(id=r['id'],tokens=new,text=text,metrics=parse(text,r['truth'],r['values'],r['expected'],eos)))
            S.save(path.name,dict(results=results,plan_sha256=S.file_sha(S.OUT/'plan.json'),elapsed_seconds=time.monotonic()-start))
            del output,inputs;gc.collect();torch.cuda.empty_cache();S.log('generation_batch',lo+len(rr),len(p['rows']),start)
        for h in hooks:h.remove()
    print('All generation batches complete',flush=True)

if __name__=='__main__':
    if sys.argv[-1]=='prepare':prepare()
    else:run()
