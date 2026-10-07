"""Bounded greedy continuation check on the preselected matched4B subset."""
import argparse,gc,json,re,time
from pathlib import Path
from phase2754_relation_stability import ROOT,OUT,write,sha,now,snapshot

def run(side='4B',protocol='raw'):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    prefix='generation' if side=='4B' else 'generation14B'
    if protocol=='chat':prefix+='_chat'
    assert not (OUT/(prefix+'_summary.json')).exists(),'Preserve completed generation.'
    rows=[r for r in json.loads((OUT/'material.json').read_text(encoding='utf-8'))['rows'] if r['replicate14B']]
    if side=='14B':
        worlds=sorted({r['world'] for r in rows});rows=[next(r for r in rows if r['world']==w and r['cell']==i%8) for i,w in enumerate(worlds)]
    maximum=8 if side=='4B' else 4;budget=300 if side=='4B' else 600
    write(OUT/(prefix+'_pre_run.json'),dict(created_utc=now(),source=snapshot(Path(__file__)),model=side,protocol=protocol,ids=[r['id'] for r in rows],max_new_tokens=maximum,greedy=True,
        scope='Post-confirmation diagnostic;4B192prompts,14B24prompts one per matched world with cells cycling0..7. Chat wraps the exact same task text in the local native template with enable_thinking=False; cache enabled. No new independent worlds.',
        reason14B='Native first tokens often start formatting/explanation;4token diagnostic distinguishes response onset from binary decision margin, not full-answer quality.',
        criteria='First content word yes/no; strict entire trimmed answer yes/no with optional terminal . or !; native EOS event recorded separately.',budget_seconds=budget))
    native={r['id']:r for p in (OUT/side/'confirmation').glob('chunk_*.json') for r in json.loads(p.read_text(encoding='utf-8'))}
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;start=time.time()
    modeldir=ROOT/'models/hf'/('qwen3-4b' if side=='4B' else 'Qwen3-14B')
    tok=AutoTokenizer.from_pretrained(modeldir,local_files_only=True)
    opts=dict(local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager')
    if side=='14B':opts.update(device_map='auto',max_memory={0:'10GiB','cpu':'48GiB'})
    model=AutoModelForCausalLM.from_pretrained(modeldir,**opts).eval()
    if side=='4B':model.to('cuda')
    eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos;records=[]
    try:
        for row in rows:
            if protocol=='raw':prompt_ids=row['tokenization'][side]['token_ids']
            else:
                rendered=tok.apply_chat_template([dict(role='user',content=row['text'])],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                assert isinstance(rendered,str)
                prompt_ids=tok.encode(rendered,add_special_tokens=False)
            ids=torch.tensor([prompt_ids],device=model.get_input_embeddings().weight.device)
            with torch.inference_mode():
                result=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=maximum,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
            gen=result[0,ids.shape[1]:].cpu().tolist();text=tok.decode(gen,skip_special_tokens=True);clean=text.strip()
            first=re.match(r'^(yes|no)\b',clean,re.I);strict=re.fullmatch(r'(yes|no)[.!]?',clean,re.I)
            records.append(dict(id=row['id'],world=row['world'],family=row['family'],split=row['split'],expected=row['expected'],protocol=protocol,input_token_ids=prompt_ids,token_ids=gen,text=text,
                first_token_matches_native=bool(gen and gen[0]==native[row['id']]['prediction_id']),first_word_correct=bool(first and first.group(1).lower()==row['expected']),
                strict_content_correct=bool(strict and strict.group(1).lower()==row['expected']),strict_format=bool(strict),eos_seen=any(i in eos for i in gen),censored=len(gen)==maximum and not any(i in eos for i in gen)))
            if len(records)%8==0:print(side,'generation',len(records),len(rows),time.time()-start,flush=True)
            assert time.time()-start<budget,'Bounded generation cap exceeded.'
        write(OUT/(prefix+'_rows.json'),records)
        keys=('first_token_matches_native','first_word_correct','strict_content_correct','strict_format','eos_seen','censored')
        write(OUT/(prefix+'_summary.json'),dict(created_utc=now(),protocol=protocol,prompts=len(rows),worlds=24,max_new_tokens=maximum,elapsed_seconds=time.time()-start,overall={k:sum(r[k] for r in records)/len(records) for k in keys},
            splits={s:{k:sum(r[k] for r in records if r['split']==s)/sum(r['split']==s for r in records) for k in keys} for s in ('entity','surface','depth')},
            source_sha256=sha(Path(__file__)),limitations='Capped continuation cannot certify eventual stopping or correctness of longer unrestricted explanations. Different4B/14B caps and sample sizes forbid full-generation capability comparison.'))
        print(json.dumps(json.loads((OUT/(prefix+'_summary.json')).read_text(encoding='utf-8'))['overall'],indent=2),flush=True)
    finally:del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['4B','14B'],default='4B');p.add_argument('--protocol',choices=['raw','chat','both'],default='raw');a=p.parse_args()
    for protocol in (('raw','chat') if a.protocol=='both' else (a.protocol,)):run(a.model,protocol)
