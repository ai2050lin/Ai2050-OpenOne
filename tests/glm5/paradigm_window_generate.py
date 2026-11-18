"""Greedy generation from exact native causal prefill states, with a fixed token cap."""

import ast
from itertools import batched
import re
import time

import numpy as np

import paradigm_window_native as N
import paradigm_window_run as W

R,C,S,M,OUT=W.R,W.C,W.S,W.M,W.OUT
CAP=24
KINDS=('prose','code','full','last20')
SLAB=128


def score(text,gold,stopped):
    try:
        v=ast.parse(text.strip(),mode='eval').body
    except (ValueError,SyntaxError):
        v=None
    python=(isinstance(v,ast.Call) and isinstance(v.func,ast.Name) and v.func.id=='print'
            and len(v.args)==1 and not v.keywords and isinstance(v.args[0],ast.Constant) and isinstance(v.args[0].value,str))
    message=v.args[0].value if python else text
    han='\u3400-\u9fff'
    language=bool(re.search(f'[{han}]',message)) and re.search('[A-Za-z]',message) is None
    numerals='零一二三四五六七八九'
    digits=re.findall(r'[0-9]+|[零一二三四五六七八九]',message)
    count=len(digits)==1 and digits[0] in (str(gold),numerals[gold])
    square=bool(re.search(r'[\[［]\s*['+re.escape(str(gold)+numerals[gold])+r']\s*[\]］]',message))
    square_any=bool(re.search(r'[\[［]\s*[0-9零一二三四五六七八九]+\s*[\]］]',message))
    return dict(format=python,language=language,count=count,square=square_any,square_gold=square,
                stopped=stopped,joint=python and language and count and square and stopped)


def decode(model,panels,progress):
    t=model.t
    end=getattr(model.model.generation_config,'eos_token_id',model.tok.eos_token_id)
    eos=set(end if isinstance(end,list) else (end,))
    eos.add(model.tok.eos_token_id)
    stop=t.tensor(sorted(i for i in eos if i is not None),device='cuda')
    for p in panels:
        positions=p['q'][p['recipient'],0].repeat(len(KINDS))
        p['ends']=positions
        p['last']=t.as_tensor(p['initial'].ravel(),device='cuda')
        p['tokens']=t.full((len(p['last']),CAP),-1,dtype=t.int64,device='cuda')
        p['alive']=t.ones_like(p['last'],dtype=t.bool)
        p['stopped']=t.zeros_like(p['alive'])
        p['check']=None
    for step in range(CAP):
        for p in panels:
            p['tokens'][:,step]=t.where(p['alive'],p['last'],-1)
            finish=p['alive']&t.isin(p['last'],stop)
            p['stopped']|=finish
            p['alive']&=~finish
        progress(step+1)
        if step==CAP-1 or not any(p['alive'].any() for p in panels):
            break
        for p in panels:
            ids=p['last'][:,None]
            pos=p['ends']+step+1
            p['hidden']=model.decoder.embed_tokens(ids)
            p['rope']=model.rope(p['hidden'],pos[:,None])
            pre=t.arange(p['cache_end'],device='cuda')[None,:]<=p['ends'][:,None]
            gen=t.ones((len(ids),step+1),device='cuda',dtype=t.bool)
            p['visible']=t.cat((pre,gen),dim=-1)[:,None,None,:]
        for i in range(40):
            with model.block(i) as block:
                for p in panels:
                    x=p['hidden']
                    q,k,v=model.qkv(block,x,p['rope'])
                    k0,v0=p['cache'][i]
                    new_k=t.cat((k0.cuda(),k),2)
                    new_v=t.cat((v0.cuda(),v),2)
                    assert new_k.size(2)==p['visible'].size(-1)
                    p['hidden']=model.residual(block,x,model.attention(block,q,new_k,new_v,p['visible']))
                    p['cache'][i]=(new_k.cpu(),new_v.cpu())
        for p in panels:
            p['last']=model.logits(p.pop('hidden')[:,0]).argmax(-1)
            if step==0:
                p['check']=p['last'].cpu().numpy().reshape(len(KINDS),2)
    out=[]
    for p in panels:
        cases=p['tokens'].cpu().numpy().reshape(len(KINDS),2,CAP)
        stops=p['stopped'].cpu().numpy().reshape(len(KINDS),2)
        for i,kind in enumerate(KINDS):
            for j,code_index in enumerate((1,5)):
                source=p['rows'][code_index-(kind=='prose')]
                ids=[int(k) for k in cases[i,j] if k>=0]
                text=model.tok.decode(ids,skip_special_tokens=True)
                hit=int(p['initial'][i,j])==source['token']['ids'][source['query'][0]+1]
                next_check=dict(eligible=hit,cached=None,full_prefix=None,equal=None)
                if hit and p['check'] is not None:
                    cached,expected=int(p['check'][i,j]),int(p['expected_next'][i,j])
                    next_check.update(cached=cached,full_prefix=expected,equal=cached==expected)
                ended=bool(stops[i,j])
                out.append(dict(id=source['id'],family=source['family'],marker=source['marker'],layout=source['layout'],
                                frame=source['frame'],arm=kind,gold=source['gold'],foil=source['foil'],tokens=ids,
                                text=text,stopped=ended,score=score(text,source['gold'],ended),check=next_check))
        p['cache'].clear()
    return out


def run(model,views,weights,binding):
    native=S.load('native',binding)
    checked={key:i for i,key in enumerate(native['ids'])}
    rows=N.material(views,M.GENERATE,(2,3))
    files=[];started=time.monotonic()
    total=(len(rows)+SLAB-1)//SLAB
    for part,group in enumerate(batched(rows,SLAB)):
        name=f'generation-{part:02}.json'
        if (OUT/name).exists():
            receipt=R.read(name)
            assert receipt['holdout']==binding and receipt['native']==W.sha(OUT/'native-files.json')
        else:
            result,panels=N.rollout(model,group,weights,N.GEN_ARMS,
                lambda i:C.N.progress('window-generation-prefill',part*40+i,total*40,started),cache=True)
            for j,key in enumerate(result['ids']):
                i=checked[key]
                assert result['reference_top'][j]==native['reference_top'][i]
                assert result['original_top'][j]==native['original_top'][i]
                for arm in ('full','last20'):
                    assert result['top'][j,N.GEN_ARMS.index(arm)]==native['top'][i,N.ARMS.index(arm)]
            decode_start=time.monotonic()
            generated=decode(model,panels,lambda step:C.N.progress('window-generation-decode',step,CAP,decode_start,part=part+1,parts=total))
            R.write(name,dict(at=C.stamp(),holdout=binding,native=W.sha(OUT/'native-files.json'),cap=CAP,views=generated))
        files.append(dict(file=name,sha256=W.sha(OUT/name)))
    R.write('generation-files.json',dict(holdout=binding,files=files))
