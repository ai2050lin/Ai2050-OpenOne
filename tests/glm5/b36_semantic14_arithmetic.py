"""Independent calibration of familiar arithmetic composition, after lookup failure.

Arithmetic expressions and worded operations are two realizations of one family,
not independent semantic domains. No heldout model response is used for design.
"""
import itertools,argparse
import numpy as np
import b36_semantic14 as S
from shared.dist_protocol import grammar

OPS=[('add',k) for k in range(1,6)]+[('sub',k) for k in range(1,6)]+[('mul',2),('mul',3)]+[('mod',k) for k in range(2,6)]

def step(x,op):
    kind,k=op
    return x+k if kind=='add' else x-k if kind=='sub' else x*k if kind=='mul' else x%k

def states(g,depth=3):
    h=[g['start']]
    for op in g['ops'][:depth]:h.append(step(h[-1],op))
    return h

def families():
    out=[]
    for ops in itertools.product(OPS,repeat=3):
        if len({op[0] for op in ops})<2:continue
        valid=[]
        for start in range(10):
            g=dict(start=start,ops=ops);h=states(g)
            if all(0<=x<=9 for x in h) and all(a!=b for a,b in zip(h,h[1:])):valid.append(start)
        if not valid:continue
        key=[[step(x,op) for x in range(10)] for op in ops]
        fid='a'+S.digest(key)[:15];rng=np.random.default_rng(int(fid[1:9],16))
        out.append(dict(id=fid,ops=ops,start=int(rng.choice(valid)),valid_starts=valid,type='modular' if any(o[0]=='mod' for o in ops) else 'affine',semantic_key=key))
    assert len({S.digest(g['semantic_key']) for g in out})==len(out)
    spec=dict(family_id='B36-ARITHMETIC14-20261113',fillers=[dict(id=g['id'],levels=dict(type=g['type'])) for g in out],splits=dict(salt='new-programs',fractions=dict(discovery=.4,confirm=.4,lockbox=.2)))
    parts=grammar.assign_splits(spec)
    for g in out:g['partition']=parts[g['id']]
    return out

def render(g,depth,wording,fmt,interface,tok):
    hist=states(g,depth);expr=str(g['start']);lines=[f'Start with the number {g["start"]}.'];marks=[]
    for op in g['ops'][:depth]:
        kind,k=op;symbol={'add':'+','sub':'-','mul':'*','mod':'%'}[kind];expr=f'({expr} {symbol} {k})'
        text={'add':f'Add {k} to it.','sub':f'Subtract {k} from it.','mul':f'Multiply it by {k}.','mod':f'Replace it with the remainder after division by {k}.'}[kind]
        lines.append(text);marks.append(len('\n'.join(lines)))
    core=('Evaluate '+expr+'. The symbol % means the integer remainder after division.' if wording=='expression' and g['type']=='modular' else 'Evaluate '+expr+'.') if wording=='expression' else '\n'.join(lines)
    rng=np.random.default_rng(int(S.digest([g['id'],depth])[:8],16));pool=[x for x in reversed(hist[:-1]) if x!=hist[-1]]
    pool=list(dict.fromkeys(pool+[int(x) for x in rng.permutation(10) if x!=hist[-1]]))[:3]
    choices=[hist[-1]]+pool;order=rng.permutation(4);choices=[choices[i] for i in order];gold=choices.index(hist[-1])
    rotation=int(fmt[-1]) if fmt.startswith('code') else 0;codes=['ABCD'[(i+rotation)%4] for i in range(4)]
    book='\nChoices:\n'+'\n'.join(codes[i]+') '+str(v) for i,v in enumerate(choices)) if fmt!='direct' else ''
    cue='Output exactly one character: the result digit. No spaces, punctuation or explanation.' if fmt=='direct' else 'Output exactly one character: the correct option letter. No spaces, punctuation or explanation.'
    user=core+book+'\n'+cue
    if interface==10:prompt=user+'\nAnswer:\n'
    else:
        prompt=tok.apply_chat_template([dict(role='user',content=user)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        if interface==12:prompt+='Final answer:\n'
    enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'];end=prompt.index(core)+len(core)
    content_end=max(i for i,(_,b) in enumerate(enc['offset_mapping']) if 0<b<=end)
    answers=[str(x) for x in choices] if fmt=='direct' else codes;choice_ids=[]
    for a in answers:
        full=tok.encode(prompt+a,add_special_tokens=False);assert full[:-1]==ids,(interface,fmt,a);choice_ids.append(full[-1])
    return dict(prompt=prompt,tokens=ids,answer_pos=len(ids)-1,content_end=content_end,step_positions=[],intermediate_states=hist,
        expected_answer=answers[gold],choice_ids=choice_ids,choices=answers,semantic_choices=choices,gold=gold,semantic_gold=hist[-1],
        branch=g['type']+'_'+wording,depth=depth,sequence=str(g['ops'][:depth]),format=fmt,frame=interface,wording=wording,program_type=g['type'],core=core)

def prepare():
    S.configure();assert not (S.OUT/'arithmetic-interface-plan.json').exists()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(S.MODEL,local_files_only=True);graphs=families();selected=[];rows=[]
    for type_ in ['affine','modular']:
        pool=sorted([g for g in graphs if g['partition']=='discovery' and g['type']==type_],key=lambda g:S.digest(g['id']))
        selected+=pool[:3]
    for g in selected:
      for depth in [1,2,3]:
       for wording in ['expression','worded']:
        for fmt in ['direct','code0','code2']:
         for interface in [10,11,12]:
            r=render(g,depth,wording,fmt,interface,tok);r.update(id=f'{g["id"]}:{depth}:{wording}:{fmt}:{interface}',family=g['id']);rows.append(r)
    assert max(len(r['tokens']) for r in rows)<=S.LENGTH
    source={n:S.file_sha(S.ROOT/'tests/glm5'/n) for n in ['b36_semantic14.py','b36_semantic14_score.py','b36_semantic14_arithmetic.py','recursive_branch14.py','qwen14_mlp_atlas.py','source_binding_resources.py']}
    S.save('arithmetic-families.json',dict(families=graphs,diagnostic_family_ids=[g['id'] for g in selected],split_unit='All starting values and lexical/code realizations of the same ordered3 primitive functions share a family. Two surface wordings are paired, not independent domains.',source_sha256=S.file_sha(__file__)))
    S.save('arithmetic-interface-plan.json',dict(rows=rows,n=len(rows),source_hashes=source,exposure='Six new discovery arithmetic programs; no confirmation or lockbox response. Exploratory interface calibration only.',
        motivation='Arbitrary six-state lookup failed direct behavior. Familiar arithmetic has independently computed intermediate truth and unambiguous one-character outputs. This does not establish cross-domain common semantics.',
        variants={10:'plain_Answer_newline',11:'chat_one_character',12:'chat_Final_answer_newline'},thresholds='Diagnostic only; a chosen design requires a separate larger discovery behavior pilot before independent confirmation.'))
    print('Arithmetic interface diagnostic',len(rows),'candidate programs',len(graphs),flush=True)

if __name__=='__main__':prepare()
