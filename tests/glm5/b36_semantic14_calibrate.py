"""Interface diagnostic restricted to already exposed pilot graph families."""
import numpy as np
import b36_semantic14 as S
from b36_semantic14_materials import WORDS,history

def render(g,branch,depth,fmt,variant,tok):
    w=WORDS[branch];seq=g['sequence'][:depth];hist=history(g,seq,g['start']);rotation=0 if fmt.startswith('word') else int(fmt[-1]);codes=['ABCDEF'[(i+rotation)%6] for i in range(6)]
    function=variant in [2,3,5]
    if function:
        lines=['The functions f and g are defined by this complete table:']
        for i in g['order']:lines.append(f'f({w[i]}) = {w[g["A"][i]]}; g({w[i]}) = {w[g["B"][i]]}')
        expr=w[g['start']]
        for op in seq:expr=('f' if op=='A' else 'g')+'('+expr+')'
        lines.append('Evaluate '+expr+'.')
    else:
        lines=['Transform a state using these two rules:']
        for op,name in [('A','move'),('B','jump')]:
            lines.append(name+': '+ '; '.join(w[i]+' becomes '+w[g[op][i]] for i in g['order'])+'.')
        lines.append('Start with '+w[g['start']]+'.')
        lines.append('Then '+', then '.join('move' if op=='A' else 'jump' for op in seq)+'.')
    core='\n'.join(lines)
    if fmt!='word_plain':lines.append('Options:\n'+'\n'.join(codes[i]+') '+w[i] for i in range(6)))
    lines.append('Give only the resulting state word.' if fmt.startswith('word') else 'Give only the option letter for the result.')
    text='\n'.join(lines)
    if variant in [2,4]:prompt=text+'\nAnswer:';leading=' '
    else:
        prompt=tok.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        if variant==3:prompt+='Answer:';leading=' '
        else:leading=''
    enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'];core_end=prompt.index(core)+len(core)
    at=max(i for i,(_,b) in enumerate(enc['offset_mapping']) if 0<b<=core_end)
    choices=w if fmt.startswith('word') else codes;choice_ids=[]
    for word in choices:
        full=tok.encode(prompt+leading+word,add_special_tokens=False);assert full[:-1]==ids;choice_ids.append(full[-1])
    assert len(ids)<=S.LENGTH
    return dict(prompt=prompt,tokens=ids,content_end=at,answer_pos=len(ids)-1,choice_ids=choice_ids,choices=choices,gold=hist[-1],
        intermediate_states=hist,state_words=[w[i] for i in hist],expected_answer=choices[hist[-1]],branch=branch,depth=depth,sequence=seq,format=fmt,frame=variant,core=core,leading=leading)

def prepare():
    S.configure();assert not (S.OUT/'interface-plan.json').exists()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(S.MODEL,local_files_only=True);old=S.read(S.OUT/'pilot-plan.json');exposed={r['id'] for p in (S.OUT/'pilot-score').glob('chunk-*.json') for r in S.read(p)['results']}
    oldby={r['id']:r for r in old['rows']};families=sorted({oldby[i]['family'] for i in exposed},key=lambda f:next(r['rank'] for r in old['rows'] if r['family']==f))[:3]
    graphs=[g for g in S.read(S.OUT/'graph-families.json')['graphs'] if g['id'] in families];rows=[]
    for g in graphs:
      assert all(r['id'] in exposed for r in old['rows'] if r['family']==g['id'])
      for branch in WORDS:
       for depth in [1,2,3]:
        for fmt in ['word_plain','word_book','code0','code1','code3','code5']:
         for variant in [2,3,4,5]:
            r=render(g,branch,depth,fmt,variant,tok);r.update(id=f'{g["id"]}:{branch}:d{depth}:{fmt}:v{variant}',family=g['id']);rows.append(r)
    sources=dict(old['source_hashes']);sources['b36_semantic14_calibrate.py']=S.file_sha(__file__)
    S.save('interface-plan.json',dict(rows=rows,n=len(rows),source_hashes=sources,graph_sha256=S.file_sha(S.OUT/'graph-families.json'),
        exposure='Only3 fully exposed pilot graph families. Four response interfaces, no heldout response, no formal mechanism claims. Original pilot retained.',
        rationale='Original codebook often elicited Step and failed required cells. Separate operator f/g or move/jump from output letters, use linewise options, and compare explicit Answer prefix to direct chat.',
        variants={2:'plain_function_Answer',3:'chat_function_Answer',4:'plain_move_jump_Answer',5:'chat_function_no_prefill'}))
    print('Interface pilot',len(rows),flush=True)

if __name__=='__main__':prepare()
