"""Natural transitive comparisons, with interior query nodes and code rotation."""
import itertools
import numpy as np
import b36_semantic14 as S

NAMES=['Tom','Mary','Max','Bob','Ann','Dan']

def render(order,relation,depth,rank,fmt,tok):
    rng=np.random.default_rng(S.SEED+80000+rank);edge_order=rng.permutation(5);directions=rng.integers(0,2,5)
    high,low=('taller','shorter') if relation=='height' else ('older','younger')
    facts=[]
    for i in edge_order:
        a,b=order[int(i)],order[int(i)+1]
        facts.append(f'{a} is {high} than {b}.' if directions[i] else f'{b} is {low} than {a}.')
    # Both endpoints have two incident edges; local terminal-node heuristics are insufficient.
    pair={1:(2,3),2:(1,3),3:(1,4)}[depth];gold_name=order[pair[0]];query=[order[i] for i in pair]
    if rank%2:query=query[::-1]
    core='In this fictional group, all of these comparisons are correct:\n'+'\n'.join(facts)+'\nWho is '+high+', '+query[0]+' or '+query[1]+'?'
    pool=[gold_name,order[pair[1]]]+[x for x in order if x not in query][:2]
    base=np.random.default_rng(S.SEED+90000+rank*5+depth).permutation(4).tolist();choices=[pool[i] for i in base]
    rotation=0 if fmt=='direct' else int(fmt[-1]);choices=choices[rotation:]+choices[:rotation];gold=choices.index(gold_name)
    if fmt=='direct':text=core+'\nReply with the person\'s name only.';answers=choices
    else:
        text=core+'\nOptions:\n'+'\n'.join('ABCD'[i]+') '+x for i,x in enumerate(choices))+'\nReply with the option letter only.';answers=list('ABCD')
    prompt=tok.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'];end=prompt.index(core)+len(core)
    at=max(i for i,(_,b) in enumerate(enc['offset_mapping']) if 0<b<=end);choice_ids=[]
    for answer in answers:
        full=tok.encode(prompt+answer,add_special_tokens=False);assert full[:-1]==ids,(answer,full[-4:]);choice_ids.append(full[-1])
    assert len(ids)<=S.LENGTH
    return dict(prompt=prompt,tokens=ids,content_end=at,answer_pos=len(ids)-1,choices=answers,semantic_choices=choices,choice_ids=choice_ids,gold=gold,
        expected_answer=answers[gold],semantic_answer=gold_name,branch=relation,depth=depth,frame=30,format=fmt,
        total_order=order,query=query,query_indices=list(pair),path_nodes=order[pair[0]:pair[1]+1],edge_order=edge_order.tolist(),edge_directions=directions.tolist(),
        core=core,relation=relation)

def prepare():
    S.configure();assert not (S.OUT/'order-pilot-plan.json').exists()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(S.MODEL,local_files_only=True);orders=sorted(list(itertools.permutations(NAMES)),key=lambda x:S.digest([S.SEED,list(x)]));rows=[]
    for rank,order in enumerate(orders[:12]):
      for relation in ['height','age']:
       for depth in [1,2,3]:
        for fmt in ['direct','code0','code2']:
            r=render(list(order),relation,depth,rank,fmt,tok);r.update(id=f'order{rank:03d}:{relation}:d{depth}:{fmt}',family=f'order{rank:03d}');rows.append(r)
    sources={n:S.file_sha(S.ROOT/'tests/glm5'/n) for n in ['b36_semantic14.py','b36_semantic14_score.py','b36_semantic14_order.py','recursive_branch14.py','qwen14_mlp_atlas.py','source_binding_resources.py']}
    S.save('order-pilot-plan.json',dict(rows=rows,n=len(rows),source_hashes=sources,exposure='12 new discovery ranking families only. All height/age, distance and code views belong to the same family.',
        tests='Accuracy >=.9 for each relation, distance and format; both strict candidate and unrestricted first-token. Two/three-hop judgments require links between interior nodes of the full chain.',
        source='Original fictional comparisons, CC0. No real personal attributes.',
        limits=['Repeated names and same chain topology: not an unseen-topology claim.','A model may construct a global ranking instead of sequentially following the annotated path.','Code mappings rotate values while code letters retain normal order. This is distinct from earlier shuffled letter labels.']))
    print('Order pilot',len(rows),flush=True)

if __name__=='__main__':prepare()
