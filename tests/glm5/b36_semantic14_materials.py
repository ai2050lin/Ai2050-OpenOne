"""New finite-state composition families, grouped under graph relabelling."""
import argparse,itertools
import numpy as np
import b36_semantic14 as S
from shared.dist_protocol import grammar

WORDS={'color':['red','blue','green','white','black','gold'],'animal':['cat','dog','bird','fish','horse','bear']}
PERMS=np.array(list(itertools.permutations(range(6))),dtype=np.int8);INVERSE=np.argsort(PERMS,axis=1)
FORMATS=['word_plain','word_book']+[f'code{i}' for i in range(6)]

def canonical(a,b):
    # All six-state relabellings, and exchange of operator names, stay in one family.
    rows=np.arange(len(PERMS))[:,None]
    aa=PERMS[rows,np.asarray(a)[INVERSE]];bb=PERMS[rows,np.asarray(b)[INVERSE]]
    return min(min(x.tobytes() for x in np.concatenate([aa,bb],axis=1)),min(x.tobytes() for x in np.concatenate([bb,aa],axis=1))).hex()

def history(graph,sequence,start):
    out=[start]
    for op in sequence:out.append(graph[op][out[-1]])
    return out

def body(graph,branch,sequence,start,frame,order):
    w=WORDS[branch]
    lines=['Use the two lookup rules below. Each instruction replaces the current state once.']
    if frame==0:
        for op in 'AB':lines.append('Rule '+op+': '+ '; '.join(w[i]+' -> '+w[graph[op][i]] for i in order)+'.')
    else:
        for op in 'AB':lines.append('For '+op+', '+ '; '.join('replace '+w[i]+' with '+w[graph[op][i]] for i in order)+'.')
    lines.append('Initial state: '+w[start]+'.')
    marks=[]
    for i,op in enumerate(sequence):
        lines.append(f'Step {i+1}: apply rule {op}.');marks.append(len('\n'.join(lines)))
    return '\n'.join(lines),marks

def render(graph,branch,sequence,start,frame,order,format_,tok):
    core,marks=body(graph,branch,sequence,start,frame,order);w=WORDS[branch]
    rotation=0 if format_ in ['word_plain','word_book','alt0'] else int(format_[-1])
    symbols='UVWXYZ' if format_=='alt0' else 'ABCDEF'
    codes=[symbols[(i+rotation)%6] for i in range(6)]
    extra=''
    if format_!='word_plain':extra='\nAnswer codes: '+ '; '.join(codes[i]+' = '+w[i] for i in range(6))+'.'
    cue='Return only the final state word, with no explanation.' if format_.startswith('word') else 'Return only the answer code for the final state, with no explanation.'
    user=core+extra+'\n'+cue
    prompt=tok.apply_chat_template([dict(role='user',content=user)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'];off=enc['offset_mapping'];shift=prompt.index(core)
    positions=[max(i for i,(a,b) in enumerate(off) if 0<b<=shift+m) for m in marks]
    h=history(graph,sequence,start);answers=w if format_.startswith('word') else codes;choice_ids=[]
    for word in answers:
        full=tok.encode(prompt+word,add_special_tokens=False);assert full[:-1]==ids,(word,full[-4:],ids[-3:]);choice_ids.append(full[-1])
    assert len(ids)<=S.LENGTH,(len(ids),frame,format_)
    return dict(prompt=prompt,tokens=ids,answer_pos=len(ids)-1,content_end=positions[-1],step_positions=positions,
        intermediate_states=h,state_words=[w[i] for i in h],semantic_ids=[tok.encode(x,add_special_tokens=False)[0] for x in w],
        choices=answers,choice_ids=choice_ids,gold=h[-1],expected_answer=answers[h[-1]],core=core,code_mapping=codes,
        branch=branch,depth=len(sequence),sequence=sequence,frame=frame,format=format_,start=start,table_order=order,
        prefix_tokens_before_format=ids[:positions[-1]+1],repeated_intermediate=len(set(h))<len(h))

def prepare():
    S.configure(60*2**20);assert not (S.OUT/'pilot-plan.json').exists()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(S.MODEL,local_files_only=True)
    assert all(len(tok.encode(w,add_special_tokens=False))==1 for words in WORDS.values() for w in words)
    rng=np.random.default_rng(S.SEED);graphs={}
    while len(graphs)<256:
        a=rng.permutation(6).tolist();b=rng.permutation(6).tolist()
        if all(a[b[i]]==b[a[i]] for i in range(6)):continue
        key=canonical(a,b)
        if key not in graphs:graphs[key]=dict(id='g'+S.digest(key)[:15],canonical=key,A=a,B=b)
    spec=dict(schema_version='template_grammar.v1',family_id='B36-SEMANTIC14-20261113-v1',grammar_version=1,
        name='Composed lookup states and answer codes',description='Original six-state permutation programs. Semantic states and answer-code mappings have separate factors.',
        factors=['branch','depth','frame','format'],frames=[],fillers=[dict(id=g['id'],slots={},levels={},metadata=g) for g in graphs.values()],
        splits=dict(salt='semantic-independent-20261113',fractions=dict(discovery=.375,confirm=.5,lockbox=.125)),layers='all',readouts=['target','probe_answer'],
        prereg=dict(min_behavior_acc=.9,min_cell_n=96,min_lockbox_cos=0.),source=dict(license='CC0-1.0',origin='Author constructed finite-state rule graphs, new prompts.'))
    parts=grammar.assign_splits(spec);groups={s:sorted([f for f in spec['fillers'] if parts[f['id']]==s],key=lambda f:S.digest(f['id'])) for s in ['discovery','confirm','lockbox']}
    selected=[(f,'pilot',i) for i,f in enumerate(groups['discovery'][:12])]+[(f,'discovery',i) for i,f in enumerate(groups['discovery'][12:60])]+[(f,'confirm',i) for i,f in enumerate(groups['confirm'][:96])]+[(f,'lockbox',i) for i,f in enumerate(groups['lockbox'])]
    identities=[];pilot=[];kept=[]
    for filler,split,rank in selected:
        g=filler['metadata'];g=dict(g,split=split,rank=rank,start=rank%6,sequence='ABA' if rank%2==0 else 'BAB',order=np.random.default_rng(int(g['id'][1:9],16)).permutation(6).tolist())
        identities.append(g)
        if split!='pilot':continue
        for branch in WORDS:
          for depth in [1,2,3]:
           for frame in [0,1]:
            for fmt in FORMATS:
                key=f'{branch}_d{depth}_f{frame}_{fmt}';r=render(g,branch,g['sequence'][:depth],g['start'],frame,g['order'],fmt,tok)
                r.update(id=g['id']+':'+key,family=g['id'],split=split,rank=rank,graph_canonical=g['canonical'])
                pilot.append(r);filler['slots'].update({f'PROMPT_{key}':r['prompt'],f'GOLD_{key}':r['expected_answer'],f'CORE_{key}':r['core'],'END':'End of instructions.'})
        kept.append(filler)
    for branch in WORDS:
      for depth in [1,2,3]:
       for frame in [0,1]:
        for fmt in FORMATS:
            key=f'{branch}_d{depth}_f{frame}_{fmt}'
            spec['frames'].append(dict(id=key,levels=dict(branch=branch,depth=str(depth),frame=str(frame),format=fmt),text='{CORE_'+key+'}\n{END}',target='END',probe=dict(prompt='{PROMPT_'+key+'}',answer='{GOLD_'+key+'}')))
    # Full graph partition remains immutable; a pilot-only protocol view never opens heldout model responses.
    spec['fillers']=kept;errors=grammar.validate(spec);assert not errors,errors
    generated=grammar.generate(spec,include_canary=False)
    assert {r['prompt'] for r in pilot}=={r['probe']['prompt'] for r in generated['items']}
    S.save('graph-families.json',dict(graphs=identities,partition_by_id=parts,unique_canonical_graphs=len(graphs),source_sha256=S.file_sha(__file__),
        split_unit='Whole operator pair modulo all6! state relabellings and operator-name exchange; both lexical domains, every prefix and answer-code view stay together.'))
    S.save('pilot-grammar.json',dict(specification=spec,validation_errors=errors,warnings=generated['_warnings'],visibility=generated['visibility']))
    sources={n:S.file_sha(S.ROOT/'tests/glm5'/n) for n in ['b36_semantic14.py','b36_semantic14_materials.py','b36_semantic14_score.py','recursive_branch14.py','qwen14_mlp_atlas.py','source_binding_resources.py']}
    S.save('pilot-plan.json',dict(rows=pilot,n=len(pilot),source_hashes=sources,graph_sha256=S.file_sha(S.OUT/'graph-families.json'),
        hypotheses=['Direct semantic answers and every cyclic code rotation remain reliable for2/3 composed lookups.','Adding a codebook without requesting codes controls part of the codebook/context effect.'],
        exposure='Only12 independent discovery graph families. No heldout response opened. Two domains are paired views of the same graphs, not24 independent graph families.',
        selection='For each frame, all branch/depth/output-mode cells must have >=.9 strict forced-choice AND >=.9 unrestricted next-token accuracy; each individual rotation must also have >=.9. Frame with greatest worst-cell accuracy wins, ties frame0. Failure stops affected joint mechanism; no threshold relaxation.',
        outputs='Each requested state/code is exactly one token at actual chat continuation. Record unrestricted top1 and top10 plus all6 candidate scores. This is one-token direct answering, not multi-token explanation.',
        identity='Rehash all original14 model/tokenizer files before execution; original BF16 CUDA layer streaming, no quantization.',
        resources=dict(new_peak_bytes=60*2**20,planned_full_campaign_peak_bytes=1600*2**20,reserve_bytes=20*2**30),
        limits=['Explicit finite-state lookup is a narrow constructed task, not all natural-language semantics.','The same six names per domain recur; confirm holds out graph structure, not words.','Intermediate truth remains evaluation metadata and is never supplied as an answer.']))
    print('Prepared pilot',len(pilot),'rows; new whole-graph families',len(identities),flush=True)

if __name__=='__main__':prepare()

