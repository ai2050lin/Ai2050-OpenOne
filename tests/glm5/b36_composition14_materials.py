"""Family-heldout word controls and programs composed from two primitive operators."""
from collections import Counter
import numpy as np
import b36_composition14 as B
from shared.dist_protocol import grammar
from b36_composition14_calibrate_simple import procedure

INTERFACE='plain_dual'

SEQUENCES=['A','B','AB','BA','ABA','BAB']
WORDS=[
 'acorn amber anchor anvil apron arrow atlas badger bamboo banner basil beacon birch bison blade blossom bonnet bronze cabin cactus candle canyon cedar cherry chisel clover cobalt comet coral cotton crane crystal'.split(),
 'island ivory jacket jasmine jewel juniper kettle kiwi ladder lagoon lantern lark laurel lemon leopard lilac linen lion lotus marble meadow melon mirror mitten moss nectar nettle nickel oak olive onion opal'.split()]


def evaluate(branch,values,sequence):
    state=values['n'] if branch=='number' else values['text']
    history=[state]
    for op in sequence:
        if branch=='number':state=state+values['k'] if op=='A' else 2*state
        else:state=state.upper() if op=='A' else state+values['suffix']
        history.append(state)
    return history


def display(branch,value):return str(value)


def render(branch,values,sequence,frame,mode,title,order,codes='ABC'):
    hist=evaluate(branch,values,sequence);input_=display(branch,hist[0]);answer=display(branch,hist[-1]);record=display(branch,values['record'])
    canonical={1:'A',2:'AB',3:'ABA'}[len(sequence)];opposite={1:'B',2:'BA',3:'BAB'}[len(sequence)]
    first=display(branch,evaluate(branch,values,canonical)[-1]);second=display(branch,evaluate(branch,values,opposite)[-1])
    choices=[None]*3;choices[order]=first;choices[(order+1)%3]=record;choices[(order+2)%3]=second
    assert len(set(choices))==3
    cover='matrix, tensor, vector' if title else 'paper, window, basket'
    task=procedure(branch,values,sequence,frame);archive=f'The independent record stores "{record}" for this input.'
    source=task+'\n'+archive if INTERFACE.endswith('dual') else task if mode=='program' else archive
    cue='Return the result of performing the calculation or text operations; ignore the record.' if mode=='program' else 'Return the stored value from the record; ignore the calculation or text operations.'
    body=f'The cover title is "{cover}"; it is only a catalog tag.\n'+source+'\n'+cue+'\nEnd of passage.'
    question='Choose the requested result.\n'+''.join(f'{codes[i]}. {x}\n' for i,x in enumerate(choices))+'Reply with only the option letter.\nAnswer:'
    return dict(body=body,prompt=body+'\n'+question,question=question,choices=choices,gold=choices.index(answer) if mode=='program' else (order+1)%3,
        program_answer=answer,record_answer=record,intermediate_states=[display(branch,x) for x in hist],input_text=input_)


def base_spec(branch):
    return dict(schema_version='template_grammar.v1',family_id=f'B36-COMPOSITION14-{branch}-20261113-v2',grammar_version=2,
        name='Two primitive operators and heldout compositions: '+branch,description='Original fictional procedure and archive have independent outputs. Semantic source changes while other text is shared.',
        factors=['sequence','wording','mode','title'],frames=[],fillers=[dict(id=f'f{i:03d}',slots={},levels={}) for i in range(512)],
        splits=dict(salt='b36-composition14-v1',fractions=dict(discovery=.25,confirm=.5,lockbox=.25)),layers='all',readouts=['target','probe_answer'],
        mlp_units=dict(layers=[36,37,38,39],readouts=['target','probe_answer']),prereg=dict(min_behavior_acc=.9,min_cell_n=80,min_lockbox_cos=0.),
        source=dict(license='CC0-1.0',origin='Author-constructed fictional arithmetic and pair transformation stimuli.'))


def prepare():
    B.configure();assert not (B.OUT/'trial-plan.json').exists()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(B.MODEL,local_files_only=True);rows=[];audits=[]
    for bi,branch in enumerate(['number','label']):
        spec=base_spec(branch);parts=grammar.assign_splits(spec)
        pools={s:sorted([f for f in spec['fillers'] if parts[f['id']]==s],key=lambda f:B.digest(f['id'])) for s in ['discovery','confirm']}
        selected=[(f,'pilot',i) for i,f in enumerate(pools['discovery'][:6])]+[(f,'discovery',i) for i,f in enumerate(pools['discovery'][6:54])]+[(f,'confirm',i) for i,f in enumerate(pools['confirm'][:80])]
        emitted=[];kept=[]
        for filler,split,rank in selected:
            rng=np.random.default_rng(B.SEED+bi*10000+int(filler['id'][1:]));words=rng.choice(WORDS[0 if split!='confirm' else 1],8,replace=False).tolist()
            values=dict(n=int(rng.integers(13,30)),k=int(rng.integers(2,8)),words=words,
                text=''.join(rng.choice(list('abcdefghijklm' if split!='confirm' else 'nopqrstuvwxyz'),2,replace=False)),
                suffix=str(rng.choice(list('rst' if split!='confirm' else 'uvw'))))
            candidates={display(branch,evaluate(branch,values,s)[-1]) for s in SEQUENCES}
            if branch=='number':
                record=int(rng.integers(2,120))
                while str(record) in candidates:record=int(rng.integers(2,120))
                values.update(record=record,foil=record+137)
            else:values.update(record=''.join(rng.choice(list('abcdefghijklmnopqrstuvwxyz'),4,replace=False)),foil='unused')
            family=f'{branch}:{filler["id"]}'
            filler['metadata']=dict(values=values,rank=rank,assigned_split=split)
            for seq in SEQUENCES:
              for frame in range(4):
               for mode in ['program','archive']:
                for title in [0,1]:
                 key=f'{seq}_w{frame}_{mode}_t{title}'
                 order=(rank+len(seq))%3
                 item=render(branch,values,seq,frame,mode,title,order)
                 filler['slots'].update({f'BODY_{key}':item['body'].removesuffix('\nEnd of passage.'),f'QUESTION_{key}':item['question'],f'GOLD_{key}':' '+'ABC'[item['gold']],'END':'End of passage.'})
                 allowed=SEQUENCES if split=='pilot' else ['A','B'] if split=='discovery' else ['A','AB','ABA'] if rank%2==0 else ['B','BA','BAB']
                 expected_frame=rank%2 if split!='confirm' else 2+rank%2
                 order_extra=split=='confirm' and seq not in allowed and len(seq)>1 and mode=='program' and title==0
                 if (seq not in allowed and not order_extra) or frame!=expected_frame:continue
                 for format_ in (['ABC','XYZ'] if split=='confirm' and rank<20 and title==0 else ['ABC']):
                  one=item if format_=='ABC' else render(branch,values,seq,frame,mode,title,order,format_)
                  if order_extra and format_=='XYZ':continue
                  actual_split='order' if order_extra else 'format' if format_=='XYZ' else split
                  enc=tok(one['prompt'],add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'];off=enc['offset_mapping']
                  end=max(i for i,(a,b) in enumerate(off) if b<=len(one['body'])+1 and b>a)
                  assert tok.decode(ids[end:end+1])=='.\n';assert len(ids)<=B.LENGTH,(branch,seq,frame,len(ids))
                  choice_ids=[]
                  for c in format_:
                    suffix=tok.encode(one['prompt']+' '+c,add_special_tokens=False);assert suffix[:-1]==ids
                    choice_ids.append(suffix[-1])
                  row=dict(one,id=f'{family}:{seq}:{mode}:t{title}:{format_}',family=family,branch=branch,split=actual_split,
                      sequence=seq,depth=len(seq),mode=mode,role=mode,title=title,frame=frame,format=format_,rank=rank,
                      tokens=ids,content_end=end,answer_pos=len(ids)-1,content_start=next(i for i,(a,b) in enumerate(off) if a>=one['prompt'].index('\n')+1),
                      choice_ids=choice_ids,grammar_family=spec['family_id'],grammar_partition=parts[filler['id']],license='CC0-1.0',source='author_constructed')
                  rows.append(row)
                  if format_=='ABC':emitted.append(row)
            kept.append(filler)
        for seq in SEQUENCES:
         for frame in range(4):
          for mode in ['program','archive']:
           for title in [0,1]:
            key=f'{seq}_w{frame}_{mode}_t{title}'
            spec['frames'].append(dict(id=key,levels=dict(sequence=seq,wording=str(frame),mode=mode,title=str(title)),text='{BODY_'+key+'}\n{END}',target='END',
                probe=dict(prompt='{BODY_'+key+'}\n{END}\n{QUESTION_'+key+'}',answer='{GOLD_'+key+'}')))
        spec['fillers']=kept;errors=grammar.validate(spec);assert not errors,errors
        generated=grammar.generate(spec,include_canary=False)
        promptset={r['probe']['prompt'] for r in generated['items']};assert all(r['prompt'] in promptset for r in emitted)
        B.save(f'grammar-{branch}.json',dict(specification=spec))
        audits.append(dict(branch=branch,spec_sha=grammar.spec_sha(spec),validation_errors=errors,visibility=generated['visibility'],warnings=generated['_warnings'],emitted=len(emitted)))
    order={'pilot':0,'discovery':1,'confirm':2,'format':3,'order':4}
    rows.sort(key=lambda r:(order[r['split']],r['family'],r['sequence'],r['mode'],r['title']))
    assert len({r['id'] for r in rows})==len(rows)
    counts=Counter(r['split'] for r in rows);assert counts==dict(pilot=288,discovery=768,confirm=1920,format=240,order=320),counts
    assert not ({r['family'] for r in rows if r['split']=='discovery'} & {r['family'] for r in rows if r['split'] in ['confirm','format','pilot']})
    by={r['id']:r for r in rows};length_pairs=[]
    for r in rows:
        if r['title']==0 and r['format']=='ABC' and r['split']!='order':
            other=by[r['id'].replace(':t0:',':t1:')];length_pairs.append(len(r['tokens'])-len(other['tokens']))
    assert set(length_pairs)=={0},'Title control must preserve all later token positions.'
    mode_lengths=[len(r['tokens'])-len(by[r['id'].replace(':program:',':archive:')]['tokens']) for r in rows if r['mode']=='program' and r['split']!='order']
    assert set(mode_lengths)=={0},'Source cue control must preserve readout positions.'
    order_audit=[]
    import re
    for r in rows:
        if r['split']!='order':continue
        partner={'AB':'BA','BA':'AB','ABA':'BAB','BAB':'ABA'}[r['sequence']]
        other=by[r['id'].replace(':'+r['sequence']+':',':'+partner+':')]
        assert r['choices']==other['choices'] and other['split']=='confirm'
        same_bag=sorted(re.findall(r'[A-Za-z]+|\d+',r['body']))==sorted(re.findall(r'[A-Za-z]+|\d+',other['body']))
        if r['depth']==2:assert same_bag
        order_audit.append(dict(id=r['id'],main_id=other['id'],depth=r['depth'],identical_options=True,same_word_multiset=same_bag))
    B.save('materials.json',dict(rows=rows,counts=dict(counts),source_sha256=B.file_sha(__file__),
        splits='Protocol family hashes. Six discovery families per branch are behavior pilot only; 48 other families fit primitives A/B. Eighty confirm families per branch test one primitive plus AB/BA and ABA/BAB. Format panel reuses first20 confirm families. Order panel adds opposite orders for the same80 confirm families, program/plain-title only, preserving identical choices. Reused families do not increase independent N.',
        composition_scope='Sequences are absent from predictor fitting, not unknown to the experimenter or behavior pilot. No claim of unseen numbers or unseen training data.'))
    B.save('material-audit.json',dict(audits=audits,rows=len(rows),max_tokens=max(len(r['tokens']) for r in rows),min_tokens=min(len(r['tokens']) for r in rows),
        title_token_length_differences=sorted(set(length_pairs)),all_choice_continuations_one_token=True,
        mode_token_length_differences=sorted(set(mode_lengths)),order_pairs=order_audit,
        alternatives=['Operation words and source labels can still carry task-mode information.','Record output is independently randomized and never Not specified.','All source information is visible at content readout; choices only appear afterward.','Two heldout wording frames do not represent all natural language.'],
        source_sha256=B.file_sha(__file__)))
    print(dict(counts),flush=True)


if __name__=='__main__':prepare()
