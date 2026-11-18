"""Rule-vs-record branch families, with unseen expressions and unseen fitted branches."""
import argparse
from collections import Counter
import numpy as np
import branch_reuse14 as B
from shared.dist_protocol import grammar

BRANCHES=['number','sequence','coordinate','label']
ELIGIBLE=['number','coordinate','label']
WORDS=[
 'acorn amber anchor anvil apron arrow atlas badger bamboo banner basil beacon birch bison blade blossom bonnet bronze cabin cactus candle canyon cedar cherry chisel clover cobalt comet coral cotton crane crystal'.split(),
 'island ivory jacket jasmine jewel juniper kettle kiwi ladder lagoon lantern lark laurel lemon leopard lilac linen lion lotus marble meadow melon mirror mitten moss nectar nettle nickel oak olive onion opal'.split(),
 'saffron salmon sandal sapphire satin scarf seal sesame shell shield shore silver slate snail snow sparrow spear spice spindle spruce squirrel star stone storm straw summit swan sword table talon temple thorn'.split()]
SEED=B.SEED
QUESTION='If the document supplies a general instruction, execute it on {QUERY} and choose the result. If it supplies only particular input-output records, choose "Not specified" for this unlisted input. What is the correct option?'


def content(branch,variant,values):
    r=values
    if branch=='number':
        verb='add' if r['op']==0 else 'subtract'
        rule=[f"{verb.capitalize()} {r['k']} to the input number." if verb=='add' else f"Subtract {r['k']} from the input number.",
            f"The output number is the input number {'plus' if r['op']==0 else 'minus'} {r['k']}.",
            f"An incoming number is {'increased' if r['op']==0 else 'decreased'} by {r['k']} to obtain the outgoing number.",
            f"Taking any input value, {'increase' if r['op']==0 else 'reduce'} it by {r['k']}; return the value obtained."][variant]
        function=lambda x:str(x+(r['k'] if r['op']==0 else -r['k']))
        inputs=[str(r['n']+i*5) for i in range(3)]
        outputs=[function(int(x)) for x in inputs]
        foil=str(int(inputs[2])-(r['k'] if r['op']==0 else -r['k']))
        keywords=f"input number; output number; {verb}; {r['k']}"
    elif branch=='sequence':
        rule=([
            'Reverse the order of the three entries.',
            'Exchange the first and last entries while leaving the middle one in place.',
            'Read the incoming triple from its final entry back to its initial entry to obtain the outgoing triple.',
            'The outgoing list takes the third entry first, the second entry second, and the first entry last.'
        ] if r['op']==0 else [
            'Move the first entry to the end of the three-entry list.',
            'Take the second and third entries in their original order, then append the first entry.',
            'The leading item becomes the trailing item; the two remaining items keep their relative order.',
            'The outgoing list takes the second entry first, the third entry second, and the first entry last.'
        ])[variant]
        w=r['words'];triples=[w[0:3],w[3:6],w[6:9]]
        inputs=[', '.join(t) for t in triples]
        outputs=[', '.join(t[::-1] if r['op']==0 else t[1:]+t[:1]) for t in triples]
        t=triples[2];foil=', '.join(t[1:]+t[:1] if r['op']==0 else t[::-1])
        keywords='input order; output order; first entry; last entry; reverse; move'
    elif branch=='coordinate':
        rule=[
            f"Add {r['k']} to the first coordinate and keep the second coordinate unchanged.",
            f"An output pair has its first coordinate increased by {r['k']}; its second coordinate equals the input's second coordinate.",
            f"A point moves {r['k']} units along the first axis and zero units along the second axis.",
            f"Keep a point's second component fixed, and form its new first component by adding {r['k']} to the old one."
        ][variant]
        pairs=[(r['n']+i*3,r['n']+11-i*2) for i in range(3)]
        inputs=[f'({a}, {b})' for a,b in pairs];outputs=[f'({a+r["k"]}, {b})' for a,b in pairs]
        a,b=pairs[2];foil=f'({a}, {b+r["k"]})';keywords='input point; output point; first coordinate; second coordinate; add'
    else:
        marker=r['words'][9]
        rule=[f'Attach the marker "{marker}" before every input label, separated by a space.',
            f'Write "{marker}" first and then a space and the unchanged input label to form the output.',
            f'Every name is extended at its left edge with "{marker} "; its existing spelling is retained.',
            f'An outgoing label consists of the prefix "{marker} ", followed immediately by the original incoming label.'][variant]
        inputs=r['words'][:3];outputs=[marker+' '+s for s in inputs];foil=inputs[2]+' '+marker
        keywords=f'input label; output label; {marker}; before; prefix; unchanged'
    return dict(rule=rule,examples=f'input {inputs[0]} → output {outputs[0]}; input {inputs[1]} → output {outputs[1]}',
        query=inputs[2],answer=outputs[2],foil=foil,keywords=keywords)


def specification(branch):
    fillers=[dict(id=f'f{i:03d}',slots={'BODY':'placeholder','QUERY':'placeholder','C0':'placeholder','C1':'placeholder','C2':'placeholder','RULE_GOLD':' A','RECORD_GOLD':' B','END':'End of passage.'},levels={}) for i in range(256)]
    frames=[]
    for frame in range(4):
      for role in ['rule','record']:
       for title in [0,1]:
        frames.append(dict(id=f'f{frame}-{role}-t{title}',levels={'role':role,'title':str(title),'wording':str(frame)},
            text='{BODY}\n{END}',target='END',probe={'prompt':'{BODY}\n{END}\n'+QUESTION+'\nA. {C0}\nB. {C1}\nC. {C2}\nGive only the option letter.\nAnswer:',
                'answer':'{RULE_GOLD}' if role=='rule' else '{RECORD_GOLD}'}))
    return dict(schema_version='template_grammar.v1',family_id=f'BR14-{branch}-20261113-v1',grammar_version=3,
        name='Universal transformation rule versus finite records: '+branch,
        description='Original fictional records. Rules determine unlisted inputs; finite records explicitly do not. This is not a claim that relationship concepts are absent in controls.',
        factors=['role','title','wording'],frames=frames,fillers=fillers,
        splits={'salt':'branch-reuse14-v1','fractions':{'discovery':.25,'confirm':.5,'lockbox':.25},'heldout_frames':[f['id'] for f in frames if f['id'].startswith(('f2-','f3-'))]},
        layers='all',readouts=['target','probe_answer'],mlp_units={'layers':'all','readouts':['target','probe_answer']},
        prereg={'min_behavior_acc':.9,'min_cell_n':80,'min_lockbox_cos':0.0},shards=1,
        source={'license':'CC0-1.0','origin':'Original constructed stimuli; no claim about facts of the world.'})


def body_text(info,branch,frame,role,title):
    titletext='algorithm, transformation, equation, matrix' if title else 'notebook, margin, caption, folder'
    prefix=f'The file title is "{titletext}". The title is a catalog tag, not an instruction.\n'
    if role=='rule':
        lead=['This document defines a rule for every permitted input: ',
              'The instruction below applies to all inputs of this type: ',
              'Unlisted inputs are governed by this general specification: ',
              'For any admissible input, its output is determined as follows: '][frame]
        text=lead+info['rule']+'\nTwo illustrative records are '+info['examples']+'.'
    else:
        lead=['This document specifies only the two input-output records printed below. ',
              'The entries below are an exhaustive list of the outputs that this document specifies. ',
              'Unlisted inputs have no stipulated output; only the following individual cases are documented. ',
              'Each association below is a separate recorded case. Nothing specifies an output for any other input. '][frame]
        text=lead+info['examples']+'.\nThe archive keyword line is "'+info['keywords']+'"; keywords do not specify further entries.'
    return prefix+text


def prepare():
    from transformers import AutoTokenizer
    B.configure(100*2**20);assert not (B.OUT/'materials.json').exists()
    tok=AutoTokenizer.from_pretrained(B.MODEL,local_files_only=True)
    rows=[];audits=[]
    for bi,branch in enumerate(BRANCHES):
        if branch not in ELIGIBLE:continue
        spec=specification(branch);parts=grammar.assign_splits(spec);counts=Counter()
        byid={x['id']:x for x in spec['fillers']}
        # Actual per-frame text is rendered below, because role/title is an external
        # controlled factor. The grammar still owns family partitions and audit.
        branch_rows=[]
        for fl in sorted(spec['fillers'],key=lambda x:x['id']):
            part=parts[fl['id']];rank=counts[part];counts[part]+=1
            if part=='discovery' and rank<16:
                split='pilot';frame=rank%2
            elif part=='discovery' and branch in ['number','coordinate']:
                split='discovery';frame=rank%2
            elif part=='confirm' and rank<80:
                split='confirm';frame=2+rank%2
            else:continue
            rng=np.random.default_rng(SEED+bi*10000+int(fl['id'][1:]))
            pool=WORDS[0 if part=='discovery' else 1]
            values=dict(n=int(rng.integers(12,63)),k=int(rng.integers(2,9)),op=rank%2,words=rng.choice(pool,10,replace=False).tolist())
            info=content(branch,frame,values);order=rank%3
            choices=[None]*3;choices[order]=info['answer'];choices[(order+1)%3]=info['foil'];choices[(order+2)%3]='Not specified'
            family=f'{branch}:{fl["id"]}';grammar_frame=[]
            for role in ['rule','record']:
              for title in [0,1]:
                body=body_text(info,branch,frame,role,title)+'\nEnd of passage.'
                prompt=body+'\n'+QUESTION.replace('{QUERY}',info['query'])+'\n'+''.join(f'{"ABC"[j]}. {c}\n' for j,c in enumerate(choices))+'Give only the option letter.\nAnswer:'
                enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'];off=enc['offset_mapping']
                end=max(i for i,(a,b) in enumerate(off) if b<=len(body)+1 and b>a)
                assert tok.decode(ids[end:end+1])=='.\n'
                assert len(ids)<=256,(branch,role,len(ids))
                choice_ids=[tok.encode(' '+c,add_special_tokens=False)[0] for c in 'ABC']
                for c,cid in zip('ABC',choice_ids):assert tok.encode(prompt+' '+c,add_special_tokens=False)==ids+[cid]
                row=dict(id=f'{family}:{role}:t{title}',family=family,branch=branch,split=split,frame=frame,role=role,title=title,
                    prompt=prompt,body=body,tokens=ids,content_end=end,answer_pos=len(ids)-1,
                    content_start=next(i for i,(a,b) in enumerate(off) if a>=prompt.index('\n')+1),
                    gold=order if role=='rule' else (order+2)%3,choice_ids=choice_ids,
                    query=info['query'],rule_answer=info['answer'],choices=choices,operation=values['op'],lexical_words=values['words'],
                    grammar_family=spec['family_id'],grammar_partition=part,
                    license='CC0-1.0',source='author_constructed_rule_record_family')
                branch_rows.append(row)
            fl['slots'].update(BODY=body_text(info,branch,frame,'rule',0),QUERY=info['query'],C0=choices[0],C1=choices[1],C2=choices[2],
                RULE_GOLD=' '+'ABC'[order],RECORD_GOLD=' '+'ABC'[(order+2)%3])
            fl['metadata']=dict(rank=rank,operation=values['op'],values=values)
        # Save a fully instantiated protocol grammar: every frame renders its own
        # controlled body using role/title-specific slots, with a shared family ID.
        for fl in spec['fillers']:
            if 'metadata' not in fl:
                for frame in spec['frames']:fl['slots']['BODY_'+frame['id'].replace('-','_')]='Unused family, never emitted for this experiment.'
                continue
            for frame in spec['frames']:
                fi=int(frame['levels']['wording']);info=content(branch,fi,fl['metadata']['values'])
                fl['slots']['BODY_'+frame['id'].replace('-','_')]=body_text(info,branch,fi,frame['levels']['role'],int(frame['levels']['title']))
        for frame in spec['frames']:
            slot='{BODY_'+frame['id'].replace('-','_')+'}'
            frame['text']=frame['text'].replace('{BODY}',slot);frame['probe']['prompt']=frame['probe']['prompt'].replace('{BODY}',slot)
        errors=grammar.validate(spec);assert not errors,errors
        generated=grammar.generate(spec,include_canary=False)
        actual={x['id']:x for x in generated['items']}
        # Cross-check every emitted prompt against the protocol expansion.
        bycontent={x['probe']['prompt'] for x in actual.values()}
        assert all(r['prompt'] in bycontent for r in branch_rows)
        path=B.OUT/f'grammar-{branch}.json';B.save(path.name,dict(specification=spec))
        audits.append(dict(branch=branch,grammar_sha256=grammar.spec_sha(spec),validation_errors=errors,
            emitted=len(branch_rows),causal_visibility=generated['visibility'],warnings=generated['_warnings']))
        rows+=branch_rows
        print('Prepared',branch,'rows',len(branch_rows),flush=True)
    rows.sort(key=lambda r:({'pilot':0,'discovery':1,'confirm':2}[r['split']],r['family'],r['role'],r['title']))
    assert len({r['id'] for r in rows})==len(rows)
    assert all(len({r['split'] for r in rows if r['family']==f})==1 for f in {r['family'] for r in rows})
    counts=Counter(r['split'] for r in rows)
    # Sixteen pilot families per branch are excluded from fitting/confirmation.
    assert counts=={'pilot':192,'discovery':384,'confirm':960},counts
    B.save('materials.json',dict(rows=rows,counts=dict(counts),families={s:len({r['family'] for r in rows if r['split']==s}) for s in counts},
        source_sha256=B.file_sha(__file__),selection='Original protocol family partitions retained; first 16 discovery families are pilot, remaining 48 each number/coordinate fit, first 80 confirm families in each of three eligible branches.',
        restrictions='Sequence excluded after failed behavioral calibration; v2 and option-position failures retained. Label pilot checks behavior only and its activations never enter fitting/selection. Every formal confirmation frame and lexical pool differs from fitting. Number sets may overlap; this is not an unseen-number claim.'))
    B.save('material-audit.json',dict(audits=audits,all_choices_token_aligned=True,max_tokens=max(len(r['tokens']) for r in rows),
        min_tokens=min(len(r['tokens']) for r in rows),same_content_end_token=True,
        limitation='Finite-record controls still express relationships. They test specified general applicability, not absence of a concept. Wording/length remain controlled statistically and by transfer, not perfectly matched tokens.'))


if __name__=='__main__':prepare()
