"""Already exposed pilot: native chat reply tokens and compact operation syntax."""
import b36_composition14 as B
import b36_composition14_calibrate as Engine
from b36_composition14_calibrate_expand import expanded_body

PARENT=B.OUT


def compact(row,v,expression=False):
    if row['branch']=='number':
        expr=str(v['n'])
        for op in row['sequence']:
            expr=f'({expr} + {v["k"]})' if op=='A' else f'({expr} * 2)'
        task='Evaluate '+expr+'.' if expression else 'Start with '+str(v['n'])+'. '+ ' Then '.join(f'add {v["k"]}' if op=='A' else 'multiply by 2' for op in row['sequence'])+'. What is the final number?'
    else:
        task='Start with the ordered pair ['+', '.join(v['words'][:2])+']. '+ ' Then '.join('swap the first and second items' if op=='A' else f'replace the first item with {v["words"][2]}' for op in row['sequence'])+'. What is the final ordered pair?'
    return task if row['mode']=='program' else f'A record states that input {row["input_text"]} has output {row["record_answer"]}. What is the recorded output?'


def prepare():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(B.MODEL,local_files_only=True)
    allrows=B.read(PARENT/'materials.json')['rows'];rows=[]
    values={b:{f['id']:f['metadata']['values'] for f in B.read(PARENT/f'grammar-{b}.json')['specification']['fillers']} for b in ['number','label']}
    variants=['chat_conflict','chat_focused','chat_short','plain_short','plain_expression','chat_expression']
    for row in allrows:
        if row['split']!='pilot' or row['title']!=0 or row['rank']>=3:continue
        v=values[row['branch']][row['family'].split(':')[1]]
        for variant in variants:
            if variant in ['chat_conflict','chat_focused']:
                body=expanded_body(row,v,'expanded_conflict' if variant=='chat_conflict' else 'focused')
                body+='\n'+('What is the final calculated value?' if row['mode']=='program' else 'What is the stored output?')
            else:body=compact(row,v,'expression' in variant)
            prompt=body+'\n'+''.join(f'{"ABC"[i]}. {c}\n' for i,c in enumerate(row['choices']))+'Respond with exactly one option letter (A, B, or C), without explanation.'
            chat=variant.startswith('chat')
            if chat:prompt=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            else:prompt+='\nAnswer:'
            ids=tok.encode(prompt,add_special_tokens=False);choice=[]
            for c in 'ABC':
                full=tok.encode(prompt+('' if chat else ' ')+c,add_special_tokens=False);assert full[:-1]==ids;choice.append(full[-1])
            assert len(ids)<=256
            rows.append(dict(row,id=row['id']+':'+variant,variant=variant,body=body,prompt=prompt,tokens=ids,answer_pos=len(ids)-1,choice_ids=choice))
    B.OUT=PARENT/'calibration-chat';B.configure();assert not (B.OUT/'calibration-plan.json').exists()
    B.save('calibration-plan.json',dict(rows=rows,n=len(rows),source_sha256=B.file_sha(Engine.__file__),generator_sha256=B.file_sha(__file__),
        exposure='Same72 previously exposed pilot examples only. Six interface alternatives, no confirmation opened.',variants=variants,length=256,batch=4,
        token_audit='Chat letters are unspaced A/B/C at generation start; plain Answer colon uses spaced letters. Earlier chat calibration had evaluated spaced continuations.'))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);a=p.parse_args()
    if a.stage=='prepare':prepare()
    else:B.OUT=PARENT/'calibration-chat';Engine.run()
