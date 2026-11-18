"""Final pilot interface comparison, explicit arithmetic and string composition."""
import b36_composition14 as B
import b36_composition14_calibrate as Engine
PARENT=B.OUT


def evaluate(branch,v,seq):
    x=v['n'] if branch=='number' else v['text'];hist=[x]
    for op in seq:
        if branch=='number':x=x+v['k'] if op=='A' else 2*x
        else:x=x.upper() if op=='A' else x+v['suffix']
        hist.append(x)
    return hist


def procedure(branch,v,seq,frame=0):
    if branch=='number':
        expr=str(v['n'])
        for op in seq:expr=f'({expr} + {v["k"]})' if op=='A' else f'({expr} * 2)'
        return ['Evaluate '+expr+'.','Calculate the value of '+expr+'.','Find the numerical result of '+expr+'.','Compute this expression: '+expr+'.'][frame]
    start=['Start with the text','Begin with the string','The initial text is','Use the starting string'][frame]+' "'+v['text']+'".'
    a=['convert every letter to uppercase','make all letters uppercase','change all letters to capitals','uppercase the entire current string'][frame]
    b=[f'append the lowercase letter "{v["suffix"]}"',f'add lowercase "{v["suffix"]}" at the end',f'attach the lowercase letter "{v["suffix"]}" to the end',f'append "{v["suffix"]}" in lowercase to the current string'][frame]
    return start+' In order, '+', then '.join(a if op=='A' else b for op in seq)+'.'


def prepare():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(B.MODEL,local_files_only=True);rows=[]
    old=B.read(PARENT/'materials.json')['rows']
    values={b:{f['id']:f['metadata']['values'] for f in B.read(PARENT/f'grammar-{b}.json')['specification']['fillers']} for b in ['number','label']}
    for r in old:
        if r['split']!='pilot' or r['title']!=0 or r['rank']>=3:continue
        v=dict(values[r['branch']][r['family'].split(':')[1]])
        v.update(text=['ab','cd','ef'][r['rank']],suffix=['x','y','z'][r['rank']])
        answer=str(evaluate(r['branch'],v,r['sequence'])[-1]);record=str(v['record']) if r['branch']=='number' else ['mn','pq','rs'][r['rank']]
        other={'A':'B','B':'A','AB':'BA','BA':'AB','ABA':'BAB','BAB':'ABA'}[r['sequence']]
        foil=str(evaluate(r['branch'],v,other)[-1]);assert len({answer,record,foil})==3
        order=(r['rank']+r['depth'])%3;choices=['']*3;choices[order]=answer;choices[(order+1)%3]=record;choices[(order+2)%3]=foil
        for variant in ['plain_dual','plain_focused','chat_dual','chat_focused']:
            task=procedure(r['branch'],v,r['sequence'])
            archive=f'The independent record stores "{record}" for this input.'
            body=task+'\n'+archive if variant.endswith('dual') else task if r['mode']=='program' else archive
            body+='\n'+('Return the result of performing the calculation or text operations; ignore the record.' if r['mode']=='program' else 'Return the stored value from the record; ignore the calculation or text operations.')
            prompt=body+'\n'+''.join(f'{"ABC"[i]}. {c}\n' for i,c in enumerate(choices))+'Reply with only the option letter.'
            chat=variant.startswith('chat')
            if chat:prompt=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            else:prompt+='\nAnswer:'
            ids=tok.encode(prompt,add_special_tokens=False);choice=[]
            for c in 'ABC':
                full=tok.encode(prompt+('' if chat else ' ')+c,add_special_tokens=False);assert full[:-1]==ids;choice.append(full[-1])
            assert len(ids)<=256
            rows.append(dict(r,id=r['id']+':'+variant,variant=variant,prompt=prompt,tokens=ids,answer_pos=len(ids)-1,choice_ids=choice,choices=choices,gold=order if r['mode']=='program' else (order+1)%3))
    B.OUT=PARENT/'calibration-simple';B.configure();assert not (B.OUT/'calibration-plan.json').exists()
    B.save('calibration-plan.json',dict(rows=rows,n=len(rows),source_sha256=B.file_sha(Engine.__file__),generator_sha256=B.file_sha(__file__),
        exposure='Same six exposed pilot families. Number values retained; label task changed to uppercase/append. No confirm family opened.',length=256,batch=4,
        alternatives='Uppercase and append are a different, simpler composition domain. Success cannot rehabilitate failed pair-reordering task.'))


if __name__=='__main__':
    import argparse,time
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);a=p.parse_args()
    if a.stage=='prepare':prepare()
    else:
        B.OUT=PARENT/'calibration-simple';B.configure();start=time.monotonic()
        while True:
            try:Engine.run();break
            except RuntimeError as e:
                if str(e)!='GPU is leased by another research/live process' or time.monotonic()-start>3600:raise
                B.log('waiting_simple_calibration',0,288,start);time.sleep(15)
