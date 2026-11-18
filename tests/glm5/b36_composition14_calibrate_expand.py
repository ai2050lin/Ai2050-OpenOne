"""Second exposed-pilot diagnostic: explicit operations versus symbol lookup."""
import b36_composition14 as B
import b36_composition14_calibrate as Engine

PARENT=B.OUT


def expanded_body(row,values,variant):
    cover=row['body'].split('\n')[0];branch=row['branch'];steps=[]
    for j,op in enumerate(row['sequence'],1):
        if branch=='number':text=f'Add {values["k"]} to the current number.' if op=='A' else 'Multiply the current number by 2.'
        else:text='Exchange the first and second items of the current pair.' if op=='A' else f'Replace only the first item with {values["words"][2]}; keep the second item unchanged.'
        steps.append(f'Step {j}: '+text)
    procedure='Calculation starts from '+row['input_text']+'.\n'+'\n'.join(steps)
    archived=row['input_text']
    if variant=='expanded_unrelated' and row['mode']=='program':
        archived=str(values['n']+137) if branch=='number' else '['+', '.join(values['words'][5:7])+']'
    archive=f'Archive record: input {archived} -> stored output {row["record_answer"]}.'
    if variant=='focused':
        material=procedure if row['mode']=='program' else archive
    else:material='The calculation and archive are independent.\n'+procedure+'\n'+archive
    task='Perform the calculation; ignore the archive.' if row['mode']=='program' else 'Read the archive; ignore the calculation.'
    return cover+'\n'+material+'\nTask: '+task+'\nEnd of passage.'


def prepare():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(B.MODEL,local_files_only=True)
    allrows=B.read(PARENT/'materials.json')['rows'];rows=[]
    values={branch:{f['id']:f['metadata']['values'] for f in B.read(PARENT/f'grammar-{branch}.json')['specification']['fillers']} for branch in ['number','label']}
    for row in allrows:
        if row['split']!='pilot' or row['title']!=0 or row['rank']>=3:continue
        v=values[row['branch']][row['family'].split(':')[1]]
        for variant in ['expanded_conflict','expanded_unrelated','focused']:
            body=expanded_body(row,v,variant)
            task='What is the final calculated value?' if row['mode']=='program' else 'What is the stored output for '+row['input_text']+'?'
            prompt=body+'\n'+task+'\n'+''.join(f'{"ABC"[i]}. {c}\n' for i,c in enumerate(row['choices']))+'Reply with only the option symbol.\nAnswer:'
            ids=tok.encode(prompt,add_special_tokens=False);choice=[]
            for c in 'ABC':
                full=tok.encode(prompt+' '+c,add_special_tokens=False);assert full[:-1]==ids;choice.append(full[-1])
            assert len(ids)<=256
            rows.append(dict(row,id=row['id']+':'+variant,variant=variant,body=body,prompt=prompt,tokens=ids,answer_pos=len(ids)-1,choice_ids=choice))
    B.OUT=PARENT/'calibration-expand';B.configure()
    assert not (B.OUT/'calibration-plan.json').exists()
    B.save('calibration-plan.json',dict(rows=rows,n=len(rows),source_sha256=B.file_sha(Engine.__file__),generator_sha256=B.file_sha(__file__),
        exposure='Same72 previously exposed pilot examples; no new confirmation. Native score engine reused unchanged, explicit nested output destination.',variants=['expanded_conflict','expanded_unrelated','focused'],length=256,batch=4))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);a=p.parse_args()
    if a.stage=='prepare':prepare()
    else:B.OUT=PARENT/'calibration-expand';Engine.run()
