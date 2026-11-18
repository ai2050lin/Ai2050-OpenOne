"""Fixed demonstrations for already exposed arithmetic pilot programs only."""
import b36_semantic14 as S

def prepare():
    S.configure();assert not (S.OUT/'fewshot-plan.json').exists()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(S.MODEL,local_files_only=True);old=S.read(S.OUT/'arithmetic-interface-plan.json');rows=[]
    for r in old['rows']:
        if r['frame']!=11:continue
        direct=r['format']=='direct';demos=[]
        if direct:
            demos=[('Evaluate (((2 + 1) * 2) - 1). Output only the result digit.','5'),('Evaluate (((8 - 2) / 3) + 1). Output only the result digit.','3')]
        else:
            demos=[('Evaluate (((2 + 1) * 2) - 1).\nChoices:\nA) 3\nB) 5\nC) 4\nD) 6\nOutput only the correct option letter.','B'),
                   ('Evaluate (((8 - 2) / 3) + 1).\nChoices:\nD) 5\nA) 4\nB) 2\nC) 3\nOutput only the correct option letter.','C')]
        user=r['prompt'].split('<|im_start|>user\n',1)[1].split('<|im_end|>',1)[0]
        for variant in [20,21]:
            messages=[]
            if variant==21:messages.append(dict(role='system',content='You are a precise calculator. Respect every pair of parentheses. In a sequence of instructions, use the result of the previous instruction as the input to the next one. Return only the requested answer.'))
            for question,answer in demos:messages.extend([dict(role='user',content=question),dict(role='assistant',content=answer)])
            messages.append(dict(role='user',content=user));prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'];end=prompt.rindex(r['core'])+len(r['core'])
            at=max(i for i,(_,b) in enumerate(enc['offset_mapping']) if 0<b<=end);choice_ids=[]
            for a in r['choices']:
                full=tok.encode(prompt+a,add_special_tokens=False);assert full[:-1]==ids;choice_ids.append(full[-1])
            assert len(ids)<=S.LENGTH,(len(ids),variant,r['id'])
            rows.append(dict(r,id=r['id']+':few'+str(variant),prompt=prompt,tokens=ids,answer_pos=len(ids)-1,content_end=at,choice_ids=choice_ids,frame=variant))
    source=dict(old['source_hashes']);source['b36_semantic14_fewshot.py']=S.file_sha(__file__)
    S.save('fewshot-plan.json',dict(rows=rows,n=len(rows),source_hashes=source,exposure='Only the same six arithmetic diagnostic families; no heldout response. Two fixed demonstrations, with/without a calculator system instruction.',
        examples='Fixed examples are not target intermediate values and have different full programs; still constitute additional in-context supervision.',selection='Diagnostic only; subsequent larger pilot and unopened confirmation required. No threshold relaxation.'))
    print('Fewshot rows',len(rows),'max tokens',max(len(r['tokens']) for r in rows),flush=True)

if __name__=='__main__':prepare()
