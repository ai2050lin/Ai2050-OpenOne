"""Prospective system-conditioned register test after user-instruction pilots.

The entire user token sequence is identical within each pair, at identical
positions. Only equal-token system register requests differ. The role is a
stated scope change, not a replacement of the failed user-level results.
"""
import phase2777_materials_v3 as V3

SEED=27772029
INSTRUCTION=V3.INSTRUCTION;JUDGE_DESCRIPTIONS=V3.JUDGE_DESCRIPTIONS
fact=V3.fact;content=V3.content;rubric=V3.rubric;references=V3.references
chinese_number=V3.chinese_number;COLLOQ=V3.COLLOQ;ACADEMIC=V3.ACADEMIC


class Builder(V3.Builder):
    def pair(self,i,frame,length,split):
        f=fact(i);body=V3.V2.source(f,frame);user=body+'\n\n'+INSTRUCTION
        def wrap(request,txt):
            return self.tok.apply_chat_template([{'role':'system','content':request},{'role':'user','content':txt}],
                tokenize=False,add_generation_prompt=True,enable_thinking=False)
        maps={}
        for reg in ['acad','colloq']:
            d={}
            for request in V3.requests(frame,reg):d.setdefault(self.count(wrap(request,user)),request)
            maps[reg]=d
        for n in sorted(set(maps['acad'])&set(maps['colloq'])):
            prefixes=self.neutral(i%2)
            for k in range(length-n-4,length-n+5):
                if k not in prefixes or prefixes[k].count('。')<4:continue
                prefix=prefixes[k];user_text=prefix+'\n'+user
                prompts=[wrap(maps[reg][n],user_text) for reg in ['acad','colloq']]
                ids=[self.tok.encode(t,add_special_tokens=False) for t in prompts]
                if not all(len(x)==length for x in ids):continue
                end=self.tok.convert_tokens_to_ids('<|im_end|>')
                j=ids[0].index(end);assert ids[1].index(end)==j
                assert ids[0][j+1:]==ids[1][j+1:],'User input/positions must be exactly identical'
                return [{'id':f'nat4-{i:04d}-F{frame}-L{length}-{reg}','family_id':f'nat4-{i:04d}',
                    'record':i,'frame':frame,'length':length,'split':split,'register':reg,'text':user_text,
                    'prompt':prompt,'token_ids':tokens,'tokens':len(tokens),'fact':f,'matching':{
                        'active_token_exact':True,'common_neutral_prefix':prefix,'common_fact_source':body,
                        'register_request':maps[reg][n],'register_role':'system','same_user_token_ids':True,
                        'system_end_token_position':j,'no_truncation':True,'no_masked_padding':True,
                        'scope':'system-conditioned natural generation, not a rescue of user-level pilot failures'}}
                    for reg,prompt,tokens in zip(['acad','colloq'],prompts,ids)]
        raise ValueError(f'No equal system request for {i}, F{frame}, L{length}')
