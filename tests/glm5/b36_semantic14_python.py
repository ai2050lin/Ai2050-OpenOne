"""Final bounded interface diagnostic on the six exposed arithmetic programs."""
import b36_semantic14 as S
from b36_semantic14_arithmetic import states

def prepare():
    S.configure();assert not (S.OUT/'python-plan.json').exists()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(S.MODEL,local_files_only=True)
    families=S.read(S.OUT/'arithmetic-families.json'); selected=set(families['diagnostic_family_ids']);rows=[]
    import numpy as np
    for g in families['families']:
      if g['id'] not in selected:continue
      for depth in [1,2,3]:
        h=states(g,depth);lines=[f'x = {g["start"]}']
        for kind,k in g['ops'][:depth]:lines.append(f'x {dict(add="+=",sub="-=",mul="*=",mod="%=")[kind]} {k}')
        code='\n'.join(lines+['print(x)'])
        rng=np.random.default_rng(int(S.digest([g['id'],depth])[:8],16))
        others=list(dict.fromkeys([x for x in reversed(h[:-1]) if x!=h[-1]]+[int(x) for x in rng.permutation(10) if x!=h[-1]]))[:3]
        choices=[h[-1]]+others;choices=[choices[i] for i in rng.permutation(4)]
        for fmt in ['direct','code0','code2']:
          rotated=list(np.roll(choices,0 if fmt=='direct' else int(fmt[-1])))
          gold=rotated.index(h[-1]); answers=[str(x) for x in rotated] if fmt=='direct' else list('ABCD')
          book='' if fmt=='direct' else '\nChoices:\n'+'\n'.join(f'{c}) {v}' for c,v in zip('ABCD',rotated))
          core='What does this Python program print?\n```python\n'+code+'\n```'
          user=core+book+'\nReply with only '+('the printed digit.' if fmt=='direct' else 'the correct option letter.')
          for frame in [40,41]:
            prompt=(user+'\nOutput:\n') if frame==40 else tok.apply_chat_template([dict(role='user',content=user)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'];choice_ids=[]
            for a in answers:
              full=tok.encode(prompt+a,add_special_tokens=False);assert full[:-1]==ids;choice_ids.append(full[-1])
            end=prompt.index(core)+len(core);at=max(i for i,(_,b) in enumerate(enc['offset_mapping']) if 0<b<=end)
            rows.append(dict(id=f'{g["id"]}:{depth}:{fmt}:{frame}',family=g['id'],prompt=prompt,tokens=ids,answer_pos=len(ids)-1,content_end=at,step_positions=[],intermediate_states=h,expected_answer=answers[gold],choice_ids=choice_ids,choices=answers,semantic_choices=[int(v) for v in rotated],gold=gold,semantic_gold=h[-1],branch=g['type'],depth=depth,format=fmt,frame=frame,core=core))
    assert len(rows)==108 and max(len(r['tokens']) for r in rows)<=S.LENGTH
    source={n:S.file_sha(S.ROOT/'tests/glm5'/n) for n in ['b36_semantic14.py','b36_semantic14_score.py','b36_semantic14_python.py','b36_semantic14_arithmetic.py','recursive_branch14.py','qwen14_mlp_atlas.py','source_binding_resources.py']}
    S.save('python-plan.json',dict(rows=rows,n=len(rows),source_hashes=source,exposure='Only six previously exposed arithmetic program families; standard ordered option letters, rotated values. No confirmation response.',selection='Last interface diagnostic this stage. Any promising interface requires a fresh, larger discovery pilot and separate unopened confirmation; failed cells never enter mechanism-positive analysis.'))
    print('Python diagnostic',len(rows),flush=True)

if __name__=='__main__':prepare()
