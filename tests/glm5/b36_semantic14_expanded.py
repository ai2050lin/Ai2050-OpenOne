"""Frozen larger discovery pilot after Python-interface diagnostics.

Independent from exposed program prefixes, but still discovery, not confirmation.
"""
import b36_semantic14 as S
from b36_semantic14_arithmetic import states
import numpy as np

def prefix(g,d):return S.digest([g['start'],g['ops'][:d]])

def prepare():
    S.configure();assert not (S.OUT/'expanded-plan.json').exists()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(S.MODEL,local_files_only=True)
    fs=S.read(S.OUT/'arithmetic-families.json');old=set(fs['diagnostic_family_ids'])
    seen={prefix(g,d) for g in fs['families'] if g['id'] in old for d in [1,2,3]}
    selected=[]
    for g in sorted(fs['families'],key=lambda x:S.digest(['fresh-python-pilot',x['id']])):
        if g['partition']!='discovery' or g['type']!='affine' or g['id'] in old:continue
        if any(prefix(g,d) in seen for d in [1,2,3]):continue
        selected.append(g);seen.update(prefix(g,d) for d in [1,2,3])
        if len(selected)==48:break
    assert len(selected)==48,len(selected)
    rows=[]
    for g in selected:
      for d in [1,2,3]:
        h=states(g,d); lines=[f'x = {g["start"]}']
        for kind,k in g['ops'][:d]:lines.append(f'x {dict(add="+=",sub="-=",mul="*=")[kind]} {k}')
        core='What does this Python program print?\n```python\n'+'\n'.join(lines+['print(x)'])+'\n```'
        rng=np.random.default_rng(int(S.digest([g['id'],d])[:8],16))
        others=list(dict.fromkeys([x for x in reversed(h[:-1]) if x!=h[-1]]+[int(x) for x in rng.permutation(10) if x!=h[-1]]))[:3]
        choices=[h[-1]]+others;choices=[choices[i] for i in rng.permutation(4)]
        for fmt in ['direct','code0','code2']:
            values=[int(x) for x in np.roll(choices,0 if fmt=='direct' else int(fmt[-1]))];gold=values.index(h[-1]);answers=[str(x) for x in values] if fmt=='direct' else list('ABCD')
            book='' if fmt=='direct' else '\nChoices:\n'+'\n'.join(f'{c}) {v}' for c,v in zip('ABCD',values))
            user=core+book+'\nReply with only '+('the printed digit.' if fmt=='direct' else 'the correct option letter.')
            prompt=tok.apply_chat_template([dict(role='user',content=user)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'];choice_ids=[]
            for a in answers:
                full=tok.encode(prompt+a,add_special_tokens=False);assert full[:-1]==ids;choice_ids.append(full[-1])
            end=prompt.index(core)+len(core);at=max(i for i,(_,b) in enumerate(enc['offset_mapping']) if 0<b<=end)
            rows.append(dict(id=f'{g["id"]}:{d}:{fmt}:51',family=g['id'],prompt=prompt,tokens=ids,answer_pos=len(ids)-1,content_end=at,intermediate_states=h,expected_answer=answers[gold],choice_ids=choice_ids,choices=answers,semantic_choices=values,gold=gold,semantic_gold=h[-1],branch='affine',depth=d,format=fmt,frame=51,core=core,prefix_sha256=prefix(g,d)))
    assert len(rows)==432 and max(len(r['tokens']) for r in rows)<=S.LENGTH
    sources={n:S.file_sha(S.ROOT/'tests/glm5'/n) for n in ['b36_semantic14.py','b36_semantic14_score.py','b36_semantic14_expanded.py','b36_semantic14_arithmetic.py','recursive_branch14.py','qwen14_mlp_atlas.py','source_binding_resources.py']}
    S.save('expanded-plan.json',dict(rows=rows,n=len(rows),families=selected,source_hashes=sources,exposure='48 new discovery programs; none of their depth1/2/3 semantic prefixes duplicates the six old diagnostic families or another selected family at that depth. Confirmation/lockbox responses unopened.',
        selection='Python chat, affine only, chosen after diagnostic frame41 showed perfect2-step performance on3 old affine families. This is a new discovery calibration, not a blinded confirmatory claim. No further interface search this stage.',
        prereg=dict(min_behavior_acc=.9,min_cell_n=48,require_all_depth_format_cells=True,metrics=['strict_forced_choice','unrestricted_next_token'],decision='At least44/48 successes on BOTH metrics in EACH of9 cells. If any required cell fails, stop joint2/3-step B36 mechanism-positive study. Do not retain only correct individual trials as a substitute. Confirmation requires a separate frozen plan.'),
        power_scope='No intervention or effect significance test. N=48 percell is a calibration screen, not the96-family confirm sample. Atp=.9, binomial SD of estimated accuracy is .0433; report raw counts and Wilson intervals. Family/code views are dependent.',
        truth='Intermediate values are metadata only; never inserted into prompt. Answer codes stay A/B/C/D in order and semantic options rotate. Same code context is not controlled by direct mode.',
        limits=['Only two of four code rotations in this pilot.','Numbers and primitive operations recur; heldout prefixes are not novel primitives.','The48 complete three-operation families are independent construction units, not432 independent samples.']))
    print('Expanded frozen pilot',len(rows),'families',len(selected),flush=True)

if __name__=='__main__':prepare()
