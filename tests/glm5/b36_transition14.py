"""Frozen paired controls for extra printed updates and terminal readout."""
import re,itertools,operator
import b36_semantic14 as S
S.OUT=S.ROOT/'tests/glm5/result/b36_transition14_20261114_v1'
S.TEMP=S.ROOT/'tests/glm5_temp/b36_transition14_20261114_v1'
S.RUN='B36-TRANSITION14-20261114-v1';S.SEED=2026111402
ARMS=['no_table','table','code','full','no_steps','counted0','counted2']
HISTORY=['replay','boundary','gold','foil']
FIELDS={a:(['Steps','Value'] if a in ['no_table','table'] else ['Steps','Value','Code'] if a=='code' else ['Value','Slot','Code'] if a=='no_steps' else ['Steps','Value','Slot','Code']) for a in ARMS}
CAPS=dict(no_table=48,table=48,code=64,full=64,no_steps=32,counted0=64,counted2=64)

def evaluate(text,row,eos):
    clean=text.replace('<|im_end|>','').strip();fields={}
    for name in ['Steps','Value','Slot','Code']:
        m=re.search(r'(?m)^'+name+r':[ \t]*([^\r\n]*)',clean);fields[name]=m[1].strip() if m else None
    integer=lambda v:int(v) if v is not None and re.fullmatch(r'\d+',v) else None
    steps=[int(v) for v in fields['Steps'].split(',')] if fields['Steps'] and re.fullmatch(r'\d+(?:[ \t]*,[ \t]*\d+)*',fields['Steps']) else None
    value=integer(fields['Value']);slot=integer(fields['Slot']);code=fields['Code'] if fields['Code'] in list('ABCD') else None
    if slot not in [1,2,3,4]:slot=None
    needs=row['fields'];patterns={'Steps':r'\d+(?:[ \t]*,[ \t]*\d+)*','Value':r'\d+','Slot':r'[1-4]','Code':r'[A-D]'}
    fmt=bool(re.fullmatch(r'[ \t]*\n'.join(n+r':[ \t]*'+patterns[n] for n in needs),clean))
    m=dict(step_count_correct=steps is not None and len(steps)==row['depth'],steps_correct=steps==row['truth'],value_correct=value==row['truth'][-1],slot_correct=slot==row['slot'],code_correct=code==row['code'],format_correct=fmt,eos=bool(eos))
    metric={'Steps':'steps_correct','Value':'value_correct','Slot':'slot_correct','Code':'code_correct'}
    m['joint']=bool(fmt and eos and all(m[metric[n]] for n in needs))
    m['code_matches_slot']=None if slot is None or code is None else code==row['codes'][slot-1]
    m['slot_matches_value']=None if slot is None or value is None else row['values'][slot-1]==value
    m['value_matches_last_printed_step']=None if not steps or value is None else value==steps[-1]
    excess=None if steps is None else len(steps)-row['depth'];kind='unparseable' if steps is None else 'correct_count' if excess==0 else 'extra' if excess>0 else 'missing'
    if excess==1:
        lastop,k=row['ops'][-1];repeat={'add':operator.add,'sub':operator.sub,'mul':operator.mul}[lastop](row['truth'][-1],k)
        if steps==[row['start']]+row['truth']:kind='included_initial_assignment'
        elif steps[:row['depth']]==row['truth'] and steps[-1]==repeat:kind='appended_repeated_last_update'
        elif steps[:row['depth']]==row['truth']:kind='appended_other_value'
        else:kind='extra_with_changed_prefix'
    return dict(fields=fields,steps=steps,value=value,slot=slot,code=code,metrics=m,step_count_offset=excess,step_pattern=kind)

def prior_materials():
    root=S.ROOT/'tests/glm5/result'
    paths=[root/'b36_generated14_20261113_v1/plan.json',root/'b36_semantic14_20261113_v1/arithmetic-families.json',root/'b36_semantic14_20261113_v1/expanded-plan.json',root/'b36_map14_20261114_v1/plan.json']
    ar=S.read(paths[1]);used=set(ar['diagnostic_family_ids'])|{g['id'] for g in S.read(paths[2])['families']}
    old=list(S.read(paths[0])['families'])+[g for g in ar['families'] if g['id'] in used]+S.read(paths[3])['families']
    return old,paths

def families():
    old,paths=prior_materials();prior_recipes={S.digest(g['ops'][:d]) for g in old for d in [2,3]};prefixes={S.digest([g['start'],g['ops'][:d]]) for g in old for d in [1,2,3]}
    ops=[(k,v) for k in ['add','sub'] for v in range(1,10)]+[('mul',2),('mul',3)];selected={2:[],3:[]};used_recipes=set()
    for depth in [2,3]:
        candidates=sorted(itertools.product(ops,repeat=depth),key=lambda p:S.digest([S.SEED,p]))
        for program in candidates:
            recipe=S.digest(program)
            if recipe in prior_recipes or recipe in used_recipes or len({op for op,k in program})<2:continue
            start=10+int(S.digest([S.SEED,program])[:8],16)%20;first=S.digest([start,program[:1]])
            if first in prefixes:continue
            x=start;truth=[]
            for op,k in program:x={'add':operator.add,'sub':operator.sub,'mul':operator.mul}[op](x,k);truth.append(x)
            if min(truth)<0 or max(truth)>99:continue
            selected[depth].append(dict(id='t'+S.digest([start,program])[:15],depth=depth,start=start,ops=program,truth=truth))
            used_recipes.add(recipe);prefixes.add(first)
            if len(selected[depth])==48:break
    assert [len(selected[d]) for d in [2,3]]==[48,48],{d:len(q) for d,q in selected.items()}
    out=[]
    for i in range(48):
        for depth in [2,3]:
            g=selected[depth][i];v=g['truth'][-1];slot=i%4;near=v+1 if v<99 else v-1
            pool=[near]+g['truth'][:-1]+[g['start']]+list(range(100));others=list(dict.fromkeys(w for w in pool if w!=v))[:3];values=others[:];values.insert(slot,v)
            # Match the number of digit tokens for the gold/foil history intervention.
            low=0 if v<10 else 10;high=9 if v<10 else 99;foil=v+1 if v<high else low
            assert foil!=v and len(str(foil))==len(str(v))
            out.append(dict(g,slot=slot+1,values=values,rotation=(i//4)%4,foil=foil))
    return out,paths

def prompt_for(g,arm,tok):
    codes=['ABCD'[(j+g['rotation']+(2 if arm=='counted2' else 0))%4] for j in range(4)]
    lines=[f'x = {g["start"]}']+[f'x {dict(add="+=",sub="-=",mul="*=")[op]} {k}' for op,k in g['ops']]
    user='Execute these Python assignments in order:\n```python\n'+'\n'.join(lines)+'\n```'
    if arm!='no_table':user+='\nOptions:\n'+'\n'.join(f'Row {j+1}: value {v}, code {c}' for j,(v,c) in enumerate(zip(g['values'],codes)))
    desc={'Steps':'values after each update separated by commas (exclude the initial assignment)','Value':'the final numeric value','Slot':'the row number whose value equals that final value','Code':"the code of the row whose value equals that final value"}
    need=FIELDS[arm]
    user+='\nWrite exactly '+{2:'two',3:'three',4:'four'}[len(need)]+' lines: '+'; '.join('"'+n+': " then '+desc[n] for n in need)+'.'
    if arm.startswith('counted'):user+=f' There are exactly {g["depth"]} update statements. Steps must contain exactly {g["depth"]} values, one per update statement.'
    user+=' Stop immediately after the requested lines. No explanation or code fences.'
    prompt=tok.apply_chat_template([dict(role='user',content=user)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    return dict(g,id=g['id']+':'+arm,family=g['id'],arm=arm,fields=need,codes=codes,code=codes[g['slot']-1],prompt=prompt,tokens=tok.encode(prompt,add_special_tokens=False))

def history_prefix(text,ids,depth,tok):
    m=re.match(r'Steps:[ \t]*([0-9]+(?:[ \t]*,[ \t]*[0-9]+)*)',text)
    if not m:return None
    numbers=list(re.finditer(r'\d+',m[1]))
    if len(numbers)<depth:return None
    end=m.start(1)+numbers[depth-1].end();prefix=text[:end]
    for k in range(1,len(ids)+1):
        decoded=tok.decode(ids[:k],skip_special_tokens=False)
        if decoded==prefix:return dict(text=prefix,tokens=ids[:k],cut=k,values=[int(n[0]) for n in numbers[:depth]])
        if len(decoded)>len(prefix):break
    return None

def replace_steps(prefix,values):
    spans=list(re.finditer(r'\d+',prefix));assert len(spans)==len(values)
    for m,v in reversed(list(zip(spans,values))):prefix=prefix[:m.start()]+str(v)+prefix[m.end():]
    return prefix

def prepare():
    S.configure(500*2**20);assert not (S.OUT/'plan.json').exists()
    from transformers import AutoTokenizer
    from tools.power_analysis import check_phase
    tok=AutoTokenizer.from_pretrained(S.MODEL,local_files_only=True);ff,prior=families();rows=[prompt_for(g,a,tok) for a in ARMS for g in ff]
    tok.padding_side='left'
    for arm in ARMS:
        rr=[r for r in rows if r['arm']==arm]
        aa=tok([r['prompt'] for r in rr],add_special_tokens=False,padding=True)
        bb=tok.pad(dict(input_ids=[r['tokens'] for r in rr]),padding=True)
        assert aa['input_ids']==bb['input_ids'] and aa['attention_mask']==bb['attention_mask']
    power=check_phase(.2,96,p_d=.3,alpha=.05/6);assert power['pass']
    sources={n:S.file_sha(S.ROOT/'tests/glm5'/n) for n in ['b36_transition14.py','b36_transition14_run.py','b36_semantic14.py','recursive_branch14.py','qwen14_mlp_atlas.py','source_binding_resources.py']}
    S.save('plan.json',dict(families=ff,rows=rows,n_families=96,n_natural_generations=672,history_max_generations=384,source_hashes=sources,prior_files={str(p.relative_to(S.ROOT)):S.file_sha(p) for p in prior},model_files=S.read(S.ROOT/'tests/glm5/result/b36_map14_20261114_v1/plan.json')['model_files'],generation=dict(batch=96,dtype='originalBF16',do_sample=False,thinking=False,cache=True,max_tokens=CAPS,history_caps=dict(replay=64,boundary=32,gold=32,foil=32)),prereg=dict(primary=[dict(left='no_table',right='table',metric='step_count_correct'),dict(left='table',right='code',metric='step_count_correct'),dict(left='code',right='full',metric='step_count_correct'),dict(left='full',right='counted0',metric='step_count_correct'),dict(left='replay',right='boundary',metric='value_correct'),dict(left='gold',right='foil',metric='value_correct')],correction='Exact paired McNemar; Holm over all6 planned tests, including unperformed tests conservatively as p=1.',power=power,history_admission='All96 source prefixes extractable at a real token boundary; replay tail must equal original free-generation tail for all96 to admit replay-vs-boundary inference. Otherwise histories are descriptive only and missing families are explicit.',mechanism_gate='Only predeclared counted0/counted2 target interface: each depth×rotation cell at least44/48 joint successes. All96 families retained, no correct-subset selection. This is a new explicitly scoped interface and cannot retroactively pass the previous raw-interface gate. Passing requires a separate frozen native-path protocol before new interventions.',mechanism_arms=['counted0','counted2'],min_cell_n=48,min_joint_correct=44),history_protocol=dict(source='full',cut_location='After the d-th printed integer, before observing a further delimiter as an input. Cut location is extracted retrospectively from the whole generated text; it is a conditional continuation intervention, not a prospective detector or future prediction.',replay='Same prompt tokens plus exact original generated prefix token IDs; natural continuation.',boundary='Same generated prefix plus externally injected newline and Value label; numeric content unchanged.',gold='Same prefix formatting but all d printed values replaced with oracle truth, followed by injected Value label.',foil='Gold prefix with only last value replaced by a different equal-digit-length integer, followed by same Value label.',evidence='Gold/foil and injected field boundaries are supplied-text interventions, never counted as autonomous computation or native-neuron mechanism proof.'),design_scope='Paired main-effect controls, not a complete factorial. no_table vs table differs only option text. table/code/full add requested output fields. full/no_steps removes Steps requirement and changes history. counted adds explicit update-count instruction (wording/length not orthogonal). Depth2/3 use different families; no paired causal depth claim.',novelty='96 unique visible recipes; both actual visible operations and start+first-operation prefixes disjoint from prior families and each other. Initial10..29, intermediate0..99; same primitive range as Phase2932. IDs hash actual visible program and initial value, with no unused third operation.',budget=dict(new_peak_bytes=500*2**20,reserve_bytes=20*2**30,expected_minutes=100,hard_runtime_minutes=180,estimated_gpu_peak_gib=10,retention='Frozen materials, every raw generated token, derived history edits, logs, audit, source archive; no native fields before gate.'),source='Author-created arithmetic tasks; CC0.'))
    print('Frozen96 new visible recipes;672 natural and up to384 history continuations. Max inputtokens',max(len(r['tokens']) for r in rows),flush=True)

if __name__=='__main__':prepare()
