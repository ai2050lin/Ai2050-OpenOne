"""Independent prospective decomposition: compute -> row match -> code."""
import sys,time,re,itertools,operator,math,random,gc
import b36_semantic14 as S
S.OUT=S.ROOT/'tests/glm5/result/b36_map14_20261114_v1'
S.TEMP=S.ROOT/'tests/glm5_temp/b36_map14_20261114_v1'
S.RUN='B36-MAP14-20261114-v1';S.SEED=2026111401
ARMS=['compute','integrated0','integrated2','match0','match2','encode0','encode2']

def evaluate(text,row,eos):
    clean=text.replace('<|im_end|>','').strip();fields={}
    for name in ['Steps','Value','Slot','Code']:
        m=re.search(r'(?m)^'+name+r':\s*([^\r\n]+)',clean)
        fields[name]=m[1].strip() if m else None
    ints=lambda value:[int(v) for v in re.findall(r'\d+',value)] if value is not None and re.fullmatch(r'[0-9 ,]+',value) else None
    steps=ints(fields['Steps']);value=int(fields['Value']) if fields['Value'] and re.fullmatch(r'\d+',fields['Value']) else None
    slot=int(fields['Slot']) if fields['Slot'] and re.fullmatch(r'[1-4]',fields['Slot']) else None
    code=fields['Code'] if fields['Code'] in list('ABCD') else None
    arm=row['arm'];need=['Steps','Value'] if arm=='compute' else ['Steps','Value','Slot','Code'] if arm.startswith('integrated') else ['Slot','Code'] if arm.startswith('match') else ['Code']
    patterns={'Steps':r'\d+(?:\s*,\s*\d+)*','Value':r'\d+','Slot':r'[1-4]','Code':r'[A-D]'}
    fmt=bool(re.fullmatch(r'\s*\n'.join(n+':\s*'+patterns[n] for n in need),clean))
    metrics=dict(steps_parseable=steps is not None and len(steps)==row['depth'],steps_correct=steps==row['truth'],value_correct=value==row['truth'][-1],slot_correct=slot==row['slot'],code_correct=code==row['code'],format_correct=fmt,eos=eos)
    relevant={'Steps':'steps_correct','Value':'value_correct','Slot':'slot_correct','Code':'code_correct'}
    metrics['joint']=bool(fmt and eos and all(metrics[relevant[n]] for n in need))
    metrics['code_consistent_with_generated_slot']=None if slot is None or code is None else code==row['codes'][slot-1]
    metrics['slot_matches_generated_value']=None if slot is None or value is None else row['values'][slot-1]==value
    return dict(fields=fields,steps=steps,value=value,slot=slot,code=code,metrics=metrics)

def prepare():
    S.configure(300*2**20);assert not (S.OUT/'plan.json').exists()
    from transformers import AutoTokenizer
    from tools.power_analysis import check_phase
    tok=AutoTokenizer.from_pretrained(S.MODEL,local_files_only=True)
    old=S.ROOT/'tests/glm5/result';prior=S.read(old/'b36_generated14_20261113_v1/plan.json')['families'];oldarith=S.read(old/'b36_semantic14_20261113_v1/arithmetic-families.json');ids=set(oldarith['diagnostic_family_ids'])|{g['id'] for g in S.read(old/'b36_semantic14_20261113_v1/expanded-plan.json')['families']}
    prior+= [g for g in oldarith['families'] if g['id'] in ids]
    opkey=lambda ops:S.digest(ops)
    prior_ops={opkey(g['ops']) for g in prior};prefixes={S.digest([g['start'],g['ops'][:d]]) for g in prior for d in [1,2,3]}
    ops=[(k,v) for k in ['add','sub'] for v in range(1,10)]+[('mul',2),('mul',3)]
    candidates=sorted(itertools.product(ops,repeat=3),key=lambda x:S.digest([S.SEED,x]));families=[]
    for program in candidates:
        if len({o[0] for o in program})<2 or opkey(program) in prior_ops:continue
        start=10+int(opkey(program)[:8],16)%20;key=S.digest([start,program[:1]])
        if key in prefixes:continue
        x=start;truth=[]
        for kind,k in program:x={'add':operator.add,'sub':operator.sub,'mul':operator.mul}[kind](x,k);truth.append(x)
        if min(truth)<0 or max(truth)>99:continue
        rank=len(families);depth=2+rank%2;slot=(rank//2)%4
        target=truth[depth-1];near=target+(1 if (rank//8)%2==0 else -1)
        if near<0:near=target+1
        pool=[near]+[v for v in truth[:depth-1]+[start]+list(range(100)) if v!=target and v!=near]
        others=list(dict.fromkeys(pool))[:3];values=others[:];values.insert(slot,target);assert len(set(values))==4
        g=dict(id='m'+opkey(program)[:15],ops=program,start=start,truth=truth[:depth],depth=depth,slot=slot+1,values=values,code_rotation=(rank//8)%4)
        families.append(g);prefixes.add(key)
        if len(families)==96:break
    assert len(families)==96
    rows=[]
    for arm in ARMS:
      for g in families:
        rotation=(g['code_rotation']+(2 if arm.endswith('2') else 0))%4;codes=['ABCD'[(j+rotation)%4] for j in range(4)]
        code=codes[g['slot']-1];lines=[f'x = {g["start"]}']+[f'x {dict(add="+=",sub="-=",mul="*=")[kind]} {k}' for kind,k in g['ops'][:g['depth']]]
        table='\n'.join(f'Row {j+1}: value {v}, code {c}' for j,(v,c) in enumerate(zip(g['values'],codes)))
        if arm=='compute':user='Execute these Python assignments in order:\n```python\n'+'\n'.join(lines)+'\n```\nWrite exactly two lines: "Steps: " then the values after each update, separated by commas (exclude the initial assignment); "Value: " then the final numeric value.'
        elif arm.startswith('integrated'):user='Execute these Python assignments in order:\n```python\n'+'\n'.join(lines)+'\n```\nOptions:\n'+table+'\nWrite exactly four lines: "Steps: " then values after each update separated by commas (exclude the initial assignment); "Value: " then the final numeric value; "Slot: " then the row number whose value equals that final value; "Code: " then that row\'s code.'
        elif arm.startswith('match'):user=f'Target value: {g["truth"][-1]}\nOptions:\n'+table+'\nFind the row whose value equals the target value. Write exactly two lines: "Slot: " then that row number; "Code: " then that row\'s code.'
        else:user=f'Selected row: {g["slot"]}\nOptions:\n'+table+'\nCopy the code of the selected row. Write exactly one line: "Code: " then that code.'
        user+=' Stop immediately after the requested lines. No explanation or code fences.'
        prompt=tok.apply_chat_template([dict(role='user',content=user)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        rows.append(dict(g,id=g['id']+':'+arm,family=g['id'],arm=arm,codes=codes,code=code,prompt=prompt,tokens=tok.encode(prompt,add_special_tokens=False)))
    power=check_phase(.16,96,p_d=.2,alpha=.025);assert power['pass']
    sources={n:S.file_sha(S.ROOT/'tests/glm5'/n) for n in ['b36_map14.py','b36_map14_run.py','b36_semantic14.py','recursive_branch14.py','qwen14_mlp_atlas.py','source_binding_resources.py']}
    S.save('plan.json',dict(rows=rows,families=families,source_hashes=sources,model_files=S.read(old/'b36_semantic14_20261113_v1/model-identity.json')['files'],n_families=96,n_rows=len(rows),generation=dict(batch=96,dtype='originalBF16',max_tokens=dict(compute=48,integrated=64,match=24,encode=16),do_sample=False,thinking=False,cache=True),prereg=dict(min_cell_n=48,min_joint_accuracy=.9,required_cells='All14 arm×depth cells; at least44/48 each. No correct-subset filtering. Passing gate permits subsequent frozen native-path plan, not automatic mechanism claim.',primary_comparisons=['integrated0 vs match0 final-code accuracy','match0 vs encode0 final-code accuracy'],power=power,alpha_family=.05,correction='Holm over two paired exact McNemar tests; all other tests/descriptions exploratory'),hypotheses=['If supplied true result removes errors, loss is upstream of or depends on computed-result history.','If supplied true row removes errors remaining with a supplied result, row matching is implicated behaviorally.','Correct generated value with wrong slot vs correct slot with wrong code separates visible error stages, not hidden causal modules.'],controls='Same programs/table/row order across paired arms; true values/rows in bypass controls explicitly supplied, not model-computed. Code rotation keeps row and value fixed. No provided intermediate truth in integrated or compute arms.',independence='96 new whole operation programs, exclude prior108 generated-plan candidates and54 arithmetic families; semantic first prefixes disjoint from those prior families and each other. All views stay in same family. Depth assigned48/48; gold rows balanced12 perdepth. New primitive magnitudes6..9 expand earlier scope; no cross-stage causal comparison.',source='Author-created arithmetic tasks; CC0.',budget=dict(new_peak_bytes=300*2**20,reserve_bytes=20*2**30,estimated_peak_gpu_gib=12,estimated_minutes=65),scientific_limit='Independent prospective behavioral confirmation of failure-stage hypotheses, not an intervention experiment. Small differences under16 percentage points may be underpowered. All arms run even if an early cell fails.'))
    print('Frozen96 families672 prompts',max(len(r['tokens']) for r in rows),flush=True)

if __name__=='__main__':prepare()
