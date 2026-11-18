"""Author-generated templates, family splits and preregistration; no model outputs."""
import argparse,hashlib,json,random,shutil,time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
import numpy as np
import state_formation_core as C
from shared.dist_protocol import grammar
from shared.dist_protocol.natural_register_runner import snapshot
from tools.power_analysis import check_phase

DISCOVERY_FRAMES=[
 'The verified register describes {a} as {x} and {b} as {y}.',
 'Current inspection: {a} carries a {x} label; {b} carries a {y} label.',
 'The latest color list reads {a}: {x}, and {b}: {y}.',
 'According to the accepted entry, the color of {a} is {x}; that of {b} is {y}.',
 'For the active record, pair {a} with {x}, and pair {b} with {y}.',
 'The two current markings are {x} for {a} and {y} for {b}.',
 'A fresh check shows a {x} marking on {a} and a {y} marking on {b}.',
 'The confirmed color assignment is {a} = {x}; {b} = {y}.']
CONFIRM_FRAMES=[
 'The approved report names {x} as the color of {a}, and {y} as the color of {b}.',
 'Read the final entries together: the {a} entry has color {x}; the {b} entry has color {y}.',
 'On the corrected chart, {a} appears under {x} while {b} appears under {y}.',
 'The active labels, in order, give {a} the color {x} and {b} the color {y}.',
 'These are the authorized pairings: {x} with {a}; {y} with {b}.',
 'The most recent audit associates {a} with the color {x}, and {b} with the color {y}.',
 'For the purpose of this lookup, {a} has been designated {x} and {b} designated {y}.',
 'The valid reference specifies a {x} color for {a}; a {y} color is specified for {b}.',
 'In the updated account, the color beside {a} is {x}, and the color beside {b} is {y}.',
 'Use the accepted color mapping: {a} maps to {x}, whereas {b} maps to {y}.',
 'The final ledger calls the color of {a} {x} and the color of {b} {y}.',
 'There are two certified associations: {a} to {x}, followed by {b} to {y}.',
 'The current tag for {a} reads {x}; the current tag for {b} reads {y}.',
 'As entered in the definitive table, {a} is marked {x} and {b} is marked {y}.',
 'The authoritative description assigns the shade {x} to {a} and the shade {y} to {b}.',
 'Only the following entries are active: {a}, with color {x}; {b}, with color {y}.']
NOUNS={'discovery':['seal','token','badge','tile','ribbon','sticker','card','disk'],
       'confirm':['vial','spool','pouch','sleeve','chip','panel','peg','capsule']}
BACKGROUND=['The office has a long wooden table beside a closed window. A clerk checked the date before leaving for lunch. Several ordinary forms remain in the lower cabinet. The lamp and the empty chair were moved yesterday. Nothing in this background note determines the requested color.',
 'The quiet workshop contains a narrow shelf and an unused desk. A visitor signed a sheet near the entrance. The clock was inspected before the afternoon meeting. A spare notebook rests beside an empty cup. These background details do not determine any of the colors in the active record.']

def build_grammar(split,frame_index):
    frames=DISCOVERY_FRAMES if split=='discovery' else CONFIRM_FRAMES
    count=32 if split=='discovery' else 48;fillers=[]
    for j in range(count):
        noun=NOUNS[split][j%8];history=(j//8)%2;rep=j//16
        # Each entity pair belongs to exactly one split and all eight variants.
        identity=f'{split[0]}{frame_index:02d}{j:03d}';number=700000+frame_index*100+j+(20000 if split=='confirm' else 0)
        a=f'{noun} X-{number}';b=f'{noun} Y-{number}'
        rng=random.Random(C.SEED+number);old_x,old_y=rng.choice([('red','blue'),('blue','red')])
        prior=(f'Obsolete entry, superseded and not to be used: {a} was {old_x}; {b} was {old_y}.' if history else
               'Archive note: an earlier file has no usable color information. It is superseded by the current verified record below.')
        fillers.append({'id':identity,'levels':{'history':history,'noun':noun},
            'slots':{'a':a,'b':b,'background':BACKGROUND[rep%2],'prior':prior,'tail':'Answer:'},
            'meta':{'frame':frame_index,'local_family':j,'noun':noun,'history':history,'replicate':rep,'number':number}})
    variants=[]
    for q in [-1,1]:
        for r in [-1,1]:
            for lang in [-1,1]:
                x,y=('red','blue') if r==1 else ('blue','red');target='{a}' if q==1 else '{b}'
                record=frames[frame_index].format(a='{a}',b='{b}',x=x,y=y)
                question=f'Using only the active record, what color is assigned to {target}?'
                language=('English','Chinese') if lang==1 else ('Chinese','English')
                instruction=f'Respond in {language[0]}, not {language[1]}. Return only the color word.'
                prefix='Use the latest active record, even if an earlier entry disagrees.\n{background}\n{prior}\n'
                text=prefix+record+'\n'+question+'\n'+instruction+'\n{tail}'
                answer=('red' if q*r==1 else 'blue') if lang==1 else ('红' if q*r==1 else '蓝')
                other=('blue' if q*r==1 else 'red') if lang==1 else ('蓝' if q*r==1 else '红')
                variants.append({'id':f'q{q+1}r{r+1}l{lang+1}','levels':{'query':q,'record':r,'language':lang},
                    'text':text,'target':'tail','cue':'Using only the active record,','probe':{'prompt':text,'answer':answer,'answers':[answer],'contrast':[other]},
                    'meta':{'record_text':record,'question_text':question,'instruction_text':instruction}})
    fractions={'discovery':float(split=='discovery'),'confirm':float(split=='confirm'),'lockbox':0.0}
    return {'schema_version':'template_grammar.v1','family_id':f'STATE-FORM-{split[0].upper()}{frame_index:02d}',
        'grammar_version':1,'module':'KN','name':f'新条件状态形成 {split}/{frame_index}',
        'factors':['query','record','language','history','noun'],'frames':variants,'fillers':fillers,'layers':'all',
        'readouts':['cue','final'],'splits':{'salt':str(C.SEED),'fractions':fractions},
        'components':{'readouts':['cue','final'],'layers':'all'},'shards':1,
        'prereg':{'min_behavior_acc':.90,'min_cell_n':2},
        'description':'Family-preserving deterministic split; entire confirm grammars have unseen frames/nouns. Dedicated local runner adds query/instruction readouts and exact module boundaries; shared grammar protocol is unchanged.'}

def material(tok,spec):
    errors=grammar.validate(spec);assert not errors,errors
    corpus=grammar.generate(spec,include_canary=False);fillers={v['id']:v for v in spec['fillers']};frames={v['id']:v for v in spec['frames']}
    by_family={}
    for item in corpus['items']:
        fl=fillers[item['filler_id']];fr=frames[item['frame_id']];slots=fl['slots'];meta=fl['meta'];cond=item['cond']
        body=item['text'].removesuffix('\nAnswer:')
        prompt=tok.apply_chat_template([{'role':'user','content':body}],tokenize=False,add_generation_prompt=True,enable_thinking=False)+'Answer:'
        tokenized=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);ids=tokenized['input_ids'];offsets=tokenized['offset_mapping']
        segments={name:grammar._fill(fr['meta'][name+'_text'],slots) for name in ['record','question','instruction']}
        def span(text):
            start=prompt.index(text);end=start+len(text)
            return [i for i,(a,b) in enumerate(offsets) if b>start and a<end and b>a]
        spans={name:span(text) for name,text in segments.items()};spans['neutral']=span(slots['background']);spans['prior']=span(slots['prior'])
        def end(text):
            limit=prompt.index(text)+len(text)
            return max(i for i,(a,b) in enumerate(offsets) if b<=limit and b>a)
        positions=[end(segments['record']),end(segments['question']),end(segments['instruction']),len(ids)-1]
        assert positions==sorted(positions) and all(0<=i<len(ids) for i in positions)
        assert len(spans['neutral'])>=max(len(spans['record']),len(spans['instruction']))
        q,r,lang=cond['query'],cond['record'],cond['language']
        row={'id':item['id'],'filler_id':item['filler_id'],'frame':meta['frame'],'noun':meta['noun'],'history':meta['history'],
            'local_family':meta['local_family'],'query':q,'record':r,'language':lang,'color':q*r,'prompt':prompt,'tokens':ids,
            'positions':positions,'spans':spans,'grammar_sha256':grammar.spec_sha(spec),'split':item['split']}
        by_family.setdefault(item['filler_id'],[]).append(row)
    groups=[]
    for fid,rows in sorted(by_family.items(),key=lambda kv:fillers[kv[0]]['meta']['local_family']):
        rows.sort(key=lambda r:(r['query'],r['record'],r['language']));assert len(rows)==8
        for subset in [[0,1,4,5],[2,3,6,7]]:
            prefix=[rows[i]['tokens'][:rows[i]['positions'][0]+1] for i in subset];assert all(v==prefix[0] for v in prefix)
        groups.append(rows)
    return groups,{'grammar_sha256':grammar.spec_sha(spec),'visibility':corpus['visibility'],'warnings':corpus['_warnings']}

def freeze():
    assert not (C.OUT/'plan.json').exists();assert C.OUT.is_dir();C.reserve();start=time.monotonic()
    assert snapshot(C.MODEL)==C.SNAP
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(C.MODEL,local_files_only=True);allrows={};grammar_records=[];hashes={}
    old=set()
    for split in ['discovery','holdout']:
        for row in C.read(C.ROOT/f'tests/glm5/result/mlp_puzzle_v1/{split}-materials.json'):
            old.add(hashlib.sha256(json.dumps(row['tokens']).encode()).hexdigest())
    unique=set();preview={}
    for split,nframes in [('discovery',8),('confirm',16)]:
        matrix=[]
        for frame in range(nframes):
            spec=build_grammar(split,frame);groups,summary=material(tok,spec);matrix.append(groups)
            grammar_records.append(spec);preview[f'{split}-{frame}']=summary
        # Frame, noun, history, replicate are crossed instead of coupled.
        groups=[matrix[frame][j] for j in range(len(matrix[0])) for frame in range(nframes)]
        for f,rows in enumerate(groups):
            for row in rows:
                row['family']=f;identity=hashlib.sha256(json.dumps(row['tokens']).encode()).hexdigest()
                assert identity not in old and identity not in unique;unique.add(identity)
        allrows[split]=[r for g in groups for r in g]
        C.save(split+'-materials.json',{'rows':allrows[split],'families':len(groups),'frames':nframes,'fresh_against_previous_run':True})
        hashes[split+'-materials.json']=C.sha(C.OUT/(split+'-materials.json'))
    assert len(allrows['discovery'])==256*8 and len(allrows['confirm'])==768*8
    C.save('grammars.json',{'schema':'state_formation_grammar_bundle.v1','grammars':grammar_records})
    C.save('grammar-preview.json',{'schema':'state_formation_preview.v1','entries':preview,'items':len(unique)})
    hashes['grammars.json']=C.sha(C.OUT/'grammars.json')
    capability_path=C.ROOT/'tests/glm5/result/phase2773_w6_v2/capability-inputs.json'
    capability=[r for r in C.read(capability_path) if r['gate_eligible']]
    random.Random(C.SEED).shuffle(capability)
    C.save('capability-materials.json',{'rows':capability[:512],'source_sha256':C.sha(capability_path),'scope':'Previously exposed historical pool, descriptive side effects only'})
    hashes['capability-materials.json']=C.sha(C.OUT/'capability-materials.json')
    variants={'en_red':[' red','red'],'en_blue':[' blue','blue'],'zh_red':['红',' 红'],'zh_blue':['蓝',' 蓝']}
    ids={k:sorted({tok.encode(s,add_special_tokens=False)[0] for s in v}) for k,v in variants.items()}
    assert len(set(t for a in ids.values() for t in a))==sum(map(len,ids.values()))
    pilot=[f for f in range(256) if (f//8)%32 in [0,1,8,9]]
    assert len(pilot)==32 and len({allrows['discovery'][f*8]['frame'] for f in pilot})==8
    power=check_phase(.05,768,p_d=.10,alpha=.05/8);assert power['pass']
    deps=['tests/glm5/'+name for name in ['state_formation_core.py','state_formation_design.py','state_formation_math.py','state_formation_runner.py','state_formation_analyze.py']]
    deps+=['tools/provenance.py','tools/power_analysis.py','shared/dist_protocol/grammar.py','shared/dist_protocol/mechanism_trace.py','shared/dist_protocol/natural_register_runner.py',
           '.venv/Lib/site-packages/transformers/models/qwen3/modeling_qwen3.py','.venv/Lib/site-packages/transformers/integrations/sdpa_attention.py']
    plan={'schema':'state_formation_plan.v1','created_at':datetime.now().astimezone().isoformat(),
        'dependencies':{name:C.sha(C.ROOT/name) for name in deps},'materials':hashes,'pilot_families':pilot,
        'choice_ids':ids,'answer_variants':variants,'design':{'families':{'discovery':256,'confirm':768},'conditions':['query','record','output_language'],
            'nuisance_crossed':['16 confirm frames','8 unseen nouns','history: none or obsolete','3 replicates per cell'],
            'readouts':C.READOUTS,'arrays':'BF16 bits: embedding + all36 raw block outputs; all36 post-attention residuals, attention writes, MLP writes; final norm separately; all2560 coordinates at four positions',
            'family_split':'24 disjoint protocol grammars:8 discovery,16 confirm; no filler family crosses split; no new lockbox claim',
            'same_token_positions':'actual full-tokenization offsets, all conditioning information visible before answer_prefix',
            'causal_sample':'one balanced condition per confirm family selected from frame/noun/history/replicate indices before outputs; 768 samples, not6144 independent cases'},
        'protocol':{'grammar':'shared dist_protocol template_grammar.v1 validated/generated','runner':'dedicated local module-boundary runner; no distributed formal task execution or changes to shared runner'},
        'behavior_gate':{'four_choice_per_frame_history_min':.90,'first_top1_overall_min':.85,'pilot_only_discovery':True,
            'failure':'stop discovery if pilot fails; confirm failing cells retained as numeric counterexamples, not semantic mechanism positives'},
        'center_test':{'primary_hidden_index':34,'secondary_hidden_indices':[33,35],
            'methods':['fixed conditional center','discovery color-conditioned center, oracle color','same correction using color decoded from current state','pooled language linear probe','vector norm'],
            'primary':'mean language accuracy gain of self-decoded center over fixed center >=.05 and frame-bootstrap lower bound>0; both select the opposite-color trained direction using decoded source-state color, never confirmation labels',
            'negative_control':'opposite-sign center correction and permuted language labels; report all, no selecting the successful direction',
            'fit':'first128 discovery for selection, then refit256; shared diagonal variance; no confirm labels/centers used',
            'claim_limit':'a more informed center uses both colors in discovery; not proof of a unique native component'},
        'formation':{'language_blocks':list(range(12,19)),'color_blocks':list(range(18,27)),
            'selection':'per factor/module, discovery validation accuracy gain from before to after using same post-module decoder; tie-break signed gain then earlier block;4 candidates',
            'normalization':'compare raw Bpre→post_attention→post_mlp; hs36raw distinct from finalnorm; signed layer writes are arithmetic contributions,not causal proof'},
        'intervention':{'candidates':4,'doses':[.5,1.],'primary_dose':1.,'calibration_families':list(range(32))+list(range(64,96)),
            'methods':['zero selected attention/MLP write at final token','same realized residual-norm random state control','identity replay','attention source exclusion vs equal-token-count neutral exclusion'],
            'effect_prediction':'freeze discovery64 stratum-specific4-score changes; compare no-change and global-mean effect; actual baseline scores allowed',
            'effect_gate':{'target_margin_loss_min':.2,'excess_over_norm_control_lower_gt':0,'effect_prediction_mse_ratio_max':.8,'prediction_baseline':'best of no change and discovery global mean','control_norm_relative_error_max':.02,'minimum_valid_control_fraction':.95},
            'source_exclusion':'all heads, final query, record span for color or instruction span for language; includes attention renormalization; not single-head mechanism',
            'side_effects':['other-factor margin/accuracy','full-vocabulary KL','historical capability512, descriptive only']},
        'baselines':{'state':['constant','additive q,r,L','four-state c,L,cL','eight-condition'],'center':['fixed','oracle-centered','decoded-centered','pooled','norm','wrong-center'],
            'inputs':'q,r,L are supplied experiment descriptors; c=qr is an authored symbolic relation, not a discovered parser'},
        'statistics':{'primary_tests':8,'alpha':.05/8,'bootstrap_replicates':4096,'cluster':'family effects and template-frame resampling both reported; require frame-bootstrap lower>0 for primary gain','multiple_layers':'only hs34 primary, other layers descriptive'},
        'power':{'expected_absolute_delta':.05,'required_n':power['n_required'],'actual_n':768,'power':power,'scope':'paired binary under discordance .10; template clusters and continuous metrics additionally reported, not guaranteed by this calculation'},
        'resources':{'physical_output':str(C.OUT.resolve()),'output_budget_GiB':32,'C_reserve_GiB':30,'gpu':'single Qwen3-4B CUDA BF16, shared GPU lease','batch':1,'max_minutes_per_stage':120,'raw_estimated_GiB':1024*8*(37+36*3+1)*4*2560*2/2**30},
        'runtime':{name:version(name) for name in ['torch','transformers','numpy','accelerate']},
        'limitations':['author-created controlled lookup,not general translation','single physical node','no training trajectory','whole module interventions may be broad','16 confirm frames are not random samples of all language','no new SAE comparison in this bounded experiment'],
        'formal_evidence_promotion':False}
    C.save('plan.json',plan);C.save('storage-audit.json',C.read(C.ROOT/'tests/glm5_temp/state-formation-storage.json'))
    C.log('frozen',1,1,start,unique_inputs=len(unique),power=power,estimated_GiB=plan['resources']['raw_estimated_GiB'])

if __name__=='__main__':freeze()
