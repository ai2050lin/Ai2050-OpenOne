"""New author-generated validity × position × conflict design, family split."""
import hashlib,shutil
from datetime import datetime
from pathlib import Path
import record_selection_core as R
from record_selection_core import C
from shared.dist_protocol import grammar
from tools.power_analysis import check_phase

DISC=[
('Entry {i} [{status}]: {a} has color {x}; {b} has color {y}.','Look up the color of {target} using the approved entry.'),
('File {i} is {status}. Its color assignments are {a}: {x}; {b}: {y}.','Which color does the approved file assign to {target}?'),
('Record {i}, status {status}, gives {a} = {x} and {b} = {y}.','Report the approved color attached to {target}.'),
('{status} entry {i}: the color for {a} is {x}, and the color for {b} is {y}.','Find {target} in the entry that is approved and state its color.'),
('Entry {i} has status {status}; {a} is listed as {x}, while {b} is listed as {y}.','According to the approved entry, what is the color of {target}?'),
('The {status} file numbered {i} pairs {a} with {x} and {b} with {y}.','Retrieve the approved color assignment for {target}.'),
('Status: {status}. Entry number: {i}. Colors: {a}—{x}; {b}—{y}.','Name the color associated with {target} in the approved entry.'),
('Under entry {i} ({status}), {a} carries a {x} label and {b} a {y} label.','Give the color of {target} from the approved entry only.')]
CONF=[
('Registration {i} ({status}) specifies {x} for {a} and {y} for {b}.','For {target}, which color is stated in the approved registration?'),
('{status} report {i} records the color of {a} as {x}; it records the color of {b} as {y}.','Read the approved report and return the color assigned to {target}.'),
('In sheet {i}, marked {status}, the assignments read {a}: {x}, {b}: {y}.','Use the approved sheet to identify the color of {target}.'),
('The status of list {i} is {status}. It lists a {x} {a} and a {y} {b}.','What color is {target} in the approved list?'),
('List {i} — {status} — assigns the label {x} to {a}, and the label {y} to {b}.','Consult the approved assignment for {target} and answer with its color.'),
('Report {i} is marked {status}: {a} has a {x} marking; {b} has a {y} marking.','Tell me the approved marking color for {target}.'),
('Status {status} belongs to entry {i}, where {a} is {x} and {b} is {y}.','Resolve the color of {target} from the approved entry.'),
('A {status} register, number {i}, identifies {a} as {x} and identifies {b} as {y}.','Find the approved register and extract the color of {target}.'),
('Reference {i} has status {status}. The color beside {a} is {x}, and beside {b} is {y}.','What color is shown beside {target} in the approved reference?'),
('Record {i} is {status}, with a {x} label for {a} and a {y} label for {b}.','Choose the approved record, then state the label color of {target}.'),
('Document {i} carries status {status}: it pairs {x} with {a} and {y} with {b}.','From the approved document, retrieve the color paired with {target}.'),
('The {status} document numbered {i} says: {a} is colored {x}; {b} is colored {y}.','Locate {target} in the approved document. What is its color?'),
('Index {i} ({status}) associates {a} with color {x} and {b} with color {y}.','Using the approved index, provide the color linked to {target}.'),
('Number {i}, status {status}: color {x} belongs to {a}; color {y} belongs to {b}.','Determine which color belongs to {target} under the approved assignment.'),
('The status is {status} for table {i}; {a} appears with {x}, and {b} appears with {y}.','Answer from the approved table: which color appears with {target}?'),
('Log {i} is designated {status} and contains {a} = {x}; {b} = {y}.','Read only the approved log to answer the color query for {target}.')]
NOUNS={'discovery':['urn','folder','packet','crystal'],'confirm':['cylinder','crate','sheet','pendant','swatch','drawer','beacon','casket']}

def spec(split,frame):
    frames=DISC if split=='discovery' else CONF;nouns=NOUNS[split];fillers=[]
    for j in range(len(nouns)*4):
        q=1 if (j//len(nouns))%2 else -1;lang=1 if (j//(2*len(nouns)))%2 else -1
        number=810000+(10000 if split=='confirm' else 0)+frame*100+j
        a=f'{nouns[j%len(nouns)]} A{number}';b=f'{nouns[j%len(nouns)]} B{number}'
        valid,old=('ACTIVE','VOID') if split=='discovery' else ('CURRENT','RETIRED')
        intro=f'Two records describe the same two objects. A record marked {valid} is approved; a record marked {old} is obsolete. Use the approved record regardless of its position. The entry numbers only indicate display order.'
        target=a if q==1 else b
        instruction=(f'Write only the color name in {"English" if lang==1 else "Chinese"}.' if split=='discovery' else
                     f'Your answer must contain just the color, expressed in {"English" if lang==1 else "Chinese"}.')
        fillers.append({'id':f'{split[0]}{frame:02d}{j:02d}','slots':{'a':a,'b':b,'target':target,'intro':intro,'instruction':instruction,'tail':'Answer:'},
            'levels':{'query':q,'language':lang},'meta':{'j':j,'q':q,'language':lang,'valid':valid,'old':old}})
    variants=[];entry,question=frames[frame]
    for pos in [-1,1]:
        for mapping in [-1,1]:
            for conflict in [0,1]:
                valid,old=('ACTIVE','VOID') if split=='discovery' else ('CURRENT','RETIRED');oldmap=mapping*(-1 if conflict else 1)
                lines=[]
                for i in [1,2]:
                    active=(i==1)==(pos==-1);m=mapping if active else oldmap
                    lines.append(entry.format(i=i,status=valid if active else old,a='{a}',b='{b}',x='red' if m==1 else 'blue',y='blue' if m==1 else 'red'))
                query=question.format(target='{target}')
                text='{intro}\n'+lines[0]+'\n'+lines[1]+'\n'+query+'\n{instruction}\n{tail}'
                variants.append({'id':f'p{pos+1}r{mapping+1}c{conflict}','levels':{'valid_position':pos,'mapping':mapping,'conflict':conflict},
                    'text':text,'target':'tail','cue':query.split('{')[0].strip(),'meta':{'record1':lines[0],'record2':lines[1],'query':query}})
    return {'schema_version':'template_grammar.v1','family_id':f'VALIDITY-{split[0]}{frame:02d}','grammar_version':1,'module':'KN',
        'name':f'Record validity {split} {frame}','description':'Authored controlled materials; positions and conflicts crossed within families',
        'factors':['valid_position','mapping','conflict','query','language'],'frames':variants,'fillers':fillers,'layers':'all','readouts':['cue','final'],
        'splits':{'salt':str(C.SEED),'fractions':{'discovery':float(split=='discovery'),'confirm':float(split=='confirm'),'lockbox':0.0}},'shards':1,
        'prereg':{'min_behavior_acc':.9,'min_cell_n':4}}

def material(tok,g):
    assert not grammar.validate(g),grammar.validate(g);corpus=grammar.generate(g,include_canary=False)
    fl={v['id']:v for v in g['fillers']};fr={v['id']:v for v in g['frames']};groups={}
    for item in corpus['items']:
        f=fl[item['filler_id']];frame=fr[item['frame_id']];slots=f['slots'];body=item['text'].removesuffix('\nAnswer:')
        prompt=tok.apply_chat_template([{'role':'user','content':body}],tokenize=False,add_generation_prompt=True,enable_thinking=False)+'Answer:'
        tokens=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);offsets=tokens['offset_mapping']
        def span(text):
            start=prompt.index(text);end=start+len(text)
            return [i for i,(a,b) in enumerate(offsets) if b>start and a<end and b>a]
        r1=grammar._fill(frame['meta']['record1'],slots);r2=grammar._fill(frame['meta']['record2'],slots);query=grammar._fill(frame['meta']['query'],slots)
        spans={'record1':span(r1),'record2':span(r2),'query_instruction':span(query+'\n'+slots['instruction'])}
        positions=[span(slots['intro'])[-1],spans['record1'][-1],spans['record2'][-1],len(tokens['input_ids'])-1]
        c=item['cond'];q=f['meta']['q'];row={'id':item['id'],'family_id':item['filler_id'],'frame':int(g['family_id'][-2:]),'local_family':f['meta']['j'],
            'query':q,'language':f['meta']['language'],'valid_position':c['valid_position'],'mapping':c['mapping'],'conflict':c['conflict'],'color':q*c['mapping'],
            'prompt':prompt,'tokens':tokens['input_ids'],'positions':positions,'spans':spans,'split':item['split'],'grammar_sha256':grammar.spec_sha(g)}
        groups.setdefault(f['meta']['j'],[]).append(row)
    return [sorted(groups[j],key=lambda r:(r['valid_position'],r['mapping'],r['conflict'])) for j in sorted(groups)]

def freeze():
    assert not(C.OUT/'plan.json').exists();C.reserve()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(C.MODEL,local_files_only=True);grammars=[];hashes={};unique=set()
    old={hashlib.sha256(str(r['tokens']).encode()).hexdigest() for split in ['discovery','confirm'] for r in C.read(C.ROOT/f'tests/glm5/result/state_formation_v1/{split}-materials.json')['rows']}
    for split,frames in [('discovery',DISC),('confirm',CONF)]:
        byframe=[]
        for i in range(len(frames)):
            g=spec(split,i);grammars.append(g);byframe.append(material(tok,g))
        groups=[byframe[i][j] for j in range(len(byframe[0])) for i in range(len(frames))];rows=[]
        for f,group in enumerate(groups):
            assert len(group)==8
            prefix=group[0]['tokens'][:group[0]['positions'][0]+1]
            for row in group:
                row['family']=f;digest=hashlib.sha256(str(row['tokens']).encode()).hexdigest();assert digest not in old|unique;unique.add(digest)
                assert prefix==row['tokens'][:row['positions'][0]+1];rows.append(row)
        C.save(split+'-materials.json',{'rows':rows,'families':len(groups),'independent_inputs':True});hashes[split+'-materials.json']=C.sha(C.OUT/(split+'-materials.json'))
    C.save('grammars.json',{'schema':'record_selection_grammars.v1','grammars':grammars});hashes['grammars.json']=C.sha(C.OUT/'grammars.json')
    choice={k:sorted(set(tok.encode(s,add_special_tokens=False)[0] for s in values)) for k,values in {'en_red':[' red','red'],'en_blue':[' blue','blue'],'zh_red':['红',' 红'],'zh_blue':['蓝',' 蓝']}.items()}
    deps=['tests/glm5/'+n for n in ['record_selection_core.py','record_selection_design.py','record_selection_runner.py','state_formation_core.py']]
    deps+=['shared/dist_protocol/grammar.py','tools/provenance.py','tools/power_analysis.py','.venv/Lib/site-packages/transformers/models/qwen3/modeling_qwen3.py']
    power=check_phase(.06,512,alpha=.05/4);assert power['pass']
    C.save('plan.json',{'schema':'record_selection_plan.v1','created_at':datetime.now().astimezone().isoformat(),'dependencies':{n:C.sha(C.ROOT/n) for n in deps},'materials':hashes,
        'choice_ids':choice,'behavior_min':.9,'pilot_families':[f for f in range(128) if f//8 in [0,4,8,12]],
        'families':{'discovery':128,'confirm':512},'unique_inputs':len(unique),'power':power,
        'resources':{'physical_output':str(C.OUT),'result_budget_GiB':18,'C_reserve_GiB':30,'max_minutes_per_stage':90,'single_cuda_model':True},
        'head_layers':R.HEAD_LAYERS,'unit_layers':R.UNIT_LAYERS,'head_group_size':2,'unit_group_size':32,'null_draws':1024,'random_group_controls':4,
        'primary_dose':1.,'doses':[.5,1.],'qkv_relative_tolerance':.02,
        'primary_claims':['selected head pair chooses valid over obsolete at both display positions','selected native MLP unit group contributes to correct color margin beyond same-layer controls'],
        'selection':'rank on discovery frames0-3, choose/refit on4-7; freeze before confirm; no confirmation sample or threshold selection',
        'analysis_freeze':'additional analysis source hashes must be frozen before full discovery; pilot uses discovery only',
        'causal_sample':'one prespecified conflict case per confirm family, 512 independent families; agree case separately for side-effect control',
        'effect_gate':{'mean_margin_loss_min':.2,'template_cluster_excess_lower_gt':0.,'alpha':.05/4},
        'failure_action':'failed behavioral cells cannot support semantic positive claims; diagnostic execution status reported separately if needed',
        'retention':'all native fields retained for new source-position and unit analyses, plus client numerical views; no old evidence removed',
        'limitations':['controlled explicit approval labels, not open-ended time reasoning','no natural-text dictionary comparison in this phase','single node and model','first-token choices, not free generation'],
        'formal_evidence_promotion':False})
    print({'unique':len(unique),'power':power,'C_free_GiB':shutil.disk_usage(C.OUT).free/2**30},flush=True)

if __name__=='__main__':freeze()
