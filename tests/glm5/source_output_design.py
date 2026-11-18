"""Author-generated crossed source rules and answer codebooks, frozen once."""
import hashlib,math,shutil
from datetime import datetime
from statistics import NormalDist
import source_output_core as S
from source_output_core import C
from shared.dist_protocol import grammar

DISC=[
 'Record {i}: item {item}; approval {status}; updated {date}; color {color}.',
 'Entry {i} for {item} has status {status}, date {date}, and color {color}.',
 'File {i} lists {item}, {color}, with status {status} and update date {date}.',
 'Number {i} concerns {item}. Status: {status}. Date: {date}. Color: {color}.',
 'For {item}, record {i} was updated on {date}; it is {status} and gives {color}.',
 'Entry {i} | {item} | date {date} | approval {status} | color {color}.']
CONF=[
 'Register {i} assigns color {color} to {item}; its date is {date}, and its status is {status}.',
 'Document {i}, dated {date}, describes {item} as {color}. Approval label: {status}.',
 'Reference {i} ({status}) reports a {color} {item} as of {date}.',
 'Sheet {i} has date {date} and approval {status}; its entry for {item} is {color}.',
 'The {status} report numbered {i} identifies {item} as {color}; update date {date}.',
 'Index {i}: {item} corresponds to {color}. This index is {status}, last updated {date}.',
 'List {i}, for item {item}, carries date {date}, status {status}, and color {color}.',
 'Under reference {i}, {item} is labeled {color}. Recorded date {date}; approval {status}.',
 'The date on table {i} is {date}. For {item} it states {color}; the status is {status}.',
 'Report {i} is {status}. It gives {color} for {item}, with an update on {date}.',
 'Update {date} belongs to record {i}: item {item}, status {status}, color {color}.',
 'Register {i} has approval {status} and update {date}; the listed color for {item} is {color}.']

def query(split,frame,task,lang,item,requested,valid):
    en={
     'status':['Choose the record with {valid} approval for {item}, ignoring its date.',
      'For {item}, select by approval status only: use the {valid} record.',
      'Find which record is {valid} for {item}; do not choose by date.',
      'Identify the approved ({valid}) entry for {item}, regardless of when it was updated.',
      'Locate the {valid} record about {item}. The update date is irrelevant.',
      'Use approval status to select the {valid} file for {item}. Ignore chronology.',
      'Resolve this lookup for {item} by selecting approval {valid}, not by the update date.',
      'The requested criterion for {item} is approval {valid}. Identify that record.',
      'Select the entry for {item} carrying approval {valid}; disregard relative age.',
      'For this {item} lookup, approval {valid} determines the required record.',
      'Find the {valid} registration for {item}, even when another registration is more recent.',
      'Consult the record of {item} whose approval is {valid}, with no date preference.'],
     'time':['Choose the record with the later update date for {item}, ignoring approval status.',
      'For {item}, select the most recently updated record even if its approval was withdrawn.',
      'Compare update dates for {item}; choose the newer record, regardless of status.',
      'Identify which entry for {item} has the later date. Approval does not matter.',
      'Locate the newest dated record about {item}; ignore the approval labels.',
      'Use chronology only to select the later file for {item}.',
      'Resolve this lookup for {item} using the later timestamp, even if that record is obsolete.',
      'The requested criterion for {item} is the latest update date. Disregard approval.',
      'Select the entry for {item} whose date comes later, whether approved or withdrawn.',
      'For this {item} lookup, the most recent date determines the required record.',
      'Find the registration for {item} updated last in time. Approval is irrelevant.',
      'Consult the record of {item} with the newer date; do not use approval to decide.'],
     'explicit':['Choose record {requested} for {item}; ignore approval and dates.',
      'For {item}, the required record number is explicitly {requested}.',
      'Select record {requested} about {item}, without examining approval or chronology.',
      'This lookup of {item} requires entry {requested}, irrespective of its attributes.',
      'Locate record {requested} for {item}; its status and date are irrelevant.',
      'Use file {requested} for {item}. Do not apply an approval or date rule.',
      'Resolve this lookup of {item} with the explicitly specified record {requested}.',
      'The requested record for {item} is number {requested}. Disregard its other fields.',
      'Select entry {requested} for {item}, whether it is older, newer, approved or withdrawn.',
      'For this {item} lookup, use record number {requested} exactly as requested.',
      'Find registration {requested} for {item}, without interpreting approval or dates.',
      'Consult record {requested} of {item}; the number is given explicitly.']}
    cn={
     'status':['请按有效状态为{item}选择{valid}记录，不考虑日期。','查询{item}时只依据批准状态，选取标为{valid}的条目。','找出{item}的{valid}记录，不要按更新时间选择。','请确定{item}的有效条目，批准标签为{valid}，日期先后无关。','定位{item}被标记为{valid}的记录，忽略日期。','依据批准状态选取{item}的{valid}档案。','本次{item}查询以状态{valid}为选择依据，时间新旧不参与判断。','针对{item}，需要的是批准标签为{valid}的那条记录。','请选择{item}具有{valid}状态的条目，不比较记录的新旧。','本次查询{item}时，状态{valid}决定应使用的记录。','即使另一条记录更新，也要选取{item}的{valid}登记。','查阅{item}批准状态为{valid}的记录，日期不作为依据。'],
     'time':['请为{item}选择更新日期更晚的记录，忽略批准状态。','查询{item}时选择最近更新的记录，即使其批准已撤销。','比较{item}记录的更新日期，选较新的一条，不看状态。','请确定{item}哪个条目的日期更晚，批准标签无关。','定位{item}日期最新的记录，忽略批准标签。','只按时间先后选取{item}较晚更新的档案。','本次{item}查询以更晚的时间为依据，即使那条记录已作废。','针对{item}，需要更新日期最新的一条，忽略有效状态。','请选择{item}日期较晚的条目，无论是否获批。','本次查询{item}时，最新日期决定应使用哪条记录。','找出{item}最后更新的登记，不考虑批准状态。','查阅{item}日期更新的记录，不以有效性决定选择。'],
     'explicit':['请选{item}的第{requested}条记录，忽略状态和日期。','查询{item}时，明确要求的记录编号是{requested}。','直接选择{item}的记录{requested}，不判断状态或时间。','本次{item}查询要求条目{requested}，与其属性无关。','定位{item}的记录{requested}，其状态和日期无关。','使用{item}的档案{requested}，不要应用日期或批准规则。','本次{item}查询必须使用明确指定的记录{requested}。','针对{item}，所需记录编号已经给定为{requested}。','请选择{item}的条目{requested}，无论新旧或是否获批。','本次查询{item}时严格使用编号为{requested}的记录。','找出{item}的登记{requested}，不解释其日期和批准信息。','查阅{item}的记录{requested}，编号已经直接指定。']}
    calibration_en={
     'status':['Read the approval tag, and pick the {valid} source for {item}.','The source to use for {item} is the one approved as {valid}; date is not the criterion.','Which record about {item} is authorized by the label {valid}?','Retrieve {item} from the source marked {valid}, ignoring temporal order.','Approval decides this {item} lookup: identify {valid}.','Determine the usable source for {item} under approval {valid}.'],
     'time':['Read the dates and pick the later source for {item}; approval is not the criterion.','The source to use for {item} is the newest dated one, irrespective of approval.','Which record about {item} was updated more recently? Disregard the approval tags.','Retrieve {item} from the chronologically later source, even when withdrawn.','Temporal order decides this {item} lookup: identify the most recent date.','Determine the newest source for {item} from its date, without using approval.'],
     'explicit':['Read this instruction: record {requested} is required for {item}.','The source to use for {item} is specified as number {requested}; other fields do not decide.','Which record is requested here? It is number {requested} for {item}.','Retrieve {item} from the explicitly named source, record {requested}.','The given number decides this {item} lookup: it is {requested}.','The required source for {item} has been specified directly: record {requested}.']}
    calibration_cn={
     'status':['读取批准标签，为{item}挑选状态为{valid}的来源。','{item}所用来源必须被批准为{valid}，不能依据日期判断。','关于{item}，哪份记录获得了{valid}标签的许可？','从标有{valid}的来源检索{item}，不考虑时间先后。','此次{item}检索以批准状态为准，找出{valid}。','根据批准标签{valid}，确定{item}可用的来源。'],
     'time':['读取日期，为{item}挑选时间较晚的来源，批准状态不参与。','{item}所用来源必须日期最新，与批准标签无关。','关于{item}，哪份记录的更新时间更靠后？不要参考批准状态。','从时间更晚的来源检索{item}，即使该来源已被撤回。','此次{item}检索以时间为准，找出日期最新的一条。','根据日期确定{item}最新的来源，不使用批准标签。'],
     'explicit':['请读取这项指令：{item}需要记录{requested}。','{item}所用来源已指定为编号{requested}，不依据其他字段。','这里请求的记录是哪份？是{item}的编号{requested}。','从明确点名的来源检索{item}，即记录{requested}。','此次{item}检索以给定编号为准，该编号是{requested}。','{item}需要的来源已直接指定：记录{requested}。']}
    # Calibration and confirm use disjoint query wording, not modulo reuse.
    bank=en if lang==1 else cn
    if split=='calibration':
        text=(calibration_en if lang==1 else calibration_cn)[task][frame]
    else:
        text=bank[task][frame]
        text=('New request: '+text+' Apply only this requested selection rule.') if lang==1 else ('新的检索请求：'+text+'请只应用这项选择规则。')
    return text.format(item=item,requested=requested,valid=valid)

def spec(split,frame,task,alphabet):
    calibration=split=='calibration';count=8 if calibration else 16;entry=(DISC if calibration else CONF)[frame]
    pairs=[('APPROVED','WITHDRAWN'),('VALID','INVALID'),('CURRENT','EXPIRED')] if calibration else [('ACTIVE','REVOKED'),('ACCEPTED','CANCELLED'),('IN FORCE','SUPERSEDED'),('VALID','VOID')]
    valid,old=pairs[frame%len(pairs)];fillers=[]
    for j in range(count):
        time_relation=1 if j%2 else -1;request_relation=1 if (j//2)%2 else -1;lang=1 if (j//4)%2 else -1
        nouns=['tile','box','parcel','badge'] if calibration else ['vial','panel','scroll','casket','token','plaque']
        item=f'{nouns[j%len(nouns)]} X{3000000+(0 if calibration else 100000)+frame*100+j}'
        low=f'2022-{1+j%6:02d}-{2+j%20:02d}';high=f'2025-{7+j%5:02d}-{3+j%19:02d}'
        slots={'intro':f'Two records describe {item}. Approval {valid} means valid; approval {old} means withdrawn. Approval and update date are separate attributes. Select by the requested rule, then translate the selected record number using the output codebook.',
            'separator':'Selection rule ends here.','tail':'Answer: '}
        for s in [-1,1]:
            suffix='plus' if s==1 else 'minus';ts=s*time_relation;requested=1 if s*request_relation==1 else 2
            for i in [1,2]:
                pos=1 if i==1 else -1
                slots[f'record{i}_{suffix}']=entry.format(i=i,item=item,status=valid if pos==s else old,date=high if pos==ts else low,color='red' if (j+i)%2 else 'blue')
            slots['query_'+suffix]=query(split,frame,task,lang,item,requested,valid)
        for m in [-1,1]:
            labels=['1','2'] if alphabet=='digits' else ['A','B'];a,b=labels if m==1 else labels[::-1]
            slots['book_'+('plus' if m==1 else 'minus')]=f'Output codebook: record 1 -> {a}; record 2 -> {b}. Return the output code only, not a record number or explanation.'
        fillers.append({'id':f'{split[0]}{frame:02d}{j:02d}','slots':slots,'levels':{'language':lang,'time_relation':time_relation,'request_relation':request_relation},
            'meta':{'j':j,'item':item,'valid':valid,'old':old,'time_relation':time_relation,'request_relation':request_relation,'language':lang}})
    variants=[]
    for s in [-1,1]:
        for m in [-1,1]:
            ss='plus' if s==1 else 'minus';ms='plus' if m==1 else 'minus'
            variants.append({'id':f's{s+1}m{m+1}','levels':{'status_position':s,'code_sign':m},'target':'tail','cue':'{separator}',
                'text':'{intro}\n{record1_'+ss+'}\n{record2_'+ss+'}\n{query_'+ss+'}\n{separator}\n{book_'+ms+'}\n{tail}',
                'meta':{'record1':'record1_'+ss,'record2':'record2_'+ss,'query':'query_'+ss,'book':'book_'+ms}})
    return {'schema_version':'template_grammar.v1','family_id':f'SOURCE-OUTPUT-{split}-{frame}-{task}-{alphabet}','grammar_version':1,'module':'KN',
        'name':f'{split} {frame} {task} {alphabet}','description':'Authored source rule and output codebook factorial; fixed old native candidates',
        'factors':['status_position','code_sign','language','time_relation','request_relation'],'frames':variants,'fillers':fillers,
        'layers':'all','readouts':['cue','final'],'splits':{'salt':str(C.SEED),'fractions':{'discovery':float(calibration),'confirm':float(not calibration),'lockbox':0.},},
        'shards':1,'prereg':{'min_behavior_acc':.9,'min_cell_n':48}}

def material(tok,g,split,frame,task,alphabet):
    assert not grammar.validate(g),grammar.validate(g);items=grammar.generate(g,include_canary=False)['items'];fl={v['id']:v for v in g['fillers']};fr={v['id']:v for v in g['frames']};rows=[]
    for item in items:
        filler=fl[item['filler_id']];variant=fr[item['frame_id']];slots=filler['slots'];meta=filler['meta'];cond=item['cond']
        body=item['text'].removesuffix('\nAnswer: ');prompt=tok.apply_chat_template([{'role':'user','content':body}],tokenize=False,add_generation_prompt=True,enable_thinking=False)+'Answer: '
        tokenized=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);offsets=tokenized['offset_mapping']
        def span(text):
            start=prompt.index(text);end=start+len(text);return [i for i,(a,b) in enumerate(offsets) if b>start and a<end and b>a]
        spans={k:span(slots[variant['meta'][k]]) for k in ['record1','record2','query','book']};spans['instruction']=sorted(spans['query']+spans['book'])
        s=cond['status_position'];target=s*(meta['time_relation'] if task=='time' else meta['request_relation'] if task=='explicit' else 1)
        rows.append({'id':item['id'],'split':split,'frame':frame,'local_family':meta['j'],'task':task,'alphabet':alphabet,
            'status_position':s,'target_sign':target,'code_sign':cond['code_sign'],'output_sign':target*cond['code_sign'],
            **{k:meta[k] for k in ['language','time_relation','request_relation']},'prompt':prompt,'tokens':tokenized['input_ids'],
            'positions':[span(slots['separator'])[-1],len(tokenized['input_ids'])-1],'spans':spans,'grammar_sha256':grammar.spec_sha(g)})
    return rows

def freeze():
    assert not (C.OUT/'plan.json').exists();C.reserve()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(C.MODEL,local_files_only=True);allgrammars=[];hashes={};unique=set();max_tokens=0
    old_tokens={hashlib.sha256(str(r['tokens']).encode()).hexdigest() for sp in ['discovery','confirm'] for r in C.read(S.OLD/(sp+'-materials.json'))['rows']}
    for split,nframes,nfamily in [('calibration',6,8),('confirm',12,16)]:
        rows=[]
        for frame in range(nframes):
            for task in S.TASKS:
                for alphabet in S.ALPHABETS:
                    g=spec(split,frame,task,alphabet);allgrammars.append(g);rows+=material(tok,g,split,frame,task,alphabet)
        rows.sort(key=lambda r:(r['local_family'],r['frame'],S.TASKS.index(r['task']),S.ALPHABETS.index(r['alphabet']),r['status_position'],r['code_sign']))
        for i,r in enumerate(rows):
            r['family']=i//24;r['variant']=i%24;digest=hashlib.sha256(str(r['tokens']).encode()).hexdigest();assert digest not in unique|old_tokens;unique.add(digest);max_tokens=max(max_tokens,len(r['tokens']))
        for f in range(nframes*nfamily):
            group=rows[f*24:(f+1)*24]
            for task in S.TASKS:
                for pos in [-1,1]:
                    sub=[r for r in group if r['task']==task and r['status_position']==pos];prefixes=[r['tokens'][:r['positions'][0]+1] for r in sub]
                    assert len(sub)==4 and all(v==prefixes[0] for v in prefixes)
        C.save(split+'-materials.json',{'rows':rows,'families':nframes*nfamily});hashes[split+'-materials.json']=C.sha(C.OUT/(split+'-materials.json'))
    C.save('grammars.json',{'grammars':allgrammars});hashes['grammars.json']=C.sha(C.OUT/'grammars.json')
    old=C.read(S.OLD/'selection.json');candidates=old['candidates'];ids={}
    for alphabet,labels in [('digits',['1','2']),('letters',['A','B'])]:
        values=[tok.encode(v,add_special_tokens=False) for v in labels];assert all(len(v)==1 for v in values);ids[alphabet]=[v[0] for v in values]
    deps=['tests/glm5/'+n for n in ['source_output_core.py','source_output_design.py','source_output_runner.py','state_formation_core.py','record_selection_core_v3.py','record_selection_match.py']]+['shared/dist_protocol/grammar.py','tools/provenance.py','.venv/Lib/site-packages/transformers/models/qwen3/modeling_qwen3.py']
    sd=.5;delta=.2;alpha=.05/6;z=NormalDist().inv_cdf(1-alpha/2);need=math.ceil(((z+NormalDist().inv_cdf(.8))*sd/delta)**2)
    upper=(37*2*2560*2+2*2560*2+2*32*128*2+2*32*max_tokens*4+2*32*4*4+2*2*4*128*4+3*2*9728*2+5000)*5760+200*2**20
    assert upper<3.8*2**30 and shutil.disk_usage(C.OUT).free-upper>30*2**30,(upper/2**30,max_tokens)
    C.save('plan.json',{'schema':'source_output_plan.v1','created_at':datetime.now().astimezone().isoformat(),'dependencies':{n:C.sha(C.ROOT/n) for n in deps},'materials':hashes,
        'resources':{'physical_output':str(C.OUT),'budget_GiB':3.8,'conservative_uncompressed_upper_bytes':upper,'C_reserve_GiB':30,'single_cuda_model':True,'max_minutes_per_stage':60},
        'answer_ids':ids,'minimum_cell_n':48,'families':{'calibration':48,'confirm':192},'unique_inputs':5760,'max_tokens':max_tokens,
        'fixed_candidates':candidates,'old_selection_sha256':C.sha(S.OLD/'selection.json'),'no_new_candidate_search':True,
        'behavior_gate':'Strict pair margin and full-vocabulary first-token top1 >=.90 in every position x map cell and query-language cell within an entire task x alphabet branch.',
        'branch_failure_action':'Keep all behavior and failure rows; stop that entire branch for semantic features and effects. No family or condition filtering and no prompt refreeze to rescue the gate.',
        'readouts':['selection_end_before_codebook','answer_prefix'],'capture':'all37 hidden layers x2 positions; all B26 native gate/value/units x2; all B29 head contexts/QK x2; selected-head source V contexts x2',
        'power':{'paired_margin_delta':delta,'assumed_family_sd':sd,'alpha':alpha,'required_family_n':need,'per_alphabet_causal_family_n':96,'approx_power':NormalDist().cdf(delta*math.sqrt(96)/sd-z),'limitations':'normal mean approximation; shared frames and actual variance can reduce power; no claim of small answer-flip sensitivity'},
        'causal_selection':'Per confirm family all3 tasks, both codebook signs; alphabet=digits if j<8 else letters; status_position parity(frame+j+j//2+j//4). No selection on model answers.',
        'causal_controls':'first two frozen same-layer control groups, each original ablation plus actual residual-direction norm matching; one random norm-matched direction; max scale4 for valid matched-native inference, otherwise control-invalid stop for that claim',
        'causal_edits':['H29 pair','U26 group','joint H29+U26'],'dose':1.,'effect_min':.2,'alpha':alpha,'null_draws':2048,
        'analysis_freeze':'Additional analysis/causal code frozen before collecting calibration native fields; predictors frozen before confirm behavior.',
        'retention':'native fields used for client and future source/output decomposition; sparse prespecified full-vocabulary audit cases; every causal output gets hash and exact answer logits',
        'formal_evidence_promotion':False})
    print({'inputs':5760,'max_tokens':max_tokens,'upper_GiB':upper/2**30,'power':C.read(C.OUT/'plan.json')['power']},flush=True)

if __name__=='__main__':freeze()
