"""Same records and same required fields; result/trace/neutral field order is crossed."""
import re
from rdc_conditional_common import *
from rdc_conditional_material import NAMES,OBJECTS
FAMILIES=('handover','category_chain','quantity_update')
ORDERS={'result_first':('Result','Trace','Neutral'),'trace_first':('Trace','Result','Neutral'),'neutral_first':('Neutral','Result','Trace')}


def case(f,unit,truth,lang):
    zh=lang=='zh';ae,be,az,bz=NAMES[unit+8];a,b=(az,bz) if zh else (ae,be);obj=OBJECTS[unit+8][int(zh)]
    c=f'保管员{unit+31}' if zh else f'Keeper{unit+31}';depth=5 if unit%2==0 else 8
    if f=='handover':
        owners=([a,b,c,a,c] if depth==5 else [a,b,c,a,b,c,a,c])+[b if truth else a]
        source=(f'开始时，{a}持有{obj}。' if zh else f'Initially {a} holds the {obj}. ')
        source+=' '.join((f'{x}把它交给{y}。' if zh else f'{x} passes it to {y}.') for x,y in zip(owners,owners[1:]))
        result=owners[-1];trace=' -> '.join(owners)
        fields='Result写最终持有人名字；Trace按顺序列出初始及每次交接后的持有人，以 -> 连接。' if zh else 'Result is the final holder name. Trace lists the initial holder and every subsequent holder, joined by ->.'
    elif f=='category_chain':
        labels=[f'C{unit+1}{chr(65+j)}' for j in range(depth+1)]
        source=(f'{a}属于{labels[0]}。' if zh else f'{a} is a member of {labels[0]}. ')
        clauses=[(f'每个{labels[j]}都属于{labels[j+1]}。' if zh else f'Every {labels[j]} is a member of {labels[j+1]}.') for j in range(depth-1)]
        clauses.append((f'{labels[-2]}的成员'+('都属于' if truth else '都不属于')+f'{labels[-1]}。') if zh else (f'Every {labels[-2]}' if truth else f'No {labels[-2]}')+f' is a member of {labels[-1]}.')
        source+=' '.join(clauses[::2]+clauses[1::2]);result='Included' if truth else 'Excluded'
        trace=' -> '.join(labels[:-1])+(' -> ' if truth else ' -/> ')+labels[-1]
        fields=(f'Result判断{a}是否属于{labels[-1]}，只写Included或Excluded。Trace从{labels[0]}开始列出类别链，属于边用 ->，不属于边用 -/>。' if zh else f'Result states whether {a} belongs to {labels[-1]}, using Included or Excluded. Trace starts at {labels[0]} and lists the category chain, using -> for membership and -/> for nonmembership.')
    else:
        depth=2;av,bv=unit+4,unit+8;delta=2 if truth else -2;af,bm=av-delta,bv+delta;bf=bm+3
        source=(f'起初{a}有{av}枚硬币，{b}有{bv}枚。'+(f'{a}给{b}2枚。' if truth else f'{b}给{a}2枚。')+f'随后{b}又得到3枚。没有其他变动。') if zh else (f'Initially {a} has {av} coins and {b} has {bv}. '+(f'{a} gives 2 to {b}. ' if truth else f'{b} gives 2 to {a}. ')+f'Then {b} receives 3 more. There are no other changes.')
        result=f'{a}={af}; {b}={bf}; Total={af+bf}';trace=f'[{av},{bv}] -> [{af},{bm}] -> [{af},{bf}]'
        fields=(f'Result用“{a}=数量; {b}=数量; Total=总数”写最终数量。Trace写初始、转移后、增加后的[前者数量,后者数量]，以 -> 连接。' if zh else f'Result gives final counts as "{a}=COUNT; {b}=COUNT; Total=COUNT". Trace gives [first person count,second person count] initially, after transfer, and after receipt, joined by ->.')
    neutral=' -> '.join(f'X{k}' for k in range(depth+1))
    return dict(family=f,unit=unit,truth=truth,language=lang,source=source,field_instruction=fields,
      expected_fields={'Result':result,'Trace':trace,'Neutral':neutral},depth=depth,u=a,v=b,
      limit='Depth alternates with entity group, not an isolated depth factor. Neutral length is not exactly token matched to Trace. All three required fields occur in every condition.')


def build(tok):
    rows=[]
    for unit in range(8):
     for f in FAMILIES:
      for truth in (0,1):
       for lang in ('en','zh'):
        base=case(f,unit,truth,lang)
        for order,labels in ORDERS.items():
            fields=base['expected_fields'];seq='、'.join(labels) if lang=='zh' else ', '.join(labels)
            instruction=(base['field_instruction']+f' Neutral只原样抄写“{fields["Neutral"]}”。仅输出三行，标签顺序必须为{seq}，每行用“标签: 内容”。不要其他说明。') if lang=='zh' else (base['field_instruction']+f' Neutral copies exactly "{fields["Neutral"]}". Output exactly three lines in this label order: {seq}. Each line is LABEL: CONTENT. Add nothing else.')
            system='严格按用户指定格式输出。' if lang=='zh' else 'Follow the requested output format exactly.'
            body=instruction+'\n\n'+base['source'];prompt=tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':body}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);start=prompt.index(base['source']);stop=start+len(base['source'])
            record_tokens=[i for i,(a,b) in enumerate(enc['offset_mapping']) if b>start and a<stop and b>a]
            row=dict(base,order=order,labels=list(labels),sample_id=f'm-{f}-{unit}-{truth}-{lang}-{order}',base_id=f'm-{f}-{unit}-{truth}-{lang}',
              word_split='train' if unit<4 else 'validation' if unit<6 else 'test',system=system,user=body,prompt=prompt,
              prompt_ids=enc['input_ids'],tokens=tok.convert_ids_to_tokens(enc['input_ids']),record_token_positions=record_tokens,
              reference='\n'.join(label+': '+fields[label] for label in labels))
            rows.append(row)
    assert len(rows)==288 and len({r['prompt'] for r in rows})==288
    for bid in {r['base_id'] for r in rows}:
        rr=[r for r in rows if r['base_id']==bid];assert len(rr)==3 and len({r['source'] for r in rr})==1 and all(r['expected_fields']==rr[0]['expected_fields'] for r in rr)
    return rows


def norm(s):return re.sub(r'\s+','',s).casefold().replace('：',':').replace('；',';').replace('，',',')


def score(row,text,eos,truncated):
    lines=[s.strip() for s in text.strip().splitlines() if s.strip()];parsed={};labels=[]
    for line in lines:
        m=re.fullmatch(r'(Result|Trace|Neutral)\s*[:：]\s*(.*)',line,re.I)
        if m:
            label=m[1].title();labels.append(label);parsed.setdefault(label,[]).append(m[2])
    field_correct={k:len(parsed.get(k,[]))==1 and norm(parsed[k][0])==norm(v) for k,v in row['expected_fields'].items()}
    return {'result_correct':field_correct['Result'],'trace_correct':field_correct['Trace'],'neutral_correct':field_correct['Neutral'],
      'all_content_correct':all(field_correct.values()),'field_order_correct':labels==row['labels'],'format_structure':len(lines)==3 and sorted(labels)==sorted(ORDERS['result_first']),
      'eos':bool(eos),'truncated':bool(truncated),'parsed_fields':parsed,'exact_reference':norm(text)==norm(row['reference']),
      'limits':'Deterministic field constraints, whitespace/punctuation normalization only. Does not infer the model internal reasoning process.'}


def result_boundary(text):
    return bool(re.search(r'(?:^|\n)\s*Result\s*[:：][ \t]*$',text,re.I))


def source_labels(tok,row,generated):
    # Assign each actual source token to the text field it occupies. Prompt record spans use tokenizer offsets.
    labels=['record' if i in row['record_token_positions'] else 'prompt_other' for i in range(len(row['prompt_ids']))]
    for i in range(len(generated)):
        before=tok.decode(generated[:i],skip_special_tokens=True);after=tok.decode(generated[:i+1],skip_special_tokens=True)
        line=after.rsplit('\n',1)[-1] if not after.endswith('\n') else before.rsplit('\n',1)[-1]
        m=re.match(r'\s*(Result|Trace|Neutral)\s*[:：]',line,re.I)
        labels.append('generated_'+m[1].lower() if m else 'generated_other')
    return labels
