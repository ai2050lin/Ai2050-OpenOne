"""Eight multi-output operations with inspectable references and conservative scoring."""
import re
from rdc_conditional_common import *
from rdc_conditional_material import NAMES,OBJECTS
FAMILIES=('clause_reorder','reference_chain','taxonomy_explain','role_table','translation','style_rewrite','structured_extract','temporal_revision')


def case(f,unit,language):
    a,b,az,bz=NAMES[unit+8];obj,oz=OBJECTS[unit+8];zh=language=='zh'
    if zh:a,b,obj=az,bz,oz
    c=(f'保管员{unit+21}' if zh else f'Keeper{unit+21}')
    source='';reference='';terms=[];ordered=[];rows=[];depth=None
    if f=='clause_reorder':
        clauses=([f'09:00，{a}打开仓库。',f'12:00，{b}送来{obj}。',f'15:00，{c}检查货物。',f'18:00，{a}关闭仓库。'] if zh else [f'09:00, {a} opened the warehouse.',f'12:00, {b} delivered the {obj}.',f'15:00, {c} inspected the goods.',f'18:00, {a} closed the warehouse.'])
        order=(2,0,3,1) if unit%2 else (3,1,0,2)
        source='\n'.join(clauses[k] for k in order)
        instruction='将下列四句按时间先后排列。逐句原样保留，每句一行，不加标题或说明。' if zh else 'Put the following four sentences in chronological order. Preserve each sentence exactly, one per line, without a title or explanation.'
        reference='\n'.join(clauses);terms=clauses;ordered=['09:00','12:00','15:00','18:00'];rows=clauses
    elif f=='reference_chain':
        depth=5 if unit%2==0 else 8
        owners=[a,b,c,a,c,b,a,b,c][:depth+1]
        clauses=[(f'开始时，{a}持有{obj}。' if zh else f'Initially, {a} holds the {obj}.')]
        for j in range(depth):
            clauses.append((f'接着，{owners[j]}把它交给{owners[j+1]}。' if zh else f'Next, {owners[j]} passes it to {owners[j+1]}.'))
            if j in (1,3):clauses.append('仓库的灯一直亮着。' if zh else 'The warehouse lights remain on.')
        source=' '.join(clauses)
        instruction=('根据交接记录输出两行：第一行“持有人：姓名”，第二行“路径：”后列出初始及每次交接后的持有人，用 -> 连接。不加说明。' if zh else 'From this handover record, output two lines: "Owner: NAME" and "Path: " followed by the initial and each subsequent holder joined by ->. Add no explanation.')
        reference=(f'持有人：{owners[-1]}\n路径：' if zh else f'Owner: {owners[-1]}\nPath: ')+' -> '.join(owners)
        terms=[owners[-1],' -> '.join(owners)];ordered=[];rows=reference.splitlines()
    elif f=='taxonomy_explain':
        depth=5 if unit%2==0 else 8;labels=[f'R{unit+1}{chr(65+j)}' for j in range(depth+1)]
        clauses=[(f'所有{labels[j]}都属于{labels[j+1]}。' if zh else f'Every {labels[j]} is a member of {labels[j+1]}.') for j in range(depth)]
        order=list(range(depth));order=order[::2]+order[1::2]
        source=' '.join(clauses[j] for j in order)
        instruction=(f'仅按记录解释{labels[0]}到{labels[-1]}的类别链。输出“结论：属于”，再另起一行“路径：”并按推导顺序用 -> 连接全部类别，不加其他内容。' if zh else f'Use only the record to explain the category chain from {labels[0]} to {labels[-1]}. Output "Decision: Included", then a new line "Path: " with all categories in derivation order joined by ->. Add nothing else.')
        reference=('结论：属于\n路径：' if zh else 'Decision: Included\nPath: ')+' -> '.join(labels)
        terms=[' -> '.join(labels)];ordered=labels;rows=reference.splitlines()
        a,b=labels[0],labels[-1]
    elif f=='role_table':
        triples=[(a,'lend' if not zh else '借出',b,obj),(c,'show' if not zh else '展示',a,'map' if not zh else '地图'),(b,'send' if not zh else '寄送',c,'letter' if not zh else '信件')]
        source=(f'{obj}由{a}借给{b}。{c}向{a}展示了地图。信件由{b}寄给{c}。' if zh else f'The {obj} was lent to {b} by {a}. {c} showed a map to {a}. A letter was sent to {c} by {b}.')
        instruction=('按事件顺序提取三行，格式“施事|动作|接收者|物品”，不要表头。动作依次用“借出”“展示”“寄送”，保留原名字。' if zh else 'Extract three rows in event order as agent|action|recipient|item, without a header. Use the action words lend, show, send respectively; preserve names.')
        rows=['|'.join(t) for t in triples];reference='\n'.join(rows);terms=rows;ordered=rows
    elif f=='translation':
        # Names stay literal in both languages, making entity retention auditable without transliteration scoring.
        a,b=NAMES[unit+8][:2];oe,oz=OBJECTS[unit+8]
        english=f'{a} put the red {oe} on the desk. {b} did not move the blue {oe}.'
        chinese=f'{a}把红色的{oz}放在桌上。{b}没有移动蓝色的{oz}。'
        source=chinese if zh else english;reference=english if zh else chinese
        instruction='翻译为英语，保留英文人名及两句的顺序、颜色、动作和否定。只输出译文。' if zh else 'Translate into Chinese, preserving the literal names and the order, colors, actions and negation of both sentences. Output only the translation.'
        terms=[a,b,oe,'red','blue','desk',"not|didn't|did not"] if zh else [a,b,oz,'红','蓝','桌','没|未']
        ordered=[a,b]
    elif f=='style_rewrite':
        count=7+unit
        plural=obj+'es' if obj=='brush' else obj+'s'
        source=(f'嗨，{a}，周二前能把{count}个{obj}送给{b}吗？谢啦！' if zh else f'Hey {a}, can you send {count} {plural} to {b} by Tuesday? Thanks!')
        instruction='将消息改为礼貌正式的请求，保留收信人、收件人、物品、数量及截止日期。不新增事实，不解释改写过程。' if zh else 'Rewrite this as a polite formal request, preserving the addressee, recipient, item, quantity and deadline. Add no facts or explanation.'
        reference=(f'尊敬的{a}，请您于周二前将{count}个{obj}送交{b}。谢谢。' if zh else f'Dear {a}, please send {count} {plural} to {b} by Tuesday. Thank you.')
        terms=[a,b,obj,str(count),'周二|星期二' if zh else 'Tuesday'];ordered=[a,b]
    elif f=='structured_extract':
        quantities=[unit+3,unit+6,unit+9];items=[obj,'蜡烛' if zh else 'candle','墨水' if zh else 'ink']
        people=[a,b,c]
        source=('仓库清点记录如下。' if zh else 'Warehouse inventory follows. ')+ ' '.join((f'{p}保管{i}，数量为{q}。' if zh else f'{p} keeps {q} units of {i}.') for p,i,q in zip(people,items,quantities))+('房间里有两扇窗，这不是库存项目。' if zh else 'There are two windows in the room; these are not inventory items.')
        instruction='按记录顺序提取三个库存项目，每行“保管人|物品|数量”，不加表头，不列出窗户。' if zh else 'Extract the three inventory items in record order, one keeper|item|quantity per line. No header; do not list windows.'
        rows=[f'{p}|{i}|{q}' for p,i,q in zip(people,items,quantities)];reference='\n'.join(rows);terms=rows;ordered=rows
    else:
        av,bv=unit+4,unit+8;transfer=2;gain=3;af,bf=av-transfer,bv+transfer+gain
        source=(f'起初{a}有{av}枚硬币，{b}有{bv}枚。{a}给{b}{transfer}枚。随后{b}又得到{gain}枚。两人没有其他硬币变动。' if zh else f'Initially {a} has {av} coins and {b} has {bv}. {a} gives {transfer} coins to {b}. Then {b} receives {gain} additional coins. Neither person has any other coin changes.')
        instruction='计算最终数量，输出两行：第一行“姓名=数量; 姓名=数量”（按最初出现顺序），第二行“总计=数量”。不要说明。' if zh else 'Compute the final amounts. Output two lines: NAME=COUNT; NAME=COUNT in initial mention order, then Total=COUNT. Do not explain.'
        rows=[f'{a}={af}; {b}={bf}',('总计' if zh else 'Total')+f'={af+bf}'];reference='\n'.join(rows);terms=rows;ordered=[a,b]
    return dict(family=f,unit=unit,language=language,source=source,instruction=instruction,reference=reference,
      constraint_terms=terms,ordered_terms=ordered,expected_rows=rows,depth=depth,u=a,v=b,
      scoring_limit='Exact structure tasks use deterministic references. Translation/style score only declared lexical constraints and order, not complete semantic equivalence or no-added-fact correctness. Depth alternates with unit; not an isolated depth experiment.')


def build(tok):
    rows=[]
    for unit in range(8):
      for f in FAMILIES:
       for lang in ('en','zh'):
        r=case(f,unit,lang);system='Follow the user instruction exactly. Do not include hidden reasoning.' if lang=='en' else '严格按用户指令输出，不展示隐藏推理。'
        body=r['instruction']+'\n\n'+r['source']
        prompt=tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':body}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);spans={}
        for role,term in [('u',r['u']),('v',r['v'])]:
            start=prompt.index(r['source'])+r['source'].index(term);end=start+len(term)
            positions=[j for j,(a,b) in enumerate(enc['offset_mapping']) if b>start and a<end and b>a];assert positions
            spans[role]={'term':term,'chars':[start,end],'positions':positions}
        rows.append(dict(r,sample_id=f'k-{f}-{unit}-{lang}',base_id=f'k-{f}-{unit}',entity_group=unit,
          word_split='train' if unit<4 else 'validation' if unit<6 else 'test',
          system=system,user=body,prompt=prompt,prompt_ids=enc['input_ids'],tokens=tok.convert_ids_to_tokens(enc['input_ids']),spans=spans,
          source_mode='live_model',full_prefill_panel=unit==0))
    assert len(rows)==128 and len({r['prompt'] for r in rows})==128
    return rows


def compact(s):return re.sub(r'\s+','',s).casefold().replace('：',':').replace('；',';')


def score(row,text,eos,truncated):
    s=text.strip();norm=compact(s);reference=compact(row['reference'])
    hits=[bool(re.search(term if '|' in term and not row['expected_rows'] else re.escape(term),s,re.I)) for term in row['constraint_terms']]
    positions=[norm.find(compact(term)) for term in row['ordered_terms']]
    order=all(p>=0 for p in positions) and positions==sorted(positions)
    # For repeated holders, the entire required path is a content constraint, not a repeated .find heuristic.
    lines=[compact(x) for x in s.splitlines() if x.strip()]
    strict=norm==reference
    expected=[compact(x) for x in row['expected_rows']]
    structure=(len(lines)==len(expected) and lines==expected) if expected else bool(s) and not any(x in s for x in ('```','<think>','</think>'))
    return {'exact_reference':strict,'declared_content_constraints':all(hits),'constraint_fraction':sum(hits)/max(len(hits),1),
      'order_constraints':order,'format_structure':structure,'eos':eos,'truncated_at_limit':truncated,
      'constraint_hits':hits,'full_semantic_correctness':'not automatically established for translation/style',
      'content_kind':'lexical_checklist_only' if row['family'] in ('translation','style_rewrite') else 'deterministic_reference_constraints'}
