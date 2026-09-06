"""Transparent character/token role annotations for future native source-coordinate ledgers.

These are externally declared text regions, not discovered semantic modules. The
assignment never uses a model answer, activation value, or expected truth label.
"""
import re

ROLES=('chat_scaffold','body_other','body_entity_a','body_entity_b','query_other',
       'query_entity_a','query_entity_b','answer_rule','example_1','example_2',
       'task_footer','task_separator','mixed','zero_width')
ROLES=ROLES+('assistant_factual_prefix',)


def character_regions(row):
    prompt,text,body=row['prompt'],row['text'],row['body']
    assert prompt.count(text)==1 and text.startswith(body)
    user_start=prompt.index(text);body_end=user_start+len(body)
    labels=['chat_scaffold']*len(prompt)
    labels[user_start:body_end]=['body_other']*len(body)
    def entities(start,end,prefix):
        for key in ('a','b'):
            word=row['entity_'+key]
            for m in re.finditer(re.escape(word),prompt[start:end]):
                a,b=start+m.start(),start+m.end();assert len(set(labels[a:b]))==1
                labels[a:b]=[prefix+'_entity_'+key]*(b-a)
    entities(user_start,body_end,'body')
    pos=body_end
    for line in text[len(body):].splitlines(keepends=True):
        stripped=line.strip()
        if not stripped:role='task_separator'
        elif stripped.startswith(('Statement:','陈述：','Question:','问题：')):role='query_other'
        elif stripped.startswith(('Example 1:','示例一：')):role='example_1'
        elif stripped.startswith(('Example 2:','示例二：')):role='example_2'
        elif stripped.startswith(('Now output','现在输出')):role='task_footer'
        elif stripped.startswith(('Output code:','If the requested','编码表：','所问判断','Output rule:','输出规则：','Completion instruction:','续写要求：')):role='answer_rule'
        else:raise ValueError(('Unclassified task text; extend explicitly before experiments',stripped))
        end=pos+len(line);labels[pos:end]=[role]*len(line)
        if role=='query_other':entities(pos,end,'query')
        pos=end
    assert pos==user_start+len(text) and len(labels)==len(prompt)
    if row.get('prefill'):
        prefill=row['prefill'];assert prompt.endswith(prefill)
        start=len(prompt)-len(prefill);assert start>=pos
        labels[start:]=['assistant_factual_prefix']*len(prefill)
    regions=[];start=0
    for i in range(1,len(labels)+1):
        if i==len(labels) or labels[i]!=labels[start]:
            regions.append({'start':start,'end':i,'role':labels[start],'text':prompt[start:i]});start=i
    assert ''.join(r['text'] for r in regions)==prompt
    return regions


def token_regions(offsets,regions):
    assert regions and regions[0]['start']==0
    n=regions[-1]['end'];out=[]
    for i,(start,end) in enumerate(offsets):
        assert 0<=start<=end<=n
        overlapping=[r['role'] for r in regions if r['end']>start and r['start']<end] if start<end else []
        roles=sorted(set(overlapping))
        role='zero_width' if not roles else roles[0] if len(roles)==1 else 'mixed'
        out.append({'token':i,'start':start,'end':end,'role':role,'overlap_roles':roles})
    return out
