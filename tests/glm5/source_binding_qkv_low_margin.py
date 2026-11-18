"""Freeze, screen and intervene on new low-margin Chinese source-binding questions."""
import argparse
from datetime import datetime
import itertools
import json
from pathlib import Path

import numpy as np

import source_binding_qkv_suffix_token as token_test
import source_binding_transfer as transfer
import source_binding_transfer_factors as io
from source_binding_design import sha
from source_binding_readheads import H16

OUT = token_test.OUT.parent / 'low-margin'
SEED = 202611130431
FAMILIES = 96
CHUNK = 32
SCREEN_GROUP = 12
MAX_MARGIN = 1.0
TIGHT_MARGIN = .5
POOL = (
    '松队 岩队 柳队 兰队 泉队 云队 山队 川队 竹队 霞队 晨队 星队 森队 海队 鹤队 鹿队'.split(),
    '陈舟 梁宁 吴帆 何静 郭远 田悦 唐雨 方雪 蒋宁 马云 杜欣 许峰 丁凡 谢青 姚远 钱琳 彭亮 任雨 邵阳 金宁 郑雪 万山 邹明 董静'.split(),
)
NOTES = (
    '核对时只按正式名单，不采用另存的旧草稿。\n旧草稿（尚未更新，不能作为答案）：{src}的{ent}分到的是{foil}。\n',
    '以下旧草稿之中的记录已经作废，正式名单才有效。\n旧草稿：{ent}在{src}的名单里对应{foil}。\n作废记录的摘要：{ent}对应{foil}。\n',
    '以正式名单为准。下面是待纠正的记录，请勿将它们当作正式答案。\n待纠正：{src}把{foil}分给了{ent}。\n旧记录再次显示：{ent}在{src}分得{foil}。\n旧记录关键词：{foil}。\n',
    '必须根据正式名单回答。以下错误示范中答案不正确。\n错误示范：{question}\n错误答案：{foil}\n',
    '以下是另一支队伍的速查笔记，仅在题目问该队时使用。\n速查：按照{other}的名单，{ent}分到的是{foil}。\n',
    '旧草稿不能作为答案。\n旧草稿：{ent}在{src}的名单里分到{foil}。\n',
)
ARMS = ('mask_key','mask_adjacent','mask_all','reopen_key','reopen_adjacent','zero_all',
        'restore_key','restore_adjacent','restore_all','self_all','self','noop')


def materials():
    from source_binding_relay import tokenizer

    assert not (OUT/'design.json').exists()
    tok = tokenizer()
    frozen = token_test.OUT/'selected.json'
    selected = json.loads(frozen.read_bytes())['templates']
    prior = []
    for split in ('discover','confirm'):
        prior.extend(json.loads((token_test.OUT/f'{split}-materials.json').read_bytes()))
    assert not set(POOL[0]).intersection(n for row in prior for n in row['sources'])
    assert not set(POOL[1]).intersection(n for row in prior for n in row['names'])
    rng = np.random.default_rng(SEED)
    family_pairs = list(itertools.product(itertools.combinations(POOL[0],2),itertools.combinations(POOL[1],2)))
    chosen = rng.choice(len(family_pairs),FAMILIES,replace=False)
    rows = []
    for family,index in enumerate(chosen):
        srcs,ents = family_pairs[index]
        values = [str(v) for v in rng.choice(transfer.POOLS['zh']['values'],4,replace=False)]
        frame = family % 4
        for variant,pattern in enumerate(NOTES):
            for side in (0,1):
                s = (family+variant+side)%2
                e = int(rng.integers(2))
                o = (family+side)%2
                x = int(rng.integers(2))
                row = transfer.material('zh',family,'low',s,e,o,x,frame,srcs,ents,values)
                text = row['content']
                qstart = text.rindex('\n问题：')+1
                card_end = len(text)-len(transfer.TEXT['zh']['tail'])
                card = text[qstart:card_end]
                before = text[:qstart].replace(transfer.TEXT['zh']['intro'],transfer.TEXT['zh']['intro']+'正式名单如下。\n',1)
                intended_foil = 2*(1-s)+e
                args = dict(src=srcs[s],ent=ents[e],foil=values[intended_foil],question=card,other=srcs[1-s])
                note = pattern.format(**args)
                content = before + note + card + transfer.TEXT['zh']['tail']
                prompt = tok.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)+'Answer:\n'
                enc = tok(prompt,add_special_tokens=False,return_offsets_mapping=True)
                ids, offsets = enc['input_ids'], enc['offset_mapping']
                question_offset = prompt.index(content)+len(before)+len(note)
                end = question_offset+len(card)
                names = []
                for word in (srcs[s],ents[e]):
                    a = question_offset+card.index(word)
                    names.extend(i for i,(l,r) in enumerate(offsets) if r>a and l<a+len(word))
                suffix = [i for i in range(max(names)+1,len(ids)) if offsets[i][0]<end]
                parts = [tok.decode([ids[i]]) for i in suffix]
                f = selected[str(frame)]
                assert parts[f['offset']] in f['token'] and 5<=len(parts)<=7,(frame,parts,f)
                choices = []
                for v in values:
                    ext = tok.encode(prompt+v,add_special_tokens=False)
                    assert ext[:-1]==ids
                    choices.append(ext[-1])
                assert len(set(choices))==4
                record = 2*s+e
                rows.append(dict(id=f'low-{family:04d}-{variant}-{side}',family=f'low-{family:04d}',frame=frame,variant=variant,
                                 prompt=prompt,content=content,tokens=ids,choice_ids=choices,gold_id=choices[record],target_record=record,
                                 foil=intended_foil,values=values,sources=srcs,names=ents,suffix=suffix,parts=parts,name_slots=sorted(names),
                                 key=suffix[f['offset']],adjacent=suffix[(f['offset']+1)%len(suffix)]))
    assert len({tuple(r['tokens']) for r in rows})==len(rows)
    io.save('materials.json',rows)
    scripts = ('source_binding_qkv_low_margin.py','source_binding_qkv_low_margin_analysis.py','source_binding_qkv_suffix_token_engine.py',
               'source_binding_transfer_factor_engine.py','source_binding_runtime.py','source_binding_transfer.py','source_binding_readheads.py')
    design = dict(created=datetime.now().astimezone().isoformat(),seed=SEED,families=FAMILIES,prompts=len(rows),chunk=CHUNK,
                  heads=H16,materials_sha256=sha(OUT/'materials.json'),selected_sha256=sha(frozen),
                  code={n:sha(Path(__file__).with_name(n)) for n in scripts},
                  population='96 new source/person families, six perturbing texts including a cross-source distractor, two order layouts per variant. Original questions and four answer animal tokens. Prompt ends Answer:.',
                  screen='Only unedited scores. Select all rows with four-choice correct and 0 < correct-minus-best-other <= 1.0 BF16 logit, before viewing intervention results. Report <=0.5 and <=0.25 descriptive. Full-vocabulary top1 separate.',
                  arms='At fixed H16 and layers 23–27, answer row only. Mask key, cyclic adjacent, or all question-suffix tokens. Reopen key or adjacent only. Independently zero V on entire suffix and restore key/adjacent/all. Same prompt and suffix positions throughout.',
                  restoration='Removing a single mask is base conditions restored; test selective rescue using full-suffix disruption and reopen one token, compared with reopening adjacent. V all restored must equal own-native V replay.',
                  analysis='Paired family bootstrap. Direct comparison key vs adjacent and selective recovery under full disruption. Also report subset all native shams agree with baseline, with no interfered outcomes used for selection.')
    io.save('design.json',design)
    print('frozen',len(rows),'prompts',FAMILIES,'families',flush=True)


def batches(rows):
    return [rows[i:i+CHUNK] for i in range(0,len(rows),CHUNK)]


def make_arms(rows):
    own = list(range(len(rows)))
    for r in rows:
        suffix = r['suffix']
        key = r['key']
        adj = r['adjacent']
        r['groups'] = dict(key=[key],adjacent=[adj],full=suffix,names=r['name_slots'],
                           except_key=[k for k in suffix if k!=key],except_adjacent=[k for k in suffix if k!=adj])
    actions = dict(mask_key=[('mask','key')],mask_adjacent=[('mask','adjacent')],mask_all=[('mask','full')],
                   reopen_key=[('mask','except_key')],reopen_adjacent=[('mask','except_adjacent')],
                   zero_all=[('zero','full')],restore_key=[('zero','full'),('own_v','key')],
                   restore_adjacent=[('zero','full'),('own_v','adjacent')],restore_all=[('zero','full'),('own_v','full')],
                   self_all=[('own_v','full')],self=[('kv','names')])
    return {name:dict(donor=own,heads=H16,actions=action) for name,action in actions.items()}


def run(mode):
    from source_binding_transfer_factor_engine import run as base_run
    from source_binding_runtime import Runtime
    from source_binding_qkv_suffix_token_engine import run as intervention_run
    import torch
    import transformers

    design = io.read('design.json')
    for file,digest in design['code'].items():
        assert sha(Path(__file__).with_name(file))==digest,file
    assert sha(OUT/'materials.json')==design['materials_sha256']
    assert sha(token_test.OUT/'selected.json')==design['selected_sha256']
    rows = io.read('materials.json')
    chunks = batches(rows)
    screen = None
    if mode=='test':
        assert io.read('screen-manifest.json')['passed']
        screen = io.read('selected.json')
        assert screen['manifest_sha256']==sha(OUT/'screen-manifest.json')
        wanted = set(screen['chunks'])
    else:
        wanted = set(range(len(chunks)))
    planned = sorted(wanted)
    size = SCREEN_GROUP if mode=='screen' else 4
    paths = []
    print(mode,'torch',torch.__version__,'transformers',transformers.__version__,'chunks',len(wanted),flush=True)
    for i in range(0,len(planned),size):
        indices = planned[i:i+size]
        name = f'{mode}-{i//size:02d}.json'
        if not (OUT/name).exists():
            args = ([chunks[j] for j in indices],)
            check = lambda:io.check_write(OUT,1<<28,operation='qkv-low-margin')
            if mode=='screen':
                results,meta = base_run(*args,Runtime,None,name,OUT,check)
            else:
                results,meta = intervention_run(*args,make_arms,name,OUT,check)
                assert all(g['restore_all']==g['self_all'] for g in results)
            for j,g in zip(indices,results,strict=True):
                assert [r['id'] for r in chunks[j]]==[r['id'] for r in g['base']]
            io.save(name,dict(indices=indices,items=results,meta=meta))
        paths.append(dict(name=name,sha256=sha(OUT/name)))
    io.save(f'{mode}-manifest.json',dict(passed=True,shards=paths,design_sha256=sha(OUT/'design.json'),
        selected_sha256=sha(OUT/'selected.json') if screen else None))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('design','screen','test'))
    mode = p.parse_args().mode
    io.OUT = OUT
    io.setup()
    (materials if mode=='design' else lambda:run(mode))()


if __name__=='__main__':
    main()
