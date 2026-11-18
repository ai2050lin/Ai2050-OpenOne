"""Fixed prompts with independently movable six-token archive and five-token request."""

from itertools import product
from random import Random

import paradigm_four_material as F
import paradigm_path_material as P
import paradigm_archive_material as A
from paradigm_routes_material import digest


RUN='PARADIGM-POSITION-20261118-v1'
SEED=29742026
SIZES=dict(pilot=16,discovery=32,confirm=128,lockbox=128)
NOUNS=dict(pilot='saucers needles anvils couplers panes baffles links anchors',
    discovery='lathes cartons grates domes bales planks paddles fabric locks castings helmets spokes',
    confirm='joists funnels sheaves struts forks tins ropes pegs awnings collars chains rafters blocks drills cages wrenches',
    lockbox='sails flanges staves chutes skids kilns pens brushes grinders chandeliers hammers mounts nests totems horns moldings')
TARGETS=(11,15)
LAYOUTS=tuple(product(range(2),repeat=2))
ARCHIVES=(('| We noted four charts.','| print("four charts")'),
          ('| We confirm six images.','| print("six images")'))
EMPTY_A='| No entry was kept.'
FORM=('Write in academic form.','Write in Python form.')
EMPTY_R='No form is given.'
ANCHORS=('after_a0','after_r0','fact0','fact1','after_a1','after_r1','language','rule','grammar','closing')
REGION_IDS=(0,0,1,1,2,2,3,3,3,4)
CAUSAL=64
FUSION=64


def rows_for(tokenizer,split):
    entries=[(w,a,b) for w,a,b in product(NOUNS[split].split(),range(2,10),range(2,10)) if a!=b]
    Random(SEED+list(SIZES).index(split)).shuffle(entries)
    groups=[]
    for n,(noun,early,late) in enumerate(entries[:SIZES[split]]):
        family=digest([RUN,split,noun,early,late])[:20]
        bank=(n//2)%2
        groups.append(dict(family=family,noun=noun,early=early,late=late,bank=bank))
    return groups


def compile_case(tokenizer,family,cell,layout,variant):
    a,r=LAYOUTS[layout]
    lang,style,logic,grammar=F.LEVELS[cell]
    values={
        'a0':ARCHIVES[family['bank']][variant] if a==0 else EMPTY_A,
        'a1':ARCHIVES[family['bank']][variant] if a else EMPTY_A,
        'r0':FORM[style] if r==0 else EMPTY_R,
        'r1':FORM[style] if r else EMPTY_R,
        'fact0':f"The first inspection counted {family['early']} {family['noun']}.",
        'fact1':f"The revised inspection counted {family['late']} {family['noun']}.",
        'language':('English','Chinese')[lang],
        'rule':('initial','revised')[logic],
        'grammar':('round','square')[grammar],
    }
    pieces=(
        ('Archive slot: ','a0'),('\nForm slot: ','r0'),('\n','fact0'),('\n','fact1'),
        ('\nArchive slot: ','a1'),('\nForm slot: ','r1'),('\nAnswer language: ','language'),
        ('.\nUse the ','rule'),(' count and put the number inside ','grammar'))
    body=''; spans={}
    for prior,key in pieces:
        body+=prior
        spans[key]=(len(body),len(body)+len(values[key]))
        body+=values[key]
    body+=' brackets.\n'+F.GUIDE
    prompt=tokenizer.apply_chat_template([dict(role='user',content=body)],tokenize=False,
                                         add_generation_prompt=True,enable_thinking=False)
    enc=tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True)
    cut=prompt.index(body)
    ids=list(enc.input_ids)
    token_spans={key:[j for j,(x,y) in enumerate(enc.offset_mapping) if x<cut+hi and y>cut+lo]
                 for key,(lo,hi) in spans.items()}
    assert [len(token_spans[k]) for k in ('a0','a1','r0','r1')]==[6,6,5,5],(values,token_spans)
    prefix=prompt+('print("' if style else '')
    code=F.encode(tokenizer,prefix)
    prefix+=('The reported count is','报告的计数为')[lang]
    core=F.encode(tokenizer,prefix)
    endings=[F.encode(tokenizer,prefix+b) for b in (' (',' [')]
    assert all(e[:-1]==core for e in endings)
    gold=family[('early','late')[logic]]
    foil=family[('late','early')[logic]]
    decorated=prefix+(' (',' [')[grammar]
    targets=[F.encode(tokenizer,decorated+str(v)) for v in (gold,foil)]
    assert all(z[:-1]==endings[grammar] for z in targets) and targets[0][-1]!=targets[1][-1]
    assert code[:len(ids)]==ids and core[:len(code)]==code and len(endings[grammar])<=F.WIDTH
    anchors=[token_spans[k][-1]+1 if k in ('a0','a1','r0','r1') else token_spans[k][-1]
             for k in ('a0','r0','fact0','fact1','a1','r1','language','rule','grammar')]
    anchors.append(len(ids)-4)
    assert len(set(anchors))==len(anchors) and max(anchors)<len(ids)-1
    source=token_spans['a'+str(a)]
    return dict(family,id=f"{family['family']}-{cell:04b}-l{layout}-a{variant}",cell=cell,layout=layout,
        dims=[lang,style,logic,grammar],archive=variant,prompt=prompt,spans=token_spans,source=source,
        sites=anchors,token=dict(ids=endings[grammar],slots=[len(endings[grammar])-1]),
        query=[len(ids)-1,len(code)-1,len(core)-1,len(endings[grammar])-1],
        delimiters=[z[-1] for z in endings],number=[z[-1] for z in targets],gold=gold)


def verify_group(rows):
    common=rows[0]
    for row in rows:
        assert (row['spans'],row['sites'],row['query'])==(common['spans'],common['sites'],common['query'])
    table={(r['layout'],r['archive']):r for r in rows}
    for layout,v in table:
        a,r=LAYOUTS[layout]
        def compare(source,other,allowed):
            changed={j for j,(x,y) in enumerate(zip(source['token']['ids'],other['token']['ids'],strict=True)) if x!=y}
            assert changed and changed<=set(allowed)
            return changed
        here=table[(layout,v)]
        same_archive=table[(layout,1-v)]
        compare(here,same_archive,here['source'])
        other=table[(layout^2,v)]
        aa=here['spans']['a0']; ab=here['spans']['a1']
        assert compare(here,other,aa+ab)&set(aa) and compare(here,other,aa+ab)&set(ab)
        other=table[(layout^1,v)]
        ra=here['spans']['r0']; rb=here['spans']['r1']
        assert compare(here,other,ra+rb)&set(ra) and compare(here,other,ra+rb)&set(rb)
        for other,keys in ((table[(layout^2,v)],('a0','a1')),(table[(layout^1,v)],('r0','r1'))):
            left,right=(here['spans'][k] for k in keys)
            ids=here['token']['ids']; alt=other['token']['ids']
            assert [ids[i] for i in left]==[alt[i] for i in right]
            assert [ids[i] for i in right]==[alt[i] for i in left]


def material(tokenizer,split):
    groups=rows_for(tokenizer,split)
    sweep=[];fusion=[]
    for n,group in enumerate(groups):
        for cell in TARGETS:
            arms=[compile_case(tokenizer,group,cell,layout,arc) for layout,arc in product(range(4),range(2))]
            verify_group(arms)
            sweep.extend(arms)
        if split=='pilot' or (split=='confirm' and n<FUSION):
            layout=n%4
            arc=(n//4)%2
            fusion.extend(compile_case(tokenizer,group,cell,layout,arc) for cell in range(16))
    return sweep,fusion


def inventory(tokenizer):
    words=' '.join(NOUNS.values()).split()
    prior=' '.join([*F.OBJECTS.values(),*P.NOUNS.values(),*A.NOUNS.values()]).split()
    assert len(words)==len(set(words)) and not set(words)&set(prior),set(words)&set(prior)
    result={}
    for split in SIZES:
        data=material(tokenizer,split)
        result[split]=dict(families=SIZES[split],views=list(map(len,data)),identity=digest(data))
    return result
