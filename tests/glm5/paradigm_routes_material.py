"""Shared-prefix academic/Python source competition, crossed with output language."""

from itertools import product
from random import Random
from hashlib import sha256
import json


RUN = 'PARADIGM-ROUTES-20261117-v1'
SEED = 29642026
SIZES = dict(pilot=16,discovery=48,confirm=64,lockbox=128)
LEVELS = tuple(product(range(2),repeat=3))  # Requested style, reference style, answer language.
STYLE = ('academic','Python')
LANGUAGE = ('English','Chinese')
REFS = (('Results indicate six observations.', 'print("six observations")'),
        ('The sample contains eight records.', 'print("eight distinct records")'),
        ('Observed count equals nine.', 'print("nine distinct events")'),
        ('Evidence indicates five events.', 'print("five events")'))
FRAMES = (
    ('Format requested: {style}. Output language: {language}.',
     'The reference is from a different task; the requested format governs your output.'),
    ('Output language: {language}. Required format: {style}.',
     'Use the indicated format. Treat the historical reference as background.'),
    ('Target language: {language}. Export format: {style}.',
     'The template example may have a different format. Honor the export specification.'),
    ('Rendering format: {style}. Response language: {language}.',
     'Follow the rendering setting even when the archived example uses another format.'),
)
TOPICS = ('sensors','receivers','samples','articles','sectors','bundles','records','stations',
          'parcels','switches','buffers','folders','modules','fields','routes','panels',
          'files','signals','requests','tickets','channels','batches','cells','assets',
          'pages','blocks','lines','boxes','motors','trials','gates','assets')


def packed(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()


def digest(value):
    return sha256(packed(value)).hexdigest()


def alphabet(tokenizer):
    names=('print','The','Results','A','In','According','This','Based','结果','研究','本','该','共','数据显示','观察','实')
    ids={word:tokenizer.encode(word,add_special_tokens=False)[0] for word in names}
    assert tokenizer.encode('print',add_special_tokens=False)==[ids['print']]
    return ids


def selected_reference(tokenizer, left, right):
    """Freeze equal-token spans with the same suffix by using a length-matched archive prefix."""
    # Variants are semantically inert formatting identifiers for historical entries.
    tails = ('',' ','  ','   ')
    probes = []
    for example in (left,right):
        probes.append({len(tokenizer.encode(' '+example+tail+'\n',add_special_tokens=False)):example+tail for tail in reversed(tails)})
    common=sorted(set(probes[0]) & set(probes[1]))
    if not common:
        raise ValueError('No token-length matched archive variants')
    return tuple(options[common[0]] for options in probes)


def material(tokenizer,split):
    archive=[selected_reference(tokenizer,*pair) for pair in REFS]
    split_idx=list(SIZES).index(split)
    cursor=sum(list(SIZES.values())[:split_idx])
    facts=list(range(21,500))
    Random(SEED).shuffle(facts)
    for n in range(SIZES[split]):
        index=cursor+n
        frame=(n%2)+(2 if split in ('confirm','lockbox') else 0)
        context=FRAMES[frame]
        topic=TOPICS[(n+8*split_idx)%len(TOPICS)]
        first=facts[index]
        foil=1000-first
        fact=('The first inspection counted {a} {t}. A later correction counted {b} {t}.',
              'In the preliminary review there were {a} {t}. A verified recount found {b} {t}.',
              'The opening count registered {a} {t}. The confirmed tally recorded {b} {t}.',
              'Initially {a} {t} were reported. The revised count identified {b} {t}.')[frame].format(a=first,b=foil,t=topic)
        use_final=bool((n//2)%2)
        rule=('Report the later count as the result.' if use_final else 'Report the first count as the result.')
        example=archive[frame]
        family=digest([RUN,index,topic,first,foil])[:18]
        for target,reference,language in LEVELS:
            ask=context[0].format(style=STYLE[target],language=LANGUAGE[language])
            guide='In academic format write one formal sentence. In Python format use a bare print statement with a double-quoted message. In both formats put the message in the requested language. Begin directly with the answer, without labels or Markdown fences.'
            sections=(ask,context[1], 'Archived example: '+example[reference],fact,rule,guide)
            # Use a shared free-generation boundary. The conditional language readout consumes a specified Python subprefix only.
            plain='\n'.join(sections)
            prompt=tokenizer.apply_chat_template([dict(role='user',content=plain)],tokenize=False,
                                                 add_generation_prompt=True,enable_thinking=False)
            encoded=tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True)
            ids=encoded.input_ids
            def span(value,start):
                lo=prompt.index(value,start);hi=lo+len(value)
                return [i for i,(a,b) in enumerate(encoded.offset_mapping) if a<hi and b>lo]
            offset=prompt.index(plain)
            request=span(' '+STYLE[target],offset)
            language_span=span(' '+LANGUAGE[language],offset)
            source=span(' '+example[reference],prompt.index('Archived example:',offset))
            facts_span=span(fact,prompt.index('Archived example:',offset))
            grammar=span(guide,prompt.index(fact,offset))
            assert len(request)==len(language_span)==1 and max(grammar)<len(ids)-1
            assert len(set([*request,*language_span,*source,*facts_span,*grammar]))==sum(map(len,(request,language_span,source,facts_span,grammar)))
            prefix=tokenizer.encode(prompt+'print("',add_special_tokens=False)
            assert prefix[:len(ids)]==ids
            yield dict(id=f'{family}-{target}{reference}{language}',family=family,split=split,frame=frame,topic=topic,
                       regime=[target,reference,language],gold=foil if use_final else first,later=use_final,
                       spans=dict(request=request,reference=source,language=language_span,facts=facts_span,guide=grammar),
                       plain=plain,token=dict(ids=ids,slots=[len(ids)-1]),prefix_code=prefix)


def inspect(tokenizer):
    from itertools import batched

    summary={}
    spans=('request','reference','language')
    for split,size in SIZES.items():
        rows=list(material(tokenizer,split))
        assert len(rows)==8*size
        for group in batched(rows,8):
            base=group[0]
            for row in group:
                assert row['family']==base['family'] and row['token']['slots']==base['token']['slots']
                assert row['spans']==base['spans'] and row['gold']==base['gold']
                diff={i for i,(a,b) in enumerate(zip(base['token']['ids'],row['token']['ids'],strict=True)) if a!=b}
                allowed={i for axis,on in enumerate(row['regime']) if on for i in row['spans'][spans[axis]]}
                assert diff<=allowed and all(diff & set(row['spans'][spans[axis]]) for axis,on in enumerate(row['regime']) if on)
        assert all(len(r['prefix_code'])<=160 for r in rows)
        summary[split]=dict(n=len(rows),families=size,sha=digest(rows),frame=sorted({r['frame'] for r in rows}),
                            min_tokens=min(len(r['token']['ids']) for r in rows),max_tokens=max(len(r['prefix_code']) for r in rows))
    return summary
