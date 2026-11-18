"""Larger matched replication of the two authorization settings that failed Phase 2959."""

from collections import Counter
from itertools import batched, combinations
from random import Random

import record_authorization_material as base


M = base.M
RUN = 'RECORD-AUTHORIZATION-RETEST-20261117-v1'
SEED = 29602026
SIZES = dict(pilot=256,discovery=192,confirm=384,lockbox=384)
LEXICONS = dict(original=('refused','granted'),synonym=('blocked','allowed'))
SETTINGS = ((False,True),(True,False))  # Earlier request targets East, old label has code Q.
FRAME = 0


def pools(tokenizer):
    old_pools = base.pools(tokenizer)
    former = ' '.join((*M.VOCABULARY.values(),base.PRIOR_V2,base.PRIOR_V3)).split()
    used = {tokenizer.encode(' '+word,add_special_tokens=False)[0]
            for word in (*former,*(word for pool in old_pools.values() for word in pool))}
    available = []
    for word in base.WORDS.split():
        ids = tokenizer.encode(' '+word,add_special_tokens=False)
        if len(ids)==1 and ids[0] not in used and tokenizer.decode(ids)==' '+word:
            used.add(ids[0])
            available.append(word)
    Random(SEED).shuffle(available)
    result = {}
    offset = 0
    for panel,size in dict(pilot=40,discovery=38,confirm=56,lockbox=48).items():
        assert offset+size <= len(available)
        result[panel] = available[offset:offset+size]
        offset += size
    return result


def families(tokenizer,split,lexicon):
    words = pools(tokenizer)[split]
    label_pairs = list(combinations(words,2))
    Random(M.digest([SEED,split,'unordered-label-pairs'])).shuffle(label_pairs)
    assert len(label_pairs)>=SIZES[split]
    for i in range(SIZES[split]):
        rng = Random(M.digest([SEED,split,i]))
        old,proposal = rng.sample(label_pairs[i],2)
        other,earlier = rng.sample([w for w in words if w not in (old,proposal)],2)
        names = (old,other,earlier,proposal)
        earlier_target,reverse = SETTINGS[i%2]
        proposal_first = bool((i//2)%2)
        codes = ('Q','P','Q','P') if reverse else ('P','Q','P','Q')
        order = rng.sample(range(4),4)
        if (order.index(3)<order.index(0))!=proposal_first:
            order = [3 if j==0 else 0 if j==3 else j for j in order]
        key = ', '.join(f'{names[j]} -> {codes[j]}' for j in order)
        family = M.digest([RUN,split,i,names])[:20]
        for update in ((False,True) if (i//4)%2==0 else (True,False)):
            status = LEXICONS[lexicon][update]
            index = 3 if update else 0
            yield dict(id=f'{family}-{int(update)}',family=family,split=split,frame=FRAME,
                       names=names,codes=codes,actions=(1 if earlier_target else 3,0 if update else 1),
                       truth=[old,old,names[index]],answer=codes[index],active=0,update=update,
                       setting=[earlier_target,reverse],status=status,lexicon=lexicon,proposal_first=proposal_first,
                       sections=base.render(FRAME,names,earlier_target,status,key))


def rows(tokenizer,split,lexicon):
    return [base.encode(tokenizer,r) for r in families(tokenizer,split,lexicon)]


def verify_cue_only(a,b):
    x,y = a['token'],b['token']
    assert a['family']==b['family'] and a['setting']==b['setting'] and a['names']==b['names'] and a['codes']==b['codes']
    assert a['sections'][:3]==b['sections'][:3] and a['sections'][4:]==b['sections'][4:]
    assert len(a['sections'][3])==len(b['sections'][3])
    assert a['sections'][3].replace(a['status'],b['status'])==b['sections'][3]
    assert x['sections']==y['sections'] and x['labels']==y['labels']
    assert x['cue']==y['cue'] and x['proposal']==y['proposal'] and x['e']==y['e']
    diff = [i for i,(left,right) in enumerate(zip(x['prompt'],y['prompt'],strict=True)) if left!=right]
    assert diff==[x['cue']]


def inspect(tokenizer):
    summary = {arm:{} for arm in LEXICONS}
    ids = {status:tokenizer.encode(' '+status,add_special_tokens=False) for words in LEXICONS.values() for status in words}
    assert len(set(tuple(x) for x in ids.values()))==4 and all(len(x)==1 for x in ids.values())
    identities = set()
    for panel,size in SIZES.items():
        by_arm = {arm:rows(tokenizer,panel,arm) for arm in LEXICONS}
        for original,synonym in zip(*by_arm.values(),strict=True):
            assert original['id']==synonym['id'] and original['truth']==synonym['truth'] and original['answer']==synonym['answer']
            verify_cue_only(original,synonym)
        for arm,group in by_arm.items():
            assert len(group)==2*size
            for a,b in batched(group,2):
                assert a['update']!=b['update'] and a['answer']!=b['answer']
                verify_cue_only(a,b)
                assert a['token']['draft'][-1]==b['token']['draft'][-1]==tokenizer.eos_token_id
                if arm=='original':
                    identity = tuple(sorted((a['names'][0],a['names'][3])))
                    assert identity not in identities
                    identities.add(identity)
            parts = Counter((r['setting'][0],r['proposal_first']) for r in group if r['update'])
            assert len(parts)==4 and set(parts.values())=={size//4}
            summary[arm][panel] = dict(families=size,views=len(group),sha256=M.digest(group),
                                      families_per_setting=size//2,families_per_key_order=size//4,
                                      min_prompt=min(len(r['token']['prompt']) for r in group),
                                      max_prompt=max(len(r['token']['prompt']) for r in group))
    return summary,{status:tokens[0] for status,tokens in ids.items()}
