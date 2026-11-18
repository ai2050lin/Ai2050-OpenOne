"""Descriptive audit of frozen B28–B35 keyword candidates on the original bank."""
from collections import defaultdict, Counter
from pathlib import Path
from time import monotonic

import numpy as np
from tokenizers import Tokenizer

import mlp_tail_goodfire as C
from mlp_tail_extension import OUT as EARLY
from mlp_tail_finalize import words

OUT = C.ROOT / 'tests/glm5/result/mlp_b31_b32_authority_v1'
SEED = 20261109031


def run(prov):
    rng = np.random.default_rng(SEED)
    started = monotonic()
    base = C.read(C.OUT / 'documents.json')
    tokens = np.load(C.TOKENS, mmap_mode='r')
    tok = Tokenizer.from_file(str(C.OUT / 'tokenizer.json'))
    positions = base['positions']
    windows = [words(tok.decode(tokens[i,p-128:p+1].tolist())) for i in range(2048) for p in positions]
    current = tokens[:, positions].reshape(-1)
    source = [doc['source'] for doc in base['docs'] for p in positions]
    holding = slice(24576,32768)
    by_key = defaultdict(list)
    for i in range(holding.start, holding.stop):
        by_key[source[i],int(current[i])].append(i)
    matched = {}
    for i in range(holding.start,holding.stop):
        alternatives = [j for j in by_key[source[i],int(current[i])] if j//16!=i//16]
        if alternatives:
            matched[i] = int(rng.choice(alternatives))
    index = defaultdict(set)
    for i in range(holding.start, holding.stop):
        for term in windows[i]:
            index[term].add(i)
    report = []
    witnesses = []
    for b in range(28,36):
        path = EARLY if b<32 else C.OUT
        profile = C.read(path / f'B{b}-profile.json')
        cards = C.read(path / f'B{b}-cues.json')['units']
        g = profile['fields']['G']
        pos = g['holdout']['positive_indices']
        a = profile['fields']['A']['holdout']['positive_indices']
        raw = np.load(path/f'B{b}-G.npy',mmap_mode='r')[holding]
        native = np.load(path/f'B{b}-A.npy',mmap_mode='r')[holding]
        sums = Counter()
        for start in range(0,C.WIDTH,256):
            end = min(C.WIDTH,start+256)
            x = C.unbits(raw[:,start:end])
            y = C.unbits(native[:,start:end])
            top64 = np.argpartition(x,-64,axis=0)[-64:] + holding.start
            bydoc = x.reshape(512,16,-1)
            maxima = bydoc.max(1)
            best_docs = np.argpartition(maxima,-8,axis=0)[-8:]
            columns = np.arange(end-start)
            distinct8 = (best_docs*16 + bydoc.argmax(1)[best_docs,columns] + holding.start)
            for j,u in enumerate(range(start,end)):
                cue = set(cards[u]['positive_keywords'])
                if not cue:
                    continue
                available = set().union(*(index[t] for t in cue))
                real = sum(i in available for i in pos[u])
                assert real == cards[u]['repeat']['holdout']['cue_hit_count'], (b,u)
                broadened = any(int(i) in available for i in top64[:,j])
                distinct = any(int(i) in available for i in distinct8[:,j])
                a_hit = any(i in available for i in a[u])
                sums.update(units=1,zero=int(real==0),absent_everywhere=int(not available),
                    zero_available=int(real==0 and bool(available)),
                    zero_rescued_top64=int(real==0 and broadened),zero_rescued_distinct8=int(real==0 and distinct),
                    zero_rescued_a8=int(real==0 and a_hit),
                    discovery_hits=sum(bool(cue.intersection(windows[i])) for i in g['discovery']['positive_indices'][u]),
                    holdout_hits=real, holdout_unique_docs=len({i//16 for i in pos[u]}),
                    repeated_current_tokens=8-len({int(current[i]) for i in pos[u]}))
                for i in pos[u]:
                    if i in matched:
                        sums.update(matched_n=1, matched_true_hits=int(i in available),
                                    matched_control_hits=int(matched[i] in available))
                # Signed G/A agreement on the same extreme G positions.
                loc = np.asarray(pos[u])-holding.start
                sums['g_positive_a_nonpositive'] += int(np.count_nonzero((x[loc,j]>0)&(y[loc,j]<=0)))
                if u in (0,9727):
                    witnesses.append({'block':b,'unit':u,'keywords':sorted(cue),'hits_top8':real,'hits_anywhere':len(available),
                                      'hit_top64':broadened,'hit_distinct8':distinct,'hit_A8':a_hit})
            if (start+256) % 2048 == 0:
                C.log('keyword_audit',(b-28)*C.WIDTH+end,8*C.WIDTH,started,block=b)
        report.append({'block':b,**sums})
        C.log('keyword_audit',(b-27)*C.WIDTH,8*C.WIDTH,started,block=b)
    C.save_json(OUT/'keyword-audit.json',{'kind':'real','schema':'mlp_tail_keyword_audit.v1','provenance':prov,
        'scope':'原有文本库的回顾性观察。逐层重算既有指标，对照取同来源、同当前 token、不同文档的确定性随机位置。Top64 与换用 A 为诊断，暴露量与指标口径不同。',
        'layers':report,'witnesses':witnesses,'seed':SEED})


if __name__=='__main__':
    run(C.read(OUT/'prereg.json')['provenance'])
