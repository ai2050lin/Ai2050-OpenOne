"""Choose challenge material on the pilot, then audit new heldout generation."""
import argparse
from collections import defaultdict

import numpy as np
import torch

import mlp_kv_common as C
from mlp_kv_materials import CUT, PROFILES, WORDS
from mlp_fidelity_v2_collect_v4 import greedy, first_word
from mlp_functional_trace import load_model
from shared.dist_protocol.mechanism_trace import GpuLease


def frozen():
    corpus=C.read(C.OUT/'corpus.json');plan=C.read(C.OUT/'prereg.json')
    assert C.digest(corpus)==plan['corpus_sha256']
    return corpus['rows'],plan


def word_indices(tokenizer):
    result={w:sorted({tokenizer.encode(v,add_special_tokens=False)[0] for v in [w,w.capitalize(),' '+w,' '+w.capitalize()]}) for w in WORDS}
    assert set(result[WORDS[0]]).isdisjoint(result[WORDS[1]])
    return result


def score(row, logits, generated, tokenizer, word_ids):
    text=tokenizer.decode(generated,skip_special_tokens=True);word=first_word(text)
    good=logits[word_ids[row['expected']]].max();bad=logits[word_ids[row['contrast']]].max()
    return {'text':text,'word':word,'correct':word==row['expected'],'tokens':generated,'gap':float(good-bad)}


def by_cell(rows, fields):
    cells=defaultdict(list)
    for row in rows:cells[tuple(row[f] for f in fields)].append(row['behavior']['correct'])
    return [dict(zip(fields,k))|{'n':len(v),'correct':sum(v),'accuracy':float(np.mean(v))} for k,v in sorted(cells.items())]


def evaluate(part):
    output=C.OUT/f'{part}-behavior.json'
    if output.exists():return
    rows,plan=frozen();selected=None
    if part=='formal':selected=C.read(C.OUT/'pilot-behavior.json')['selected_profile']
    items=[r for r in rows if (r['split']=='pilot' if part=='pilot' else r['split']!='pilot' and r['profile']==selected)]
    freeze={**C.identity(),'schema':'mlp_kv_behavior_execution.v1','part':part,'n':len(items),'profile':selected,
            'source_sha256':C.file_sha(__file__),'prereg_sha256':C.file_sha(C.OUT/'prereg.json'),'layout':'prefix+cached suffix for all cases'}
    C.save(C.OUT/f'{part}-behavior-prereg.json',freeze)
    outcomes=[];parity=[];torch.set_num_threads(4)
    with GpuLease(),torch.inference_mode():
        model,tok,eos=load_model();words=word_indices(tok)
        for lo in range(0,len(items),4):
            batch=items[lo:lo+4];x=torch.tensor([r['tokens'] for r in batch],device='cuda')
            past=model(input_ids=x[:,:CUT],use_cache=True,logits_to_keep=1).past_key_values
            suffix=model(input_ids=x[:,CUT:],past_key_values=past,use_cache=True,logits_to_keep=1)
            logits=suffix.logits[:,-1].float()
            if part=='pilot' and lo<16:
                whole=model(input_ids=x,use_cache=False,logits_to_keep=1).logits[:,-1].float()
                lp=whole.log_softmax(-1);lq=logits.log_softmax(-1)
                parity.append({'case_ids':[r['id'] for r in batch],'bitwise_equal':torch.equal(whole,logits),
                               'max_abs_logit_difference':float((whole-logits).abs().max()),
                               'KL_whole_to_segmented':(lp.exp()*(lp-lq)).sum(-1).cpu().tolist(),
                               'argmax_equal':(whole.argmax(-1)==logits.argmax(-1)).cpu().tolist()})
            generated=greedy(model,suffix.past_key_values,logits,eos,limit=4)
            for j,row in enumerate(batch):outcomes.append({k:v for k,v in row.items() if k!='tokens'}|{'behavior':score(row,logits[j],generated[j],tok,words)})
            if (lo+len(batch))%64==0:print(part,lo+len(batch),'/',len(items),flush=True)
    if part=='pilot':
        profiles={}
        for name in PROFILES:
            subset=[r for r in outcomes if r['profile']==name]
            cells=by_cell(subset,['frame','expected'])
            margins=[r['behavior']['gap'] for r in subset];correct=[r['behavior']['correct'] for r in subset]
            profiles[name]={'n':len(subset),'accuracy':float(np.mean(correct)),'cells':cells,'median_gap':float(np.median(margins)),
                            'low_margin_correct':sum(0<v<=8 and c for v,c in zip(margins,correct)),
                            'eligible':np.mean(correct)>=.8 and min(c['accuracy'] for c in cells)>=.65}
            profiles[name]['eligible']=bool(profiles[name]['eligible'])
        eligible=[n for n in PROFILES[1:] if profiles[n]['eligible']]
        selected=min(eligible,key=lambda n:(-profiles[n]['low_margin_correct'],profiles[n]['median_gap'],PROFILES.index(n))) if eligible else 'direct'
        extra={'profiles':profiles,'selected_profile':selected,'challenge_calibration_available':bool(eligible),'layout_parity':parity}
        print('pilot selected',selected,{k:(v['accuracy'],v['median_gap'],v['low_margin_correct']) for k,v in profiles.items()},flush=True)
    else:
        cells=by_cell(outcomes,['split','frame','expected']);accuracy=float(np.mean([r['behavior']['correct'] for r in outcomes]))
        assert all(c['n']>=30 for c in cells)
        extra={'cells':cells,'accuracy':accuracy,'passed':accuracy>=.80 and min(c['accuracy'] for c in cells)>=.65,
               'median_gap':float(np.median([r['behavior']['gap'] for r in outcomes]))}
        print('formal gate',extra['passed'],'accuracy',accuracy,'median gap',extra['median_gap'],flush=True)
    C.save(output,{**freeze,'status':'complete',**extra,'rows':outcomes})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=['pilot','formal']);evaluate(p.parse_args().part)
