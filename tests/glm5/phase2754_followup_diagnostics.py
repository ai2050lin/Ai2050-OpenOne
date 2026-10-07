"""Explicitly post-hoc operator readout and label-blind answer-prefix parsing."""
import argparse,json,re
from pathlib import Path
import numpy as np
from phase2754_relation_stability import OUT,ROOT,write,sha,now,snapshot

def answer(text):
    s=text.strip()
    s=re.sub(r'^(?:\*\*)?(?:final\s+answer|answer)\s*:(?:\*\*)?\s*','',s,flags=re.I)
    s=s.lstrip('*_`"\' ')
    direct=re.match(r'^(yes|no)\b',s,re.I)
    if direct:return direct.group(1).lower()
    boxed=re.match(r'^\\(?:\[|\()?\s*\\?boxed\{\s*(?:\\text\{)?(yes|no)\}\}?',s,re.I)
    if boxed:return boxed.group(1).lower()
    return None

def seal_parser():
    cases={'Answer: yes\nExplanation: ...':'yes',' **No**.':'no','yes indeed':'yes','Yesterday no':None,'We should say yes':None,'Answer: unknown':None,r'\boxed{no}':'no',r'\boxed{\text{yes}}':'yes'}
    for text,expected in cases.items():assert answer(text)==expected,(text,answer(text),expected)
    if (OUT/'answer_parser_design.json').exists():return
    write(OUT/'answer_parser_design.json',dict(created_utc=now(),source=snapshot(Path(__file__)),status='Post-hoc scoring diagnostic, preserves original literal-first-token metrics',
        rule='Only a leading direct yes/no or a leading explicit Answer:/Final answer:/boxed answer is recognized, allowing ordinary markup. No scanning explanations for convenient answer words.',
        tests=len(cases),censoring='Unrecognized short prefixes are unresolved, not proof of an incorrect eventual answer. Conditional accuracy among recognized prefixes is selection-biased.',
        finite_response='Strict completion and EOS retained separately from parsed answer prefix. Original instructions permit an explanation after the yes/no decision.'))

def parse_generation():
    seal_parser()
    records={}
    for prefix in ('generation','generation_chat','generation14B','generation14B_chat'):
        path=OUT/(prefix+'_rows.json')
        if not path.exists():continue
        rows=json.loads(path.read_text(encoding='utf-8'));pred=[answer(r['text']) for r in rows];recognized=[i for i,p in enumerate(pred) if p is not None]
        records[prefix]=dict(prompts=len(rows),recognized=len(recognized),coverage=len(recognized)/len(rows),correct_observed=sum(p==r['expected'] for p,r in zip(pred,rows))/len(rows),
            accuracy_among_recognized=sum(pred[i]==rows[i]['expected'] for i in recognized)/len(recognized) if recognized else None,
            unresolved=len(rows)-len(recognized),cases=[dict(id=r['id'],world=r['world'],text=r['text'],expected=r['expected'],parsed=p,eos=r['eos_seen']) for r,p in zip(rows,pred)])
    write(OUT/'parsed_generation_diagnostic.json',dict(created_utc=now(),status='Post-confirmation scoring diagnostic',parser_sha256=sha(OUT/'answer_parser_design.json'),results=records))
    print({k:{n:v[n] for n in ('prompts','coverage','correct_observed','accuracy_among_recognized')} for k,v in records.items()},flush=True)

def operator():
    sel=json.loads((OUT/'selection.json').read_text(encoding='utf-8'));method=sel['selected']['semantic'];reports={}
    for root,pred_file,material in [(OUT,'probe_predictions.npz',OUT/'material.json'),(OUT/'fact_order_challenge','predictions.npz',OUT/'fact_order_challenge/material.json')]:
        src={r['id']:r for r in json.loads(material.read_text(encoding='utf-8'))['rows']}
        if root==OUT:rows=[src[r['id']] for r in json.loads((OUT/'probe_rows.json').read_text(encoding='utf-8'))]
        else:
            meta=[r for p in sorted((root/'4B/confirmation').glob('chunk_*.json')) for r in json.loads(p.read_text(encoding='utf-8'))];rows=[src[r['id']] for r in meta]
        with np.load(root/pred_file) as z:predicate=np.where(z[method][:,0]>=0,1,-1)
        eta=np.array([-1 if 'It is not the case that ' in r['text'] else 1 for r in rows]);truth=np.array([r['truth_sign'] for r in rows]);pred=predicate*eta
        for split in sorted({r['split'] for r in rows}):
            ix=[i for i,r in enumerate(rows) if r['split']==split]
            reports[split]=dict(rows=len(ix),worlds=len({rows[i]['world'] for i in ix}),accuracy=float(np.mean(pred[ix]==truth[ix])))
    write(OUT/'factorized_readout_diagnostic.json',dict(created_utc=now(),source=snapshot(Path(__file__)),method=method,status='Post-hoc candidate; independent confirmation of this newly assembled readout is NOT claimed',
        rule='Decode predicate using frozen probe column0, then multiply by known negation sign parsed from the explicit sentence marker.',results=reports,
        limitations='Logical negation is supplied externally; this does not identify the original model internal negation algorithm. Materials are simple chains with supplied semantic query slots. No independent selection/confirmation cycle for this newly assembled rule.'))
    print(reports,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['seal','parse','operator']);a=p.parse_args();{'seal':seal_parser,'parse':parse_generation,'operator':operator}[a.mode]()
