"""Post-primary diagnostic: match token lengths for style and negation contrasts.

These are NEW diagnostic controls, not a retroactive blind confirmation. They reuse
the original semantic sources with changed phrasing; they do not add independent
semantic samples. GPU execution must follow completion of other model arms.
"""
import argparse
import json
import re
from datetime import datetime,timezone
from transformers import AutoTokenizer
import phase2751_trusted_rebuild as base


def prepare():
    out=base.OUT/'length_control';out.mkdir(exist_ok=True)
    p=out/'material.json'
    if p.exists():return out,json.loads(p.read_text(encoding='utf-8'))
    original=json.loads((base.OUT/'material.json').read_text(encoding='utf-8'))
    tok=AutoTokenizer.from_pretrained(base.ROOT/'models/hf/qwen3-4b',local_files_only=True)
    rows=[]
    names=['Arin','Bela','Ciro','Dena','Eron','Fara','Galen','Hana',
           'Iven','Jora','Kelan','Luma','Miro','Nera','Orin','Pela']
    for r in original['rows']:
        if r['split']=='legacy':continue
        r=dict(r);text=r['text']
        # Repair an independently detected corpus limitation: the primary corpus
        # held out focal entities, but some appeared as partner entities in training.
        # Keep both participants within the same train/heldout entity cohort here.
        i=r['entity_index'];old_other=names[(i+5)%16];new_other=names[(i//8)*8+(i%8+5)%8]
        if r['family'] in ('role','chain'):
            text=re.sub(r'\b'+re.escape(old_other)+r'\b',new_other,text)
        if r['style']==0:text='Use an ordinary tone. '+text
        if r['negation']==0:
            text,n=re.subn(r'(Question: |answer this question: )(Is|Was) (\w+) ',r'\1\2 \3 really ',text)
            assert n==1,(r['id'],text)
        r['id']='length_'+r['id'];r['text']=text;r['token_count']=len(tok(text,add_special_tokens=False)['input_ids'])
        rows.append(r)
    lengths={}
    for r in rows:lengths.setdefault(r['group'],set()).add(r['token_count'])
    assert all(len(v)==1 for v in lengths.values()),{k:v for k,v in lengths.items() if len(v)!=1}
    doc=dict(created_utc=datetime.now(timezone.utc).isoformat(),rows=rows,
        status='new post-primary diagnostic; not independent semantic sources',
        manipulation='style ordinary vs formal; question really vs not; all4 conditions same tokenizer length per group; partner entities kept within entity split',
        limitation='really/not also changes pragmatics; matching token count does not remove every lexical/confounding effect.',
        primary_source_sha256=base.sha(base.OUT/'material.json'),n_source_groups=48,n_wording_groups=96)
    base.write(p,doc)
    base.write(out/'preregistered_design.json',dict(created_utc=doc['created_utc'],scope=doc['status'],
        material_sha256=base.sha(p),primary='Compare additive vs negation-only and development interaction transfer under equal lengths.',
        adaptation='Designed after 4B primary findings, before any control-model outputs; includes partner-entity leakage repair. Do not pool as independent confirmation.',
        same_length_all_groups=True,n_prompts=len(rows),model='4B',budget='same collector 1800s cap; expected <2minutes'))
    return out,doc


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','collect','analyze']);a=p.parse_args()
    out,material=prepare()
    if a.action=='prepare':print(json.dumps(dict(n=len(material['rows']),all_group_lengths_equal=True)))
    else:
        base.OUT=out;base.prepare=lambda:material
        if a.action=='collect':base.collect('4B')
        else:base.analyze('4B')
