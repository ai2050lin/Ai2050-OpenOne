"""Semantic-scoring control after negative-question ambiguity was detected.

Preserves primary corpus and predictions. Reuses the same worlds, so this is a
within-world controlled diagnostic, not an independent replication. Runs 4B only.
"""
import argparse
import copy
import json
import re
from pathlib import Path
import numpy as np
import phase2752_context_interaction as capture
import phase2752_predict_interaction as prediction
import phase2752_source_ablation as ablation
import phase2752_quality_and_plots as quality

ORIGINAL=capture.OUT
CONTROL=ORIGINAL/'assertion_control'


def prepare_control():
    path=CONTROL/'material.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    from transformers import AutoTokenizer
    original=json.loads((ORIGINAL/'material.json').read_text(encoding='utf-8'))
    material=copy.deepcopy(original)
    tokenizers={s:AutoTokenizer.from_pretrained(capture.ROOT/'models/hf'/m,local_files_only=True) for s,m in [('4B','qwen3-4b'),('14B','Qwen3-14B')]}
    starts=['Is it ', 'Would it ', 'According to these statements, is it ', 'Can we ']
    middles=[' true that ', ' be correct to say that ', ' true that ', ' conclude that ']
    for r in material['rows']:
        head,question=r['text'].split('\nQuestion: ',1)
        question=question.split('\n')[0]
        marker='not' if r['negation'] else 'really'
        prefix=starts[r['wording']]+marker+middles[r['wording']]
        assert question.startswith(prefix) and question.endswith('?')
        proposition=question[len(prefix):-1]
        r['text']=head+'\nStatement: It is '+marker+' the case that '+proposition+'.\nReturn yes if and only if the entire statement is true; otherwise return no.\nAnswer:'
        r['control_origin']=r['id']
        for side,tok in tokenizers.items():
            ids=tok.encode(r['text'],add_special_tokens=False)
            r['tokenization'][side]=dict(token_ids=ids,length=len(ids),question_start=len(tok.encode(head+'\nStatement: ',add_special_tokens=False)))
    groups={}
    for r in material['rows']:
        groups.setdefault(r['group'],[]).append(r)
    for side in tokenizers:
        assert all(len({r['tokenization'][side]['length'] for r in rr})==1 for rr in groups.values())
    material['created_utc']=capture.now()
    material['control_scope']='Same worlds/splits/conditions; replace ambiguous negative polar questions by explicit true/false assertion classification. Generator truth convention stated in prompt.'
    capture.write(path,material)
    design=json.loads((ORIGINAL/'design.json').read_text(encoding='utf-8'))
    design.update(created_utc=capture.now(),material_sha256=capture.sha(path),
        status='Diagnostic specified after primary4B outcomes, before this control capture; no independent-replication claim',
        resource='4B only,3328 prompts, same native precision/configuration and full procedure; after14B model has been released',
        reason='Negative yes/no questions are pragmatically ambiguous; primary accuracy cannot establish propositional-negation reasoning.',
        feature_change='Question-modal feature and its family products fixed0 because every query is now a declarative assertion. Fact voice/order/context metadata unchanged.')
    capture.write(CONTROL/'design.json',design)
    capture.write(CONTROL/'pre_capture_seal.json',dict(created_utc=capture.now(),material_sha256=capture.sha(path),
        files=[capture.snapshot(Path(__file__)),capture.snapshot(Path(capture.__file__)),capture.snapshot(Path(prediction.__file__))]))
    return material


def control_features(rows,side):
    graph,surface=original_features(rows,side)
    graph[:,7]=0  # four family flags, then role/order/depth/modal/...
    for family in range(4):
        graph[:,12+5*family+3]=0
    return graph,surface


original_features=prediction.features


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','collect','analyze'])
    a=p.parse_args()
    for module in (capture,prediction,ablation,quality):
        module.OUT=CONTROL
    material=prepare_control()
    capture.prepare=lambda:material
    prediction.features=control_features
    if a.mode=='collect':
        capture.collect('4B',pilot=True)
        capture.collect('4B')
    elif a.mode=='analyze':
        prediction.run('4B')
        ablation.run('4B')
        quality.audit('4B')
        quality.plots('4B')
