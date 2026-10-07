"""Phase2753: remove late-state inputs, freeze new confirmation worlds.

One model, bounded4B protocol. Old assertion data are discovery/validation;
new confirmation responses must not be read before selection is sealed.
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import phase2752_context_interaction as prior

ROOT=prior.ROOT
OLD=ROOT/'tests/glm5/result/rdc_context_interaction_20260923/assertion_control'
OUT=ROOT/'tests/glm5/result/rdc_early_interaction_20260923'
FAMILIES=prior.FAMILIES
write,sha,now=prior.write,prior.sha,prior.now


def snapshot(path):
    dest=OUT/'code_snapshots'/(sha(path)+path.suffix)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():dest.write_bytes(path.read_bytes())
    assert sha(dest)==sha(path)
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path),snapshot=str(dest.relative_to(ROOT)))


def prepare():
    path=OUT/'material.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    from transformers import AutoTokenizer
    old=json.loads((OLD/'material.json').read_text(encoding='utf-8'))
    used={n for w in old['worlds'] for n in w['entities']}
    syllables=('ba','ce','di','fo','gu','ha','ji','ko','lu','me','ni','po','ra','se','ti','vu')
    names=[(''.join((syllables[i//256],syllables[(i//16)%16],syllables[i%16]))+'x').capitalize() for i in range(4096)]
    names=[n for n in names if n not in used]
    np.random.default_rng(2753001).shuffle(names)
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True)
    worlds,rows=[],[]
    for family in FAMILIES:
        for split in ('fresh_entity','fresh_surface','fresh_role_order','fresh_depth'):
            for j in range(24):
                # Balance both truth values and path depths independently of roles.
                role,order=(1,1) if split=='fresh_role_order' else ((0,0),(0,1),(1,0))[j%3]
                depth=3 if split=='fresh_depth' else 1+(j//6)%2
                truth=(j//3)%2==0
                wid=f'{family}_{split}_{j:02d}'
                entities=names[len(worlds)*5:len(worlds)*5+5]
                assert len(entities)==5
                worlds.append(dict(id=wid,family=family,cohort=split,index=j,entities=entities,role=role,order=order,depth=depth,relation_truth=truth))
                for wording in ((4,5) if split=='fresh_surface' else (0,1)):
                    facts,prop=prior.make_world(family,entities,role,truth,depth,1 if wording in (1,5) else 0)
                    if order:facts=facts[:-1][::-1]+facts[-1:]
                    facts_text=' '.join(facts)
                    if wording==4:facts_text='Facts to use (and no others):\n'+'\n'.join('- '+s for s in facts)
                    if wording==5:facts_text='Read this record carefully: '+' / '.join(facts)
                    for style in (0,1):
                        for neg in (0,1):
                            lead='Use a formal tone. ' if style else 'Use an ordinary tone. '
                            marker='not' if neg else 'really'
                            label='Claim' if wording==5 else 'Statement'
                            instruction='Judge the whole claim. Output yes for true and no for false.' if wording==5 else 'Return yes if and only if the entire statement is true; otherwise return no.'
                            prefix=lead+facts_text+'\n'+label+': '
                            text=prefix+'It is '+marker+' the case that '+prop+'.\n'+instruction+'\nAnswer:'
                            ids=tok.encode(text,add_special_tokens=False)
                            gid=f'{wid}_t{wording}'
                            rows.append(dict(id=f'{gid}_s{style}n{neg}',group=gid,world=wid,family=family,split=split,cohort=split,
                                world_index=j,wording=wording,role=role,order=order,depth=depth,fact_count=len(facts),
                                cond=2*style+neg,style=style,negation=neg,text=text,expected='yes' if truth!=bool(neg) else 'no',
                                tokenization={'4B':dict(token_ids=ids,length=len(ids),question_start=len(tok.encode(prefix,add_special_tokens=False)))},
                                replication14B=False))
    assert len(worlds)==384 and len(rows)==3072
    assert not ({n for w in worlds for n in w['entities']}&used)
    groups={}
    for r in rows:groups.setdefault(r['group'],[]).append(r)
    assert all(len({r['tokenization']['4B']['length'] for r in rr})==1 for rr in groups.values())
    mat=dict(created_utc=now(),worlds=worlds,rows=rows,sampling_unit='world, with2 surface forms and4 conditions',
        scope='Fresh entity strings and2 novel surface protocols; same4 synthetic English relation families. New surfaces also change formatting/instructions.')
    write(path,mat)
    write(OUT/'design.json',dict(created_utc=now(),phase=2753,material_sha256=sha(path),seed=2753001,
        objective='Forecast late interaction with no late source states; test one-prefix early inputs and role-bound embeddings on new worlds.',
        model='local qwen3-4b bf16 eager CUDA; no model training or quantization',
        discovery='Old assertion train and validation only. All old heldout results are exploratory context, never pooled into fitting/selection.',
        targets=['primary I at finalnorm boundary36','secondary full h11 at finalnorm and full-vocabulary readout','all hidden-boundary interaction profiles'],
        input_scopes=['metadata/static embeddings: zero transformer blocks','one target prefix through4/8/12 blocks','one base prefix through8 blocks (control)',
                      'four prefixes through8 blocks (separate cost class, not a one-prefix mechanism)'],
        architecture='Ridge in all original coordinates, evaluated in dual kernel form; noPCA, noTopK. Native-coordinate product route as comparator.',
        alpha_grid=[.001,.01,.1,1.,10.],selection='Old validation primary interaction error; one global chosen method frozen before new model capture, separate same-input h11 readout fit.',
        material='384 new worlds,3072 prompts; both truth values; distinct new worlds peraxis; four conditions length-matched',
        inference='Paired stratified world bootstrap within family and split,2000 draws; fixed templates and one model only.',
        resource='Reuse old fields.16-prompt pilot,3072 formal4B prompts,cap600s capture. Offline fitting/readout capped600s each. No14B/GLM replication allocated this phase.',
        boundaries='Saved last-position complete coordinates at all boundaries, not full token field or full KV state.',
        no_claims=['no universal language law','no causal parameter identification from prediction','no cross-model replication','no autonomous generation equivalence']))
    write(OUT/'material_seal.json',dict(created_utc=now(),source=snapshot(Path(__file__)),material_sha256=sha(path),design_sha256=sha(OUT/'design.json'),
        old_material_sha256=sha(OLD/'material.json'),old_capture_receipt_sha256=sha(OLD/'4B/capture_done.json')))
    print(dict(worlds=len(worlds),prompts=len(rows)),flush=True)
    return mat


def collect(pilot=False):
    import threading, os
    mat=prepare()
    assert (OUT/'selection.json').exists(),'Freeze discovery selection before new outcomes.'
    # Reuse the already-validated native collector without editing its legacy code.
    prior.OUT=OUT
    prior.prepare=lambda:mat
    write(OUT/('pilot_wrapper.json' if pilot else 'capture_wrapper.json'),dict(created_utc=now(),source=snapshot(Path(__file__)),
        native_collector=snapshot(Path(prior.__file__)),selection_sha256=sha(OUT/'selection.json'),budget_note='Outer watchdog enforces600s; native collector also retains its2100s emergency cap.'))
    def stop():
        write(OUT/'capture_budget_exceeded.json',dict(created_utc=now(),seconds=600,pilot=pilot))
        os._exit(124)
    watchdog=threading.Timer(600,stop)
    watchdog.start()
    try:prior.collect('4B',pilot)
    finally:watchdog.cancel()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','collect'])
    p.add_argument('--pilot',action='store_true')
    a=p.parse_args()
    prepare() if a.mode=='prepare' else collect(a.pilot)
