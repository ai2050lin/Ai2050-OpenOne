"""Freeze and validate genuine greedy Qwen3-14B record-update behavior."""

import importlib.metadata
from itertools import batched
from pathlib import Path
import json
import os
import sys
import time

import numpy as np

import kv_chain_run as C
import record_update_material as M
from deploy.storage_policy import StorageBlocked, check_write, checked_dir, staging_file
from tools.power_analysis import check_phase
from tools.provenance import check_provenance, provenance


ROOT = C.ROOT
OUT = ROOT/'tests/glm5/result/record_update_20261117_v1'
TEMP = ROOT/'tests/glm5_temp/record_update_20261117_v1'
MODEL = ROOT/'models/hf/Qwen3-14B'
BATCH = 10
SLAB = 120
FREEZE_PEAK = 2*2**20
PHASE_PEAK = 980*2**20
HOLDOUTS = ('words','frames','combos')


def configure(peak):
    checked_dir(OUT,peak,operation='记录更新实验写入和剩余阶段预算')
    checked_dir(TEMP,peak,operation='记录更新隔离暂存')
    C.OUT,C.TMP,C.RUN,C.WIDTH,C.BATCH,C.SLAB,C.BUDGET = OUT,TEMP,M.RUN,M.WIDTH,BATCH,SLAB,peak
    C.setup()


def write(name, payload):
    raw = M.canonical(payload)
    filename = check_write(OUT/name,len(raw)+2**14,operation='记录更新实验结果')
    with staging_file(filename) as temp:
        temp.write_bytes(raw)
    return C.A.digest(filename)


def read(name):
    return json.loads((OUT/name).read_bytes())


def source_hashes():
    paths = set(C.sources()) | {
        'tests/glm5/record_update_material.py','tests/glm5/record_update_run.py',
        'shared/dist_protocol/transcoder.py','tools/power_analysis.py','tools/provenance.py',
    }
    paths.add('tests/glm5/record_update_study.py')
    return {p:C.A.digest(ROOT/p) for p in sorted(paths)}


def versions():
    return {p:importlib.metadata.version(p) for p in ('torch','numpy','scipy','safetensors','transformers','tokenizers')}


def powers():
    records = {k:check_phase(.08,384,.20,.01) for k in (*HOLDOUTS,'lockbox')}
    records['discovery'] = check_phase(.12,M.SIZES['discovery'],.20,.01)
    records['pilot'] = check_phase(.44,M.SIZES['pilot'],.44,.01)
    if not all(r['pass'] for r in records.values()):
        raise ValueError(f'Power preflight failed: {records}')
    return records


def metadata():
    for index,(key,value) in enumerate((('safe.directory',ROOT.as_posix()),)):
        os.environ['GIT_CONFIG_COUNT'] = str(index+1)
        os.environ[f'GIT_CONFIG_KEY_{index}'] = key
        os.environ[f'GIT_CONFIG_VALUE_{index}'] = value
    p = provenance(model_path=MODEL,seed=M.SEED,tool=M.RUN)
    if errors:=check_provenance({'provenance':p}):
        raise ValueError(errors)
    return p


def plan():
    from transformers import AutoTokenizer

    if (OUT/'plan.json').is_file():
        raise FileExistsError('Existing immutable plan. A changed design requires a new run ID.')
    tok = AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    p = dict(run=M.RUN,at=C.stamp(),provenance=metadata(),snapshot=C.Prior.identity(),versions=versions(),
             sources=source_hashes(),material=M.inspect(tok),power=powers(),batch=BATCH,width=M.WIDTH,
             design=dict(language='Original authored experimental prompts with executable truth and actual CausalLM responses',
                         prediction_inputs='only actual prefix at the query, earlier L20 native states, and frozen fit',
                         pilot_roles='development only; disjoint labels, no confirmation outcomes accessed',
                         natural='unrestricted vocabulary greedy, deterministic draft verification, preserve first divergent actual token, fixed EOS or 32-token horizon',
                         behavior='per frame by number of applied selected-record updates, require >=.85 joint State + Code + format + stop',
                         behavior_floor=M.BEHAVIOR_FLOOR,leaveout='new label pool, new verbal frame, and held-out operation pair; names and initial pairs never shared',
                         added_inputs=32,dictionary_atoms=256,dictionary_active=24,
                         dictionary_steps=240,ridge_regularization=12.,
                         read_layer=M.READ_LAYER,source=[M.SOURCE_LAYER,M.SOURCE_HEAD],
                         primary_prediction='family-balanced 99% bootstrap confidence; >=10% relative full-vocabulary KL reduction vs matched text baseline at both State and Code, lower confidence bound positive, all 3 panels',
                         minimum_power_families=365,predictive_tie='prespecify both native and dictionary losses; a path is admitted when an input representation satisfies the complete held-out gate',
                         intervention='predictions frozen before edits, single-query State Value attenuation + projected-dose matched same-layer random heads outside GQA group, sham and native self-recovery, preserve state and raw branch trace',
                         dose_rel_p95=.03,minimum_matched_margin=.10,minimum_restore_margin=.10,
                         lockbox='unopened during behavior qualification and model selection'))
    print('Frozen',write('plan.json',p),p['snapshot']['snapshot'],flush=True)
    return p


def verified():
    p = read('plan.json')
    assert p['sources']==source_hashes() and p['versions']==versions()
    return p


def candidates(tokenizer, row):
    eos = tokenizer.eos_token
    choices = [tuple(row['token']['draft'])]
    for name in row['names']:
        for letter in 'PQ':
            for spacing in ('\n','  \n'):
                for end in (eos,'\n'+eos):
                    text=f' {name}{spacing}Code: {letter}{end}'
                    tail=tuple(tokenizer.encode(text,add_special_tokens=False))
                    if tail not in choices:choices.append(tail)
    return choices


def pick(context, drafts):
    for trial in drafts:
        if len(trial)>len(context) and trial[:len(context)]==tuple(context):
            return trial[len(context):]
    return ()


def generation_batch(model, rows, stage):
    """Verify drafts one causal position at a time and commit only original argmax."""
    by_id={r['id']:r for r in rows}
    offered={r['id']:candidates(model.tok,r) for r in rows}
    gen={r['id']:[] for r in rows}
    eos = model.model.generation_config.eos_token_id
    term = set(eos if isinstance(eos,list) else [eos])
    term.add(model.tok.eos_token_id)
    drafts = {r['id']:0 for r in rows}
    divergent = {r['id']:0 for r in rows}
    active=list(by_id)
    start=time.monotonic()

    def progress(n,done):
        C.N.progress(stage,done,len(rows),start,decode_pass=n,pending=len(rows)-done)

    progress(0,0)
    for repetition in range(1,M.MAX_NEW+1):
        if not active:break
        packets=[]
        for key in active:
            ordinal=by_id[key]
            base=ordinal['token']['prompt']+gen[key]
            proposal=pick(gen[key],offered[key])[:M.MAX_NEW-len(gen[key])]
            # [slot] is required by the existing padded forward engine. Decode does
            # not use it to select tokens.
            packets.append(dict(ordinal,token=dict(ids=[*base,*proposal],slots=[len(base)-1],
                                                  verified_prefix=len(base),proposal=proposal)))
        missing=(-len(packets))%BATCH
        if missing:
            packets.extend([dict(packets[-1],virtual=True)]*missing)
        advance=[]
        for batch in C.forward(model,packets,stage+f'-pass{repetition:02d}'):
            real=[r for r in batch['rows'] if not r.get('virtual')]
            left=min(r['token']['verified_prefix']-1 for r in batch['rows'])
            right=int(batch['c'].max())+1
            full=model.logits(batch['x'][:,left:right].cuda()).argmax(-1).cpu().numpy()
            for j,row in enumerate(real):
                key=row['id']; t=row['token']; proposed=t['proposal']; current=gen[key]
                begin=t['verified_prefix']-1-left
                actual=full[j,begin:begin+len(proposed)+1]
                for k,token in enumerate(actual):
                    token=int(token)
                    current.append(token)
                    if k<len(proposed) and token==proposed[k]:
                        drafts[key]+=1
                    elif k<len(proposed):
                        divergent[key]+=1
                        break
                    if token in term or len(current)>=M.MAX_NEW:
                        break
                if current[-1] not in term and len(current)<M.MAX_NEW:
                    advance.append(key)
        assert len(advance)==len(set(advance))
        active=advance
        progress(repetition,len(rows)-len(active))
    assert not active
    return [behavior(model.tok,r,gen[r['id']],term,drafts[r['id']],divergent[r['id']]) for r in rows]


def behavior(tokenizer,row,tokens,term,matched,diverged):
    '''Score output without supplying a label, correct response or end marker.'''
    stopped=bool(tokens and tokens[-1] in term)
    body=tokens[:-1] if stopped else tokens
    text=tokenizer.decode(body,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    parsed=M.OUTPUT.fullmatch(text)
    state=parsed[1] if parsed else None
    code=parsed[2] if parsed else None
    state_ok=state==row['truth'][-1]
    code_ok=code==row['answer']
    reason=('ok' if parsed and stopped and state_ok and code_ok else
            'format' if not parsed else 'no_stop' if not stopped else 'state' if not state_ok else 'code')
    return dict(id=row['id'],actual=tokens,text=text,stop=stopped,format=bool(parsed),state=state,code=code,state_ok=state_ok,
                code_ok=code_ok,joint=reason=='ok',failure=reason,draft_matches=matched,draft_divergences=diverged)


def gate(rows, results):
    from collections import defaultdict

    cells=defaultdict(list)
    seen=defaultdict(list)
    for r,y in zip(rows,results,strict=True):
        assert r['id']==y['id']
        seen[r['family']].append(y['joint'])
        cells[r['frame'],r['actions'].count(0)].append(y)
    values=[]
    for (frame,update), cell in sorted(cells.items()):
        joint=sum(y['joint'] for y in cell)
        values.append(dict(frame=frame,updates=update,n=len(cell),joint=joint,passed=joint/len(cell)>=M.BEHAVIOR_FLOOR,
                           format=sum(y['format'] for y in cell),stop=sum(y['stop'] for y in cell),
                           state=sum(y['state_ok'] for y in cell),code=sum(y['code_ok'] for y in cell)))
    return dict(cells=values,passed=all(v['passed'] for v in values),
                complete_families=[k for k,v in seen.items() if len(v)==2 and all(v)])


def stage(model,p,split):
    rows=[M.encode(model.tok,row) for row in M.families(model.tok,split)]
    assert M.digest(rows)==p['material'][split]['sha256']
    name=split+'-behavior.json'
    if (OUT/name).exists():
        report=read(name)
        assert report['plan']==C.A.digest(OUT/'plan.json') and report['material']==M.digest(rows)
        assert M.digest(gate(rows,report['records']))==M.digest(report['gate'])
        return rows,report
    observed=[]
    for i,rows_batch in enumerate(batched(rows,SLAB),1):
        observed.extend(generation_batch(model,rows_batch,f'{split}-greedy-slab{i:02d}'))
    result=dict(run=M.RUN,provenance=p['provenance'],plan=C.A.digest(OUT/'plan.json'),material=M.digest(rows),
                at=C.stamp(),records=observed,gate=gate(rows,observed))
    print(split,result['gate']['passed'],len(result['gate']['complete_families']),len(rows)//2,flush=True)
    write(name,result)
    return rows,result


def main(action):
    if action=='freeze':
        configure(FREEZE_PEAK); plan()
        return
    if action not in ('pilot','behavior','all'):
        raise ValueError('Expected freeze, pilot, behavior or all')
    configure(PHASE_PEAK)
    p=verified()
    import torch
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    with C.P.GpuLease(),torch.no_grad():
        t=time.monotonic()
        C.N.progress('record-update-load',0,2,t)
        assert C.Prior.identity()==p['snapshot']
        C.N.progress('record-update-load',1,2,t)
        model=C.N.Engine()
        C.N.progress('record-update-load',2,2,t)
        try:
            _,trial=stage(model,p,'pilot')
            if action=='pilot' or not trial['gate']['passed']:
                return
            discovery,result=stage(model,p,'discovery')
            if not result['gate']['passed']:
                return
            if action=='all':
                from record_update_study import run
                run(model,p,discovery,result)
            else:
                print('Discovery behavior complete. Confirmation stays unopened until prediction freeze.',flush=True)
        finally:
            model.close()


if __name__=='__main__':
    sys.modules['record_update_run']=sys.modules[__name__]
    try:
        main(sys.argv[1])
    except StorageBlocked as error:
        raise SystemExit(str(error)) from None
