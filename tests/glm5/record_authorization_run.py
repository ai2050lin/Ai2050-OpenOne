"""Run paired Qwen3-14B authorization probes with frozen admission gates."""

from collections import Counter
from itertools import batched
from pathlib import Path
import argparse
import os

import record_authorization_material as A
import record_update_run as R
from deploy.storage_policy import StorageBlocked, check_write


C,M = R.C,R.M
BASE_SOURCES = R.source_hashes
BASE_CANDIDATES = R.candidates
STUDY_FILES = ('record_authorization_material.py','record_authorization_run.py',
               'record_authorization_study.py','record_authorization_report.py')
FLOOR = .85
PAIR_FLOOR = .80


def hashes():
    paths = set(BASE_SOURCES()) | {f'tests/glm5/{file}' for file in STUDY_FILES}
    paths.update(('tests/glm5/record_update_v2.py','tests/glm5/record_update_v3.py'))
    return {path:C.A.digest(R.ROOT/path) for path in sorted(paths)}


def gate(rows, results):
    counts = {}
    families = {}
    complete = []
    for (left,right),(first,second) in zip(batched(rows,2), batched(results,2), strict=True):
        pair = sorted(zip((left,right),(first,second),strict=True),key=lambda x:x[0]['update'])
        (no,a),(yes,b) = pair
        assert no['family']==yes['family'] and not no['update'] and yes['update']
        assert a['id']==no['id'] and b['id']==yes['id']
        key = (no['frame'],*no['setting'])
        family = families.setdefault(key,Counter())
        family['n'] += 1
        family['both'] += int(a['joint'] and b['joint'])
        if a['joint'] and b['joint']:
            complete.append(no['family'])
        for row,output in pair:
            condition = counts.setdefault((*key,row['update']),Counter())
            condition['n'] += 1
            for value in ('joint','state_ok','code_ok','format','stop'):
                condition[value] += int(output[value])
    cells = [dict(frame=f,earlier_target=target,reverse=reverse,updates=int(update),
                  n=c['n'],joint=c['joint'],state=c['state_ok'],code=c['code_ok'],format=c['format'],stop=c['stop'],
                  passed=c['joint']/c['n']>=FLOOR)
             for (f,target,reverse,update),c in sorted(counts.items())]
    pairs = [dict(frame=f,earlier_target=target,reverse=reverse,n=c['n'],both=c['both'],passed=c['both']/c['n']>=PAIR_FLOOR)
             for (f,target,reverse),c in sorted(families.items())]
    assert cells and len(cells)==2*len(pairs)
    return dict(cells=cells,pairs=pairs,complete_families=complete,passed=all(x['passed'] for x in cells+pairs))


def candidates(tokenizer,row):
    # Use the spacing observed before this experiment as the first verification draft.
    all_ = BASE_CANDIDATES(tokenizer,row)
    spaced = tuple(tokenizer.encode(f" {row['truth'][-1]}  \nCode: {row['answer']}{tokenizer.eos_token}",
                                    add_special_tokens=False))
    assert spaced in all_
    return [spaced]+[draft for draft in all_ if draft!=spaced]


def freeze():
    from transformers import AutoTokenizer

    if (R.OUT/'plan.json').exists():
        raise FileExistsError('The authorization plan has already been frozen')
    R.configure(R.FREEZE_PEAK)
    tok = AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    material = A.inspect(tok)
    power = R.powers()
    previous = R.OUT.with_name('record_update_20261117_v3')
    prior = {file:C.A.digest(previous/file) for file in ('plan.json','behavior-audit.json')}
    p = dict(run=A.RUN,at=C.stamp(),provenance=R.metadata(),snapshot=C.Prior.identity(),
             versions=R.versions(),sources=R.source_hashes(),material=material,power=power,previous=prior,
             batch=R.BATCH,slab=R.SLAB,width=M.WIDTH,max_new=M.MAX_NEW,
             design=dict(material='Handwritten template grammar and labels. One matched seven-character word and one token changes in focal notice 2. Earlier request always refused. Each family retains labels, full codebook and all prompt boundaries.',
                         independent='Pilot, known, new word and lockbox vocabulary pools disjoint and previously used label IDs excluded. Each unordered initial/proposed label pair allocated only once.',
                         natural='With State: chat prefix, verify each proposed continuation token using original full-vocabulary greedy. Commit only observed argmax, no future token visible. Stop at actual EOS or 32 tokens.',
                         behavior='For each frame by earlier destination by key orientation by authorization, joint correctness >=.85. At least .80 of families in every frame by setting must get both views correct.',
                         behavior_floor=FLOOR,pair_floor=PAIR_FLOOR,
                         condition_pairs=['State selection and exact joint result for both sides','Pairwise proposal-following change with family bootstrap, diagnostic by earlier destination and frame'],
                         leaveout='Unseen lexical pool; unseen frames; unseen combination of earlier refused request for East and reversed key. Always compare both authorization levels.',
                         lockbox='Keep joint lockbox unopened. Holdout behavior starts only after discovery prediction and native target/control freeze.',
                         prediction_inputs='Actual visible prefix at State prelabel and Code after the produced State. Text accepts explicit authorization, roles and rule-routed label as inputs. Native/dictionary receive the same text plus 32 discovery-selected features from B20 input.',
                         read_layer=M.READ_LAYER,source=[M.SOURCE_LAYER,M.SOURCE_HEAD],
                         added_inputs=32,dictionary_atoms=256,dictionary_active=24,dictionary_steps=240,ridge_regularization=12.,
                         primary_prediction='>=10% full-vocabulary KL reduction over role-aware text on both queries in all three panels plus family-bootstrap 99% lower bound >0. At least 365 complete stable pairs. Same augmented method in all comparisons.',
                         minimum_power_families=365,
                         intervention='At State only, clear activation from the one focal authorization token Value in one B25 query head. Choose head with largest discovery signed projected cue write on proposal-vs-old lm_head contrast. Seeded same-layer random control from eight closest discovery native doses outside same GQA group. Include sham, matched-dose control, double block with proposal token, exact all-source restoration, and authorization-only restoration.',
                         causal_endpoints='At least 365 complete families per panel, exact sham/full-native restore, <=.03 p95 projected dose relative error. >=.10 logit matched loss and partial restore with Bonferroni 99.8333% two-sided intervals across the six panel/endpoints. Score full-vocab argmax and old-vs-proposal margins separately.',
                         causal_confidence=1-.01/6,dose_rel_p95=.03,minimum_matched_margin=.10,minimum_restore_margin=.10,
                         resource=dict(volume='D:',added_peak_bytes=R.PHASE_PEAK,source='original BF16 safetensors one block at a time to CUDA; layer-boundary copy to host')))
    print('Frozen',R.write('plan.json',p),p['snapshot']['snapshot'],flush=True)


def configure():
    M.RUN,M.SEED,M.SIZES,M.WIDTH = A.RUN,A.SEED,A.SIZES,192
    M.families,M.encode,M.inspect = A.families,A.encode,A.inspect
    R.OUT = R.OUT.with_name('record_authorization_20261117_v1')
    R.TEMP = R.TEMP.with_name(R.OUT.name)
    R.BATCH,R.SLAB = 12,192
    R.source_hashes,R.candidates,R.gate = hashes,candidates,gate


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','all','close'))
    args=cli.parse_args()
    configure()
    if args.action=='freeze':
        freeze()
    elif args.action=='all':
        from record_authorization_study import install

        install()
        log=check_write(R.OUT/'run-start.json',2**20,operation='授权成对实验执行状态')
        if not log.exists():
            R.write(log.name,dict(at=C.stamp(),pid=os.getpid(),plan=C.A.digest(R.OUT/'plan.json')))
        R.main('all')
    else:
        from record_authorization_report import close

        close()


if __name__=='__main__':
    try:
        main()
    except StorageBlocked as error:
        raise SystemExit(str(error)) from None
