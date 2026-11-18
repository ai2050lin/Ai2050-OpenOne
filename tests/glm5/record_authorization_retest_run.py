"""Run both frozen lexical arms sequentially, with each arm's mechanism admission audited."""

from contextlib import contextmanager
from itertools import batched
from os import getpid
import argparse
import json
import time

import record_authorization_retest_material as A
import record_authorization_run as old
from deploy.storage_policy import StorageBlocked, check_write, checked_dir
from tools.power_analysis import check_phase


R,M,C = old.R,old.M,old.C
OUT = R.OUT.with_name('record_authorization_retest_20261117_v1')
TEMP = R.TEMP.with_name(OUT.name)
PREVIOUS = OUT.with_name('record_authorization_20261117_v1')
SCRIPTS = ('record_authorization_retest_material.py','record_authorization_retest_run.py','record_authorization_retest_report.py')
PRIORS = ('plan.json','pilot-behavior.json','behavior-audit.json','manifest.json')


def sources():
    added = set(SCRIPTS) | {'record_authorization_study.py'}
    paths = {f'tests/glm5/{name}' for name in added}
    return old.hashes() | {path:C.A.digest(R.ROOT/path) for path in sorted(paths)}


def candidates(tokenizer,row):
    drafts = old.candidates(tokenizer,row)
    # These strings were observed failure outputs before freeze. Drafts only
    # schedule original argmax checks and cannot supply a generated token.
    eos = tokenizer.eos_token
    for name in ('East','West'):
        for letter in 'PQ':
            for separator in ('  \n','\n'):
                draft = tokenizer.encode(f' {name}{separator}Code: {letter}{eos}',add_special_tokens=False)
                drafts.append(tuple(draft))
    return drafts


def configure():
    old.configure()
    M.RUN,M.SEED,M.SIZES = A.RUN,A.SEED,A.SIZES
    R.OUT,R.TEMP = OUT,TEMP
    R.SLAB,R.PHASE_PEAK = 516,2*2**30
    R.HOLDOUTS = ('confirm',)
    R.source_hashes,R.candidates = sources,candidates


@contextmanager
def arm_scope(arm):
    previous,tables = R.OUT,M.families
    try:
        R.OUT = OUT/arm
        M.families = lambda tokenizer,split:A.families(tokenizer,split,arm)
        yield
    finally:
        R.OUT,M.families = previous,tables


def freeze():
    from transformers import AutoTokenizer

    if (OUT/'plan.json').exists():
        raise FileExistsError('An immutable replication plan is already present')
    R.configure(R.FREEZE_PEAK)
    tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    material,ids=A.inspect(tok)
    failed = json.loads((PREVIOUS/'pilot-behavior.json').read_bytes())['gate']
    targets = {(c['earlier_target'],c['reverse']) for c in failed['cells']+failed['pairs'] if not c['passed']}
    assert targets==set(A.SETTINGS) and all(c['frame']==A.FRAME for c in failed['cells']+failed['pairs'] if not c['passed'])
    for arm in A.LEXICONS:
        checked_dir(OUT/arm,R.FREEZE_PEAK,operation='配对复测分支')
    power=dict(pilot_per_setting=check_phase(.12,A.SIZES['pilot']//2,.18,.05),
               confirmation_per_arm=check_phase(.08,A.SIZES['confirm'],.20,.01))
    plan=dict(run=A.RUN,at=C.stamp(),provenance=R.metadata(),snapshot=C.Prior.identity(),versions=R.versions(),
              sources=R.source_hashes(),previous={f:C.A.digest(PREVIOUS/f) for f in PRIORS},material=material,
              prompt_cue_ids=ids,lexicons=A.LEXICONS,power=power,batch=R.BATCH,slab=R.SLAB,width=M.WIDTH,max_new=M.MAX_NEW,
              design=dict(settings=A.SETTINGS,frame=A.FRAME,invariant='Use exact Phase 2959 frame 0. The first request is refused. No new words in the rule, codebook, or instructions. Across four views for each family, change only one seven-character, one-token status in the focal second notice; proposal ranked before old in half of the families by design.',
                          order='Complete pilot original before pilot synonym. Both measured even if one pilot fails. Each arm requires both settings to pass the same behavioral thresholds to advance. Within-arm discovery, forecast and intervention only, with untouched independent pools.',
                          behavior_floor=old.FLOOR,pair_floor=old.PAIR_FLOOR,
                          behavior='Each arm separately requires >=.85 joint State, Code, format and EOS in both authorization sides of both settings, and >=.80 families with both sides correct in each setting. Check all thresholds in pilot, discovery and confirmation; no across-cell averaging to pass.',
                          diagnostics='Approved side retains old label; refused side emits focal proposal. Count exact paired original/synonym transitions per setting, descriptive 99% family-bootstrap intervals for risk differences, and prespecified stratification by proposal-before-old in codebook. Earlier destination and code orientation vary together across the two settings.',
                          natural='State: supplied as the shared prefix. Original full-vocabulary greedy checked token by token on CUDA with verified drafts including East/West; maximum 32 tokens, actual EOS. Drafts schedule evaluation only.',
                          read_layer=M.READ_LAYER,source=[M.SOURCE_LAYER,M.SOURCE_HEAD],
                          feature_comparison='On the same queries compare role-aware text baseline with +32 B20 native coordinates and +32 fitted dictionary features. Both use true observable prefix. Code uses model-emitted history. Fit/target/control freeze on disjoint discovery labels before confirm.',
                          added_inputs=32,dictionary_atoms=256,dictionary_active=24,dictionary_steps=240,ridge_regularization=12.,
                          primary_prediction='Within one independently qualifying arm: require the same method to lower full-vocabulary KL >=10% on State and Code, family-bootstrap 99% interval lower >0, and >=365 complete stable pairs in disjoint confirm families across the two targeted settings.',
                          minimum_power_families=365,
                          intervention='At State, B25 head chosen on discovery by signed focal cue Value write proxy. Same-layer control outside GQA group, matched output-projected dose. Compare matched occlusion loss and partial native cue-only restoration after joint cue/proposal block. Include sham, exact full native restore, native unembedding and top-1 flips.',
                          causal_confidence=1-.01/4,dose_rel_p95=.03,minimum_matched_margin=.10,minimum_restore_margin=.10,
                          lockbox='Joint lockbox remains unopened.'),
              resource=dict(added_peak_bytes=R.PHASE_PEAK,volume='D:',reserve_gib=20,placement='Original BF16 safetensors streamed block by block from host to CUDA, full vocabulary greedy; fixed-width right padding.'))
    master=R.write('plan.json',plan)
    for arm in A.LEXICONS:
        with arm_scope(arm):
            R.write('plan.json',dict(master_sha256=master,run=A.RUN,lexicon=arm,material=material[arm]))
    print('Frozen',master,plan['snapshot']['snapshot'],flush=True)


def verify():
    p=R.verified()
    assert p['previous']=={f:C.A.digest(PREVIOUS/f) for f in PRIORS}
    assert p['design']['behavior_floor']==old.FLOOR and p['design']['pair_floor']==old.PAIR_FLOOR
    master=C.A.digest(OUT/'plan.json')
    for arm in A.LEXICONS:
        with arm_scope(arm):
            assert R.read('plan.json')==dict(master_sha256=master,run=A.RUN,lexicon=arm,material=p['material'][arm])
    return p


def measured(model,p,split,arm):
    rows=A.rows(model.tok,split,arm)
    assert M.digest(rows)==p['material'][arm][split]['sha256']
    name=split+'-behavior.json'
    plan=C.A.digest(R.OUT/'plan.json')
    material=p['material'][arm][split]['sha256']
    if (R.OUT/name).exists():
        report=R.read(name)
        assert report['plan']==plan and report['material']==material
        assert old.gate(rows,report['records'])==report['gate']
        return rows,report
    observed=[]
    for i,batch in enumerate(batched(rows,R.SLAB),1):
        C.P.check_write(OUT,R.PHASE_PEAK,operation='授权复测逐批磁盘余量')
        observed.extend(R.generation_batch(model,batch,f'{arm}-{split}-greedy-slab{i:02d}'))
    result=dict(run=A.RUN,lexicon=arm,at=C.stamp(),provenance=p['provenance'],plan=plan,
                material=material,records=observed,gate=old.gate(rows,observed))
    R.write(name,result)
    print(arm,split,result['gate']['passed'],len(result['gate']['complete_families']),len(rows)//2,flush=True)
    return rows,result


def run_mechanism(model,p,arm,discovery,behavior):
    import record_update_study as S

    rows,z=S.collect(model,discovery,behavior,'discovery',p)
    dict_,fit,frozen=S.freeze_models(model,p,rows,z)
    material,confirmation=measured(model,p,'confirm',arm)
    if not confirmation['gate']['passed']:
        return dict(outcome='confirm_behavior_failed',admitted=[])
    observed,native=S.collect(model,material,confirmation,'confirm',p)
    forecast=S.forecast(model,p,observed,native,dict_,fit,'confirm')
    admitted=[m for m in S.METHODS[1:] if all(forecast['metric'][f'{q}.{m}']['improvement']['passed'] for q in S.NAMES)]
    if admitted:
        S.perturb(model,p,observed,native,confirmation,frozen,'confirm')
    return dict(outcome='intervention_finished' if admitted else 'prediction_gate_failed',admitted=admitted)


def run():
    import torch
    from record_authorization_study import install

    p=verify()
    R.configure(R.PHASE_PEAK)
    install()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    log=check_write(OUT/'run-start.json',2**20,operation='记录授权复测进程')
    if not log.exists():
        R.write(log.name,dict(at=C.stamp(),pid=getpid(),plan=C.A.digest(OUT/'plan.json')))
    with C.P.GpuLease(),torch.no_grad():
        start=time.monotonic()
        C.N.progress('retest-load',0,2,start)
        assert C.Prior.identity()==p['snapshot']
        C.N.progress('retest-load',1,2,start)
        model=C.N.Engine()
        C.N.progress('retest-load',2,2,start)
        try:
            decision={}
            for arm in A.LEXICONS:
                with arm_scope(arm):
                    _,result=measured(model,p,'pilot',arm)
                    decision[arm]=dict(passed=result['gate']['passed'],behavior_sha=C.A.digest(R.OUT/'pilot-behavior.json'))
            R.write('pilot-decision.json',dict(at=C.stamp(),arms=decision))
            for arm,data in decision.items():
                if not data['passed']:
                    data.update(outcome='pilot_behavior_failed',admitted=[])
                    continue
                with arm_scope(arm):
                    rows,b=measured(model,p,'discovery',arm)
                    if not b['gate']['passed']:
                        data.update(outcome='discovery_behavior_failed',admitted=[])
                    else:
                        data.update(run_mechanism(model,p,arm,rows,b))
            R.write('stage-decision.json',dict(at=C.stamp(),plan=C.A.digest(OUT/'plan.json'),arms=decision))
        finally:
            model.close()


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','all','close'))
    args=cli.parse_args()
    configure()
    if args.action=='freeze':
        freeze()
    elif args.action=='all':
        run()
    else:
        from record_authorization_retest_report import close

        close(verify(),arm_scope)


if __name__=='__main__':
    try:
        main()
    except StorageBlocked as error:
        raise SystemExit(str(error)) from None
