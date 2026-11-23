"""Matched original-model perturbation and restoration for an admitted native path."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from clt_first_record_predict import HERE, ROOT, OUT, MATERIAL, SEED, dependencies, registered_context
from deploy.test_archive_client import runtime_identity, save
from shared.dist_protocol.test_archive import CORE_ROLES, digest, validate_plan
from source_binding_design import read, sha

SCRIPT = HERE/'clt_first_record_causal.py'
PLAN = OUT/'causal-plan.json'
QUESTIONS = OUT/'causal-materials.json'
ARMS = ('base','target','random','sham','restore','restore_mlp')
BATCH = 384


def code_identity():
    return {**dependencies(),str(SCRIPT.relative_to(ROOT)):sha(SCRIPT)}


def build_plan(prediction_record):
    prediction_record = prediction_record.resolve(strict=True)
    if not prediction_record.is_relative_to(HERE/'result/routine'):
        raise ValueError('留出记录必须来自本机登记工作区')
    review = read(prediction_record/'review.json')
    receipt = read(prediction_record/'server-receipt.json')
    result = review.get('conclusion') or {}
    if (receipt.get('archive_state') != 'server_verified' or not receipt.get('readback_verified')
            or receipt.get('run_id')!=review.get('server_run_id')
            or not result.get('prediction', {}).get('gate_passed')):
        raise ValueError('须先完成归档和留出预测过门')
    candidate = Path(review['scratch'])/'candidate.npz'
    if sha(candidate) != result['candidate_sha256']:
        raise ValueError('候选文件不匹配')
    manifest = read(MATERIAL/'manifest.json')
    split = manifest['splits']['intervention']
    source = MATERIAL/split['file']
    if sha(source) != split['material_sha256']:
        raise ValueError('干预题目文件哈希不匹配')
    population = [row for row in read(source) if row['cell']==2+3*((row['family']//4)%4)]
    if len(population)!=192 or len({r['family'] for r in population}) != 192:
        raise ValueError('干预家族代表未覆盖')
    questions = [dict(row, id=f"{row['id']}-{arm}", source_id=row['id'], arm=arm) for row in population for arm in ARMS]
    save(QUESTIONS,questions)
    data_hash = sha(QUESTIONS)
    selection = {'prediction_run':review['server_run_id'],'prediction_receipt_sha256':sha(prediction_record/'server-receipt.json'),
                 'candidate':str(candidate),'candidate_sha256':sha(candidate),
                 'block':result['block'],'head':result['head'],'arms':list(ARMS),'target_head_fraction':.5,
                 'control':'seeded same-layer head, actual FP32 O-write L2 dose matched to target within 10%',
                 'restore':'same-input head original value before O; separate selected MLP units restored before down_proj',
                 'decision':'Baseline accuracy ≥90%, ≥95% dose match, sham and full restore exactly reproduce full logits; lower 97.5% family-bootstrap bound on random-minus-target margin positive'}
    entry, code = sha(SCRIPT), digest(code_identity())
    plan = {
        'schema':'research_test_plan.v1','kind':'real','title':'First-record native matched perturbation and restoration',
        'objective':'Test whether an admitted parallel head changes answer margin through its O write and measured MLP units.',
        'principle':'Change original head output immediately before O; match actual write dose and restore original same-input values.',
        'hypothesis':'The selected head contributes selectively to the physical-first answer and influences the chosen MLP units.',
        'alternatives':'Nonspecific residual damage, parallel compensation, or a readout sensitivity with no output role.',
        'criteria':selection['decision'],
        'domains':['KN','RS'],'family_ids':['CLT-FIRST-RECORD-PATH','SOURCE-BINDING-20261110'],
        'dataset':{'id':'first-record-causal-v1','revision':'one-prespecified-physical-row-per-family',
                   'content_sha256':data_hash,'questions_file_sha256':data_hash,'split_manifest_sha256':sha(source),
                   'case_count':len(questions),'exposure':'192 intervention families unused by fitting or confirmation.'},
        'entrypoint_sha256':entry,'execution_args':[],
        'acquisition':{'model_id':'Qwen3-14B','model_snapshot':manifest['model_snapshot'],'tokenizer_sha256':manifest['tokenizer_sha256'],
                       'entrypoint_sha256':entry,'questions_file_sha256':data_hash,
                       'inputs_sha256':digest([[r['id'],r['tokens'],r['arm']] for r in questions]),
                       'code_sha256':code,'environment_sha256':digest(runtime_identity()),
                       'precision':'BF16','seed':SEED,'generation':{'script_args':[],'one_step':True,'sample':False},
                       'capture':{'layers':[result['block']],'readouts':['last input token'],
                                  'boundary':'head before original O, selected units before original down projection'}},
        'analysis':{'method_version':'clt_first_record_causal.v1','code_sha256':code,'scoring_sha256':digest(selection),
                    'selection':selection,'required_artifact_roles':sorted(CORE_ROLES|{'evidence'})},
    }
    validate_plan(plan)
    save(PLAN,plan)


def collect(rt, rows, block, head, units, progress):
    t = rt.t
    F = t.nn.functional
    origin = rt.block
    saved, offsets, letters = {}, {'row':0}, {}
    H = slice(head*rt.hd,(head+1)*rt.hd)

    def add(key, value):
        saved.setdefault(key,[]).append(value)

    def hook_layer(number):
        layer=origin(number)
        if number!=block:
            return layer
        O=layer.self_attn.o_proj.weight.float()

        def edit_o(_, args):
            raw=args[0]
            changed=raw.clone()
            rr=rows[offsets['row']:offsets['row']+len(raw)]
            offsets['row']+=len(raw)
            offsets['rows']=rr
            local=[]
            for i,row in enumerate(rr):
                selected=raw[i,-1,H]
                decreased=(selected.float()*.5).to(raw.dtype)
                desired=F.linear(decreased.float()-selected.float(),O[:,H]).norm()
                arm=row['arm']
                realized=0.0
                if arm in {'target','restore','restore_mlp'}:
                    changed[i,-1,H]=decreased
                    realized=float(desired)
                    if arm=='restore':
                        changed[i,-1,H]=selected
                elif arm=='random':
                    ctl=(head+1+(row['family']*7+SEED)%(rt.heads-1))%rt.heads
                    col=slice(ctl*rt.hd,(ctl+1)*rt.hd)
                    original=raw[i,-1,col].float()
                    power=F.linear(original,O[:,col]).norm()
                    scale=desired/power.clamp_min(1e-12)
                    trial=(original*(1-scale)).to(raw.dtype)
                    changed[i,-1,col]=trial
                    realized=float(F.linear(trial.float()-original,O[:,col]).norm())
                local.append({'dose':realized,'target_dose':float(desired)})
            add('dose',local)
            return (changed,)

        def edit_mlp(_, args):
            raw=args[0]
            changed=raw.clone()
            block_distance=[]
            for i,row in enumerate(offsets['rows']):
                k=row['source_id']
                original=raw[i,-1,units]
                if row['arm']=='base':
                    letters[k]=original.clone()
                baseline=letters[k]
                if row['arm']=='restore_mlp':
                    changed[i,-1,units]=baseline
                block_distance.append(float((changed[i,-1,units].float()-baseline.float()).norm()))
            add('mlp_delta',block_distance)
            return (changed,)

        layer.self_attn.o_proj.register_forward_pre_hook(edit_o)
        layer.mlp.down_proj.register_forward_pre_hook(edit_mlp)
        return layer

    def step(b, layer, x, y, rr, length, offset=0, **_):
        if b>=0 and offset+len(rr)==len(rows) and (b+1)%5==0:
            progress(b+1)

    rt.block=hook_layer
    try:
        _, logits=rt.forward(rows,step)
    finally:
        rt.block=origin
    trials=[]
    for row,logit,dose,mlp in zip(rows,logits,sum(saved['dose'],[]),sum(saved['mlp_delta'],[]),strict=True):
        values=logit[row['choice_ids']].cpu().tolist()
        gold=row['target_record']
        token=int(logit.argmax())
        trials.append(dict(dose,case_id=row['id'],family_id=row['family'],arm=row['arm'],margin=values[gold]-max(values[:gold]+values[gold+1:]),
                           output_id=token,correct=token==row['gold_id'],mlp_distance=mlp,
                           logits=values,logits_sha256=hashlib.sha256(logit.contiguous().cpu().numpy()).hexdigest()))
    return trials


def estimate(rows):
    groups={arm:sorted((r for r in rows if r['arm']==arm),key=lambda r:r['family_id']) for arm in ARMS}
    keys=[[r['family_id'] for r in entries] for entries in groups.values()]
    if any(k!=keys[0] for k in keys) or len(keys[0])!=192:
        raise ValueError('干预臂没有完整家族配对')
    vector=lambda arm,k:np.array([row[k] for row in groups[arm]])
    base,target,random=(vector(k,'margin') for k in ('base','target','random'))
    dose,ctl=vector('target','dose'),vector('random','dose')
    match=float(((dose>0)&(np.abs(ctl-dose)<=.10*dose)).mean())
    exact={k:bool(np.array_equal(vector('base','logits_sha256'),vector(k,'logits_sha256'))) for k in ('sham','restore')}
    baseline_accuracy=float(vector('base','correct').mean())
    rng=np.random.default_rng(SEED+2)
    ix=rng.integers(len(base),size=(10000,len(base)))
    diff=random-target
    ci=np.quantile(diff[ix].mean(1),[.0125,.9875]).tolist()
    recovery=vector('restore_mlp','margin')-target
    return {'dose_match':match,'exact_sham_and_restore':exact,'margin_drop_over_random':float(diff.mean()),
            'interval':ci,'causal_gate_passed':baseline_accuracy>=.90 and match>=.95 and all(exact.values()) and ci[0]>0,
            'mlp_distance':{k:float(vector(k,'mlp_distance').mean()) for k in ARMS},
            'separate_mlp_restore_margin_gain':float(recovery.mean()),
            'accuracy':{k:float(vector(k,'correct').mean()) for k in ARMS}}


def execute():
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    from source_binding_design import MODELS
    from source_binding_resources import resources
    from source_binding_runtime import Runtime

    work,questions,plan,run_id=registered_context(code_identity,Path(__file__))
    metadata=plan['analysis']['selection']
    path=Path(metadata['candidate'])
    if sha(path)!=metadata['candidate_sha256'] or snapshot(MODELS['qwen14'])!=plan['acquisition']['model_snapshot']:
        raise ValueError('干预候选或模型版本不符')
    rows=read(questions)
    rt=Runtime('qwen14')
    rt.t.backends.cuda.matmul.allow_tf32=False
    with np.load(path,allow_pickle=False) as candidate:
        units=rt.t.as_tensor(candidate['units'],device='cuda',dtype=rt.t.long)
    observations=[]
    zero=time.monotonic()
    total=40*((len(rows)+BATCH-1)//BATCH)
    with GpuLease(),rt.t.inference_mode(),(work/'case_results.jsonl').open('x',encoding='utf-8',buffering=1) as stream:
        for shard,lo in enumerate(range(0,len(rows),BATCH)):
            def progress(block):
                done=40*shard+block
                elapsed=time.monotonic()-zero
                print(json.dumps({'stage':'causal','completed':done,'total':total,'elapsed':elapsed,
                                  'eta':elapsed*(total-done)/done if done else None,**resources()}),flush=True)
            progress(0)
            trial=collect(rt,rows[lo:lo+BATCH],metadata['block'],metadata['head'],units,progress)
            observations.extend(trial)
            for row in trial:
                stream.write(json.dumps(dict(row,run_id=run_id),ensure_ascii=False)+'\n')
    measures=estimate(observations)
    result={'run_id':run_id,'families':192,'block':metadata['block'],'head':metadata['head'],
            'prediction_run':metadata['prediction_run'],'causal':measures}
    save(work/'matched-effects.json',measures)
    save(work/'artifacts.json',{'schema':'research_test_files.v1','files':[
        {'path':'matched-effects.json','role':'evidence','retention':'important'}]})
    save(work/'conclusion.json',result)


if __name__ == '__main__':
    execute()
