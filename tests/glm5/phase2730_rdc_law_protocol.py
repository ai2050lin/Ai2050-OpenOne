"""Prospective native deployment, model replication and architecture-scope tests."""
from rdc_law_common import *


def protocol():
    path=BASE/'deployment/protocol.json'
    if path.exists():return read(path)
    main=gzread(BASE/'material.json.gz');fresh=gzread(BASE/'confirmation_material.json.gz')
    natural=[];scale=[]
    for cohort in ('gum','ewt','cmrc'):
        pool=sorted([r for r in fresh if r['cohort']==cohort],key=lambda r:rank('deploy:'+r['sample_id']))
        if cohort!='cmrc':
            natural.extend([r for flag in (True,False) for r in [x for x in pool if bool(x['held_relation_combinations'])==flag][:8]])
        else:natural.extend(pool[:16])
        for split,n in [('train',24),('validation',8),('confirmation',16)]:
            pp=fresh if split=='confirmation' else main
            candidates=sorted([r for r in pp if r['cohort']==cohort and r['split']==split],key=lambda r:rank('scale:'+r['sample_id']))
            if split=='confirmation' and cohort!='cmrc':
                chosen=[r for flag in (True,False) for r in [x for x in candidates if bool(x['held_relation_combinations'])==flag][:8]]
            else:chosen=candidates[:n]
            assert len(chosen)==n;scale.extend(chosen)
    qa=[r for r in main if r['kind']=='QA' and r['split']=='test'];assert len(natural)==48 and len(qa)==48 and len(scale)==144
    scaleqa=[]
    for cohort in ('squad_qa','cmrc_qa','hotpot_qa'):
        pool=sorted([r for r in qa if r['cohort']==cohort],key=lambda r:rank('scaleqa:'+r['sample_id']))
        if cohort=='hotpot_qa':scaleqa.extend([r for typ in ('bridge','comparison') for r in [x for x in pool if x['question_type']==typ][:4]])
        else:scaleqa.extend(pool[:8])
    p={'timestamp':stamp(),'source':snapshot(Path(__file__)),'natural_ids':[r['sample_id'] for r in natural],
        'QA_ids':[r['sample_id'] for r in qa],'scale_ids':[r['sample_id'] for r in scale],'scale_QA_ids':[r['sample_id'] for r in scaleqa],
        'rollout_branches':['native','early_prediction_L16','early_prediction_L35','coherent_seed2728','prefix_order_control_seed2728','coherent_seed2729','prefix_order_control_seed2729'],
        'rollout':'Greedy with native EOS,48new tokens max. Natural prefix ends at final preregistered anchor; QA full original prompt. Each branch has independent own-selected IDs/KV; no reference-state refresh.',
        'learned_deployment':'Frozen block16/35 validation winners consume only actual H12 query,embedding,allcausalH12history and known task/language/position. Replace respective native MLP output at every prefill/decode position; untouched modules stay native.',
        'training_deployment':'All three native last-MLP matrices set to originalBF16 interpretedFP32 plus saved64stepdelta, then castBF16. Originalcheckpointfiles remain read-only. All four fixed runs deployed; no confirmation-best selection.',
        'scope_audit_rows':{'natural':12,'QA':12},'scope_audit':'Same exact known token history, compare final-query-only versus all-prefill learned replacement and trained last-MLP; inspect every layer/key/value coordinate and prompt logits, then matched known next-ID decode. Block16 is upstream-of-upper-KV contrast.',
        'predeclared_architecture_prediction':'LastMLP follows every attention computation; changing only it cannot directly change any layerKV on identical known history. Thus final-query-only and all-prefill lastMLP must have equal query output and KV modulo explicitly measured execution rounding. A middleMLP may alter upper-layer KV. KV differences between independently chosen token histories are not evidence of direct memory corruption.',
        'automatic_followup':'After main integrated measurements, apply same-history native reference on autonomous approximate histories to separate output-rule error from directKVdrift; do not inject compensatingKV. This is same authorized mechanism question, not an assumed hallucination cure.',
        'scale':'qwen4,qwen14,glm4 strictly sequential nonquantizedCUDA withCPU/offloadifneeded.144same natural windows each72train24validation48confirmation,3character-aligned anchors;24same realQA,32new tokens cap. Own widths/units/layers retained, nocoordinateisomorphism.',
        'scale_predictors':'Own Hfloor(depth/3) and available all-source mean; earlylinear/additive/multiplicative/taskconditioned and one shuffled-history control, effectiveDF32/128; directMLP/predictedxnative/productoffactors/jointproduct, fullnativeunits. Validation selects before own confirmation capture. Common protocol fixed now; Q4 confirmation already observed so matched-subsetQ4 is a reanalysis, not a second independent confirmation.',
        'resource':'Per-model2natural+1QA pilot first, predict remainingtime/diskbeforeexpansion. Record slowoffloadratherthanquantize.168sources per model are limited replication,not rerunning allfull4Btraining. FullKV arrays auditedstreaming/hashes, not duplicatedallcaches on disk.',
        'selection_boundary':'Protocol fixed after offline confirmation became available but before any new live rollout,scopeorlargermodeloutput. Dataset selections hash-only; not conditioned on correctness or prediction errors.'}
    immutable(path,p);print('LAW_DEPLOYMENT_PROTOCOL_FROZEN',len(scale),len(scaleqa),len(natural)+len(qa),flush=True);return p


if __name__=='__main__':protocol()
