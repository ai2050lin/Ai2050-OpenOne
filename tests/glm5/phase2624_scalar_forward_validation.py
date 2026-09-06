"""Temporary native scalar interventions with unconditional weight restoration.

No donor state, no replacement direction. Validate the last-block real-arithmetic
predictor against complete, same-shape BF16 model forwards. Disk weights untouched.
"""
import json
import numpy as np
import torch
from phase2620_native_coordinate_contract import *
from phase2621_native_behavior_run import load_model
from phase2622_native_field_capture import arr,tensor_output,eps_of
from phase2623_native_parameter_algorithms import field,finite_rank_one,SOURCE

OUT=RESULT/'phase2624_scalar_forward_validation'

def digest_tensor(x):return __import__('hashlib').sha256(x.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()

@torch.inference_mode()
def validate(model,tok,cases,source_records,out,source_dir=SOURCE,effect_field='neuron_delete_effect'):
    last=model.model.layers[-1];W=last.mlp.down_proj.weight
    before=digest_tensor(W);D=W.shape[0];K=W.shape[1];epsilon=eps_of(model.model.norm)
    holder={};handles=[]
    handles.append(last.mlp.down_proj.register_forward_pre_hook(lambda m,inp:holder.update(a=arr(inp[0][0,-1]))))
    handles.append(last.register_forward_hook(lambda m,inp,out:holder.update(h=arr(tensor_output(out)[0,-1]))))
    records=[]
    def forward(ids,y,z):
        output=model.model(input_ids=ids,use_cache=False)
        logits=model.lm_head(output.last_hidden_state[:,-1]).float()[0]
        return float(logits[y]-logits[z]),int(logits.argmax()),arr(output.last_hidden_state[0,-1])
    try:
        for ci,row in enumerate(cases):
            sr=source_records[row['case_id']];y,z=sr['objective_token_ids']
            ids=torch.tensor([row['prompt_ids']],device=model.get_input_embeddings().weight.device)
            base,top,normed=forward(ids,y,z);h=holder['h'].astype('float64');a=holder['a'].astype('float64')
            U=model.lm_head.weight;w=arr((U[y].float()-U[z].float())*model.model.norm.weight.float()).astype('float64')
            cs=np.argsort(abs(h),kind='stable');ns=np.argsort(abs(a),kind='stable')
            js=[int(cs[min(D-1,int((q+.5)*D/4))]) for q in range(4)]
            ks=[int(ns[min(K-1,int((q+.5)*K/4))]) for q in range(4)]
            r_h=float(np.sqrt(np.mean(h*h)));r_a=float(np.sqrt(np.mean(a*a)))
            tests=[('noop',0,0,0.0)]
            for q in range(4):
                for sign in (-1,1):
                    tests.append(('coordinate',js[q],ks[q],sign*.1*r_h))
                    tests.append(('neuron',js[q],ks[q],sign*.1*r_a))
                    for dose in (.02,.2):tests.append(('weight',js[q],ks[q],sign*dose))
            # High-effect unit is a diagnostic selected from the exhaustive formula,
            # not a discovery claim or a sparse core defining the atlas.
            effects=np.asarray(field(effect_field,source=source_dir)[sr['source_index']])
            kmax=int(np.argmax(abs(effects)));tests.append(('neuron_delete_diagnostic',0,kmax,-float(a[kmax])))
            for kind,j,k,dose in tests:
                extra=None;oldweight=None;realized=0.0
                try:
                    if kind=='coordinate':
                        def patch(module,inputs,output):
                            nonlocal realized
                            raw=tensor_output(output);modified=raw.clone();original=raw[0,-1,j].clone()
                            modified[0,-1,j]=(original.float()+dose).to(raw.dtype);realized=float(modified[0,-1,j].float()-original.float())
                            return (modified,*output[1:]) if isinstance(output,tuple) else modified
                        extra=last.register_forward_hook(patch)
                    elif kind in ('neuron','neuron_delete_diagnostic'):
                        def patch(module,inputs):
                            nonlocal realized
                            modified=inputs[0].clone();original=modified[0,-1,k].clone()
                            modified[0,-1,k]=(original.float()+dose).to(modified.dtype);realized=float(modified[0,-1,k].float()-original.float())
                            return (modified,*inputs[1:])
                        extra=last.mlp.down_proj.register_forward_pre_hook(patch)
                    elif kind=='weight':
                        oldweight=W[j,k].clone();W[j,k]=(oldweight.float()*(1+dose)).to(W.dtype)
                        realized=float(W[j,k].float()-oldweight.float())
                    observed,newtop,_=forward(ids,y,z)
                    if kind=='coordinate':v=np.eye(1,D,j,dtype='float64')[0];delta=realized
                    elif kind in ('neuron','neuron_delete_diagnostic'):v=arr(W[:,k]).astype('float64');delta=realized
                    elif kind=='weight':v=np.eye(1,D,j,dtype='float64')[0];delta=realized*a[k]
                    else:v=np.zeros(D);delta=0
                    prediction=finite_rank_one(h,w,v,delta,epsilon)
                    result={'case_id':row['case_id'],'family':row['family'],'language':row['language'],'kind':kind,'j':j,'k':k,
                        'requested_delta_or_relative_weight_dose':dose,'realized_native_delta':realized,'base_margin':base,
                        'observed_margin_change':observed-base,'predicted_margin_change':prediction,'absolute_prediction_error':abs(observed-base-prediction),
                        'native_next_token_changed':newtop!=top,'semantic_first_token_distinct':sr['semantic_first_token_distinct'],
                        'objective_token_ids':[y,z],'weight_restored_after_case':True}
                    records.append(result)
                finally:
                    if extra is not None:extra.remove()
                    if oldweight is not None:W[j,k].copy_(oldweight)
            print('native scalar validation',ci+1,'/',len(cases),flush=True)
    finally:
        for hook in handles:hook.remove()
    after=digest_tensor(W);assert before==after,'final down matrix was not restored'
    save(Path(out)/'analysis/interventions.json',records)
    summary={}
    for kind in sorted({r['kind'] for r in records}):
        rr=[r for r in records if r['kind']==kind];den=sum(abs(r['observed_margin_change']) for r in rr)
        summary[kind]={'n':len(rr),'mean_absolute_error':float(np.mean([r['absolute_prediction_error'] for r in rr])),
            'max_absolute_error':max(r['absolute_prediction_error'] for r in rr),
            'aggregate_l1_error_over_observed_effect':sum(r['absolute_prediction_error'] for r in rr)/den if den else None,
            'zero_observed_fraction':float(np.mean([r['observed_margin_change']==0 for r in rr])),
            'native_next_token_changed':float(np.mean([r['native_next_token_changed'] for r in rr])),
            'sign_agreement_for_observed_abs_ge_0.5':float(np.mean([np.sign(r['observed_margin_change'])==np.sign(r['predicted_margin_change']) for r in rr if abs(r['observed_margin_change'])>=.5])) if any(abs(r['observed_margin_change'])>=.5 for r in rr) else None}
    save(Path(out)/'analysis/weight_restore.json',{'before':before,'after':after,'disk_weights_modified':False})
    return records,summary,before==after

def main():
    cases=read(SOURCE/'material/cases.json');source=read(SOURCE/'analysis/native_records.json')
    mapping={r['case_id']:{**r,'source_index':i} for i,r in enumerate(source)}
    selected=[r for r in cases if r['index']==6 and r['form']==0]
    save(OUT/'protocol/frozen.json',{'cases':[r['case_id'] for r in selected],'selection':'32 heldout cases, every family/language/both variants; amplitude-quartile median indices for validation only',
        'conditions':'noop + 4 strata x 2 signs x (coordinate, neuron, weight .02, weight .2) + 1 exhaustive predicted largest-delete diagnostic',
        'donor_used':False,'persist_model_weights':False})
    model,tok=load_model('qwen4');records,summary,restored=validate(model,tok,selected,mapping,OUT)
    result={'provenance':str(Path(__file__)),'summary':summary,'checks':{'all32_cases':len(selected)==32,'all1088_forwards':len(records)==1088,'weights_restored':restored,'no_op_identical':all(r['observed_margin_change']==0 for r in records if r['kind']=='noop')}}
    finish(2624,'1088次完整BF16前向的单坐标、单神经元与单参数精度裁决',OUT,result,
        '只用自然recipient当前激活和真实权重；在最终block真实坐标/MLP单位上±小量，或临时修改一个down标量权重后完整重新前向。每次恢复权重，矩阵SHA256前后相同；不保存修改后的模型。',
        r'\Delta m_{pred}=m(h+\eta v)-m(h),\quad v=e_j\ \text{or}\ W_{:,k};\qquad\Delta W_{jk}\Rightarrow\eta=a_k\Delta W_{jk}.',
        '全部16语言族单元×2条件共32留出例，每例34条件=1088；坐标和神经元按幅值四分位中位索引选审计点，不用Top-K定义机制；单权重±2%/20%实际BF16变化记录。另对全神经元解析扫描预测最大删除效应者作局部诊断，明确是按本例选中。',
        '以真实前向核对解析预测，同时揭示多小的单参数变化会被BF16舍入掩盖。严格区分公式的实数准确性、实际数值实现准确性、语义功能三个层次。',
        '最后block单站点和首token margin；不是完整生成或自然必要性。±10%RMS对低激活单元相对较大，诊断零化最大效应单元有本例选择偏差。若小权重效应被舍入淹没，应记录数值分辨率，不得据此宣称参数无用。',
        '用真实误差限定逐参数工具适用尺度；随后三模型顺序复核原生计算，最终扩大新材料确认，早中层条件载体仍未闭合。')

if __name__=='__main__':main()
