"""Native shared-weight finite interventions: whole-token vs boundary-only predictors."""
import json,numpy as np,torch
from phase2620_native_coordinate_contract import *
from phase2621_native_behavior_run import load_model
from phase2622_native_field_capture import arr
from phase2624_scalar_forward_validation import digest_tensor
from phase2632_fulltoken_native_adjoints import module_at,LAYERS,MODULES

SOURCE=RESULT/'phase2632_fulltoken_native_adjoints'
OUT=RESULT/'phase2633_shared_weight_forward_tests'

@torch.inference_mode()
def run(model,frames,records,out):
    info=read(SOURCE/'protocol/weights.json');before={f'L{l}_{n}':digest_tensor(module_at(model,l,n).weight) for l in LAYERS for n in MODULES}
    outputs=[]
    def forward(ids,y,z):
        result=model.model(input_ids=ids,use_cache=False);state=result.last_hidden_state[0,-1]
        logits=model.lm_head(state).float();contrast=model.lm_head.weight[y].float()-model.lm_head.weight[z].float()
        return float(logits[y]-logits[z]),float((state.float()*contrast).sum()),int(logits.argmax()),arr(state)
    for fi,frame in enumerate(frames):
        sr=records[frame['frame_id']];y,z=frame['chosen_id'],frame['runnerup_id'];ids=torch.tensor([frame['prefix_ids']],device='cuda:0')
        base16,base32,baseid,bstate=forward(ids,y,z);noop16,noop32,_,noopstate=forward(ids,y,z)
        outputs.append({'frame_id':frame['frame_id'],'family':frame['family'],'language':frame['language'],'kind':'noop',
            'margin16_change':noop16-base16,'margin32_change':noop32-base32,'state_l2_change':float(np.linalg.norm(noopstate-bstate))})
        for site in sr['sites']:
            l=site['layer'];name=site['module'];W=module_at(model,l,name).weight;rms=info[f'L{l}_{name}']['rms']
            for selector in ('diagnostic_max','matched'):
                j=site[selector+'_j'] if selector=='diagnostic_max' else site['matched_index_j']
                k=site[selector+'_k'] if selector=='diagnostic_max' else site['matched_index_k']
                g=site[selector+'_full_gradient'];lastg=site[selector+'_last_gradient']
                for scale in (.2,1.0):
                    for sign in (-1,1):
                        original=W[j,k].clone()
                        try:
                            W[j,k]=(original.float()+sign*scale*rms).to(W.dtype);actual=float(W[j,k].float()-original.float())
                            observed16,observed32,newid,state=forward(ids,y,z)
                            delta16=observed16-base16;delta32=observed32-base32
                            outputs.append({'frame_id':frame['frame_id'],'case_id':frame['case_id'],'family':frame['family'],'language':frame['language'],
                                'kind':'shared_weight','layer':l,'module':name,'selector':selector,'j':j,'k':k,'rms_scale':scale,'sign':sign,
                                'original_weight':float(original),'actual_weight_delta':actual,'gradient_full':g,'gradient_boundary':lastg,
                                'predicted_full':g*actual,'predicted_boundary':lastg*actual,'margin16_change':delta16,'margin32_change':delta32,
                                'full_absolute_error32':abs(delta32-g*actual),'boundary_absolute_error32':abs(delta32-lastg*actual),
                                'state_l2_change':float(np.linalg.norm(state-bstate)),'native_next_token_changed':newid!=baseid,
                                'baseline32_vs_adjoint_error':abs(base32-sr['fp32_loss'])})
                        finally:W[j,k].copy_(original)
        save(Path(out)/'analysis/progress.json',{'frames':fi+1,'total':len(frames),'forwards':len(outputs)})
        print('shared weight',fi+1,'/',len(frames),'forwards',len(outputs),flush=True)
    after={f'L{l}_{n}':digest_tensor(module_at(model,l,n).weight) for l in LAYERS for n in MODULES};assert before==after
    save(Path(out)/'analysis/interventions.json',outputs);save(Path(out)/'analysis/weight_restoration.json',{'before':before,'after':after,'disk_weights_changed':False})
    return outputs

def summarize(outputs):
    results={}
    for key in ['all','diagnostic_max','matched']+[f'L{l}/{n}' for l in LAYERS for n in MODULES]:
        rr=[r for r in outputs if r['kind']=='shared_weight' and (key=='all' or key==r['selector'] or key==f'L{r["layer"]}/{r["module"]}')]
        den=sum(abs(r['margin32_change']) for r in rr);strong=[r for r in rr if abs(r['margin32_change'])>=.05]
        results[key]={'n':len(rr),'fp32_head_nonzero_while_bf16_head_zero':sum(abs(r['margin32_change'])>1e-6 and r['margin16_change']==0 for r in rr),
            'internal_state_nonzero':sum(r['state_l2_change']>0 for r in rr),'next_token_changed':sum(r['native_next_token_changed'] for r in rr),
            'full_gradient_l1_relative_error':sum(r['full_absolute_error32'] for r in rr)/den if den else None,
            'boundary_only_l1_relative_error':sum(r['boundary_absolute_error32'] for r in rr)/den if den else None,
            'n_effect_ge_0.05':len(strong),'full_sign_agreement_effect_ge_0.05':float(np.mean([np.sign(r['margin32_change'])==np.sign(r['predicted_full']) for r in strong])) if strong else None,
            'mean_absolute_effect32':den/len(rr)}
    return results

def main():
    allframes=read(SOURCE/'material/frames.json');records={r['frame_id']:r for r in read(SOURCE/'analysis/records.json')}
    frames=[f for f in allframes if f['index']==24 and f['variant']==0 and f['step']==0]
    save(OUT/'protocol/frozen.json',{'frames':[f['frame_id'] for f in frames],'sites':28,'selectors':['diagnostic_max','matched_index'],
        'rms_scales':[.2,1.0],'signs':[-1,1],'maximum_gradient_selection_is_numeric_diagnostic_only':True,'no_donor':True,'no_op_per_frame':True})
    model,tok=load_model('qwen4');outputs=run(model,frames,records,OUT);summary=summarize(outputs)
    result={'provenance':str(Path(__file__)),'summary':summary,'checks':{'all16_frames':len(frames)==16,'all3600_condition_forwards':len(outputs)==3600,
        'all_noops_identical':all(r['margin16_change']==0 and r['margin32_change']==0 and r['state_l2_change']==0 for r in outputs if r['kind']=='noop'),
        'all28_weight_hashes_restored':read(OUT/'analysis/weight_restoration.json')['before']==read(OUT/'analysis/weight_restoration.json')['after']}}
    finish(2633,'早中晚28矩阵3600次真实单权重前向与全token求和裁决',OUT,result,
        '每个测试只临时改一个真实共享标量参数，作用于该前缀全部token；同时读原BF16头margin、同状态FP32诊断头margin和原生归一化状态变化。比较完整token求和导数与末位置近似，恢复全部权重哈希。',
        r'\Delta m_{full}\approx\Delta W_{jk}\sum_t\bar Y_{t,j}X_{t,k},\quad\Delta m_{last}\approx\Delta W_{jk}\bar Y_{T,j}X_{T,k};\quad\Delta W=\pm\alpha\operatorname{RMS}(W),\ \alpha\in\{0.2,1\}.',
        '八族中英各1个index24/variant0/step0例，共16；每例28矩阵×两个坐标选法×两幅度×双方向，加no-op=225，共3600。最大导数坐标按本例全矩阵扫描选，仅数值诊断；匹配索引对照同幅度，不把选中坐标叫主干。',
        '可以直接判断小效应在内部是否已经改变，而不是只看被末端BF16量化后的margin。全token共享参数测试把原生坐标算法推进到早层attention及MLP，不依赖donor激活或替换方向。',
        '一倍权重RMS的单参数变化相对该标量可大，非无限小；梯度误差可来自曲率和中间舍入。每组只有一个数值审计例，不能作普遍语义结论；native top2可能是格式。最大梯度选点有条件选择偏差，所有全矩阵图谱仍保留。',
        '将可重复的全token共享参数规律与行为、词形和输出身份分开检验，再扩大到另一组新组合及发布跨位置参数查询。')

if __name__=='__main__':main()
