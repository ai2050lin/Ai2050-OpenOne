"""Sequential native gated-MLP replication, no crossmodel coordinate-index matching."""
import argparse,gc,itertools,shutil
import numpy as np
import torch
from phase2620_native_coordinate_contract import *
from phase2621_native_behavior_run import MODELS
from phase2670_native_mlp_contract import OUT as CONTRACT,encoded
from phase2671_native_mlp_field import run,unbits
from phase2662_symmetric_mapping_contract import load_native
from phase2663_symmetric_mapping_calibration import behavior_groups

OUT=RESULT/'phase2675_native_mlp_crossmodel'

def run_one(key):
    folder=OUT/key;assert not (folder/'analysis/completion.json').exists()
    model,tok=load_native(key);L=model.config.num_hidden_layers;D=model.config.hidden_size;K=model.config.intermediate_size;selected=tuple(sorted({L//3,2*L//3,L-1}));mlp=model.model.layers[0].mlp
    layout='split_gate_up' if hasattr(mlp,'gate_proj') else 'merged_gate_up'
    assert hasattr(mlp,'down_proj') and (hasattr(mlp,'gate_proj') or hasattr(mlp,'gate_up_proj'))
    rows=[]
    for old in read(CONTRACT/'material/cases.json'):
        if old['unit'] not in (2,3) or (old['form'],old['mention_order'],old['probe_index'])!=(0,0,0):continue
        r=encoded(tok,{**old,'case_index':len(rows),'published':old['published'] and old['family'] in ('chronology','word_sense')});rows.append(r)
    assert len(rows)==512 and sum(r['published'] for r in rows)==4
    eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos or []);maxT=max(len(r['prompt_ids']) for r in rows)
    budget=512*2*((L+1)*D+L*K)+4*maxT*2*((L+1)*D+len(selected)*(3*K+2*D))+16*16*((L+1)*D+3*L*K+2*L*D)
    assert shutil.disk_usage(ROOT).free>budget+8*1024**3,('crossmodel_disk',key,budget)
    save(folder/'material/cases.json',rows);save(folder/'protocol/model.json',{'model':MODELS[key],'dtype':str(model.dtype),'nonquantized':not getattr(model,'is_quantized',False),'layers':L,'hidden':D,'mlp_units':K,
        'layout':layout,'mlp_class':str(type(mlp)),'selected_native_layers':selected,'actual_parameter_devices':sorted({str(p.device) for p in model.parameters()}),'device_map':getattr(model,'hf_device_map',None),
        'all_eos_ids':eos,'storage_upper_bytes':budget,'fulltoken_published':4,'scope':'512balancedconditions, not8192factorial. WholeH/a taskboundary percase, alltoken fullcoordinate6field moments.4fulltoken examples, all families retained. Selectedlayerfractions are samplinglocations, not matchedsemanticindices.',
        'generation':'16newtokens, no-cache, ownchattemplate; reasoning-model budget may prevent an answer. Lowbehavior not absenceofmechanism.'})
    if key=='qwen14':save(folder/'protocol/runtime_recovery.json',{'first_attempt':'Loaded BF16auto12GiB/20GiB, then CUDA CUBLAS memoryallocation failure atfirstlm_head. Durablebehaviorrecords=0; no scientificcases completed.',
        'second_attempt':'CUDA cache release passed readout; systemcommit then exhausted: NumPy failed allocating5.31MiB momentstack. Still0durablecases. OwnedCPUartifactbackend used2.969GBprivatecommit; stopped only verifiedPIDs34044/28268 beforethirdattempt, torestartbeforeliveUIchecks. Otherapplications/systempagefile unchanged.',
        'change':'Release unusedCUDAallocator cache after fullfield prefill and before offloaded readout; no live tensors, checkpointvalues, dtype, material or mathematical operation changed. All otherlargeoffloadedcompact models use same safety rule. Thirdattempt frees ownedartifactbackend commit first.'})
    rr=run(model,tok,rows,folder,selected=selected,raw_all=True,compact=True,extra_eos=eos);del model;gc.collect();torch.cuda.empty_cache()
    # Native statement-truth pattern per family/language/entity/content. No use of4B indices.
    patterns={};basecounts={};ng=0
    table={tuple(r[k] for k in ('family','language','unit','content_instance','polarity','mapping','target_index')):r for r in rows}
    for fam,lang,e,c in itertools.product(sorted({r['family'] for r in rows}),('en','zh'),(2,3),(0,1)):
        fields={}
        for q,m,v in itertools.product((0,1),repeat=3):
            r=table[fam,lang,e,c,q,m,v]
            with np.load(folder/f'field/case_{r["case_index"]:04d}.npz') as z:
                fields[q,m,v]={k:unbits(z[k+'_prompt']) if k+'_prompt' in z else unbits(z[k])[:,-1] for k in ('h','a')}
        for metric in ('h','a'):
            signs={(q,m):np.sign(fields[q,m,0][metric]-fields[q,m,1][metric]) for q,m in itertools.product((0,1),repeat=2)}
            for hypothesis in ('statement_truth','question_affirmative','answer_label'):
                good=signs[0,0]!=0
                for q,m in itertools.product((0,1),repeat=2):good&=signs[q,m]==signs[0,0]*(-1)**(0 if hypothesis=='statement_truth' else q if hypothesis=='question_affirmative' else q+m)
                name=metric+'__'+hypothesis
                if name not in patterns:patterns[name]=np.zeros_like(good,dtype='int16')
                patterns[name]+=good
        ng+=1
    np.savez_compressed(folder/'maps/full_coordinate_pattern_counts.npz',**patterns)
    result={'cases':len(rr),'groups':ng,'natural':behavior_groups(rr),'all64group_counts':{k:(v==ng).sum(-1).tolist() for k,v in patterns.items()},'q14_frozen':None}
    if key=='qwen14':
        with np.load(CONTRACT/'maps/frozen_masks.npz') as z:
            result['q14_frozen']={metric:{'original':int(z[mk].sum()),'same_all64_truth_pattern':int((z[mk].astype(bool)&(patterns[metric+'__statement_truth']==ng)).sum()),
                'indices':np.argwhere(z[mk].astype(bool)&(patterns[metric+'__statement_truth']==ng)).tolist()} for metric,mk in [('h','q14__hidden_boundary__statement_truth'),('a','q14__mlp_boundary__statement_truth')]}
    save(folder/'analysis/completion.json',result);print('CROSSMODEL FINISHED',key,json.dumps({'cases':len(rr),'q14_frozen':result['q14_frozen']},ensure_ascii=True),flush=True)

def finalize():
    results={k:read(OUT/k/'analysis/completion.json') for k in ('qwen14','glm4','ds7')};models={k:read(OUT/k/'protocol/model.json') for k in results}
    checks={'three_models_512each':all(v['cases']==512 for v in results.values()),'three_models_nonquantized':all(v['nonquantized'] and v['dtype']=='torch.bfloat16' for v in models.values()),'64pattern_groups_each':all(v['groups']==64 for v in results.values()),'actual_cuda_each':all('cuda:0' in v['actual_parameter_devices'] for v in models.values())}
    assert all(checks.values())
    finish(2675,'三模型顺序非量化512条件原生MLP跨材料复验',OUT,{'provenance':str(Path(__file__)),'summary':results,'checks':checks},
        '一次只加载一个本地模型；按实际split/merged gate-up架构与本模型分词测量。512条件所有H/MLP坐标及六种全token坐标矩，事实/问题/答案方向假说分账。',
        r'N_{model}=8\times2\times2\times2\times2\times2\times2=512;\quad C_j=\sum_g\mathbf1[\operatorname{sgn}D_{g,q,m,j}=\operatorname{sgn}D_{g,0,0,j}\ne0].',
        'C001Qwen14B512；C002GLM4 512；C003DS7B512；每模型八族双语、两实体对、两内容、双目标/极性/映射。C004本模型64组全场符号模式；14B另核对冻结62H/125MLP而非移植4B下标。',
        '跨模型对照能区分某个坐标纹理属于本模型固定基底、任务支架或更广的条件关系。不同维度、不同MLP布局不是同下标同义；低胜任度结果作为解释限制保留。',
        '固定form/order/probe0，不能冒称复制8192全交叉。16token可能不足推理模型自然回答；相同任务模板不是最优chat能力比较。符号规律对其他未测形式是否成立仍未知。四例保留全token，其他逐坐标背景通过原值边界和流式矩保留。',
        '按实际证据扩大重要阳性、汇总客户端真实参数查询与全坐标热力图，终审并只清理本轮白名单中未展示原场；连续追加完成2676。')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['qwen14','glm4','ds7','finalize']);a=p.parse_args()
    if a.action=='finalize':finalize()
    else:run_one(a.action)
