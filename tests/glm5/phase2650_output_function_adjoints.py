"""Full physical coordinates, dual objectives and all-token scalar V factors."""
import gc,shutil
from collections import defaultdict
import numpy as np
import torch
from phase2620_native_coordinate_contract import *
from phase2622_native_field_capture import arr
from phase2632_fulltoken_native_adjoints import AdjointCapture,LAYERS
from phase2636_precision_engine import load_precision
from phase2643_matched_dual_adjoint_engine import clear_grads

MATERIAL=RESULT/'phase2648_output_function_contract';BF=RESULT/'phase2649_output_function_behavior'
INITIAL=RESULT/'phase2650_output_function_adjoints';CONFIRM=RESULT/'phase2652_output_function_confirmation'

def grads(em,cap,label):
    p={label+'__hidden_adjoint_boundary':np.stack([arr(v.grad[0,-1]) for v in [em]+cap.hidden]),
       label+'__mlp_adjoint_boundary':np.stack([arr(v.grad[0,-1]) for v in cap.a])}
    for l in LAYERS:p[f'{label}__L{l}_v_g']=arr(cap.linears[(l,'v_proj')][1].grad[0])
    return p

def capture_set(field_set,out):
    if (out/'analysis/capture_completion.json').exists():
        prior=read(out/'analysis/capture_completion.json');assert prior['all_checks_passed'];return prior
    cases=[r for r in read(MATERIAL/'material/cases.json') if r['field_set']==field_set];firsts={r['case_index']:r for r in read(BF/'analysis/first_decisions.json')}
    out.joinpath('field').mkdir(parents=True,exist_ok=True)
    save(out/'protocol/frozen.json',{'field_set':field_set,'case_ids':[r['case_id'] for r in cases],'material_sha256':sha(MATERIAL/'material/cases.json'),
        'native':'frozen actual BF16 native output IDs on FP32 core','common':'fixedYes-No/是-否 fortruth; entityA-B forname/cloze, no truth-oriented sign applied to gradients',
        'scope':'H3anchors, MLPboundary and both gradients atboundary; alltoken V factors4layers; alltoken h scanned into coordinateRMS;64initial exemplary fullhidden arrays only',
        'fixed_rule':'2648 preregistered coordinate envelope and semantic/truth-oriented signs; no effect selection'})
    model,info=load_precision('fp32');save(out/'protocol/model.json',info);U=model.lm_head.weight.detach();cap=AdjointCapture(model)
    records=[];manifest=[];alltoken={};token_counts=defaultdict(int)
    for i,r in enumerate(cases):
        if shutil.disk_usage(out).free<8*1024**3:raise RuntimeError('8GiB disk floor')
        torch.cuda.empty_cache();cap.reset();ci=r['case_index'];native=firsts[ci]['native_ids'];common=r['common_readout_ids'];assert r['common_readout_available']
        pos=[r['entity_spans']['a'][-1],r['entity_spans']['b'][-1],len(r['prompt_ids'])-1]
        with torch.no_grad():em=model.get_input_embeddings()(torch.tensor([r['prompt_ids']],device='cpu')).to('cuda:0')
        em=em.detach().requires_grad_(True);result=model.model(inputs_embeds=em,use_cache=False);state=result.last_hidden_state[0,-1]
        h=np.stack([arr(v[0]) for v in [em]+cap.hidden]);pack={'hidden_positions':h[:,pos],'mlp_boundary':np.stack([arr(v[0,-1]) for v in cap.a]),'normalized_boundary':arr(state)}
        if r['published']:pack['hidden_fulltoken']=h
        key=r['family']+'/'+r['language']+'/'+r['mode'];hh=h.astype('float64')
        if key not in alltoken:alltoken[key]=[np.zeros((37,2560)),np.zeros((37,2560))]
        alltoken[key][0]+=hh.sum(1);alltoken[key][1]+=(hh*hh).sum(1);token_counts[key]+=h.shape[1]
        for l in LAYERS:
            x,y=cap.linears[(l,'v_proj')];pack[f'L{l}_v_x']=arr(x[0]);pack[f'L{l}_v_value']=arr(y[0])
        loss=(state*(U[native[0]]-U[native[1]]).to('cuda:0')).sum();loss.backward(retain_graph=True);pack.update(grads(em,cap,'native'))
        clear_grads(em,cap);cm=(state*(U[common[0]]-U[common[1]]).to('cuda:0')).sum();cm.backward();pack.update(grads(em,cap,'common'))
        identity='same' if native==common else 'opposite' if native==common[::-1] else 'different';identity_error=None
        if identity!='different':
            sign=1 if identity=='same' else -1;identity_error=max(float(np.max(np.abs(pack['native__'+suffix]-sign*pack['common__'+suffix]))) for suffix in ('hidden_adjoint_boundary','mlp_adjoint_boundary')+tuple(f'L{l}_v_g' for l in LAYERS));assert identity_error==0
        with torch.no_grad():logits=U@state.detach().cpu();chosen=int(logits.argmax());logits[chosen]=-float('inf');runner=int(logits.argmax())
        with np.load(BF/f'field/case_{ci:04d}.npz') as b:
            bh=b['hidden_positions'];embedding_exact=np.array_equal(bh[0],pack['hidden_positions'][0]);error=float(np.linalg.norm(bh-pack['hidden_positions'])/max(np.linalg.norm(bh),1e-30))
        assert embedding_exact and all(np.isfinite(v).all() for v in pack.values())
        path=out/f'field/case_{ci:04d}.npz';np.savez(path,**pack);manifest.append({'path':str(path),'bytes':path.stat().st_size,'case_index':ci,'published':r['published'],'field_set':field_set})
        records.append({'case_index':ci,'case_id':r['case_id'],'mode':r['mode'],'positions':pos,'native_ids':native,'common_ids':common,'native_common_identity':identity,
            'identity_max_absolute_error':identity_error,'native_margin':float(loss.detach()),'common_margin':float(cm.detach()),'fp32_own_top2':[chosen,runner],
            'embedding_exact':embedding_exact,'bf16_fp32_hidden_relative_l2':error,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated()})
        clear_grads(em,cap);cap.reset();del em,result,state,loss,cm,pack,h,hh,x,y,logits
        if (i+1)%32==0:
            gc.collect();save(out/'analysis/progress.json',{'cases':i+1,'total':len(cases)});print(field_set,'output adjoints',i+1,'/',len(cases),flush=True)
    cap.close();no_grad=all(p.grad is None for p in model.parameters());del model,U,cap;gc.collect();torch.cuda.empty_cache()
    save(out/'analysis/records.json',records);save(out/'analysis/raw_manifest.json',manifest);save(out/'analysis/alltoken_counts.json',dict(token_counts))
    np.savez(out/'field/alltoken_coordinate_maps.npz',**{k+'__'+kind:(s[0]/token_counts[k] if kind=='mean' else np.sqrt(s[1]/token_counts[k])).astype('float32') for k,s in alltoken.items() for kind in ('mean','rms')})
    checks={'all2048_cases':len(records)==2048,'64_fullcoordinate_groups':len(alltoken)==64,'no_weight_gradients':no_grad,'all_embeddings_exact':all(r['embedding_exact'] for r in records),'readout_identity_exact':all(r['identity_max_absolute_error'] in (None,0) for r in records),'all28_weight_values_exact':all(info['all28_weight_values_exact'].values())}
    assert all(checks.values());summary={'cases':len(records),'alltoken_occurrences':sum(token_counts.values()),'fp32_first_id_changes':sum(rec['fp32_own_top2'][0]!=rec['native_ids'][0] for rec in records),
        'identity_counts':{mode:{key:sum(r['mode']==mode and r['native_common_identity']==key for r in records) for key in ('same','opposite','different')} for mode in ('name','cloze','truth_a','truth_b')}}
    report={'summary':summary,'checks':checks,'all_checks_passed':True};save(out/'analysis/capture_completion.json',report);return report

def main():
    r=capture_set('initial',INITIAL)
    finish(2650,'初始2048输出功能条件原生坐标与双读出全token参数伴随',INITIAL,{'provenance':str(Path(__file__)),'summary':r['summary'],'checks':r['checks']},
        '同值FP32参数核心，用原生BF16输出IDs及固定任务读出各自反传、清空伴随再切换目标。全部H/MLP边界坐标、全部V共享参数的全token精确因子，按64族语言模式流式保留完整token坐标RMS。',
        r'\bar H^r_{l,T,j}=\partial m_r/\partial H_{l,T,j};\quad G^r_{l,jk}=\sum_{t=0}^{T}\bar V^r_{l,t,j}X_{l,t,k};\quad r\in\{nativeIDs,task\}.',
        '预定单位0..3×八族×双语×双句式/目标/顺序×四模式=2048。H3锚点、全部9728MLP边界单位、四层V全部token，64示例完整H保留。两个读出相同/相反时逐坐标伴随恒等校验。',
        '固定truth输出头行不随实体改变，而name/cloze读出仍依人名，保留这一实质区别。FP32自己的首位改变另记，不覆盖原始BF16完整答案；相同读出伴随相等是工程核验而非共享语义。',
        '仅Qwen4B，主要梯度只存输出边界，未声称所有token的全部H梯度持久化。全token V因子精确，但不表示已找出语义参数组；FP32仍改变前向基线。',
        '按照冻结规则测绘目标/顺序/句式、语义目标与真值方向、跨实体与跨输出功能；不用高相似度自动命名机制。')

if __name__=='__main__':main()
