"""All-coordinate native/common adjoints, full hidden fields and exact V factors."""
import gc,json,shutil
from collections import defaultdict
import numpy as np
import torch
from phase2620_native_coordinate_contract import *
from phase2622_native_field_capture import arr
from phase2632_fulltoken_native_adjoints import AdjointCapture,LAYERS
from phase2636_precision_engine import load_precision

MATERIAL=RESULT/'phase2641_matched_operation_contract'
BF=RESULT/'phase2642_matched_operation_behavior'
INITIAL=RESULT/'phase2643_matched_dual_adjoints'
CONFIRM=RESULT/'phase2645_confirmation_dual_adjoints'

def clear_grads(em,capture):
    for v in [em]+capture.hidden+capture.a+[y for x,y in capture.linears.values()]:v.grad=None

def gradients(em,capture,positions,prefix):
    out={prefix+'__hidden_adjoint_positions':np.stack([arr(v.grad[0,positions]) for v in [em]+capture.hidden]),
         prefix+'__mlp_adjoint_positions':np.stack([arr(v.grad[0,positions]) for v in capture.a])}
    for l in LAYERS:out[f'{prefix}__L{l}_v_g']=arr(capture.linears[(l,'v_proj')][1].grad[0])
    return out

def behavioral_audit():
    rows=[json.loads(s) for s in (BF/'behavior/greedy.jsonl').read_text(encoding='utf-8').splitlines()]
    cells=defaultdict(list)
    for r in rows:cells[(r['family'],r['language'],r['unit'],r['form'])].append((r['target_index'],r['mention_order']))
    audit={'matched_cells':len(cells),'all_four_cells_verified':len(cells)==1024 and all(len(v)==4 and set(v)=={(0,0),(0,1),(1,0),(1,1)} for v in cells.values()),
           'all_cases_unique':len({r['case_id'] for r in rows})==4096,'strict_correct':sum(r['strict_correct'] for r in rows),'content_correct':sum(r['name_content_correct'] for r in rows),'n':len(rows),
           'note':'Independent post-run verification replaces a formerly literal True check in Phase2642 source; no model rerun or historical result rewriting.'}
    assert audit['all_four_cells_verified'] and audit['all_cases_unique']
    save(BF/'analysis/factorial_audit.json',audit)
    return audit

def run(phase,field_set,out):
    if (out/'analysis/final.json').exists():raise RuntimeError('phase already completed')
    out.joinpath('field').mkdir(parents=True,exist_ok=True)
    cases=[r for r in read(MATERIAL/'material/cases.json') if r['field_set']==field_set]
    firsts={r['case_index']:r for r in read(BF/'analysis/first_decisions.json')}
    audit=behavioral_audit()
    save(out/'protocol/frozen.json',{'field_set':field_set,'case_ids':[r['case_id'] for r in cases],'material_sha256':sha(MATERIAL/'material/cases.json'),
        'native':'frozen BF16 natural top1/top2 IDs, FP32 arithmetic adjoint; NOT the FP32 model own native choice',
        'common':'canonical entityA first token minus entityB first token, no sign orientation by answer',
        'positions':'last subtoken of first A and B mention, plus prompt boundary','full_hidden_tokens':True,'V_layers':LAYERS,
        'shared_scalar_formula':'sum over every token g[t,j]*x[t,k]','all_coordinates':True,'forward_passes_per_case':1,'backward_passes_per_case':2})
    model,info=load_precision('fp32');save(out/'protocol/model.json',info)
    U=model.lm_head.weight.detach();capture=AdjointCapture(model);records=[];manifest=[]
    for i,row in enumerate(cases):
        if shutil.disk_usage(out).free<8*1024**3:raise RuntimeError('8GiB safety floor reached; preserve collected fields')
        capture.reset();torch.cuda.empty_cache();ci=row['case_index'];native=firsts[ci]['native_ids'];common=row['common_readout_ids']
        pos=[row['entity_spans']['a'][-1],row['entity_spans']['b'][-1],len(row['prompt_ids'])-1]
        with torch.no_grad():em=model.get_input_embeddings()(torch.tensor([row['prompt_ids']],device='cpu')).to('cuda:0')
        em=em.detach().requires_grad_(True)
        result=model.model(inputs_embeds=em,use_cache=False);state=result.last_hidden_state[0,-1]
        pack={'hidden':np.stack([arr(v[0]) for v in [em]+capture.hidden]),
              'mlp_positions':np.stack([arr(v[0,pos]) for v in capture.a]),'normalized_boundary':arr(state)}
        for l in LAYERS:
            x,y=capture.linears[(l,'v_proj')];pack[f'L{l}_v_x']=arr(x[0]);pack[f'L{l}_v_value']=arr(y[0])
        native_loss=(state*(U[native[0]]-U[native[1]]).to('cuda:0')).sum()
        native_loss.backward(retain_graph=row['common_readout_available']);pack.update(gradients(em,capture,pos,'native'))
        cm=None
        if row['common_readout_available']:
            clear_grads(em,capture)
            common_loss=(state*(U[common[0]]-U[common[1]]).to('cuda:0')).sum()
            common_loss.backward();cm=float(common_loss.detach());pack.update(gradients(em,capture,pos,'common'))
        with torch.no_grad():
            logits=U@state.detach().cpu();chosen=int(logits.argmax());logits[chosen]=-float('inf');runner=int(logits.argmax())
        with np.load(BF/f'field/case_{ci:04d}.npz',allow_pickle=False) as bf:
            embedding_exact=np.array_equal(pack['hidden'][0],bf['hidden'][0])
            h_error=float(np.linalg.norm(pack['hidden']-bf['hidden'])/max(np.linalg.norm(bf['hidden']),1e-30))
        assert embedding_exact and all(np.isfinite(v).all() for v in pack.values())
        path=out/f'field/case_{ci:04d}.npz';np.savez(path,**pack)
        manifest.append({'path':str(path),'bytes':path.stat().st_size,'case_index':ci,'field_set':field_set})
        records.append({'case_index':ci,'case_id':row['case_id'],'positions':pos,'native_ids':native,'common_ids':common,
            'common_available':row['common_readout_available'],'native_margin':float(native_loss.detach()),'common_margin':cm,
            'fp32_own_top2':[chosen,runner],'bf16_native_ids':native,'bf16_vs_fp32_h_relative_l2':h_error,'embedding_exact':embedding_exact,
            'native_common_identity':'same' if native==common else ('opposite' if native==common[::-1] else 'different'),
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated()})
        clear_grads(em,capture);capture.reset()
        del em,result,state,pack,x,y,native_loss,logits
        if cm is not None:del common_loss
        if (i+1)%16==0:
            gc.collect();save(out/'analysis/progress.json',{'cases':i+1,'total':len(cases)})
            print(field_set,'dual adjoints',i+1,'/',len(cases),flush=True)
    capture.close();no_param_grad=all(p.grad is None for p in model.parameters())
    del model,U,capture;gc.collect();torch.cuda.empty_cache()
    save(out/'analysis/records.json',records);save(out/'analysis/raw_manifest.json',manifest)
    checks={'all512_cases':len(records)==512,'all_embedding_values_exact':all(r['embedding_exact'] for r in records),
            'no_parameter_gradients':no_param_grad,'same28_weight_matrices':all(info['all28_weight_values_exact'].values()),'behavior_four_cells_verified':audit['all_four_cells_verified']}
    assert all(checks.values())
    summary={'cases':len(records),'full_hidden_shape_per_case':'37 x all T x2560','mlp_positions_shape':'36 x3 x9728','V_factors':'4layers x allT x(2560input +1024value +2x1024adjoint)',
        'fp32_first_id_changed':sum(r['fp32_own_top2'][0]!=r['bf16_native_ids'][0] for r in records),
        'native_common_identity':{k:sum(r['native_common_identity']==k for r in records) for k in ('same','opposite','different')},
        'mean_bf16_fp32_h_relative_l2':float(np.mean([r['bf16_vs_fp32_h_relative_l2'] for r in records])),'behavior_audit':audit}
    finish(phase,('初始' if field_set=='initial' else '独立实体扩大')+'512条件全token原场与双读出逐坐标伴随',out,{'provenance':str(Path(__file__)),'summary':summary,'checks':checks},
        '以与磁盘BF16完全同值的FP32模型计算数值图谱。全部token隐藏场、实体A/B/输出边界的全部隐藏坐标及MLP单位、四层V全部token精确参数因子。两个读出顺次反传并清空中间梯度，绝不混合求导。',
        r'\bar H^{r}_{l,t,j}=\partial m_r/\partial H_{l,t,j};\quad G^{r,l}_{jk}=\sum_t\bar V^{r,l}_{t,j}X^l_{t,k};\quad r\in\{\mathrm{BF16nativeIDs},A_0-B_0\}.',
        '八语言族×中英×四对人名×两句式×两个正确目标×两种提及顺序=512。初始单位0/1/30/31，扩大单位12/13/14/15，分割预先冻结。所有行为失败保留；H全token、梯度在三个明确位置全坐标；V因子全token。',
        '共同A/B读出是实验者条件，若恰等于原生输出对则梯度相同是代数恒等，不算语义新发现。FP32本身可能改变原生首token，逐例另记；原始BF16完整名字生成成绩独立保留。2642四格验证原先代码常量已被独立4096行审计替代，历史文件未重写。',
        '只有Qwen3-4B的短句人名选择，不代表跨模型或开放语言。隐藏梯度只保留三个锚点，不声称全token梯度都存储；隐藏值与V参数导数确为全token。FP32改变离散计算而非恢复未知训练精度。',
        '继续用全坐标目标响应、顺序/句式对照、同实体跨族与不同实体留出分别分析，不将输出同一性或数值可预测性叫作语义齿轮。')

if __name__=='__main__':run(2643,'initial',INITIAL)
