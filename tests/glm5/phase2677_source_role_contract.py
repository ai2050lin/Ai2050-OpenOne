"""Freeze the next complete native-source campaign and test real instrumentation."""
import gc,inspect,shutil,time
from collections import Counter
import numpy as np
import torch
from transformers import AutoTokenizer
from phase2620_native_coordinate_contract import *
from phase2670_native_mlp_contract import LAYERS,SITES
from phase2662_symmetric_mapping_contract import load_native
from phase2677_source_role_material import build,evaluate
from phase2677_padded_native_runtime import PAD_LENGTH,MAX_NEW_TOKENS,PaddedCapture,padded_inputs,group_key
from phase2679_native_source_capture import NativeSourceCapture
from phase2679_source_coordinate_ledger import attention_ledger,conditional_norm_ledger,input_weight_ledger

OUT=RESULT/'phase2677_source_role_contract'
FIELD=RESULT/'phase2678_padded_source_field'
SOURCE=RESULT/'phase2679_native_source_ledger'


def prepare():
    previous=RESULT/'phase2676_native_mlp_delivery'
    assert read(previous/'analysis/terminal_audit.json')['all_checks_passed']
    nextplan=read(previous/'analysis/next_campaign.json')
    if (OUT/'protocol/frozen.json').exists():
        protocol=read(OUT/'protocol/frozen.json');assert sha(OUT/'material/cases.json')==protocol['material_sha256']
        return read(OUT/'material/cases.json'),protocol
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True)
    rows=build(tok);panel=[r for r in rows if r['source_selected']];published=[r for r in rows if r['published']]
    groups={};roles=Counter()
    for r in rows:
        r['parameter_published']=r['published'] and r['output_function']=='truth'
        assert len(r['token_regions'])==len(r['prompt_ids'])
        assert ''.join(x['text'] for x in r['character_regions'])==r['prompt']
        assert evaluate(r,r['target'])['strict_correct'] and not evaluate(r,r['alternate'])['content_correct']
        roles.update(t['role'] for t in r['token_regions'])
    for r in panel:groups.setdefault(tuple(r[k] for k in ('family','language','unit','content_instance','target_index')),[]).append(r)
    old_names={r[k] for path in (RESULT/'phase2670_native_mlp_contract/material/cases.json',previous/'expansion/material/cases.json') for r in read(path) for k in ('entity_a','entity_b')}
    new_names={r[k] for r in rows for k in ('entity_a','entity_b')}
    n,L,D,K=len(rows),36,2560,9728;maxT=max(len(r['prompt_ids']) for r in rows)
    # Include every byte of the native uint16 arrays, no compressibility assumption.
    boundary=n*2*2*((L+1)*D+L*K)
    full=len(published)*maxT*2*(L+1)*D+sum(r['parameter_published'] for r in rows)*maxT*2*len(LAYERS)*(3*K+4*D)
    moment_groups=len({group_key(r) for r in rows})
    moments=moment_groups*16*((L+1)*D+L*(3*K+2*D))
    # Subsequent source-native packs (uint16 BF16 bits), Wo and full-coordinate
    # derived ledgers must fit a separate 3GiB allocation. Do not save a giant
    # all-head x all-source x all-coordinate tensor: compute it exactly in
    # head blocks and reconstruct a selected published case from its raw QKV.
    budget={'free_bytes':shutil.disk_usage(ROOT).free,'all_boundary_bytes':boundary,'64_fulltoken_bytes':full,
            'alltoken_moment_bytes':moments,'source_and_later_analysis_reserve_bytes':3*1024**3,'floor_bytes':8*1024**3,
            'maximum_real_prompt_tokens':maxT,'moment_groups':moment_groups}
    budget['fits_without_compression']=budget['free_bytes']>=sum(budget[k] for k in ('all_boundary_bytes','64_fulltoken_bytes','alltoken_moment_bytes','source_and_later_analysis_reserve_bytes','floor_bytes'))
    checks={'8448_unique':len(rows)==len({r['prompt'] for r in rows})==8448,'512_source_cells':len(panel)==512,'64_published':len(published)==64,
        '128_same_body_4functions':len(groups)==128 and all(len(g)==4 and len({tuple(r['prompt_ids'][:r['body_end_token']+1]) for r in g})==1 and len({r['output_function'] for r in g})==4 for g in groups.values()),
        'disjoint_new_entities':not (old_names&new_names),'fixed_shape_covers_generation':maxT+MAX_NEW_TOKENS<=PAD_LENGTH,
        'uncompressed_budget':budget['fits_without_compression'],'prior_terminal_audited':True}
    save(OUT/'analysis/material_preflight.json',{'checks':checks,'budget':budget,'role_counts':dict(roles)});assert all(checks.values()),checks
    save(OUT/'material/cases.json',rows)
    protocol={'plan':nextplan['plan'],'material_sha256':sha(OUT/'material/cases.json'),'previous_evidence_sha256':sha(previous/'analysis/scientific_checks.json'),
        'previous_interpretation_sha256':sha(previous/'analysis/interpretation.json'),'prior_q4_h':nextplan['frozen_q4_h'],'prior_q4_mlp':SITES,
        'q14_observed_own_gate':nextplan['q14_observed_own_gate'],'layers':LAYERS,'storage':budget,
        'fixed_shape':{'tokens':PAD_LENGTH,'right_pad':'actual EOS token, mask0','real_tokens':'mask1; query=body_end_token/task_end_token, never -1 padded index','generation':'greedy, no cache, same160 positions through maximum16 generatedtokens; all failures retained'},
        'measurement':'All H/a coordinates both real boundaries on8448; all realtoken sixfield sum/sumsq per coordinate in96groups.64predetermined fulltoken H;16truth examples also fulltoken fourlayer gate/up/a/x/down/pre_norm/attention. Remaining48examples have fullH but NOT fulltokenMLP, clearlydeclared storage allocation. RawBF16 uint16 bits, lossless storage not quantization.',
        'source_panel':'512 cells=256truth-grid reused+256name/cloze additions;128same-body groups x4outputfunctions; f/o/p/q0,2entitypairs,2contentinstances,2targets,8families2languages. Not512additional independent observations.',
        'source_algorithm':'ActualpostRoPE Q/K, V, P and Wo for allheads/source/outputcoords; two querypositions. Headblocking exact, noTopK. Contextual V not isolatedwordexclusivecredit. Observednormalizer endogenous, allocation isconditionalarithmetic notablation.',
        'numerical_controls':'64published prefixes: baseline / capture / restored no-op same shape, true source causal mask0, wholecoordinate ledger residual separately. Samebody across4functions compare H/a; anyfailures are numerical limits, never backwardsemantic evidence.',
        'scope_limits':['Only reused8families2contenttemplates and newentities; no open-world generalization.',
            'Padding changesexecutionfrom2671; do notclaim oldrawfields are bitwise same.',
            'Direction-positive2H/5MLP observationwindows are not oldstrictgate survivors.',
            'DS7 needs response-budget calibration beforeformal crossmodel; prior16token failure not mechanism absence.',
            'Local output differences around1e-5 unresolved; FP64readout notFP64model; known identities not newmath.'],
        'validation_scope':'2677 instrumentation/algebra checks precede2678; pass means measurement implemented, not mechanismclosed.'}
    save(OUT/'protocol/frozen.json',protocol)
    return rows,protocol


@torch.inference_mode()
def native_preflight(model,tok,rows):
    selected=[r for r in rows if r['published']];checks=[];source_groups={};modules={inspect.getmodule(type(model.model.layers[l].self_attn)) for l in LAYERS}
    original={m:m.eager_attention_forward for m in modules}
    for r in selected:
        kwargs=padded_inputs(model,r['prompt_ids'],tok.eos_token_id)
        baseline=model.model(**kwargs).last_hidden_state.detach().cpu().clone()
        with NativeSourceCapture(model,LAYERS) as source:
            cap=PaddedCapture(model,LAYERS);cap.reset(r['body_end_token'],False,r['task_end_token']);cap.enabled=True
            try:
                source.reset(r['body_end_token'],r['task_end_token']);source.enabled=True
                observed=model.model(**kwargs).last_hidden_state.detach().cpu().clone()
                source.enabled=False;cap.enabled=False;data=source.pack();fields=cap.pack()
            finally:cap.close()
            same=torch.equal(baseline,observed);errors=[];future_zero=True
            for l,row in data.items():
                block=model.model.layers[l];arr=source.array
                ledger=attention_ledger(row['actual_probability'],row['actual_value'],arr(block.self_attn.o_proj.weight),row['attention_output'],row['native_head_concat'],None if block.self_attn.o_proj.bias is None else arr(block.self_attn.o_proj.bias))
                norm=conditional_norm_ledger(row['residual_before_attention'],ledger,arr(block.post_attention_layernorm.weight),block.post_attention_layernorm.variance_epsilon,row['pre_mlp_norm'],row['mlp_x'])
                j=next(j for ll,j in SITES if ll==l)
                weighted=input_weight_ledger(norm,arr(block.mlp.gate_proj.weight[j]),row['gate'][:,j])
                errors.append(max(float(np.abs(x['reconstruction_error']).max()) for x in (ledger,norm,weighted)))
                for qi,t in enumerate((r['body_end_token'],r['task_end_token'])):future_zero &= bool((row['actual_probability'][qi,:,t+1:]==0).all())
            key=(r['family'],r['language'])
            here={k:fields[k][:,0].copy() for k in ('h','a')}
            if key not in source_groups:source_groups[key]=here
            source_identical=all(np.array_equal(here[k],source_groups[key][k]) for k in here)
        restored=model.model(**kwargs).last_hidden_state.detach().cpu()
        report={'case_index':r['case_index'],'family':r['family'],'language':r['language'],'function':r['output_function'],
            'exact_noop':same and torch.equal(baseline,restored),'wrappers_restored':all(m.eager_attention_forward is original[m] for m in modules),
            'all_source_future_mask_zero':future_zero,'same_body_all_H_a_bit_identical':source_identical,'max_ledger_reconstruction_error':max(errors)}
        checks.append(report);save(OUT/'analysis/native_progress.json',{'cases':len(checks),'total':64,'last':report})
        print('2677 NATIVE NOOP',len(checks),'/64',flush=True)
        del baseline,observed,restored,data,fields,source,ledger,norm,weighted
    result={'records':checks,'all_checks_passed':all(r['exact_noop'] and r['wrappers_restored'] and r['all_source_future_mask_zero'] and r['max_ledger_reconstruction_error']<1e-10 for r in checks),
        'same_body_bit_identical_count':sum(r['same_body_all_H_a_bit_identical'] for r in checks),
        'scope':'Real pretrained Qwen3-4B nativeBF16 CUDA. Knownarithmetic plus instrumentation reliability, notsemantic mechanism. Comparisonincludes16firstgroup selfreferences and48cross-function comparisons.'}
    save(OUT/'analysis/native_preflight.json',result);assert result['all_checks_passed'],result
    return result


def main():
    assert not (OUT/'analysis/final.json').exists();rows,protocol=prepare()
    model,tok=load_native('qwen4');assert model.dtype==torch.bfloat16 and not getattr(model,'is_quantized',False)
    save(OUT/'protocol/model.json',{'dtype':str(model.dtype),'actual_devices':sorted({str(p.device) for p in model.parameters()}),'quantized':False,
        'attention_implementation':model.config._attn_implementation,'config':model.config.to_dict()})
    result=native_preflight(model,tok,rows);del model;gc.collect();torch.cuda.empty_cache()
    material=read(OUT/'analysis/material_preflight.json');checks={**material['checks'],'real_pretrained_noop_and_arithmetic':result['all_checks_passed'],'material_immutable':sha(OUT/'material/cases.json')==protocol['material_sha256']}
    finish(2677,'等长执行与四种输出功能的来源坐标大阶段冻结',OUT,{'provenance':str(Path(__file__)),'summary':{'material_checks':material,'native_noops':result,'whole_plan':protocol['plan']},'checks':checks},
        '在2676等长回放与精度审计之后固定160位置、真实token mask1/尾部mask0；测量明确取正文和真实任务边界，填充坐标不进入统计或展示。对真实Qwen4先验证只读挂钩不改变结果，再冻结来源token到原生参数的整批实验。',
        r'N=8192+256=8448;\quad N_{source}=128\times4=512;\quad P_{t,s}=0\ (s>t);\quad c_{t,s,k}=\sum_h P^h_{t,s}\sum_dW^O_{k,hd}V^h_{s,d};\quad g_j=\sum_kW^g_{jk}x_k+\epsilon_g.',
        'C001八族双语8448唯一新实体条件；C002128同正文四功能组；C003全部字符/token区域可审核、混合token单列；C004未压缩磁盘预算；C00564真实预训练前缀的baseline/capture/restored三次运行与全坐标算术/遮罩核对。',
        '保留旧7候选方向阳性与严格门失败两个事实；观察窗口不是单语义神经元。全heads、全source、全部物理坐标参与乘加，语言分区是外部标注而非模型内部天然模块。source处V已经混合更早上下文，来源记账不等于独占因果归因。',
        '沿用8族两内容模板，只改变实体并增加输出功能；不宣称全新语义独立样本。160位置是数值控制而非精确实数保证。no-op和已知算术恒等通过，不代表微小输出效应或语言机制闭合。',
        '继续2678–2684完整全场观察、QKV来源账本、原生参数输入路径、全族扩大确认、数值可分辨局部复验、三个模型顺序复验与客户端/清理交付；同目标自动续研。')


if __name__=='__main__':main()
