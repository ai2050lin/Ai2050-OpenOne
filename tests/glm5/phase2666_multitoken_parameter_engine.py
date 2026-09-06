"""Multi-token content/format/EOS exact sequence factors; lossless activation CPU offload."""
import argparse,gc,shutil
import numpy as np
import torch
from transformers import AutoTokenizer
import transformers.modeling_utils as loading
from phase2620_native_coordinate_contract import *
from phase2622_native_field_capture import arr
from phase2632_fulltoken_native_adjoints import AdjointCapture,LAYERS
from phase2636_precision_engine import load_precision
from phase2658_sequence_parameter_engine import raw_checkpoint
from phase2662_symmetric_mapping_contract import compose,encode,load_native
from phase2663_symmetric_mapping_calibration import run,behavior_groups
from phase2664_symmetric_native_field import OUT as MATERIAL

OUT=RESULT/'phase2666_multitoken_parameter_engine';PARTS=('content','format','eos')


def load_fp():
    original=loading.safe_open
    def pread(*args,**kwargs):kwargs['backend']='pread';return original(*args,**kwargs)
    loading.safe_open=pread
    try:return load_precision('fp32')
    finally:loading.safe_open=original


def prepare():
    path=OUT/'material/cases.json'
    if path.exists():return read(path)
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True);rows=[]
    for r in read(MATERIAL/'material/cases.json'):
        if not r['fp_selected']:continue
        obj=compose(*[r[k] for k in ('family','language','unit','form','target_index','mention_order','probe_index','polarity','mapping','style','shots')],multi=True)
        obj=encode(tok,{**obj,'case_index':r['case_index']});categories=[];strings=[]
        for i,answer in enumerate(obj['common_readout_words']):
            enc=tok(answer,add_special_tokens=False,return_offsets_mapping=True);assert enc['input_ids']==obj['canonical_answer_ids'][i]
            needle=obj['short_answer_words'][i];a=answer.rindex(needle);b=a+len(needle);cat=['content' if e>a and s<b else 'format' for s,e in enc['offset_mapping']]+['eos'];assert 'content' in cat and 'format' in cat
            categories.append(cat);strings.append(tok.convert_ids_to_tokens(enc['input_ids']))
        obj['answer_token_categories']=categories;obj['answer_token_strings']=strings;rows.append(obj)
    assert len(rows)==256 and sum(r['published'] for r in rows)==64 and all(min(map(len,r['canonical_answer_ids']))>1 for r in rows)
    save(path,rows);save(OUT/'protocol/frozen.json',{'material_sha256':sha(path),'case_indices':[r['case_index'] for r in rows],'answer_lengths':sorted({len(a) for r in rows for a in r['canonical_answer_ids']}),
        'branch_shape':'Right-pad both alternatives to prefix+max(answerlength), attention mask0 only onpadding. Score actual answer tokens plusEOS only. Same branch shape protects causal-prefix numerical comparison.',
        'scores':'Full vocabulary logsoftmax; content tokens overlap Yes/No/是/否; format is allother answer tokens, EOSseparate. Scores are raw sequence log probabilities, not length-normalized or answer-class masses.',
        'offload':'saved_tensors_hooks moves only requires-grad CUDA saved tensors to CPU with identicaldtype, no quantization or featurecompression; frozen weight tensors stayonGPU; restore to originaldevice forbackward.',
        'numeric':'All4V layers alltoken X/value/full/content/format/EOS adjoints. Total H/MLP boundary adjoints. Full raw prompt H for64published.'})
    return rows


def branch_inputs(model,r,i,grad):
    prefix=r['prompt_ids'];answer=r['canonical_answer_ids'][i];actual=prefix+answer;n=len(prefix)+max(map(len,r['canonical_answer_ids']));ids=actual+[r['eos_token_id']]*(n-len(actual))
    with torch.no_grad():em=model.get_input_embeddings()(torch.tensor([ids],device='cpu')).to('cuda:0')
    em=em.detach().requires_grad_(grad);mask=torch.tensor([[1]*len(actual)+[0]*(n-len(actual))],device='cuda:0');return em,mask,ids,answer+[r['eos_token_id']]


def values(model,r,i,grad=False):
    em,mask,ids,targets=branch_inputs(model,r,i,grad);result=model.model(inputs_embeds=em,attention_mask=mask,use_cache=False)
    start=len(r['prompt_ids'])-1;states=result.last_hidden_state[0,start:start+len(targets)];logits=states.cpu()@model.lm_head.weight.T
    lp=logits.log_softmax(-1)[torch.arange(len(targets)),torch.tensor(targets)];return em,states,lp,ids,targets


@torch.inference_mode()
def score(model,r):
    branches=[]
    for i in (0,1):
        _,_,lp,_,_=values(model,r,i);vv=lp.tolist();cat=r['answer_token_categories'][i];branches.append({'logprobs':vv,'total':sum(vv),**{part:sum(v for v,c in zip(vv,cat) if c==part) for part in PARTS}})
    return {'branches':branches,'contrast':branches[0]['total']-branches[1]['total'],**{part:branches[0][part]-branches[1][part] for part in PARTS}}


def branch(model,r,i):
    cap=AdjointCapture(model);traffic={'count':0,'bytes':0}
    def pack_saved(t):
        if t.is_cuda and t.requires_grad:traffic['count']+=1;traffic['bytes']+=t.numel()*t.element_size();return (t.device,t.detach().cpu())
        return t.detach()
    def unpack_saved(v):return v[1].to(v[0]) if isinstance(v,tuple) else v
    try:
        with torch.autograd.graph.saved_tensors_hooks(pack_saved,unpack_saved):
            em,states,lp,ids,targets=values(model,r,i,True);loss=lp.sum();loss.backward(retain_graph=True);boundary=len(r['prompt_ids'])-1
            pack={'hidden_adjoint_prompt_boundary':np.stack([arr(v.grad[0,boundary]) for v in [em]+cap.hidden]),
                'mlp_adjoint_prompt_boundary':np.stack([arr(v.grad[0,boundary]) for v in cap.a]),'normalized_prompt_boundary':arr(states[0])}
            outputs=[]
            for l in LAYERS:
                x,v=cap.linears[(l,'v_proj')];outputs.append(v);pack[f'L{l}_v_x']=arr(x[0]);pack[f'L{l}_v_value']=arr(v[0]);pack[f'L{l}_v_g']=arr(v.grad[0])
            cats=r['answer_token_categories'][i];vv=lp.detach().tolist()
            for pi,part in enumerate(PARTS):
                sub=lp[torch.tensor([k for k,c in enumerate(cats) if c==part])].sum();grads=torch.autograd.grad(sub,outputs,retain_graph=pi<len(PARTS)-1)
                for l,g in zip(LAYERS,grads):pack[f'L{l}_v_g_{part}']=arr(g[0])
            actual_n=len(r['prompt_ids'])+len(r['canonical_answer_ids'][i])
            assert all(not np.any(pack[f'L{l}_v_g{suffix}'][actual_n:]) for l in LAYERS for suffix in ('','_content','_format','_eos'))
            reconstruction=max(float(np.max(np.abs(sum(pack[f'L{l}_v_g_{p}'].astype('float64') for p in PARTS)-pack[f'L{l}_v_g']))) for l in LAYERS)
            info={'input_ids':ids,'actual_input_length':len(r['prompt_ids'])+len(r['canonical_answer_ids'][i]),'target_ids':targets,'prediction_positions':list(range(boundary,boundary+len(targets))),
                'logprobs':vv,'total_logprob':sum(vv),'part_logprobs':{part:sum(v for v,c in zip(vv,cats) if c==part) for part in PARTS},'categories':cats,'saved_activation_cpu_traffic':traffic,'maximum_adjoint_component_sum_error':reconstruction}
        assert all(np.isfinite(v).all() for v in pack.values());return pack,info
    finally:cap.close();cap.reset()


def natural():
    cases=prepare();model,tok=load_native('qwen4');records=run(model,tok,cases,OUT/'natural',fields=True);del model;gc.collect();torch.cuda.empty_cache();save(OUT/'analysis/natural_summary.json',behavior_groups(records))


def fp():
    cases=prepare();model,info=load_fp();save(OUT/'protocol/model.json',info);OUT.joinpath('field').mkdir(parents=True,exist_ok=True);path=OUT/'analysis/records.jsonl'
    records=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else [];assert [r['case_index'] for r in records]==[r['case_index'] for r in cases[:len(records)]]
    with path.open('a',encoding='utf-8') as stream:
        for i,r in enumerate(cases[len(records):],len(records)):
            if shutil.disk_usage(OUT).free<8*1024**3:raise RuntimeError('8GiB disk floor; completedJSONL preserved, no unscoped deletion')
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();pack,base=raw_checkpoint(model,r);branches=[]
            for bi,label in enumerate(('Y','N')):
                pp,meta=branch(model,r,bi);pack.update({label+'__'+k:v for k,v in pp.items()});branches.append(meta);del pp;gc.collect();torch.cuda.empty_cache()
            exact=np.array_equal(pack['Y__normalized_prompt_boundary'],pack['N__normalized_prompt_boundary']);assert exact
            with np.load(OUT/f'natural/field/case_{r["case_index"]:04d}.npz') as b:embedding_exact=np.array_equal(pack['hidden_boundary'][0],b['hidden_boundary'][0])
            assert embedding_exact
            rec={k:r[k] for k in ('case_index','case_id','family','language','unit','polarity','mapping','published','expected_yes')};rec.update(**base,branches=branches,
                contrast=branches[0]['total_logprob']-branches[1]['total_logprob'],parts={p:branches[0]['part_logprobs'][p]-branches[1]['part_logprobs'][p] for p in PARTS},branch_prefix_bitwise=exact,embedding_exact=embedding_exact,gpu_peak_bytes=torch.cuda.max_memory_allocated())
            np.savez_compressed(OUT/f'field/case_{r["case_index"]:04d}.npz',**pack);stream.write(json.dumps(rec,ensure_ascii=False)+'\n');stream.flush();records.append(rec);del pack
            if (i+1)%8==0:save(OUT/'analysis/progress.json',{'cases':i+1,'total':256});print('multitoken adjoints',i+1,'/256',flush=True)
    nograd=all(p.grad is None for p in model.parameters());del model;gc.collect();torch.cuda.empty_cache();save(OUT/'analysis/records.json',records)
    save(OUT/'analysis/raw_manifest.json',[{'path':str(OUT/f'field/case_{r["case_index"]:04d}.npz'),'bytes':(OUT/f'field/case_{r["case_index"]:04d}.npz').stat().st_size,'case_index':r['case_index'],'published':r['published']} for r in cases])
    save(OUT/'analysis/fp_completion.json',{'no_weight_gradients':nograd,'cases':len(records)})


def finalize():
    records=read(OUT/'analysis/records.json');info=read(OUT/'protocol/model.json');summary={'cases':len(records),'natural':read(OUT/'analysis/natural_summary.json'),
        'content_vs_full_rank_difference':sum((r['parts']['content']>0)!=(r['contrast']>0) for r in records),'max_adjoint_components_error':max(b['maximum_adjoint_component_sum_error'] for r in records for b in r['branches']),
        'max_score_components_error':max(abs(r['contrast']-sum(r['parts'].values())) for r in records),'gpu_peak_bytes':max(r['gpu_peak_bytes'] for r in records),'answer_lengths':read(OUT/'protocol/frozen.json')['answer_lengths']}
    checks={'256_cases':len(records)==256,'all_prefixes_bitwise':all(r['branch_prefix_bitwise'] for r in records),'all_embeddings_exact':all(r['embedding_exact'] for r in records),'no_weight_gradients':read(OUT/'analysis/fp_completion.json')['no_weight_gradients'],
        '28_loaded_weights_exact':all(info['all28_weight_values_exact'].values()),'score_components_identity':summary['max_score_components_error']<1e-9}
    assert all(checks.values());finish(2666,'多token答案的内容/格式/EOS全位置原生参数分解',OUT,{'provenance':str(Path(__file__)),'summary':summary,'checks':checks},
        '两规范答案分支都按最大长度右侧屏蔽填充，真实回答token加EOS做全词表归一概率；分别反传全分数、内容、格式、EOS，四层V所有输入/输出坐标保留。',
        r'L=L_{content}+L_{format}+L_{EOS};\quad G^{c}_{jk}=\sum_t\bar V^{Y,c}_{t,j}X^Y_{t,k}-\sum_t\bar V^{N,c}_{t,j}X^N_{t,k};\quad G^{all}=\sum_cG^c.',
        '256前缀：八族双语、单位0/1/4/5、v=unit%2、probe0/form0/order0、双极性/映射。先BF16自然生成再同值FP32，64展示例保留全tokenH；Answer:Yes./No.和答案：是。/否。均实际4token加EOS。配对答案已等长，无实例实际需要补padding，不能宣称已验证非等长补齐效果。',
        '单个共享权重如何通过全部位置影响答案字词、固定格式和停止，可以分别寻址和加总；公共格式前缀和分支后续上下文需要区别。这扩展测量对象，不是新语言数学。',
        '强制候选仍只有两条规范格式，非全部等价答案概率和自由生成能力。256前缀复用四实体对/语言；标准链式法则不证明语义齿轮。激活CPU搬存只改变存放设备不改值、不筛坐标，不是特征压缩。',
        '以单位4/5的128前缀和八冻结V标量检验2048有限改动，同时检查内容、格式、EOS部分误差和权重恢复。')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','natural','fp','finalize','all']);a=ap.parse_args()
    if a.action=='all':prepare();natural();fp();finalize()
    else:globals()[a.action]()
