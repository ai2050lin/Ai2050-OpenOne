"""Sequential BF16 Qwen14B replication in its own complete physical coordinate basis."""
import itertools,gc,os,faulthandler
import numpy as np
import torch
from phase2620_native_coordinate_contract import *
from phase2621_native_behavior_run import load_model
import transformers.modeling_utils as model_loading
from phase2649_output_function_behavior import Capture
from phase2655_truth_answer_contract import OUT as MATERIAL,encode,evaluate,leading_answer,FAMILIES

OUT=RESULT/'phase2660_qwen14_truth_answer_replication'


def analyze(cases):
    table={(r['family'],r['language'],r['target_index'],r['probe_index'],r['polarity'],r['mapping']):r for r in cases};counts={};profiles=[]
    OUT.joinpath('maps').mkdir(parents=True,exist_ok=True)
    for fam,lang in itertools.product(FAMILIES,('en','zh')):
        values={}
        for p,q,m in itertools.product((0,1),repeat=3):
            a,b=[table[fam,lang,v,p,q,m]['case_index'] for v in (0,1)]
            with np.load(OUT/f'field/case_{a:04d}.npz') as za,np.load(OUT/f'field/case_{b:04d}.npz') as zb:
                values[p,q,m]={k:za[k].astype('float64')-zb[k].astype('float64') for k in ('hidden_boundary','mlp_boundary')}
        pack={}
        for metric in ('hidden_boundary','mlp_boundary'):
            for p,q,m in itertools.product((0,1),repeat=3):pack[f'p{p}q{q}m{m}__'+metric]=values[p,q,m][metric].astype('float32')
            baseline=np.sign(values[0,0,0][metric]);base_nonzero=baseline!=0
            for hypothesis in ('statement_truth','question_affirmative','answer_label'):
                good=base_nonzero.copy()
                for p,q,m in itertools.product((0,1),repeat=3):
                    exponent=p if hypothesis=='statement_truth' else p+q if hypothesis=='question_affirmative' else p+q+m
                    good &= np.sign(values[p,q,m][metric])==(-1)**exponent*baseline
                key=metric+'__'+hypothesis;counts.setdefault(key,np.zeros_like(good,dtype='int16'));counts[key]+=good
            profiles.append({'family':fam,'language':lang,'metric':metric,'baseline_rms_by_layer':np.sqrt(np.mean(values[0,0,0][metric]**2,axis=-1)).tolist()})
        np.savez_compressed(OUT/f'maps/{fam}_{lang}_allcoordinate_responses.npz',**pack)
    np.savez_compressed(OUT/'maps/allcoordinate_sign_group_counts.npz',**counts);save(OUT/'analysis/response_profiles.json',profiles)
    return {k:{'coordinate_count':v.shape[-1],'all16groups_by_layer':(v==16).sum(-1).tolist()} for k,v in counts.items()}


@torch.inference_mode()
def main():
    assert not (OUT/'analysis/final.json').exists()
    # Process-local supported loader option; do not edit transformers or model files.
    # The first attempt crashed in torch_cpu.dll before any forward or saved case.
    os.environ['HF_DEACTIVATE_ASYNC_LOAD']='1';faulthandler.enable()
    save(OUT/'protocol/runtime_adaptation.json',{'initial_attempt':'2026-09-05 05:47:34 Windows Application Error1000, torch_cpu.dll exception0xc0000005 during weight loading (~151/443), no model forward or case artifacts',
        'second_attempt':'Synchronous retry also crashed in torch/storage.py __getitem__ during safetensors mmap materialization; concurrency alone does not solve it.',
        'third_attempt':'disable_mmap=True avoids mapped storage but loads all8shards into RAM together and raises MemoryError before any forward.',
        'retry':'HF_DEACTIVATE_ASYNC_LOAD=1 plus safetensors backend=pread selected in a process-local safe_open wrapper. Streams individual tensor reads without torch mmap slicing or whole-checkpoint RAM staging. Same from_pretrained device_map=auto/BF16/model/material.',
        'pread_probe':'Installed safetensors supports backend keyword; model.layers.0.self_attn.v_proj.weight read successfully as BF16 [1024,5120], entry[0,0]=0.00665283203125.',
        'causal_limit':'Native storage materialization failure is observed; specific DLL root cause is not proven. No case ran in either failed load.'})
    original_open=model_loading.safe_open
    def pread_open(*args,**kwargs):
        kwargs['backend']='pread';return original_open(*args,**kwargs)
    model_loading.safe_open=pread_open
    try:model,tok=load_model('qwen14')
    finally:model_loading.safe_open=original_open
    original=[r for r in read(MATERIAL/'material/cases.json') if r['crossmodel']]
    cases=[encode(tok,{**r,'published':(r['target_index'],r['probe_index'],r['polarity'],r['mapping'])==(0,0,0,0)}) for r in original];save(OUT/'material/cases.json',cases)
    save(OUT/'protocol/model.json',{'model':'local Qwen3-14B','dtype':str(model.dtype),'quantized':bool(getattr(model,'is_quantized',False)),
        'device_map':getattr(model,'hf_device_map',{'actual_parameter_devices':sorted({str(p.device) for p in model.parameters()})}),
        'load_policy':'device_map=auto,12GiB GPU/20GiB CPU,nonquantized BF16 local_files_only, single model; no prior4B resident; HF_DEACTIVATE_ASYNC_LOAD=1,safetensors backend=pread process-local wrapper restored after load',
        'dimensions':{'hidden':model.config.hidden_size,'layers':model.config.num_hidden_layers,'mlp':model.config.intermediate_size}})
    cap=Capture(model);OUT.joinpath('field').mkdir(parents=True,exist_ok=True);records=[];manifest=[];alltoken={};ntokens={}
    for i,r in enumerate(cases):
        ids=list(r['prompt_ids']);generated=[];first=None;pack={};decision=None
        for step in range(16):
            cap.reset();cap.enabled=step==0
            result=model.model(input_ids=torch.tensor([ids],device=model.get_input_embeddings().weight.device),use_cache=False);state=result.last_hidden_state[0,-1];logits=model.lm_head(state).float();chosen=int(logits.argmax());logits[chosen]=-float('inf');runner=int(logits.argmax())
            if step==0:
                h=np.stack(cap.h);pack={'hidden_boundary':h[:,-1],'mlp_boundary':np.stack(cap.a)}
                if r['published']:pack['hidden_fulltoken']=h
                key=r['family']+'/'+r['language'];hh=h.astype('float64')
                if key not in alltoken:alltoken[key]=[np.zeros_like(hh[:,0]),np.zeros_like(hh[:,0])];ntokens[key]=0
                alltoken[key][0]+=hh.sum(1);alltoken[key][1]+=(hh*hh).sum(1);ntokens[key]+=h.shape[1];first=[chosen,runner];del h,hh
            generated.append(chosen);ids.append(chosen);text=tok.decode(generated,skip_special_tokens=True);a=leading_answer(text,r['language'])
            if decision is None and a is not None:decision={'step':step,'answer_yes':a}
            cap.enabled=False;cap.reset()
            if chosen==tok.eos_token_id:break
        ci=r['case_index'];path=OUT/f'field/case_{ci:04d}.npz';np.savez_compressed(path,**pack);manifest.append({'path':str(path),'bytes':path.stat().st_size,'case_index':ci,'published':r['published']})
        record={k:r[k] for k in ('case_index','case_id','family','language','target_index','probe_index','polarity','mapping','target','alternate','expected_yes','published')}
        record.update(native_ids=first,generated=text,generated_ids=generated,decision=decision,eos=tok.eos_token_id in generated,**evaluate(r,text));records.append(record);del pack,result,state,logits
        save(OUT/'analysis/progress.json',{'cases':i+1,'total':256});print('Qwen14 BF16 truth/mapping',i+1,'/256',flush=True)
    cap.close();del model;gc.collect();torch.cuda.empty_cache();save(OUT/'analysis/records.json',records);save(OUT/'analysis/raw_manifest.json',manifest)
    np.savez_compressed(OUT/'maps_alltoken_coordinate_rms.npz',**{k:np.sqrt(v[1]/ntokens[k]).astype('float32') for k,v in alltoken.items()})
    groups={}
    for lang,q,m in itertools.product(('en','zh'),(0,1),(0,1)):
        rr=[r for r in records if (r['language'],r['polarity'],r['mapping'])==(lang,q,m)];groups[f'{lang}/q{q}/m{m}']={'n':len(rr),'content_correct':sum(r['content_correct'] for r in rr),'strict_correct':sum(r['strict_correct'] for r in rr),'leading_correct':sum(r['decision'] is not None and r['decision']['answer_yes']==r['expected_yes'] for r in rr)}
    coordinate=analyze(cases);checks={'256_cases':len(records)==256,'16_published':sum(r['published'] for r in records)==16,'six_allcoordinate_hypothesis_maps':len(coordinate)==6,'nonquantized':not read(OUT/'protocol/model.json')['quantized']}
    assert all(checks.values())
    finish(2660,'Qwen14B非量化事实/问题/答案映射与模型内全坐标复验',OUT,{'provenance':str(Path(__file__)),'summary':{'behavior':groups,'coordinate_hypotheses':coordinate,'model':read(OUT/'protocol/model.json')},'checks':checks},
        '串行加载本地Qwen14B BF16，自动分配GPU/CPU，不量化。相同语义材料按自己的tokenizer重新编码；全H/MLP边界及全token坐标RMS在模型自身基底内比较，不把Qwen4坐标编号硬搬过去。',
        r'D_{p,q,m}=H(v=0,p,q,m)-H(v=1,p,q,m);\quad s_t=(-1)^p,\ s_a=(-1)^{p+q},\ s_y=(-1)^{p+q+m}.',
        '八族双语、单位7、form0/order0，双事实目标/探问/极性/映射共256条件；16实例保存完整H供客户端，其他场完成分析后清理。',
        '对照更大模型能否执行反向输出约定，并观察同一模型内哪些全坐标方向跟随陈述真值、问题肯定性或答案标签。行为不通过不阻止记录；没有借用另一个模型的物理坐标假装通用基底。',
        '每语言只有1实体对、1句式/顺序，未复验目标幅度超过句式/顺序的完整门，只是跨模型因素分离的初步复验。三个简单符号规律不排除混合或非线性条件编码。两次加载storage切片发生访问异常；第三次整文件入内存触发MemoryError，均未运行样本。最终逐tensor pread读取且同步加载，仍from_pretrained/device_map=auto/BF16，不修改模型或库文件，过程内wrapper加载后恢复；此适配不证明DLL根因。大模型也未证明普遍编码机制。',
        '结合8192 Qwen4自然图谱、规范序列逐参数算法和本模型内复验，完成客户端、未展示原包清理与整批记录；同目标继续新的可判定拼图。')


if __name__=='__main__':main()
