"""Candidate-free behavior; preserve every output and exact long-sentence content."""
import json, time
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from phase2620_native_coordinate_contract import *
from phase2621_native_language_material import build,evaluate,FAMILIES

MODELS={'qwen4':'qwen3-4b','qwen14':'Qwen3-14B','glm4':'glm4-9b-chat-hf','ds7':'deepseek-r1-distill-qwen-7b'}

def load_model(key):
    torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf'/MODELS[key],local_files_only=True,trust_remote_code=True,use_fast=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(ROOT/'models/hf'/MODELS[key],dtype=torch.bfloat16,device_map='auto',
        max_memory={0:'12GiB','cpu':'20GiB'},offload_folder=str(ROOT/f'tests/glm5_temp/native_{key}_offload'),
        offload_state_dict=True,offload_buffers=True,local_files_only=True,trust_remote_code=True,low_cpu_mem_usage=True,attn_implementation='eager').eval()
    assert not getattr(model,'is_quantized',False)
    return model,tok

def stats(rows):
    return {g:{'n':len(rr),'answer_correct':sum(r['answer_correct'] for r in rr)/len(rr),
        'strict_correct':sum(r['strict_correct'] for r in rr)/len(rr),'eos_rate':sum(r['eos'] for r in rr)/len(rr),
        'content_preserved':sum(bool(r['content_preserved']) for r in rr)/len(rr) if rr[0]['family']=='long_reorder' else None}
        for g in sorted({r['family']+'/'+r['language'] for r in rows})
        for rr in [[r for r in rows if r['family']+'/'+r['language']==g]]}

@torch.inference_mode()
def run(model,tok,cases,out,batch_size=8):
    records=[];tok.padding_side='left'
    target=Path(out)/'behavior/greedy.jsonl';target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('w',encoding='utf-8') as stream:
        for start in range(0,len(cases),batch_size):
            rr=cases[start:start+batch_size]
            ids=tok.pad({'input_ids':[r['prompt_ids'] for r in rr]},padding=True,return_tensors='pt').to(model.get_input_embeddings().weight.device)
            maximum=192 if any(r['family']=='long_reorder' for r in rr) else 48
            generated=model.generate(**ids,do_sample=False,max_new_tokens=maximum,pad_token_id=tok.pad_token_id)
            for row,seq in zip(rr,generated[:,ids['input_ids'].shape[1]:].tolist()):
                text=tok.decode(seq,skip_special_tokens=True)
                r={k:row[k] for k in ('case_id','family','language','index','form','variant','base_unit','split','target','alternate')}
                r.update(generated=text,generated_ids=seq,eos=tok.eos_token_id in seq,**evaluate(row,text));records.append(r)
                stream.write(json.dumps(r,ensure_ascii=False)+'\n')
            stream.flush()
            if start%64==0:print('behavior',start+len(rr),'/',len(cases),flush=True)
    return records

def main():
    out=RESULT/'phase2621_native_language_behavior'
    model,tok=load_model('qwen4');cases=build(tok);save(out/'material/cases.json',cases)
    save(out/'protocol/frozen.json',{'material_sha256':sha(out/'material/cases.json'),'case_count':768,'unique_base_items':192,
        'max_new_tokens':{'short':48,'long_reorder':192},'precision':str(model.dtype),'device_map':getattr(model,'hf_device_map',{'actual_first_parameter':str(next(model.parameters()).device)})})
    records=run(model,tok,cases,out)
    result={'provenance':str(Path(__file__)),'by_group':stats(records),'summary':{'prompts':len(records),'unique_base_items':192,'strict_correct':sum(r['strict_correct'] for r in records)/len(records),'by_group':stats(records)},
        'checks':{'all768_outputs':len(records)==768,'no_duplicate_prompts':len({r['prompt'] for r in cases})==768,'nonquantized':not getattr(model,'is_quantized',False)}}
    finish(2621,'八族双语768自然生成与真正长句内容保持基线',out,result,
        '先观察自然运行，不做干预也不按行为筛选材料。区分名称第一行包含判分与严格答案判分；重排严格比较三句原文的内容和顺序。',
        r'A_{strict}=\frac{1}{n}\sum_i 1[\operatorname{normalize}(y_i)=\operatorname{normalize}(y_i^*)],\quad n=8\times2\times12\times2\times2.',
        '时序、苹果同词异义、分类链、主被动施事、否定、段落回指、标点、三句完整重排；中英各48/族，192基础item各复用两表面两条件。原句全部在material/cases.json；long_reorder只忽略空白，不忽略文字或标点改动。',
        '把可观察坐标响应和行为胜任分开保存，阴性族也继续测绘但不宣称其语言能力已解释。多族不同问法是覆盖扩展，不冒称严格去除了所有词汇及答案混杂。',
        '名称等答案可复制，分类链和标点任务较浅，回指含显式再次提名；苹果只有一个多义词。发现/留出按item分，但对侧名字和事件可能跨split重复。不是完整正交超图或开放世界语言。',
        '继续未干预原生全场采集，不因任何族低成功率停止。所有自然输出保留供后续成功/失败分层比较。')

if __name__=='__main__':main()
