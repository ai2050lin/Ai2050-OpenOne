"""Sequential BF16 model-local natural-operation replication; one model per process."""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/'tests/glm5';sys.path.insert(0,str(TESTS))
import model_utils
import phase2603_c643329_c659712_unique_natural_lockbox as material
import phase2605_c676097_c692480_singleprompt_source_patch as source
import phase2608_c725249_c741632_autonomous_source_band as greedy
MODELS={'qwen14':('Qwen3-14B',2610),'glm4':('glm4-9b-chat-hf',2611),'ds7':('deepseek-r1-distill-qwen-7b',2612)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=MODELS,required=True);args=parser.parse_args()
    directory,phase=MODELS[args.model];out=TESTS/f'result/phase{phase}_{args.model}_natural_replication'
    torch.set_num_threads(4)
    path=ROOT/'models/hf'/directory
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=True,use_fast=False)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,device_map='auto',
        max_memory={0:'12GiB','cpu':'20GiB'},offload_folder=str(ROOT/f'tests/glm5_temp/phase{phase}_offload'),
        offload_state_dict=True,offload_buffers=True,local_files_only=True,trust_remote_code=True,
        low_cpu_mem_usage=True,attn_implementation='eager').eval()
    assert not getattr(model,'is_quantized',False)
    source.save_json(out/'protocol/model.json',{'device_map':model.hf_device_map,'dtype':str(model.dtype),'model':directory,'max_new_tokens':24,'interface':'raw continuation, identical prompt text; not chat capability ceiling'})
    groups=defaultdict(list)
    for pair in greedy.pairs_for('external'):groups[pair[0]['family']+'/'+pair[0]['language']].append(pair)
    pairs=[]
    for g in sorted(groups):pairs+=groups[g][:5]
    for pair in pairs:
        for row in pair:row['prompt_ids']=tok.encode(row['prompt'],add_special_tokens=False)
        spans=material.source_spans(pair[0]['prompt_ids'],pair[1]['prompt_ids'])
        for row,pos in zip(pair,spans):row['source_token_positions']=pos
    source.save_json(out/'material/pairs.json',pairs)
    source.OUT=out;p=source.collect_source_means(model,tok,pairs);sf=np.load(p,mmap_mode='r')
    layers=model_utils.get_layers(model);source_layer=round(5*(len(layers)-1)/35)
    delta=sf[:,1,source_layer+1].astype('float32')-sf[:,0,source_layer+1].astype('float32')
    records=[];summary={}
    for name,condition,data in [('baseline','baseline',pairs),('natural_donor','baseline',[list(reversed(p)) for p in pairs]),('source1','source1',pairs),('source_roll','source_roll',pairs)]:
        fp=out/f'field/prefill_{name}.float16.npy';fp.parent.mkdir(parents=True,exist_ok=True)
        field=np.lib.format.open_memmap(fp,mode='w+',dtype='float16',shape=(60,len(layers)+1,model.config.hidden_size))
        rows=greedy.generate_condition(model,tok,data,condition,delta,{},batch_size=12,raw_field=field,source_layer=source_layer)
        field.flush()
        for row in rows:row['condition']=name
        records+=rows;summary[name]=greedy.metrics(rows);source.write_jsonl(out/f'behavior/{name}.jsonl',rows)
        print(f'phase{phase}',name,json.dumps(summary[name]),flush=True)
    bygroup={g:{c:greedy.metrics([r for r in records if r['group']==g and r['condition']==c]) for c in summary} for g in groups}
    result={'phase':phase,'model':directory,'timestamp':datetime.now().astimezone().isoformat(),'pairs':60,'source_layer':source_layer,
        'conditions':summary,'by_group':bygroup,'source_field_shape':list(sf.shape),
        'checks':{'all60_pairs':len(pairs)==60,'all240_generations':len(records)==240,'nonquantized':not getattr(model,'is_quantized',False),
        'all_physical_coordinates':sf.shape[-1]==model.config.hidden_size,'cuda_used':any(str(v) in ('0','cuda:0') for v in model.hf_device_map.values())},
        'language_mechanism_closed':False,'limits':['five pairs per language-family; exploratory scale replication','raw 24-token interface may disadvantage reasoning/chat models','no direct cross-model coordinate alignment','single relative depth is not architecture-optimized','oracle source direction']}
    result['all_checks_passed']=all(result['checks'].values());source.save_json(out/'analysis/final.json',result)
    stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M');memo=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
    text=rf'''

## Phase {phase}: {directory}非量化自然操作、全坐标与生成通路顺序复验 [{stamp}]

**原理、用例与公式。** 固定六族×中英×5 pair=60，不按本模型成功筛选；两侧自然生成、source干预、roll共240条，统一24-token原始续写。BF16、CUDA与device_map=auto，GPU预算12GiB、CPU20GiB，和其他模型不同进程顺序运行。

$$l_* = \operatorname{{round}}(5(L-1)/35),\quad H_{{l_*,S}}\leftarrow H_{{l_*,S}}+\delta H_{{l_*,S}}.$$

保存每模型自身词嵌入/source HiddenState全部坐标及四条件prefill全层原始block输出，禁止把不同模型相同坐标编号当相同功能。

**结果汇总。** 相对深度layer={source_layer}；{json.dumps(summary,ensure_ascii=False)}。

**相关文件。** 脚本`tests/glm5/phase2610_2612_sequential_nonquantized_replication.py --model {args.model}`；`{out}`包含模型精度/device_map、冻结文本与重分词位置、240生成、全场、12组和检查。

**分析与理论进展。** 这是接口相同、精度非量化的功能路线复验；模型本身是否自然答对和干预是否有效分开报告。

**问题硬伤。** 每组仅5 pair，本阶段是跨模型探索而非普遍性证明；24-token原始续写对推理/chat模型不一定合适，失败不等于缺少能力；相对层5并非每模型最优层；source delta仍是oracle且mean pooling。需要接口校准后再扩大重要阳性。

**结论。** 检查={json.dumps(result['checks'])}；以模型局部坐标积累拼图，不宣称跨架构同构或语言编码闭合。
'''
    if f'## Phase {phase}:' not in memo.read_text(encoding='utf-8-sig'):
        with memo.open('a',encoding='utf-8') as f:f.write(text)

if __name__=='__main__':main()
