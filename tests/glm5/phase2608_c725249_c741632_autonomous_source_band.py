"""Open-vocabulary greedy validation; all external pairs, no success selection."""
from __future__ import annotations
import json, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/'tests/glm5'; RESULT=TESTS/'result'
sys.path.insert(0,str(TESTS))
import model_utils
import phase2601_c610561_c626944_natural_singleprompt_behavior_lockbox as behavior
import phase2605_c676097_c692480_singleprompt_source_patch as source
import phase2607_c708865_c725248_multilayer_qkv_band as band
OUT=RESULT/'phase2608_c725249_c741632_autonomous_source_band'
MEMO=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
CONDITIONS=['baseline','source025','source05','source1','source15','source_roll','source_wrong','source_negative','kv','v','k','kv_roll']

def pairs_for(split):
    groups=defaultdict(list)
    path=RESULT/'phase2603_c643329_c659712_unique_natural_lockbox/material/cases.unique.jsonl'
    for line in path.read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        if row['split']==split: groups[row['pair_id']].append(row)
    return [sorted(v,key=lambda r:r['variant']) for k,v in sorted(groups.items())]

def generate_condition(model, tokenizer, pairs, condition, delta_source, delta_kv,
                       batch_size=8, raw_field=None, raw_logits=None, source_layer=5,
                       band_start=6, max_new_tokens=24):
    layers=model_utils.get_layers(model); device=model.get_input_embeddings().weight.device
    records=[]
    for start in range(0,len(pairs),batch_size):
        batch=pairs[start:start+batch_size]; rows=[p[0] for p in batch]
        ids,mask=behavior.left_pad([r['prompt_ids'] for r in rows],tokenizer.pad_token_id,device)
        width=ids.shape[1]; positions=[[p+width-len(r['prompt_ids']) for p in r['source_token_positions']] for r in rows]
        handles=[]; seen=set(); hits=defaultdict(int)
        actual_energy=np.zeros(len(batch));intended_energy=np.zeros(len(batch))
        def make_patch(key, vectors, pos):
            def hook(module,args,output):
                if key in seen: return output
                seen.add(key); hits[key]+=1
                value=output[0] if isinstance(output,tuple) else output
                patched=value.clone()
                for i,pp in enumerate(pos):
                    before=patched[i,pp].float().clone()
                    patched[i,pp]+=torch.as_tensor(vectors[i],device=value.device,dtype=value.dtype)
                    actual_energy[i]+=float((patched[i,pp].float()-before).square().sum().item())
                    intended_energy[i]+=float(np.sum(np.asarray(vectors[i],dtype='float64')**2))*len(pp)
                return (patched,)+output[1:] if isinstance(output,tuple) else patched
            return hook
        if condition.startswith('source'):
            scale={'source025':.25,'source05':.5,'source1':1.,'source15':1.5,'source_negative':-1.,'source_roll':1.,'source_wrong':1.}[condition]
            vectors=delta_source[start:start+len(batch)].copy()*scale
            if condition=='source_roll': vectors=np.roll(vectors,641,axis=-1)
            pos=positions if condition!='source_wrong' else [[max(0,min(p)-1)] for p in positions]
            handles.append(layers[source_layer].register_forward_hook(make_patch('source',vectors,pos)))
        elif condition!='baseline':
            for li in range(band_start,len(layers)):
                for comp in ('k','v') if condition.startswith('kv') else (condition,):
                    vectors=delta_kv[comp][start:start+len(batch),li].copy()
                    if condition.endswith('roll'): vectors=np.roll(vectors,193,axis=-1)
                    handles.append(getattr(layers[li].self_attn,comp+'_proj').register_forward_hook(make_patch(f'{comp}{li}',vectors,positions)))
        # Post-block raw residuals: last block is NOT replaced by final-normalized state.
        def make_capture(li):
            done=[False]
            def hook(module,args,output):
                if done[0]: return
                done[0]=True
                v=output[0] if isinstance(output,tuple) else output
                if raw_field is not None: raw_field[start:start+len(batch),li]=v[:,-1].detach().float().cpu().numpy()
            return hook
        if raw_field is not None:
            handles.append(model.get_input_embeddings().register_forward_hook(make_capture(0)))
            for li,layer in enumerate(layers): handles.append(layer.register_forward_hook(make_capture(li+1)))
        logits_seen=[False]
        def capture_logits(module,args,output):
            if logits_seen[0]: return
            logits_seen[0]=True
            if raw_logits is not None: raw_logits[start:start+len(batch)]=output[:,-1].detach().float().cpu().numpy()
        if raw_logits is not None: handles.append(model.get_output_embeddings().register_forward_hook(capture_logits))
        try:
            with torch.inference_mode():
                generated=model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=max_new_tokens,
                    do_sample=False,use_cache=True,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
        finally:
            for h in handles: h.remove()
        for ri,(row,seq) in enumerate(zip(rows,generated)):
            tokens=seq[width:].cpu().tolist(); text=tokenizer.decode(tokens,skip_special_tokens=True)
            parsed,_=behavior.parse_generation(row,text)
            records.append({'pair_id':row['pair_id'],'group':row['family']+'/'+row['language'],
                'condition':condition,'generated':text,'generated_ids':tokens,'parsed':parsed,
                'recipient_correct':parsed==row['target'],'donor_correct':parsed==row['alternate'],
                'intervention_sites_called_once':all(v==1 for v in hits.values()),'hook_sites':len(hits),
                'realized_delta_l2':float(np.sqrt(actual_energy[ri])),
                'intended_delta_l2':float(np.sqrt(intended_energy[ri]))})
    return records

def metrics(rows):
    result={'n':len(rows),'recipient_correct':float(np.mean([r['recipient_correct'] for r in rows])),
            'donor_correct':float(np.mean([r['donor_correct'] for r in rows])),
            'unparsed':float(np.mean([r['parsed'] is None for r in rows]))}
    if rows and all('realized_delta_l2' in r for r in rows):
        result.update(mean_realized_delta_l2=float(np.mean([r['realized_delta_l2'] for r in rows])),
                      mean_intended_delta_l2=float(np.mean([r['intended_delta_l2'] for r in rows])))
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pairs=pairs_for('external'); source.save_json(OUT/'material/pairs.json',pairs)
    model,tok,_=model_utils.load_model('qwen3',dtype=torch.bfloat16,use_8bit=False)
    source.OUT=OUT; source_path=source.collect_source_means(model,tok,pairs)
    sf=np.load(source_path,mmap_mode='r'); ds=sf[:,1,6].astype('float32')-sf[:,0,6].astype('float32')
    band.OUT=OUT; paths=band.collect_all_layers(model,tok,pairs)
    dk={k:np.load(p,mmap_mode='r')[:,1].astype('float32')-np.load(p,mmap_mode='r')[:,0].astype('float32') for k,p in paths.items() if k in ('k','v')}
    records=[]; summaries={}; base_tokens={}
    for condition in CONDITIONS:
        raw=np.lib.format.open_memmap(OUT/f'field/greedy_prefill_{condition}.float16.npy',mode='w+',dtype='float16',shape=(len(pairs),37,2560))
        logits=np.lib.format.open_memmap(OUT/f'field/first_logits_{condition}.float16.npy',mode='w+',dtype='float16',shape=(len(pairs),model.config.vocab_size))
        rows=generate_condition(model,tok,pairs,condition,ds,dk,raw_field=raw,raw_logits=logits)
        raw.flush(); logits.flush()
        for row in rows:
            if condition=='baseline': base_tokens[row['pair_id']]=row['generated_ids']
            row['first_divergence']=next((i for i,(a,b) in enumerate(zip(row['generated_ids'],base_tokens[row['pair_id']])) if a!=b),None)
        records+=rows; summaries[condition]=metrics(rows)
        source.write_jsonl(OUT/f'behavior/{condition}.jsonl',rows)
        print(condition,json.dumps(summaries[condition]),flush=True)
    bygroup={g:{c:metrics([r for r in records if r['group']==g and r['condition']==c]) for c in CONDITIONS} for g in sorted({r['group'] for r in records})}
    result={'phase':2608,'timestamp':datetime.now().astimezone().isoformat(),'pairs':len(pairs),'conditions':summaries,'by_group':bygroup,
        'checks':{'all_120_external_pairs':len(pairs)==120,'all_1440_greedy':len(records)==1440,'all_hooks_prefill_once':all(r['intervention_sites_called_once'] for r in records)},
        'limitations':['external core lexical combinations overlap earlier splits','oracle donor required','24-token raw continuation, not chat or full reasoning','source span mean loses order','open vocabulary parsed metric still string based'],
        'language_mechanism_closed':False}
    result['all_checks_passed']=all(result['checks'].values()); source.save_json(OUT/'analysis/final.json',result)
    stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
    memo=rf'''

## Phase 2608: 多层通路的1440次自主生成、剂量与首分歧复验 [{{stamp}}]

**测试原理、用例与公式。** 对全部120个external pair（六族×中英×10），不筛选成功样本。比较baseline、source层5剂量0.25/0.5/1/1.5、roll、错token、负向；层6—35的KV、V、K与KV-roll。每次只在prefill注入，之后使用自然生成KV cache；最多24 token：

$$\hat y_t=\arg\max_v p(v\mid x,\hat y_{{<t}};do(H_{{5,S}}\!+\!\alpha\delta H)),\quad \tau=\min\{{t:\hat y_t^{{patch}}\ne\hat y_t^{{base}}\}}.$$

所有37个embedding/原始block-output状态的2560坐标与首步完整词表logits均落盘。不同于旧output_hidden_states索引，最后状态明确为未过final norm的block35输出。

**历史修正与计划续接。** Phase2604早期勘误把前缀差异归因于索引，最终修复实际是两侧同批、同形状BF16采集；全token exemplars已重采，旧boundary大批量场没有因此获得精度不变保证。文本不重复也不等于核心实体组合独立。Phase2605—2607依赖自身反事实，是oracle诊断，未满足Phase2600所要求的无测试donor提取器。材料纠错和多层追踪占用额外Phase，原合同剩余提取器、跨模型、客户端与汇总仍须继续，不能按编号提前宣称完成。

**结果汇总。** {json.dumps(summaries,ensure_ascii=False)}

**相关文件。** 本脚本及`{OUT}`，包括1440原始生成、完整坐标、首词表logits和逐族结果。

**分析、理论进展。** 将候选似然充分性与开放词表生成分开；带效应为重复层级读出提供可测试线索，但不证明每层自然执行同一个算法。

**问题硬伤、结论。** 仍需oracle，目标词常可复制；24-token原始续写不代表chat能力；样本核心组合不是全新实体锁箱；平均span破坏次序。不以任何单个阴性门关闭路线，也不把方向搬运命名为编码机制。检查={json.dumps(result['checks'])}；机制未闭合。
'''.replace('{stamp}',stamp)
    if '## Phase 2608:' not in MEMO.read_text(encoding='utf-8-sig'):
        with MEMO.open('a',encoding='utf-8') as f:f.write(memo)
    print(json.dumps(result['checks']),flush=True)

if __name__=='__main__':main()
