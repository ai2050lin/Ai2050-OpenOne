"""All native coordinates on frozen equal-shape language/output conditions."""
import gc,shutil,time
import numpy as np
import torch
from phase2620_native_coordinate_contract import *
from phase2670_native_mlp_contract import LAYERS
from phase2677_source_role_contract import OUT as CONTRACT,FIELD as OUT
from phase2677_source_role_material import evaluate
from phase2677_padded_native_runtime import PAD_LENGTH,MAX_NEW_TOKENS,PaddedCapture,padded_inputs,native_pack,group_key,summarize_behavior
from phase2671_native_mlp_field import weight_snapshot
from phase2662_symmetric_mapping_contract import load_native


@torch.inference_mode()
def run(model,tok,cases):
    for folder in ('analysis','field','maps'):OUT.joinpath(folder).mkdir(parents=True,exist_ok=True)
    ordered=sorted(cases,key=lambda r:(group_key(r),r['case_index']))
    order=[r['case_index'] for r in ordered];recpath=OUT/'analysis/records.jsonl'
    records=[json.loads(s) for s in recpath.read_text(encoding='utf-8').splitlines()] if recpath.exists() else []
    assert [r['case_index'] for r in records]==order[:len(records)]
    begin=len(records)
    if begin<len(ordered):
        key=group_key(ordered[begin])
        while begin>0 and group_key(ordered[begin-1])==key:begin-=1
    # Grouped processing preserves deterministic per-coordinate moments without
    # 96 simultaneous dense accumulators. Case IDs/material order remain fixed.
    save(OUT/'protocol/execution.json',{'material_sha256':sha(CONTRACT/'material/cases.json'),'ordered_case_indices':order,
        'order':'lexicographic family/language/fieldset/output_function, then frozen case index',
        'padding_tokens':PAD_LENGTH,'generated_token_budget':MAX_NEW_TOKENS,'unmasked_moments_only':True,
        'published':'64fullH;16truthcasesadditionallyfulltoken4layerMLP; everycaseallH/a bothactualqueryboundaries'})
    cap=PaddedCapture(model,LAYERS);acc={};ng=nt=0;t0=time.monotonic()
    try:
        with recpath.open('a',encoding='utf-8') as stream:
            for i in range(begin,len(ordered)):
                r=ordered[i];key=group_key(r)
                if shutil.disk_usage(OUT).free<8*1024**3:raise RuntimeError('8GiB floor; preserve completed cases and do not delete unrelated files')
                ids=list(r['prompt_ids']);real_task=r['task_end_token'];assert real_task==len(ids)-1
                cap.reset(r['body_end_token'],r['published'],real_task);cap.enabled=True
                result=model.model(**padded_inputs(model,ids,tok.eos_token_id));cap.enabled=False
                logits=model.lm_head(result.last_hidden_state[0,real_task]).float();chosen=int(logits.argmax())
                pack=cap.pack();mm=cap.moment_pack();assert all(np.isfinite(v).all() for v in mm.values())
                for k,v in mm.items():
                    if k not in acc:acc[k]=np.zeros_like(v)
                    acc[k]+=v
                nt+=len(ids);ng+=1
                raw=native_pack(pack,r['published'],r['parameter_published']);path=OUT/f'field/case_{r["case_index"]:04d}.npz'
                if i<len(records):
                    with np.load(path) as old:assert set(old.files)==set(raw) and all(np.array_equal(old[k],v) for k,v in raw.items()),'Partial-group replay changed nativefield'
                else:
                    np.savez_compressed(path,**raw)
                    generated=[];native=chosen
                    for step in range(MAX_NEW_TOKENS):
                        generated.append(chosen);ids.append(chosen)
                        if chosen==tok.eos_token_id or step+1==MAX_NEW_TOKENS:break
                        result=model.model(**padded_inputs(model,ids,tok.eos_token_id))
                        chosen=int(model.lm_head(result.last_hidden_state[0,len(ids)-1]).float().argmax())
                    text=tok.decode(generated,skip_special_tokens=True)
                    record={k:r[k] for k in ('case_index','case_id','family','language','unit','content_instance','form','target_index','mention_order','probe_index','polarity','mapping','target','alternate','expected_yes','published','parameter_published','field_set','output_function','source_selected')}
                    record.update(generated=text,generated_ids=generated,native_id=native,eos=tok.eos_token_id in generated,**evaluate(r,text))
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush();records.append(record)
                del mm,pack,raw,result,logits;cap.reset(0,False)
                if i+1==len(ordered) or group_key(ordered[i+1])!=key:
                    np.savez_compressed(OUT/f'maps/alltoken_{key}.npz',**acc)
                    save(OUT/f'analysis/moments_{key}.json',{'cases':ng,'actual_unmasked_tokens':nt,'padding_tokens_included':0,
                        'meaning':'Percoordinate sum/sumsq includes every actual token and every physical coordinate; no variance or independent-observation interpretation.'})
                    acc={};ng=nt=0
                if (i+1)%16==0:
                    save(OUT/'analysis/progress.json',{'cases':i+1,'total':len(ordered),'elapsed_seconds_this_process':time.monotonic()-t0,'free_bytes':shutil.disk_usage(OUT).free,'last_case':r['case_id']})
                    print('2678 PADDED NATIVE',i+1,'/',len(ordered),flush=True)
    finally:cap.close()
    records.sort(key=lambda r:r['case_index']);save(OUT/'analysis/records.json',records)
    manifest=[{'case_index':r['case_index'],'path':str((OUT/f'field/case_{r["case_index"]:04d}.npz').resolve()),'published':r['published'],'parameter_published':r['parameter_published'],
               'bytes':(OUT/f'field/case_{r["case_index"]:04d}.npz').stat().st_size} for r in cases]
    save(OUT/'analysis/raw_manifest.json',manifest)
    return records


def main():
    assert not (OUT/'analysis/final.json').exists()
    assert read(CONTRACT/'analysis/final.json')['all_checks_passed']
    cases=read(CONTRACT/'material/cases.json');frozen=read(CONTRACT/'protocol/frozen.json')
    assert sha(CONTRACT/'material/cases.json')==frozen['material_sha256']
    model,tok=load_native('qwen4');assert model.dtype==torch.bfloat16 and not getattr(model,'is_quantized',False)
    save(OUT/'protocol/model.json',{'dtype':str(model.dtype),'quantized':False,'device_map':getattr(model,'hf_device_map',None),
        'actual_devices':sorted({str(p.device) for p in model.parameters()}),'real_boundaries':'H0embedding; H1..36block0..35outputs;H36beforefinalRMSNorm;bodylast/tasklast notpadlast',
        'shapes':{'h':[37,2,2560],'a':[36,2,9728]},'uint16':'Exact BF16 bit serialization, not new quantization.'})
    if not (OUT/'protocol/native_weights.json').exists():weight_snapshot(model,OUT)
    records=run(model,tok,cases);del model;gc.collect();torch.cuda.empty_cache()
    checks={'8448_conditions':len(records)==8448,'512_source_panel':sum(r['source_selected'] for r in records)==512,'64_fullH':sum(r['published'] for r in records)==64,
        '16_fullMLP':sum(r['parameter_published'] for r in records)==16,'96_dense_moment_groups':len(list((OUT/'maps').glob('alltoken_*.npz')))==96,
        '8448_native_raw_packs':len(read(OUT/'analysis/raw_manifest.json'))==8448,'material_immutable':sha(CONTRACT/'material/cases.json')==frozen['material_sha256']}
    assert all(checks.values()),checks
    finish(2678,'8448等长执行条件的全坐标场与四功能自然输出',OUT,{'provenance':str(Path(__file__)),'summary':{'conditions':len(records),'behavior':summarize_behavior(records),'moment_groups':96,'source_panel':512,'fullH':64,'fullMLP':16},'checks':checks},
        '所有前缀和逐步自然生成均以160位置右侧EOS遮罩填充，记录真实正文末尾与真实任务末尾。所有层所有H和MLP乘积单元保留两边界原值；全实际token的六类场按每个物理坐标累积，尾部填充不进入场矩。',
        r'H^{obs}_{\ell,q,j}=H^{(T=160)}_{\ell,q,j},\quad q\in\{t_{body},t_{task}\};\qquad M_{\ell,j}=\sum_{r}\sum_{t=0}^{t_{task}(r)}X_{r,\ell,t,j},\quad Q_{\ell,j}=\sum_{r,t}X_{r,\ell,t,j}^2.',
        'C0018192八族双语实体/内容/形式/顺序/目标/所问人/极性/映射条件；C002其中256是非条件与新增256人名/填空合成512同正文四功能格；C003完整自然输出至EOS或16token，全部失败保留；C004所有层H/a两真实边界全坐标；C00596组全部token全坐标矩；C00664例全tokenH，其中16例另保存四中层全token参数输入/输出场。',
        '把未来指令变化的执行长度混杂控制住，为来源token怎样在原生坐标分配贡献提供更可信入口。四功能格包含任务内容和上下文长度以外的支架变化，需保持同正文和实际token索引审查。',
        '固定总长度不保证跨精度精确相同；全场方向图不是语义必要性结论。8192条件非独立意义样本，人名和填空是受控模板，填空含外部给定助手前缀。未展示token矩不保留任意跨token关联；只有16例保留全部选定MLP轨迹，另外48例只有完整H。',
        '继续512同正文四功能条件的真实QKV/P/Wo全heads来源账本，以及source→归一化→实际MLP标量输入项；不把已知乘加重构当作破解语言。')


if __name__=='__main__':main()
