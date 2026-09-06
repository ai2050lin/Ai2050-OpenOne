"""Durable natural-generation runner and calibration-only instruction selection."""
import gc,shutil
import numpy as np
import torch
from phase2620_native_coordinate_contract import *
from phase2649_output_function_behavior import Capture
from phase2662_symmetric_mapping_contract import OUT as CONTRACT,load_native,evaluate,evaluate_multi,leading_answer

OUT=RESULT/'phase2663_symmetric_mapping_calibration'


def behavior_groups(records):
    groups={}
    for r in records:groups.setdefault(f'{r["language"]}/q{r["polarity"]}/m{r["mapping"]}',[]).append(r)
    return {key:{'n':len(rr),'content_correct':sum(r['content_correct'] for r in rr),'strict_correct':sum(r['strict_correct'] for r in rr),'eos':sum(r['eos'] for r in rr)} for key,rr in groups.items()}


@torch.inference_mode()
def run(model,tok,cases,out,fields=False):
    out=Path(out);out.joinpath('analysis').mkdir(parents=True,exist_ok=True);stream_path=out/'analysis/records.jsonl'
    records=[json.loads(line) for line in stream_path.read_text(encoding='utf-8').splitlines()] if stream_path.exists() else []
    assert [r['case_index'] for r in records]==[r['case_index'] for r in cases[:len(records)]]
    cap=Capture(model) if fields else None;manifest=[]
    if fields:out.joinpath('field').mkdir(parents=True,exist_ok=True)
    with stream_path.open('a',encoding='utf-8') as stream:
        for i,r in enumerate(cases[len(records):],len(records)):
            if fields and shutil.disk_usage(out).free<8*1024**3:raise RuntimeError('8GiB disk floor reached; keep records, no unscoped cleanup')
            ids=list(r['prompt_ids']);generated=[];decision=None;first=None;pack={}
            for step in range(16):
                if cap:cap.reset();cap.enabled=step==0
                result=model.model(input_ids=torch.tensor([ids],device=model.get_input_embeddings().weight.device),use_cache=False);state=result.last_hidden_state[0,-1];logits=model.lm_head(state).float();chosen=int(logits.argmax());logits[chosen]=-float('inf');runner=int(logits.argmax())
                if step==0:
                    first=[chosen,runner]
                    if fields:
                        h=np.stack(cap.h);hh=h.astype('float64');anchor=[r['entity_spans']['a'][-1],r['entity_spans']['b'][-1],h.shape[1]-1]
                        pack={'hidden_boundary':h[:,-1],'hidden_anchor':h[:,anchor],'mlp_boundary':np.stack(cap.a),'normalized_boundary':state.float().cpu().numpy(),
                            'hidden_token_sum':hh.sum(1),'hidden_token_sumsq':(hh*hh).sum(1)}
                        if r['published']:pack['hidden_fulltoken']=h
                        del h,hh
                generated.append(chosen);ids.append(chosen);text=tok.decode(generated,skip_special_tokens=True)
                if not r.get('multi'):
                    label=leading_answer(text,r['language'])
                    if decision is None and label is not None:decision={'step':step,'answer_yes':label}
                if cap:cap.enabled=False;cap.reset()
                if chosen==tok.eos_token_id:break
            if fields:
                assert all(np.isfinite(x).all() for x in pack.values());np.savez_compressed(out/f'field/case_{r["case_index"]:04d}.npz',**pack)
            rec={k:r[k] for k in ('case_index','case_id','family','language','unit','form','target_index','mention_order','probe_index','polarity','mapping','style','shots','expected_yes','target','published','field_set')}
            rec.update(generated=text,generated_ids=generated,native_ids=first,decision=decision,eos=tok.eos_token_id in generated,**(evaluate_multi(r,text) if r.get('multi') else evaluate(r,text)));records.append(rec);stream.write(json.dumps(rec,ensure_ascii=False)+'\n');stream.flush()
            del pack,result,state,logits
            if (i+1)%32==0:save(out/'analysis/progress.json',{'cases':i+1,'total':len(cases)});print(out.name,i+1,'/',len(cases),flush=True)
    if cap:cap.close()
    save(out/'analysis/records.json',records)
    if fields:
        for r in cases:
            p=out/f'field/case_{r["case_index"]:04d}.npz';assert p.exists();manifest.append({'path':str(p),'bytes':p.stat().st_size,'case_index':r['case_index'],'published':r['published']})
        save(out/'analysis/raw_manifest.json',manifest)
    return records


def main():
    assert not (OUT/'analysis/final.json').exists();cases=read(CONTRACT/'material/calibration_cases.json');model,tok=load_native('qwen4');records=run(model,tok,cases,OUT)
    del model;gc.collect();torch.cuda.empty_cache();variants={};chosen={}
    for lang in ('en','zh'):
        options=[]
        for style in (0,1):
            for shots in (0,1):
                rr=[r for r in records if (r['language'],r['style'],r['shots'])==(lang,style,shots)];groups=behavior_groups(rr);counts=[g['content_correct'] for g in groups.values()]
                obj={'style':style,'shots':shots,'groups':groups,'minimum_cell_correct':min(counts),'total_correct':sum(counts),'n':len(rr)};variants[f'{lang}/s{style}/d{shots}']=obj;options.append(obj)
        chosen[lang]=max(options,key=lambda x:(x['minimum_cell_correct'],x['total_correct'],-x['shots'],-x['style']))
    selection={'selection':chosen,'calibration_material_sha256':sha(CONTRACT/'material/calibration_cases.json'),'records_sha256':sha(OUT/'analysis/records.json'),'frozen_before_heldout':True};save(OUT/'protocol/selected.json',selection)
    checks={'2048_records':len(records)==2048,'eight_instruction_variants':len(variants)==8,'per_cell64':all(g['n']==64 for v in variants.values() for g in v['groups'].values()),'no_heldout_read':not (RESULT/'phase2664_symmetric_native_field/analysis/records.json').exists()}
    assert all(checks.values());finish(2663,'2048自然校准与仅校准集的对称问法冻结',OUT,{'provenance':str(Path(__file__)),'summary':{'variants':variants,'selected':chosen},'checks':checks},
        '单prompt BF16 greedy、16token上限。每语言按最弱极性/映射格优先选择指令，次比较总正确数，最后按预定无演示/style0破平局；不在留出集调参。',
        r'C_{s,q,m}=\sum_i\mathbf1[\operatorname{content}(y_i)=y_i^*];\quad s^*=\arg\max_s(\min C,\sum C,-d,-s).',
        '8族×2语言×2校准实体对×目标/探问/极性/映射×2指令×无或2演示=2048；每语言每指令组合256、每q/m格64。全部生成、严格/内容分数和EOS保留。',
        '冻结的赢家仅为后续可判定性改善的操作设置，不是发现语义机制。正文在指令/演示之前，可另外检查因果源前缀是否只受形状数值影响。',
        '校准只有每语言2实体对且演示是固定简单比较；选择后的校准分数有选择偏差，不能冒充新材料能力。任一格弱也继续全坐标观察，不因行为失败关闭路线。',
        '保持选择文件哈希，生成并执行8192新实体条件的完整原生场；先记录全量，再分析条件方向与坐标复用。')


if __name__=='__main__':main()
