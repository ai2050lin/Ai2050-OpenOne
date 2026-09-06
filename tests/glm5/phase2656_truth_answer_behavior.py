"""Natural BF16 behavior, every native boundary coordinate and explicit decision checkpoints."""
from collections import defaultdict
import shutil
import numpy as np
import torch
from phase2620_native_coordinate_contract import *
from phase2621_native_behavior_run import load_model
from phase2649_output_function_behavior import Capture
from phase2655_truth_answer_contract import OUT as MATERIAL,evaluate,leading_answer

OUT=RESULT/'phase2656_truth_answer_behavior'


@torch.inference_mode()
def main():
    assert not (OUT/'analysis/final.json').exists()
    cases=read(MATERIAL/'material/cases.json');model,tok=load_model('qwen4');cap=Capture(model);records=[];manifest=[];alltoken={};counts=defaultdict(int)
    OUT.joinpath('field').mkdir(parents=True,exist_ok=True);OUT.joinpath('behavior').mkdir(parents=True,exist_ok=True)
    save(OUT/'protocol/runtime_adaptation.json',{'initial_startup_failure':'hf_device_map optional attribute absent in all-GPU Transformers5.14; failure occurred in protocol serialization before any experimental forward or raw/behavior record. Fallback records actual parameter devices, no experimental rule changed.'})
    save(OUT/'protocol/frozen.json',{'material_sha256':sha(MATERIAL/'material/cases.json'),'precision':str(model.dtype),'device_map':getattr(model,'hf_device_map',{'actual_parameter_devices':sorted({str(p.device) for p in model.parameters()})}),'quantized':False,'max_new_tokens':16,
        'fullcoordinates':'allcases boundaryH andMLP; alltokenH streamed; fulltokenpublishedonly','decision':'first generated step whose decoded leading answer is identifiable, not a proof of prior internal commitment'})
    with (OUT/'behavior/greedy.jsonl').open('w',encoding='utf-8') as stream:
        for i,r in enumerate(cases):
            if shutil.disk_usage(OUT).free<8*1024**3:raise RuntimeError('8GiB floor')
            ids=list(r['prompt_ids']);generated=[];decision=None;pack={};first={};dinfo=None
            for step in range(16):
                cap.reset();cap.enabled=step==0 or (r['published'] and decision is None)
                result=model.model(input_ids=torch.tensor([ids],device='cuda:0'),use_cache=False);state=result.last_hidden_state[0,-1];logits=model.lm_head(state).float();chosen=int(logits.argmax());logits[chosen]=-float('inf');runner=int(logits.argmax())
                if step==0:
                    h=np.stack(cap.h);pack={'hidden_boundary':h[:,-1],'mlp_boundary':np.stack(cap.a),'normalized_boundary':state.float().cpu().numpy()}
                    if r['published']:pack['hidden_fulltoken']=h
                    key=f'{r["family"]}/{r["language"]}/p{r["probe_index"]}q{r["polarity"]}m{r["mapping"]}';hh=h.astype('float64')
                    if key not in alltoken:alltoken[key]=[np.zeros((37,2560)),np.zeros((37,2560))]
                    alltoken[key][0]+=hh.sum(1);alltoken[key][1]+=(hh*hh).sum(1);counts[key]+=h.shape[1]
                    first={'native_ids':[chosen,runner],'native_tokens':tok.convert_ids_to_tokens([chosen,runner])};del h,hh
                generated.append(chosen);s=tok.decode(generated,skip_special_tokens=True);answer=leading_answer(s,r['language'])
                if decision is None and answer is not None:
                    decision=step;dinfo={'step':step,'answer_yes':answer,'prefix_ids':list(ids),'native_ids':[chosen,runner]}
                    if r['published']:
                        pack['decision_hidden_fulltoken']=np.stack(cap.h);pack['decision_mlp_boundary']=np.stack(cap.a)
                ids.append(chosen);cap.enabled=False;cap.reset()
                if chosen==tok.eos_token_id:break
            path=OUT/f'field/case_{i:04d}.npz';np.savez_compressed(path,**pack);manifest.append({'path':str(path),'bytes':path.stat().st_size,'case_index':i,'published':r['published']})
            record={k:r[k] for k in ('case_index','case_id','family','language','unit','form','target_index','mention_order','probe_index','polarity','mapping','statement_truth','question_affirmative','expected_yes','target','alternate','field_set','fp_selected','published','crossmodel')}
            record.update(**first,generated=s,generated_ids=generated,eos=tok.eos_token_id in generated,decision=dinfo,**evaluate(r,s));records.append(record);stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            del result,state,logits,pack
            if (i+1)%64==0:stream.flush();save(OUT/'analysis/progress.json',{'cases':i+1,'total':8192});print('truth mapping natural',i+1,'/8192',flush=True)
    cap.close();save(OUT/'analysis/records.json',records);save(OUT/'analysis/raw_manifest.json',manifest);save(OUT/'analysis/alltoken_counts.json',dict(counts))
    np.savez_compressed(OUT/'field/alltoken_coordinate_maps.npz',**{k+'__'+kind:(s[0]/counts[k] if kind=='mean' else np.sqrt(s[1]/counts[k])).astype('float32') for k,s in alltoken.items() for kind in ('mean','rms')})
    groups={}
    for fam,lang,q,m in __import__('itertools').product(sorted({r['family'] for r in records}),('en','zh'),(0,1),(0,1)):
        rr=[r for r in records if (r['family'],r['language'],r['polarity'],r['mapping'])==(fam,lang,q,m)]
        groups[f'{fam}/{lang}/q{q}/m{m}']={'n':len(rr),'content_correct':sum(r['content_correct'] for r in rr),'strict_correct':sum(r['strict_correct'] for r in rr),
            'leading_answer_correct':sum(r['decision'] is not None and r['decision']['answer_yes']==r['expected_yes'] for r in rr),'eos':sum(r['eos'] for r in rr),
            'no_decision':sum(r['decision'] is None for r in rr),'decision_after_first':sum(r['decision'] is not None and r['decision']['step']>0 for r in rr)}
    checks={'8192_behavior_and_raw':len(records)==len(manifest)==8192,'128_fullcoordinate_token_groups':len(alltoken)==128,'64_published_raw':sum(r['published'] for r in manifest)==64,'nonquantized':not getattr(model,'is_quantized',False)}
    assert all(checks.values())
    finish(2656,'8192事实—问题—答案映射自然行为与原生全坐标',OUT,{'provenance':str(Path(__file__)),'summary':{'groups':groups,'alltoken_occurrences':sum(counts.values()),'raw_packs':len(manifest)},'checks':checks},
        '单prompt未干预BF16greedy，原生首位与首个可识别答案时刻分账。全部37检查点/2560H边界坐标和36层/9728MLP边界单位保留；全tokenH流式统计，不挑大激活。',
        r'y_s=\arg\max_v z_v(x,y_{<s});\quad s_{answer}=\min\{s:\operatorname{parse}(y_{\le s})\in\{Yes,No\}\};\quad R_{l,j}=\sqrt{\sum_{x,t}H_{x,l,t,j}^2/\sum_xT_x}.',
        '8192交叉条件、每条最多16新token；64预定实例保留完整前缀H及首个答案可识别时刻H/MLP。正反映射、真假极性和全部行为失败保留。',
        '把理解事实、理解问题与执行输出约定分开；答案识别时刻是外部可观察分叉，不声称它就是内部语义首次生成时刻。模型未遵守输出约定时仍保留其坐标轨迹。',
        '多数语义词表和模板复用，映射反转增加指令长度；严格整词评分与首词评分均不能认证任意长解释正确。非展示样本仅持久化边界而非每token全部H，完整token值已纳入全坐标聚合。',
        '继续第三批旧候选确认及新因素图谱；正反映射若失败，分析其失败条件与仍通过的轨迹，不终止整条路线。')


if __name__=='__main__':main()
