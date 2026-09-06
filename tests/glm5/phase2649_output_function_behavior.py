"""All natural greedy outputs plus streamed all-token, all-coordinate BF16 atlas."""
import json,shutil
from collections import defaultdict
import numpy as np
import torch
from phase2620_native_coordinate_contract import *
from phase2621_native_behavior_run import load_model
from phase2622_native_field_capture import arr,tensor_output
from phase2648_output_function_material import evaluate

MATERIAL=RESULT/'phase2648_output_function_contract';OUT=RESULT/'phase2649_output_function_behavior'

class Capture:
    def __init__(self,model):
        self.enabled=False;self.hooks=[];self.reset()
        self.hooks.append(model.get_input_embeddings().register_forward_hook(self.hidden))
        for block in model.model.layers:
            self.hooks.append(block.register_forward_hook(self.hidden));self.hooks.append(block.mlp.down_proj.register_forward_pre_hook(self.mlp))
    def reset(self):self.h=[];self.a=[]
    def hidden(self,m,inp,out):
        if self.enabled:self.h.append(arr(tensor_output(out)[0]))
    def mlp(self,m,inp):
        if self.enabled:self.a.append(arr(inp[0][0,-1]))
    def close(self):
        for h in self.hooks:h.remove()

@torch.inference_mode()
def main():
    if (OUT/'analysis/final.json').exists():raise RuntimeError('completed phase')
    cases=read(MATERIAL/'material/cases.json');model,tok=load_model('qwen4');cap=Capture(model)
    firsts=[];manifest=[];records=[];alltoken={};token_counts=defaultdict(int)
    OUT.joinpath('behavior').mkdir(parents=True,exist_ok=True);OUT.joinpath('field').mkdir(parents=True,exist_ok=True)
    save(OUT/'protocol/frozen.json',{'material_sha256':sha(MATERIAL/'material/cases.json'),'model':'localnonquantizedBF16Qwen3-4B','single_prompt':True,'cache':False,'maximum_new_tokens':16,
        'scope':'alltokenH streamed into fullcoordinate sum/RMS; percase H3anchors +MLPboundary; raw fulltoken H only64 predefined published examples','native_runnerup':'argmax then mask chosen, no unstabletopk tie ordering'})
    with (OUT/'behavior/greedy.jsonl').open('w',encoding='utf-8') as stream:
        for i,row in enumerate(cases):
            ids=list(row['prompt_ids']);generated=[];cap.reset();first=None
            for step in range(16):
                cap.enabled=step==0 and row['field_set']!='behavior_only'
                result=model.model(input_ids=torch.tensor([ids],device='cuda:0'),use_cache=False);state=result.last_hidden_state[0,-1];logits=model.lm_head(state).float();chosen=int(logits.argmax())
                if step==0:
                    masked=logits.clone();masked[chosen]=-float('inf');runner=int(masked.argmax());a,b=row['common_readout_ids'];common=None
                    if row['common_readout_available']:common=float((state.float()*(model.lm_head.weight[a].float()-model.lm_head.weight[b].float())).sum())
                    first={k:row[k] for k in ('case_index','case_id','family','language','unit','form','target_index','mention_order','mode','output_function','probe_index','truth','field_set','published','response_orientation')}
                    first.update(native_ids=[chosen,runner],native_tokens=tok.convert_ids_to_tokens([chosen,runner]),common_ids=[a,b],common_available=row['common_readout_available'],
                        native_margin16=float(logits[chosen]-logits[runner]),native_margin32=float((state.float()*(model.lm_head.weight[chosen].float()-model.lm_head.weight[runner].float())).sum()),common_margin32=common)
                    firsts.append(first)
                    if cap.enabled:
                        if shutil.disk_usage(OUT).free<8*1024**3:raise RuntimeError('8GiB safety floor')
                        h=np.stack(cap.h);pos=[row['entity_spans']['a'][-1],row['entity_spans']['b'][-1],len(ids)-1]
                        pack={'hidden_positions':h[:,pos],'mlp_boundary':np.stack(cap.a),'normalized_boundary':arr(state)}
                        if row['published']:pack['hidden_fulltoken']=h
                        path=OUT/f'field/case_{i:04d}.npz';np.savez(path,**pack);manifest.append({'path':str(path),'bytes':path.stat().st_size,'case_index':i,'published':row['published'],'field_set':row['field_set']})
                        key=row['family']+'/'+row['language']+'/'+row['mode'];hh=h.astype('float64')
                        if key not in alltoken:alltoken[key]=[np.zeros((37,2560)),np.zeros((37,2560))]
                        alltoken[key][0]+=hh.sum(1);alltoken[key][1]+=(hh*hh).sum(1);token_counts[key]+=h.shape[1]
                        del h,hh,pack
                    cap.enabled=False;cap.reset()
                generated.append(chosen);ids.append(chosen)
                if chosen==tok.eos_token_id:break
            text=tok.decode(generated,skip_special_tokens=True);record={**first,'target':row['target'],'alternate':row['alternate'],'generated':text,'generated_ids':generated,'eos':tok.eos_token_id in generated,**evaluate(row,text)}
            records.append(record);stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            if (i+1)%64==0:
                stream.flush();save(OUT/'analysis/progress.json',{'cases':i+1,'total':len(cases),'raw_fields':len(manifest)});print('outputfunction natural',i+1,'/',len(cases),flush=True)
    cap.close();save(OUT/'analysis/first_decisions.json',firsts);save(OUT/'analysis/raw_manifest.json',manifest);save(OUT/'analysis/alltoken_counts.json',dict(token_counts))
    np.savez(OUT/'field/alltoken_coordinate_maps.npz',**{k+'__'+kind:(s[0]/token_counts[k] if kind=='mean' else np.sqrt(s[1]/token_counts[k])).astype('float32') for k,s in alltoken.items() for kind in ('mean','rms')})
    groups={g:{'n':len(rr),'strict_accuracy':sum(r['strict_correct'] for r in rr)/len(rr),'content_accuracy':sum(r['content_correct'] for r in rr)/len(rr),'eos_rate':sum(r['eos'] for r in rr)/len(rr)} for g in sorted({r['family']+'/'+r['language']+'/'+r['mode'] for r in records}) for rr in [[r for r in records if r['family']+'/'+r['language']+'/'+r['mode']==g]]}
    checks={'8192_outputs':len(records)==8192,'4096_original_fields':len(manifest)==4096,'64_alltoken_groups':len(alltoken)==64,'64_published':sum(r['published'] for r in manifest)==64,'collision_not_zero_filled':all(r['common_margin32'] is None for r in firsts if not r['common_available'])}
    assert all(checks.values())
    finish(2649,'8192输出功能自然生成与4096全坐标BF16原场',OUT,{'provenance':str(Path(__file__)),'summary':{'groups':groups,'raw_fields':len(manifest),'alltoken_occurrences':sum(token_counts.values())},'checks':checks},
        '完整前缀单prompt无cachegreedy，原生首步排序与固定任务读出分账。对预定4096条件扫描全部token、37层检查点所有隐藏坐标，流式汇总；保存三锚点H与全部MLP边界值，64例保留完整原场。',
        r'y_s=\arg\max_v z_v(x,y_{<s});\quad m_{task}=(U_{c_0}-U_{c_1})^Th;\quad R_{l,j}=\sqrt{\sum_{x,t}H_{x,l,t,j}^2/\sum_xT_x}.',
        '八族双语16新实体对、双句式/正确实体/提及顺序×name/cloze/truth_a/truth_b，8192完整生成，最多16新token。是/否内容采用预先明确的完整词表归一，不用答案子串冒充整句正确。',
        '自然胜任度与原场一起记录，未按成功筛除样本。给定事实前缀续写与自主完整生成明确区分；人名首token与完整答案分开。所有原坐标纳入流式图谱而非挑大激活。',
        '给定前缀改变了位置和token条件；输出差异不只是头行变化。全token统计混合不同位置，仅是原场描述；后续功能比较用明确锚点和全token参数求和。语言模板仍有限。',
        '按冻结实体分割继续初始2048双读出FP32图谱，固定读出与实际输出身份分账，不因某输出功能自然表现低就停止其他全坐标观察。')

if __name__=='__main__':main()
