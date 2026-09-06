"""4096 natural single-prefix greedy runs and unmodified full-token BF16 atlas subset."""
import json,shutil
from collections import defaultdict
import numpy as np
import torch
from phase2620_native_coordinate_contract import *
from phase2621_native_behavior_run import load_model
from phase2622_native_field_capture import arr,tensor_output
from phase2641_matched_operation_material import evaluate

SOURCE=RESULT/'phase2641_matched_operation_contract';OUT=RESULT/'phase2642_matched_operation_behavior'

class Capture:
    def __init__(self,model):
        self.enabled=False;self.hooks=[];self.reset()
        self.hooks.append(model.get_input_embeddings().register_forward_hook(self.hook_hidden))
        for block in model.model.layers:
            self.hooks.append(block.register_forward_hook(self.hook_hidden));self.hooks.append(block.mlp.down_proj.register_forward_pre_hook(self.hook_mlp))
    def reset(self):self.hidden=[];self.mlp=[]
    def hook_hidden(self,module,inputs,outputs):
        if self.enabled:self.hidden.append(arr(tensor_output(outputs)[0]))
    def hook_mlp(self,module,inputs):
        if self.enabled:self.mlp.append(arr(inputs[0][0,self.positions]))
    def close(self):
        for hook in self.hooks:hook.remove()

@torch.inference_mode()
def main():
    if (OUT/'analysis/final.json').exists():raise RuntimeError('phase already complete')
    model,tok=load_model('qwen4');cases=read(SOURCE/'material/cases.json');capture=Capture(model)
    records=[];manifest=[];firsts=[];(OUT/'behavior').mkdir(parents=True,exist_ok=True)
    save(OUT/'protocol/frozen.json',{'cases_sha256':sha(SOURCE/'material/cases.json'),'single_prompt':True,'cache':False,'maximum_new_tokens':12,'greedy_tie_policy':'torch.argmax; runnerup is maximum after masking chosen index','readout':'first native top1/top2 plus canonical entity A/B first-token contrast, all on actual BF16 hidden state'})
    with (OUT/'behavior/greedy.jsonl').open('w',encoding='utf-8') as stream:
        for i,row in enumerate(cases):
            ids=list(row['prompt_ids']);generated=[];capture.positions=[row['entity_spans']['a'][-1],row['entity_spans']['b'][-1],len(ids)-1]
            capture.reset();record=None
            for step in range(12):
                capture.enabled=step==0 and row['field_set']!='behavior_only'
                result=model.model(input_ids=torch.tensor([ids],device='cuda:0'),use_cache=False)
                state=result.last_hidden_state[0,-1];logits=model.lm_head(state).float();chosen=int(logits.argmax())
                if step==0:
                    masked=logits.clone();masked[chosen]=-float('inf');runner=int(masked.argmax())
                    contrast=model.lm_head.weight[chosen].float()-model.lm_head.weight[runner].float()
                    ai,bi=row['common_readout_ids'];common=None
                    if row['common_readout_available']:common=float((state.float()*(model.lm_head.weight[ai].float()-model.lm_head.weight[bi].float())).sum())
                    record={'case_index':i,'case_id':row['case_id'],'family':row['family'],'language':row['language'],'unit':row['unit'],'form':row['form'],'target_index':row['target_index'],'mention_order':row['mention_order'],'field_set':row['field_set'],
                            'native_ids':[chosen,runner],'native_tokens':tok.convert_ids_to_tokens([chosen,runner]),'native_margin16':float(logits[chosen]-logits[runner]),
                            'native_margin32':float((state.float()*contrast).sum()),'common_margin32':common,'common_readout_ids':row['common_readout_ids'],'common_available':row['common_readout_available']}
                    firsts.append(record)
                    if capture.enabled:
                        if shutil.disk_usage(OUT).free<8*1024**3:raise RuntimeError('insufficient free disk before raw field save')
                        path=OUT/f'field/case_{i:04d}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                        np.savez(path,hidden=np.stack(capture.hidden),mlp_positions=np.stack(capture.mlp),normalized_boundary=arr(state))
                        manifest.append({'path':str(path),'bytes':path.stat().st_size,'case_index':i,'field_set':row['field_set']})
                    capture.enabled=False;capture.reset()
                generated.append(chosen);ids.append(chosen)
                if chosen==tok.eos_token_id:break
            text=tok.decode(generated,skip_special_tokens=True)
            r={**record,'target':row['target'],'alternate':row['alternate'],'generated':text,'generated_ids':generated,'eos':tok.eos_token_id in generated,**evaluate(row,text)}
            records.append(r);stream.write(json.dumps(r,ensure_ascii=False)+'\n')
            if (i+1)%64==0:
                stream.flush();save(OUT/'analysis/progress.json',{'prompts':i+1,'total':len(cases),'raw_fields':len(manifest)});print('matched natural behavior',i+1,'/',len(cases),flush=True)
    capture.close();save(OUT/'analysis/first_decisions.json',firsts);save(OUT/'analysis/raw_manifest.json',manifest)
    groups={g:{'n':len(rr),'strict_accuracy':sum(r['strict_correct'] for r in rr)/len(rr),'name_content_accuracy':sum(r['name_content_correct'] for r in rr)/len(rr),'eos_rate':sum(r['eos'] for r in rr)/len(rr)} for g in sorted({r['family']+'/'+r['language'] for r in records}) for rr in [[r for r in records if r['family']+'/'+r['language']==g]]}
    cells=defaultdict(list)
    for r in records:cells[(r['family'],r['language'],r['unit'],r['form'])].append((r['target_index'],r['mention_order']))
    checks={'all4096_behavior':len(records)==4096,'all1024_raw_fields':len(manifest)==1024,'all_four_cells_per_matching_unit':len(cells)==1024 and all(len(v)==4 and set(v)=={(0,0),(0,1),(1,0),(1,1)} for v in cells.values()),
        'collision_common_not_zero_filled':all(r['common_margin32'] is None for r in firsts if not r['common_available'])}
    finish(2642,'4096单prompt自然生成与1024全token未干预BF16场',OUT,{'provenance':str(Path(__file__)),'summary':{'groups':groups,'raw_fields':len(manifest),'all_first_decisions':len(firsts)},'checks':checks},
        '一次只运行一个prompt，完整前缀无cache自然greedy；保存完整生成与真实首步输出对。预先分层的1024条保留全部token×37检查点×2560隐藏坐标，以及36层全部9728个MLP单位在A/B实体锚点和输出边界的值。',
        r'y_s=\arg\max_v z_v(x,y_{<s});\quad m_n^{32}=(U_{y_0}-U_{z_0})^Th;\quad m_c^{32}=(U_{A_0}-U_{B_0})^Th.',
        '八族中英、32实体对、双句式/目标/顺序全交叉4096条，最多12新token，仅EOS正常结束；首token碰撞共同读出不可用而非0。原始文本在2641，实际生成、IDs、读出和原场在本Phase。',
        '先独立记录行为是否胜任，再分析场；自然输出对由模型选择，共同实体对只是受控读出。第一实体与正确实体正交，可辨认简单首名复制倾向，而不把答对直接解释为深层语义机制。',
        '受控人名输出不是长文本自由生成；任务中实体可复制，部分类别/词义材料有强词面线索。名字可能多token，首token分数与完整名字正确率分别记录。不同族命题内容不完全相同，匹配实体只是控制一个混杂。',
        '在未按成功筛选的初始512条件上，以同值FP32计算原生/共同读出双伴随及全坐标V参数因子，先建立对照再讨论复用。')

if __name__=='__main__':main()
