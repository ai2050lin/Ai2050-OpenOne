"""Natural no-cache trajectories, preserving the exact prefix behind every decision."""
import json,torch
from phase2620_native_coordinate_contract import *
from phase2621_native_language_material import build,evaluate
from phase2621_native_behavior_run import load_model,stats

OUT=RESULT/'phase2631_native_generation_paths'

@torch.inference_mode()
def trajectories(model,tok,cases,out):
    frames=[];behavior=[];device=next(model.parameters()).device
    U=model.lm_head.weight
    for ci,row in enumerate(cases):
        prefix=list(row['prompt_ids']);generated=[];decisions=[]
        maximum=160 if row['family']=='long_reorder' else 32
        for step in range(maximum):
            ids=torch.tensor([prefix],device=device)
            result=model.model(input_ids=ids,use_cache=False)
            state=result.last_hidden_state[0,-1]
            logits=model.lm_head(state).float();native=logits.topk(2).indices.tolist()
            w=U[native[0]].float()-U[native[1]].float();m32=float((state.float()*w).sum());m16=float(logits[native[0]]-logits[native[1]])
            decision={'step':step,'chosen_id':native[0],'runnerup_id':native[1],'tokens':tok.convert_ids_to_tokens(native),'bf16_head_margin':m16,'fp32_readout_margin':m32,'eos':native[0]==tok.eos_token_id}
            decisions.append(decision)
            if step in (0,1,4):
                frame={k:row[k] for k in ('case_id','family','language','index','form','variant','anchor_positions')}
                frame.update(frame_id=len(frames),prefix_ids=list(prefix),prompt_length=len(row['prompt_ids']),generated_prefix=list(generated),**decision)
                frames.append(frame)
            generated.append(native[0]);prefix.append(native[0])
            if native[0]==tok.eos_token_id:break
        text=tok.decode(generated,skip_special_tokens=True)
        b={k:row[k] for k in ('case_id','family','language','index','form','variant','base_unit','split','target','alternate')}
        b.update(generated=text,generated_ids=generated,eos=tok.eos_token_id in generated,decisions=decisions,**evaluate(row,text));behavior.append(b)
        if (ci+1)%16==0:
            save(Path(out)/'analysis/progress.json',{'completed_prompts':ci+1,'total':len(cases),'frames':len(frames)})
            print('natural path',ci+1,'/',len(cases),'frames',len(frames),flush=True)
    save(Path(out)/'material/frames.json',frames)
    with (Path(out)/'behavior/greedy.jsonl').open('w',encoding='utf-8') as f:
        for row in behavior:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    return frames,behavior

def main():
    model,tok=load_model('qwen4');cases=[r for r in build(tok,12,36,forms=(1,)) if r['index'] in (12,16,20,24,28,32)]
    save(OUT/'material/cases.json',cases);(OUT/'behavior').mkdir(parents=True,exist_ok=True)
    frames,behavior=trajectories(model,tok,cases,OUT)
    result={'provenance':str(Path(__file__)),'summary':{'prompts':len(cases),'frames':len(frames),'frame_steps':{s:sum(f['step']==s for f in frames) for s in (0,1,4)},
        'eos_frames':sum(f['eos'] for f in frames),'by_group':stats(behavior),'mean_head_rounding_abs_difference':sum(abs(f['fp32_readout_margin']-f['bf16_head_margin']) for f in frames)/len(frames)},
        'checks':{'all192_prompts':len(cases)==192,'no_duplicate_new_prompts':len({r['prompt'] for r in cases})==192,'every_selected_prefix_traceable':all(len(f['prefix_ids'])==f['prompt_length']+f['step'] for f in frames)}}
    finish(2631,'192新表面组合的真实多token轨迹与BF16/FP32双读出',OUT,result,
        '一个prompt一次全前缀无cache自然greedy，保存每步自己选择的top1/top2；不指定正确答案构造输出目标。选第0、1、4步做后续全场；已结束的生成不强行延长，EOS与格式token单独标记。',
        r'y_s=\arg\max_v z_v(x,y_{<s}),\quad m^{16}_s=z^{16}_{y_s}-z^{16}_{z_s},\quad m^{32}_s=(U_{y_s}-U_{z_s})^T h^{norm}_s.',
        '八族中英×六个item×双变体=192条新的form1组合；相同item的form0此前测试过，因此是新组合而非全新语义事件。所有真实生成prefix与token IDs保留。',
        '后续参数读取对准实际生成中的分叉，不把预设答案或第一步格式长期当语义目标。FP32读出仅提高末端测量分辨率，不改变BF16内部模型或宣称FP32生成。',
        '两个读出只比较同一对原生top2，不是重新计算全部FP32词表排名。无cache实现有别于先前cached生成接口；可复算但不能把其行为差异直接当能力变化。多数短答案的后一步是EOS，必须从内容token分析中分离。',
        '对实际存在的选择前缀执行全坐标伴随追踪，覆盖原生Q/K/V和MLP共享参数的全token作用。')

if __name__=='__main__':main()
