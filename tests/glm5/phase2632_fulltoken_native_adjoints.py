"""Native reverse-mode path tracing. Shared weight derivatives sum ALL token positions."""
import gc,json,shutil
import numpy as np
import torch
from phase2620_native_coordinate_contract import *
from phase2621_native_behavior_run import load_model
from phase2622_native_field_capture import arr,tensor_output

OUT=RESULT/'phase2632_fulltoken_native_adjoints'
LAYERS=(0,5,17,35)
MODULES=('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')
INPUT_KEY={'q_proj':'attn_x','k_proj':'attn_x','v_proj':'attn_x','o_proj':'o_x','gate_proj':'mlp_x','up_proj':'mlp_x','down_proj':'down_x'}

def module_at(model,l,name):return getattr(model.model.layers[l].self_attn if name.endswith('_proj') and name[0] in 'qkvo' else model.model.layers[l].mlp,name)

class AdjointCapture:
    def __init__(self,model):
        self.model=model;self.hooks=[];self.reset()
        for l,block in enumerate(model.model.layers):
            self.hooks.append(block.register_forward_hook(self.block(l)))
            self.hooks.append(block.mlp.down_proj.register_forward_pre_hook(self.mlp(l)))
        for l in LAYERS:
            for name in MODULES:self.hooks.append(module_at(model,l,name).register_forward_hook(self.linear(l,name)))
    def reset(self):self.hidden=[];self.a=[];self.linears={}
    def block(self,l):
        def hook(m,inp,out):
            value=tensor_output(out);value.retain_grad();self.hidden.append(value)
        return hook
    def mlp(self,l):
        def hook(m,inp):inp[0].retain_grad();self.a.append(inp[0])
        return hook
    def linear(self,l,name):
        def hook(m,inp,out):out.retain_grad();self.linears[(l,name)]=(inp[0],out)
        return hook
    def close(self):
        for hook in self.hooks:hook.remove()

def quartile_energy(value,gradient):
    # Every entry participates; quartiles are descriptive amplitude strata, never a retained subset.
    value=value.detach().float();gradient=gradient.detach().float();idx=value.abs().argsort(dim=-1);parts=torch.tensor_split(idx,4,dim=-1)
    energy=gradient.square();den=energy.sum()
    return [float(torch.gather(energy,-1,p).sum()/den) if den>0 else None for p in parts]

def main():
    model,tok=load_model('qwen4')
    for p in model.parameters():p.requires_grad_(False)
    frames=read(RESULT/'phase2631_native_generation_paths/material/frames.json');OUT.joinpath('field').mkdir(parents=True,exist_ok=True)
    weight_info={}
    for l in LAYERS:
        for name in MODULES:
            module=module_at(model,l,name);path=OUT/f'field/weights/L{l}_{name}.float32.npy';path.parent.mkdir(parents=True,exist_ok=True);np.save(path,arr(module.weight))
            weight_info[f'L{l}_{name}']={'shape':list(module.weight.shape),'sha256':sha(path),'rms':float(module.weight.detach().float().square().mean().sqrt())}
    save(OUT/'protocol/weights.json',weight_info)
    capture=AdjointCapture(model);records=[];hidden_boundary=[];hgrad_boundary=[];agrad_boundary=[];a_boundary=[];manifests=[]
    for fi,frame in enumerate(frames):
        if shutil.disk_usage(OUT).free<8*1024**3:raise RuntimeError('less than 8GiB storage; checkpoint retained, do not silently discard fields')
        capture.reset();ids=torch.tensor([frame['prefix_ids']],device='cuda:0');embedding=model.get_input_embeddings()(ids).detach().requires_grad_(True)
        result=model.model(inputs_embeds=embedding,use_cache=False);state=result.last_hidden_state[0,-1]
        logits=model.lm_head(state).detach().float();y,z=frame['chosen_id'],frame['runnerup_id']
        contrast=model.lm_head.weight[y].float()-model.lm_head.weight[z].float()
        loss=(state.float()*contrast).sum();loss.backward()
        hs=[embedding]+capture.hidden;boundary=frame['prompt_length']-1;anchor=frame['anchor_positions'][-1];positions=[anchor,boundary,len(frame['prefix_ids'])-1]
        h=np.stack([arr(v[0]) for v in hs]);g=np.stack([arr(v.grad[0]) for v in hs]);hpath=OUT/f'field/hidden/frame_{fi:04d}.npz';hpath.parent.mkdir(parents=True,exist_ok=True)
        np.savez(hpath,h=h,adjoint=g)
        hidden_boundary.append(h[:,positions]);hgrad_boundary.append(g[:,positions]);a_boundary.append(np.stack([arr(v[0,positions]) for v in capture.a]));agrad_boundary.append(np.stack([arr(v.grad[0,positions]) for v in capture.a]))
        pack={};sites=[];published=frame['index']==12 and frame['variant']==0
        with torch.no_grad():
            for l in LAYERS:
                for name in MODULES:
                    x,value=capture.linears[(l,name)];xx=x[0].float();gg=value.grad[0].float();key=f'L{l}_{name}'
                    input_key=f'L{l}_{INPUT_KEY[name]}'
                    if input_key not in pack:pack[input_key]=arr(xx)
                    elif not np.array_equal(pack[input_key],arr(xx)):raise RuntimeError('supposedly shared input differs')
                    pack[key+'__g']=arr(gg)
                    if published:pack[key+'__value']=arr(value[0])
                    full=gg.T@xx;last=torch.outer(gg[-1],xx[-1]);norm=full.norm();err=(full-last).norm()/norm if norm>0 else None
                    flat=int(full.abs().argmax());j,k=divmod(flat,full.shape[1]);rj=(fi*43+l*7+MODULES.index(name)*79)%full.shape[0];rk=(fi*137+l*29+MODULES.index(name)*31)%full.shape[1]
                    total=gg.square().sum();non_boundary=gg[:-1].square().sum()/total if total>0 else None
                    pack[key+'__row_rms']=arr(full.square().mean(-1).sqrt());pack[key+'__column_rms']=arr(full.square().mean(0).sqrt())
                    sites.append({'layer':l,'module':name,'shape':list(full.shape),'full_parameter_gradient_l2':float(norm),
                        'last_token_only_relative_l2_error':float(err) if err is not None else None,'non_boundary_adjoint_energy_fraction':float(non_boundary) if non_boundary is not None else None,
                        'diagnostic_max_j':j,'diagnostic_max_k':k,'diagnostic_max_full_gradient':float(full[j,k]),'diagnostic_max_last_gradient':float(last[j,k]),
                        'matched_index_j':rj,'matched_index_k':rk,'matched_full_gradient':float(full[rj,rk]),'matched_last_gradient':float(last[rj,rk])})
                    del full,last,xx,gg
        fpath=OUT/f'field/factors/frame_{fi:04d}.npz';fpath.parent.mkdir(parents=True,exist_ok=True);np.savez(fpath,**pack)
        records.append({'frame_id':fi,'case_id':frame['case_id'],'family':frame['family'],'language':frame['language'],'index':frame['index'],'variant':frame['variant'],
            'step':frame['step'],'eos':frame['eos'],'chosen_id':y,'runnerup_id':z,'chosen_token':tok.convert_ids_to_tokens(y),
            'autograd_same_native_argmax':int(logits.argmax())==y,'fp32_loss':float(loss.detach()),'bf16_head_margin':float(logits[y]-logits[z]),
            'trace_vs_autograd_fp32_margin_error':abs(float(loss.detach())-frame['fp32_readout_margin']),
            'hidden_amplitude_quartile_gradient_energy':[quartile_energy(v,v.grad) for v in hs],
            'mlp_amplitude_quartile_gradient_energy':[quartile_energy(v,v.grad) for v in capture.a],'sites':sites})
        for p in (hpath,fpath):manifests.append({'path':str(p),'bytes':p.stat().st_size,'frame_id':fi,'published':published})
        capture.reset();del result,state,loss,embedding,logits,hs,h,g,pack,x,value;gc.collect()
        if (fi+1)%16==0:save(OUT/'analysis/progress.json',{'frames':fi+1,'total':len(frames)});print('native adjoints',fi+1,'/',len(frames),flush=True)
    capture.close()
    for name,values in [('hidden_positions',hidden_boundary),('hidden_adjoint_positions',hgrad_boundary),('mlp_positions',a_boundary),('mlp_adjoint_positions',agrad_boundary)]:np.save(OUT/f'field/{name}.float32.npy',np.stack(values))
    save(OUT/'material/frames.json',frames);save(OUT/'analysis/records.json',records);save(OUT/'analysis/raw_manifest.json',manifests)
    summary={'frames':len(frames),'native_argmax_agreement':sum(r['autograd_same_native_argmax'] for r in records)/len(records),
        'maximum_trace_vs_autograd_fp32_margin_error':max(r['trace_vs_autograd_fp32_margin_error'] for r in records),'weight_sites':28,
        'all_hidden_coordinates':2560,'all_mlp_neurons':9728,'published_frames':sum(f['index']==12 and f['variant']==0 for f in frames)}
    structural=[s for r in records for s in r['sites'] if s['layer']==35 and s['module'] in ('q_proj','o_proj','gate_proj','up_proj','down_proj')]
    result={'provenance':str(Path(__file__)),'summary':summary,'checks':{'every_frame_traced':len(records)==len(frames),'all28_sites_per_frame':all(len(r['sites'])==28 for r in records),
        'final_pointwise_sites_no_earlier_token_adjoint':all(s['non_boundary_adjoint_energy_fraction'] in (0,None) for s in structural),
        'model_parameters_not_accumulating_gradients':all(p.grad is None for p in model.parameters())}}
    finish(2632,'真实生成全36层伴随场与四层七矩阵全token参数导数',OUT,result,
        '冻结模型参数，只令原生embedding输入可求导；沿实际前缀对固定自身top1/top2的FP32诊断读出反传。所有隐藏坐标与MLP单位参与，四层七矩阵的每个参数用全部token逐项求和计算。',
        r'G^{(l,W)}_{jk}=\sum_{t=0}^{T}\bar Y^{(l)}_{t,j}X^{(l)}_{t,k},\quad E_{last}=\|G-G^{last}\|_F/\|G\|_F;\quad \bar H_{l,t,j}=\partial m/\partial H_{l,t,j}.',
        'Phase2631中实际出现的0/1/4步，EOS单独标记；保存原始全token h和adjoint，全部四层七矩阵输入与输出伴随值（精确因子，不是低秩近似），在原生全矩阵上计算误差。',
        '把单参数在所有token位置共享的事实变成可查询算法，正式检验“只看最后token的单参数贡献”在早层和K/V是否遗漏路径。任何最大梯度坐标仅用于数值诊断选点，不定义稀疏语义核心。',
        '原生AD忽略BF16舍入不连续，敏感度需要有限扰动验证。输出对比由当前模型决定，可能是EOS或格式；不是特定语义监督，也不自动成为语义发现。仅四层线性矩阵有完整权重梯度因子，其他层只做状态/MLP伴随测绘。',
        '对早中晚层实际单权重进行双精度读出校验，比较全token预测与错误的末位置近似，继续保留所有低效应和负结果。')

if __name__=='__main__':main()
