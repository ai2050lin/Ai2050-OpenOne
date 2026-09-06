"""Unmodified forward fields and a donor-free last-MLP coordinate compiler.

No dimensional reduction: every hidden coordinate and every MLP boundary unit
is retained. Closed forms refer to the real-arithmetic extension of the captured
BF16 state and weights, not a derivative of discrete floating-point rounding.
"""
from __future__ import annotations
import argparse, gc, json, math, time
import numpy as np
import torch
from phase2620_native_coordinate_contract import *
from phase2621_native_behavior_run import load_model,MODELS
from phase2621_native_language_material import build,FAMILIES

OUT=RESULT/'phase2622_unmodified_native_fields'

def arr(x): return x.detach().float().cpu().numpy()

def tensor_output(x): return x[0] if isinstance(x,(tuple,list)) else x

def actual_parameter(module,name='weight'):
    """Read an offloaded parameter from Accelerate's CPU map, never from meta."""
    value=getattr(module,name,None)
    if value is None:return None
    if value.device.type!='meta':return value.detach()
    hook=getattr(module,'_hf_hook',None)
    if hook is not None and getattr(hook,'weights_map',None) is not None:
        return hook.weights_map[name].detach()
    for subhook in getattr(hook,'hooks',()):
        if getattr(subhook,'weights_map',None) is not None:return subhook.weights_map[name].detach()
    raise RuntimeError(f'Cannot resolve offloaded {type(module).__name__}.{name}')

def allocate(out,name,shape):
    path=Path(out)/f'field/{name}.float32.npy';path.parent.mkdir(parents=True,exist_ok=True)
    return np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=shape)

def eps_of(norm):
    return float(getattr(norm,'variance_epsilon',getattr(norm,'eps',1e-6)))

class Capture:
    def __init__(self,model,full=True):
        self.model=model;self.full=full;self.handles=[];self.positions=[0,-1];self.reset()
        self.handles.append(model.get_input_embeddings().register_forward_hook(self.embed))
        for l,block in enumerate(model.model.layers):
            self.handles.append(block.register_forward_hook(self.block(l)))
            self.handles.append(block.mlp.down_proj.register_forward_pre_hook(self.neuron(l)))
        last=model.model.layers[-1].mlp
        self.handles.append(last.register_forward_pre_hook(self.last_input))
        if hasattr(last,'gate_proj'):
            self.handles.append(last.gate_proj.register_forward_hook(self.project('gate')))
            self.handles.append(last.up_proj.register_forward_hook(self.project('up')))
        else:self.handles.append(last.gate_up_proj.register_forward_hook(self.project('gate_up')))
        self.handles.append(last.down_proj.register_forward_hook(self.project('mlp_output')))

    def reset(self):self.hidden=[];self.boundary=[];self.units=[];self.last={}
    def embed(self,module,inputs,output):
        if self.full:self.hidden.append(arr(output[0]))
        self.boundary.append(arr(output[0,self.positions]))
    def block(self,l):
        def hook(module,inputs,output):
            value=tensor_output(output)
            if self.full:self.hidden.append(arr(value[0]))
            self.boundary.append(arr(value[0,self.positions]))
            if l==len(self.model.model.layers)-1:self.last['h']=arr(value[0,-1])
        return hook
    def neuron(self,l):
        def hook(module,inputs):self.units.append(arr(inputs[0][0,self.positions]))
        return hook
    def last_input(self,module,inputs):self.last['x']=arr(inputs[0][0,-1])
    def project(self,key):
        def hook(module,inputs,output):self.last[key]=arr(output[0,-1])
        return hook
    def close(self):
        for h in self.handles:h.remove()

@torch.inference_mode()
def collect(model,tok,cases,out,save_full=True,include_native=False):
    out=Path(out);save(out/'material/cases.json',cases)
    n=len(cases);L=len(model.model.layers);D=model.config.hidden_size;K=model.model.layers[-1].mlp.down_proj.in_features
    device=torch.device('cuda:0')
    # Reuse a single full physical weight matrix. No Top-K, PCA or learned projection.
    W=actual_parameter(model.model.layers[-1].mlp.down_proj).to(device).float()
    gamma=actual_parameter(model.model.norm).to(device).float();epsilon=eps_of(model.model.norm)
    U=actual_parameter(model.get_output_embeddings())
    output_bias=actual_parameter(model.get_output_embeddings(),'bias')
    wnorm=(W*W).sum(0)
    fields={name:allocate(out,name,shape) for name,shape in {
        'hidden_anchor_boundary':(n,L+1,2,D),'mlp_anchor_boundary':(n,L,2,K),
        'normed_output':(n,D),'final_gate':(n,K),'final_up':(n,K),'final_mlp_input':(n,D),
        'output_weight_contrast':(n,D),'coordinate_attribution':(n,D),'gradient_h':(n,D),
        'gradient_a':(n,K),'gradient_gate':(n,K),'gradient_up':(n,K),
        'coordinate_delete_effect':(n,D),'neuron_delete_effect':(n,K)}.items()}
    if include_native:
        fields.update({name:allocate(out,name,shape) for name,shape in {'native_gradient_h':(n,D),'native_gradient_a':(n,K),
            'native_coordinate_attribution':(n,D),'native_neuron_delete_effect':(n,K),'native_weight_contrast':(n,D)}.items()})
    np.save(out/'field/final_down_weights.float32.npy',arr(W))
    np.save(out/'field/final_norm_weights.float32.npy',arr(gamma))
    capture=Capture(model,full=save_full);records=[];raw_manifest=[]
    for i,row in enumerate(cases):
        capture.reset();capture.positions=[row['anchor_positions'][-1],len(row['prompt_ids'])-1]
        ids=torch.tensor([row['prompt_ids']],device=device)
        output=model.model(input_ids=ids,use_cache=False)
        z=model.lm_head(output.last_hidden_state[:,-1]).float()[0]
        native=z.topk(2).indices.tolist()  # output-token identity only; no hidden-coordinate selection
        candidate=[];prefix_ok=True
        for answer in (row['target'],row['alternate']):
            combined=tok.encode(row['prompt']+answer,add_special_tokens=False)
            ok=combined[:len(row['prompt_ids'])]==row['prompt_ids'];prefix_ok &=ok
            candidate.append(combined[len(row['prompt_ids'])] if ok and len(combined)>len(row['prompt_ids']) else tok.encode(answer,add_special_tokens=False)[0])
        semantic=prefix_ok and candidate[0]!=candidate[1]
        y,zid=candidate if semantic else native
        contrast=(U[y].float()-U[zid].float()).to(device)
        bias=output_bias
        bias_delta=float((bias[y]-bias[zid]).float()) if bias is not None else 0.0
        h=torch.as_tensor(capture.last['h'],device=device);a=torch.as_tensor(capture.units[-1][-1],device=device)
        x=torch.as_tensor(capture.last['x'],device=device)
        if 'gate_up' in capture.last:gate,up=np.split(capture.last['gate_up'],2)
        else:gate,up=capture.last['gate'],capture.last['up']
        g=torch.as_tensor(gate,device=device);u=torch.as_tensor(up,device=device)
        w=contrast*gamma; s2=h.square().mean()+epsilon;s=s2.sqrt();q=(w*h).sum();m=q/s
        grad_h=w/s-q*h/(D*s**3);grad_a=W.T@grad_h
        sig=torch.sigmoid(g);silu=g*sig;grad_g=grad_a*u*(sig+g*sig*(1-sig));grad_u=grad_a*silu
        dot_w=w@W;dot_h=h@W
        neuron_delta=-a
        # One-native-unit deletion, retaining all K units. Exact in real arithmetic at the last MLP.
        neuron_new=(q+dot_w*neuron_delta)/torch.sqrt(torch.clamp(s2+(2*dot_h*neuron_delta+wnorm*neuron_delta.square())/D,min=epsilon))
        coordinate_delta=-h
        coordinate_new=(q+w*coordinate_delta)/torch.sqrt(torch.clamp(s2+(2*h*coordinate_delta+coordinate_delta.square())/D,min=epsilon))
        normed=output.last_hidden_state[0,-1].float().to(device)
        values={'hidden_anchor_boundary':np.stack(capture.boundary),'mlp_anchor_boundary':np.stack(capture.units),
            'normed_output':arr(normed),'final_gate':gate,'final_up':up,'final_mlp_input':arr(x),
            'output_weight_contrast':arr(contrast),'coordinate_attribution':arr(contrast*normed),
            'gradient_h':arr(grad_h),'gradient_a':arr(grad_a),'gradient_gate':arr(grad_g),'gradient_up':arr(grad_u),
            'coordinate_delete_effect':arr(coordinate_new-m),'neuron_delete_effect':arr(neuron_new-m)}
        for name,value in values.items():fields[name][i]=value
        if include_native:
            own_contrast=(U[native[0]].float()-U[native[1]].float()).to(device);own_w=own_contrast*gamma;own_q=(own_w*h).sum()
            own_gh=own_w/s-own_q*h/(D*s**3);own_ga=W.T@own_gh
            own_delete=(own_q+(own_w@W)*neuron_delta)/torch.sqrt(torch.clamp(s2+(2*dot_h*neuron_delta+wnorm*neuron_delta.square())/D,min=epsilon))-own_q/s
            for key,value in {'native_gradient_h':own_gh,'native_gradient_a':own_ga,'native_coordinate_attribution':own_contrast*normed,
                'native_neuron_delete_effect':own_delete,'native_weight_contrast':own_contrast}.items():fields[key][i]=arr(value)
        if save_full:
            full=np.stack(capture.hidden);path=out/f'field/fulltoken/case_{i:04d}.float32.npy';path.parent.mkdir(parents=True,exist_ok=True);np.save(path,full)
            raw_manifest.append({'path':str(path),'case_index':i,'case_id':row['case_id'],'shape':list(full.shape),'bytes':path.stat().st_size,
                'published_exemplar':row['index'] in (0,6) and row['form']==0})
        reconstructed=W@a
        if model.model.layers[-1].mlp.down_proj.bias is not None:reconstructed+=actual_parameter(model.model.layers[-1].mlp.down_proj,'bias').to(device).float()
        mlp_actual=torch.as_tensor(capture.last['mlp_output'],device=device)
        norm_formula=gamma*h/s
        records.append({'case_id':row['case_id'],'semantic_first_token_distinct':semantic,'candidate_token_ids':candidate,'objective_token_ids':[y,zid],
            'objective_tokens':tok.convert_ids_to_tokens([y,zid]),'native_top2_ids':native,'native_top2_tokens':tok.convert_ids_to_tokens(native),
            'native_margin':float(z[native[0]]-z[native[1]]),'measured_contrast':float(z[y]-z[zid]),'fp32_formula_contrast':float(m)+bias_delta,
            'logit_sum_abs_error':abs(float((contrast*normed).sum())+bias_delta-float(z[y]-z[zid])),
            'norm_formula_max_error':float((norm_formula-normed).abs().max()),
            'mlp_fp32_reconstruction_relative_l2':float(torch.linalg.vector_norm(reconstructed-mlp_actual)/(torch.linalg.vector_norm(mlp_actual)+1e-12)),
            'gate_product_relative_l2':float(torch.linalg.vector_norm(silu*u-a)/(torch.linalg.vector_norm(a)+1e-12)),
            'max_abs_single_neuron_delete':float((neuron_new-m).abs().max()),'max_abs_single_coordinate_delete':float((coordinate_new-m).abs().max()),
            'bias_difference':bias_delta,'token_count':len(row['prompt_ids'])})
        if (i+1)%32==0:
            for f in fields.values():f.flush()
            save(out/'analysis/progress.json',{'completed':i+1,'total':n})
            print('native capture',i+1,'/',n,flush=True)
    for f in fields.values():f.flush()
    capture.close()
    save(out/'analysis/native_records.json',records);save(out/'analysis/raw_manifest.json',raw_manifest)
    save(out/'protocol/model.json',{'model':type(model).__name__,'dtype':str(model.dtype),'device_map':getattr(model,'hf_device_map',{'first':str(next(model.parameters()).device)}),
        'layers':L,'hidden_size':D,'intermediate_size':K,'epsilon':epsilon,'raw_last_hidden':'raw output of final block; normed_output separate',
        'coordinate_order':'original physical indices; every unit retained','fulltensor_hooks':'embedding and raw block outputs; no source means',
        'objective':'target vs alternate FIRST token if distinct; otherwise native top1/top2; not complete answer probability',
        'weights_semantics':'all final down[j,k] parameters and exact gradient factors for down/gate/up at boundary; not all model weights',
        'native_label_free_objective_also_saved':include_native,
        'storage_precision':'FP32 containers for captured BF16 activations and full-coordinate derived FP32 quantities','nonquantized':not getattr(model,'is_quantized',False)})
    return records

def summarize(records):
    keys=['logit_sum_abs_error','norm_formula_max_error','mlp_fp32_reconstruction_relative_l2','gate_product_relative_l2','max_abs_single_neuron_delete','max_abs_single_coordinate_delete']
    return {'cases':len(records),'distinct_semantic_first_token':sum(r['semantic_first_token_distinct'] for r in records),
        **{k:{'mean':float(np.mean([r[k] for r in records])),'max':float(max(r[k] for r in records))} for k in keys}}

def main():
    model,tok=load_model('qwen4');cases=build(tok);records=collect(model,tok,cases,OUT)
    result={'provenance':str(Path(__file__)),'summary':summarize(records),'checks':{'all768_cases':len(records)==768,
        'all37_hidden_checkpoints':np.load(OUT/'field/hidden_anchor_boundary.float32.npy',mmap_mode='r').shape==(768,37,2,2560),
        'all36x9728_mlp_units':np.load(OUT/'field/mlp_anchor_boundary.float32.npy',mmap_mode='r').shape==(768,36,2,9728),
        'finite_all_fields':all(np.isfinite(np.load(p,mmap_mode='r')).all() for p in (OUT/'field').glob('*.npy'))}}
    finish(2622,'未干预全token原生场与36层全9728个MLP神经元测绘',OUT,result,
        '不用donor，也不搬运差分。每个prompt独立自然前向，保存全部token的embedding与36层原始block坐标；所有层MLP中间神经元保存锚点末subtoken和答案边界两个真实位置，无span平均。',
        r'a_{l,t,k}=\operatorname{SiLU}(g_{l,t,k})u_{l,t,k},\quad h_{l+1,t}=h_{l,t}+\mathrm{Attn}_{l,t}+W^{down}_la_{l,t};\quad z_y-z_z=\sum_j(U_{yj}-U_{zj})N(h)_j+b_y-b_z.',
        '八族全部768prompt，不剔除行为失败。每例含完整token顺序与原生2560坐标；36×9728个MLP神经元覆盖所有单位，但中间神经元只保存两个位置，不冒称全token MLP。',
        '第一次在本轮直接分开真实激活坐标、SwiGLU神经元与固定矩阵参数；能够逐坐标解释最终logit的求和，并测出BF16前向与FP32记账误差。全场不是仅答案边界，低值参数未截断。',
        '锚点由材料给定、不是自动语义定位；首token对比不是完整答案，碰撞时使用模型自己前两名token并显式标记。最后层原生公式只解释一段输出编译，不代表早中层机制；FP32容器不等于FP32推理。',
        '进入逐神经元/逐参数闭式算法、跨材料指纹及真实前向扰动验证。计算一致性不能升级为语义发现。')

if __name__=='__main__':main()
