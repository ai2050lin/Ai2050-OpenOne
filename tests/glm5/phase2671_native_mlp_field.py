"""Durable BF16 native gated-MLP fields. All dimensions observed, no donor patch."""
import argparse, gc, hashlib, shutil, time
import numpy as np
import torch
from phase2620_native_coordinate_contract import *
from phase2670_native_mlp_contract import OUT as CONTRACT,FIELD as OUT,LAYERS,SITES
from phase2662_symmetric_mapping_contract import load_native,evaluate,leading_answer
from phase2663_symmetric_mapping_calibration import behavior_groups

def bits(t):
    assert t.dtype==torch.bfloat16,t.dtype
    return t.detach().contiguous().view(torch.uint16).cpu().numpy().copy()

def unbits(a):
    assert a.dtype==np.uint16
    return (a.astype(np.uint32)<<16).view(np.float32)

def tensor(t):return t[0] if isinstance(t,tuple) else t

class Capture:
    def __init__(self,model,selected=LAYERS):
        self.enabled=False;self.hooks=[];self.selected=tuple(selected);self.layers=len(model.model.layers);self.reset(0,False)
        self.hooks.append(model.get_input_embeddings().register_forward_hook(lambda m,a,b:self.take('h',0,tensor(b))))
        for l,block in enumerate(model.model.layers):
            mlp=block.mlp
            self.hooks.append(block.register_forward_hook(lambda m,a,b,l=l:self.take('h',l+1,tensor(b))))
            self.hooks.append(mlp.register_forward_pre_hook(lambda m,a,l=l:self.take('x',l,a[0])))
            self.hooks.append(block.post_attention_layernorm.register_forward_pre_hook(lambda m,a,l=l:self.take('pre_mlp',l,a[0],moments=False)))
            self.hooks.append(block.self_attn.register_forward_hook(lambda m,a,b,l=l:self.take('attention',l,tensor(b),moments=False)))
            if hasattr(mlp,'gate_proj'):
                self.hooks.append(mlp.gate_proj.register_forward_hook(lambda m,a,b,l=l:self.take('gate',l,b)))
                self.hooks.append(mlp.up_proj.register_forward_hook(lambda m,a,b,l=l:self.take('up',l,b)))
            else:
                assert hasattr(mlp,'gate_up_proj'),type(mlp)
                self.hooks.append(mlp.gate_up_proj.register_forward_hook(lambda m,a,b,l=l:self.merged(l,b)))
            self.hooks.append(mlp.down_proj.register_forward_pre_hook(lambda m,a,l=l:self.take('a',l,a[0])))
            self.hooks.append(mlp.down_proj.register_forward_hook(lambda m,a,b,l=l:self.take('down',l,b)))
    def reset(self,body,published):self.body=body;self.published=published;self.boundaries={};self.sums={};self.squares={};self.full={}
    def merged(self,l,b):
        if self.enabled:
            g,u=b.chunk(2,dim=-1);self.take('gate',l,g);self.take('up',l,u)
    def take(self,key,l,t,moments=True):
        if not self.enabled:return
        t=t.detach()[0]
        if key in ('h','a'):value=t[[self.body,-1]]
        elif key in ('gate','up') or l in self.selected:value=t[-1]
        else:value=None
        if value is not None:self.boundaries.setdefault(key,{})[l]=bits(value)
        if moments:
            # Float64 reductions include every token and every coordinate, not an embedding projection.
            d=t.double();self.sums.setdefault(key,{})[l]=d.sum(0).cpu().numpy();self.squares.setdefault(key,{})[l]=(d*d).sum(0).cpu().numpy()
        if self.published and (key=='h' or l in self.selected):self.full.setdefault(key,{})[l]=bits(t)
    def pack(self):
        out={k:np.stack([v[l] for l in sorted(v)]) for k,v in self.boundaries.items()}
        out.update({'full__'+k:np.stack([v[l] for l in sorted(v)]) for k,v in self.full.items()})
        return out
    def moment_pack(self):
        return {k+'__'+name:np.stack([v[l] for l in sorted(v)]) for name,src in [('sum',self.sums),('sumsq',self.squares)] for k,v in src.items()}
    def close(self):
        for h in self.hooks:h.remove()

def weight_snapshot(model,out):
    fields={};metadata=[]
    for l,j in SITES:
        mlp=model.model.layers[l].mlp
        for kind in ('gate','up','down'):
            w=getattr(mlp,kind+'_proj').weight
            v=w[:,j] if kind=='down' else w[j,:]
            a=v.detach().float().cpu().numpy();key=f'L{l}_J{j}_{kind}';fields[key]=a
            indices=np.argsort(np.abs(a),kind='stable');low=int(indices[len(a)//8]);ordinary=int(indices[len(a)//2])
            metadata.append({'layer':l,'unit':j,'kind':kind,'key':key,'shape':list(w.shape),'low_coordinate':low,'ordinary_coordinate':ordinary,
                'low_value':float(a[low]),'ordinary_value':float(a[ordinary]),'vector_sha256':hashlib.sha256(a.tobytes()).hexdigest()})
    out.joinpath('weights').mkdir(parents=True,exist_ok=True);np.savez(out/'weights/native_candidate_vectors.npz',**fields)
    save(out/'protocol/native_weights.json',{'vectors':metadata,'candidate_vector_sha256':sha(out/'weights/native_candidate_vectors.npz'),
        'selection':'All15candidate gate/up/down vectors retained with every coordinate; ordinary/low scalar controls at median/12.5percentile magnitude in each vector, indexstable ties. This is control matching, not a TopK representation.'})

def moment_group(r):return f'{r["family"]}_{r["language"]}_{r["field_set"]}'

@torch.inference_mode()
def run(model,tok,cases,out=OUT,selected=LAYERS,raw_all=True,compact=False,extra_eos=(),observer=None):
    out=Path(out);out.joinpath('analysis').mkdir(parents=True,exist_ok=True);out.joinpath('field').mkdir(parents=True,exist_ok=True);out.joinpath('maps').mkdir(parents=True,exist_ok=True)
    eos_ids={tok.eos_token_id,*extra_eos}
    recpath=out/'analysis/records.jsonl';records=[json.loads(s) for s in recpath.read_text(encoding='utf-8').splitlines()] if recpath.exists() else []
    assert [r['case_index'] for r in records]==[r['case_index'] for r in cases[:len(records)]]
    cap=Capture(model,selected);acc={};nt=0;ng=0;active=None;t0=time.monotonic()
    # A crashed partial group has durable behavior/raw fields; only its moments are re-measured.
    begin=len(records)
    if begin<len(cases):
        key=moment_group(cases[begin])
        while begin>0 and moment_group(cases[begin-1])==key:begin-=1
    with recpath.open('a',encoding='utf-8') as stream:
        for i in range(begin,len(cases)):
            r=cases[i];key=moment_group(r)
            if key!=active:assert not acc;active=key
            if shutil.disk_usage(out).free<8*1024**3:raise RuntimeError('8GiB floor; preserve durable records, do not delete unrelated files')
            cap.reset(r['body_end_token'],r['published']);cap.enabled=True
            ids=list(r['prompt_ids']);generated=[];decision=None
            result=model.model(input_ids=torch.tensor([ids],device=model.get_input_embeddings().weight.device),use_cache=False)
            state=result.last_hidden_state[0,-1];cap.enabled=False
            # Offloaded large heads are materialized on CUDA by Accelerate. Full-field
            # float64 reductions can leave large *unused* allocator blocks. Releasing
            # only those cached blocks does not change live tensors or BF16 arithmetic.
            if compact and 'cpu' in getattr(model,'hf_device_map',{}).values():torch.cuda.empty_cache()
            logits=model.lm_head(state).float();chosen=int(logits.argmax())
            pack=cap.pack();mm=cap.moment_pack();assert all(np.isfinite(a).all() for a in mm.values())
            extras={} if observer is None else observer(r,pack)
            for k,v in mm.items():
                if k not in acc:acc[k]=np.zeros_like(v)
                acc[k]+=v
            nt+=len(ids);ng+=1
            if i>=len(records):
                if raw_all or r['published']:
                    stored={'h_prompt':pack['h'][:,-1],'a_prompt':pack['a'][:,-1]} if compact and not r['published'] else pack
                    np.savez_compressed(out/f'field/case_{r["case_index"]:04d}.npz',**stored)
                native=chosen
                for step in range(16):
                    generated.append(chosen);ids.append(chosen);text=tok.decode(generated,skip_special_tokens=True)
                    label=leading_answer(text,r['language'])
                    if decision is None and label is not None:decision={'step':step,'answer_yes':label}
                    if chosen in eos_ids:break
                    if step==15:break
                    result=model.model(input_ids=torch.tensor([ids],device=model.get_input_embeddings().weight.device),use_cache=False)
                    chosen=int(model.lm_head(result.last_hidden_state[0,-1]).float().argmax())
                rec={k:r[k] for k in ('case_index','case_id','family','language','unit','content_instance','form','target_index','mention_order','probe_index','polarity','mapping','expected_yes','published','field_set')}
                rec.update(generated=text,generated_ids=generated,native_id=native,decision=decision,eos=any(t in eos_ids for t in generated),**evaluate(r,text))
                rec.update(extras)
                stream.write(json.dumps(rec,ensure_ascii=False)+'\n');stream.flush();records.append(rec)
            del mm,pack,result,state,logits;cap.reset(0,False)
            if i+1==len(cases) or moment_group(cases[i+1])!=key:
                np.savez_compressed(out/f'maps/alltoken_{key}.npz',**acc)
                save(out/f'analysis/moments_{key}.json',{'cases':ng,'tokens':nt,'meaning':'Alltoken percoordinate uncentered sum/sumsq, not sample variance or an independence claim.'});acc={};nt=ng=0;active=None
            if (i+1)%16==0:
                save(out/'analysis/progress.json',{'cases':i+1,'total':len(cases),'elapsed_seconds_this_process':time.monotonic()-t0,'free_bytes':shutil.disk_usage(out).free})
                print(out.name,i+1,'/',len(cases),flush=True)
    cap.close();save(out/'analysis/records.json',records)
    manifest=[{'path':str(out/f'field/case_{r["case_index"]:04d}.npz'),'case_index':r['case_index'],'published':r['published'],'bytes':(out/f'field/case_{r["case_index"]:04d}.npz').stat().st_size} for r in cases if raw_all or r['published']]
    save(out/'analysis/raw_manifest.json',manifest);return records

def main():
    assert not (OUT/'analysis/final.json').exists();cases=read(CONTRACT/'material/cases.json');model,tok=load_native('qwen4')
    cfg=model.config;assert (cfg.num_hidden_layers,cfg.hidden_size,cfg.intermediate_size)==(36,2560,9728)
    save(OUT/'protocol/model.json',{'dtype':str(model.dtype),'quantized':getattr(model,'is_quantized',False),'device_map':getattr(model,'hf_device_map',None),'actual_parameter_devices':sorted({str(p.device) for p in model.parameters()}),'layers':36,'hidden':2560,'intermediate':9728,'native_boundaries':'H/a axis1=[bodylast,promptlast]; gate/up promptlast; x/down/pre_mlp/attention selectedlayers23,26,27,28; uint16 represents exact BF16 bits.'})
    weight_snapshot(model,OUT);records=run(model,tok,cases);del model;gc.collect();torch.cuda.empty_cache()
    checks={'8192_cases':len(records)==8192,'32_fullcoordinate_moment_groups':len(list((OUT/'maps').glob('alltoken_*.npz')))==32,'material_immutable':sha(CONTRACT/'material/cases.json')==read(CONTRACT/'protocol/frozen.json')['material_sha256'],
        '16_fulltoken_examples':sum(r['published'] for r in records)==16}
    assert all(checks.values())
    finish(2671,'8192独立实体内容交叉的自然生成与原生gate/up/down全坐标场',OUT,{'provenance':str(Path(__file__)),'summary':{'behavior':behavior_groups(records),'cases':len(records),'published':16,'moments':32},'checks':checks},
        'BF16原生单前缀无cache，先完整观察再生成最多16token。所有层所有H/MLP单元逐坐标记录，不按成功筛除。所有token六种场流式保存全坐标和/平方和；原始BF16位无损保存。',
        r'g=W_gx,\quad u=W_ux,\quad a=\operatorname{SiLU}(g)\odot u,\quad m=W_da;\quad M_{l,j}=\sum_{x,t}X_{x,l,t,j},\quad Q_{l,j}=\sum_{x,t}X_{x,l,t,j}^2.',
        'C0018192完整自然生成；C002正文/任务H与MLP乘积全部坐标；C003所有层gate/up任务边界；C004全部token H/x/gate/up/a/down矩；C005冻结四中层真实x/down/Attention/归一化前值；16预定展示例完整轨迹。',
        '这给出真实MLP计算的可复查物理入口；行为失败仍有完整纹理。实体变化与内容变化已不再绑定，后续可检验复用是否来自名字或条件本身。',
        '8192是交叉条件非独立语义样本；自然输出上限16token会限制冗长模型表达。均值/平方和不保留未展示token之间关系，完整原始轨迹只存16例；BF16细弱变化还需FP32数值对照。',
        '继续2672逐候选全部输入/输出参数贡献与2673确认对照，不能将标准SwiGLU乘法账本冒称语言机制已闭合。')

if __name__=='__main__':main()
