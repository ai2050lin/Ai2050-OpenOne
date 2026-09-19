"""Sequential full-native-width replication; fit and freeze before fresh-source capture."""
import argparse
import gc
from collections import defaultdict
from rdc_operator_common import *
from rdc_native_conditional_operator import apply_operator, save_bank, load_bank, label
from phase2726_rdc_operator_compile import causal_meta
from phase2721_rdc_joint_scale import MODELS
from rdc_operator_model import load, memory


def selected():
    result=[]
    for split,n in [('train',32),('validation',16),('confirmation',16)]:
        for lang in ('en','zh'):
            result.extend([r for r in rows() if r['split']==split and r['language']==lang][:n])
    return result


class Observer:
    def __init__(self,model,blocks):
        self.blocks=blocks;self.depth=len(model.model.layers);self.handles=[];self.enabled=False
        self.positions=[];self.data={};self.h={};self.stats={};self.hashes={};self.energy={};self.active={}
        def emb(m,a,o):
            if self.enabled:self.hidden(0,o[0])
        self.handles.append(model.get_input_embeddings().register_forward_hook(emb))
        for b,layer in enumerate(model.model.layers):
            def before(m,a,b=b):
                if self.enabled:self.active[b]={'r':a[0]}
            self.handles.append(layer.register_forward_pre_hook(before))
            def att(m,a,o,b=b):
                if self.enabled:self.active[b]['a']=o[0]
            self.handles.append(layer.self_attn.register_forward_hook(att))
            def mlp(m,a,o,b=b):
                if self.enabled:
                    self.active[b]['m']=o
                    if b in blocks:self.data[f'L{b}_mlp']=bits(o[0,self.positions])
            self.handles.append(layer.mlp.register_forward_hook(mlp))
            def after(m,a,o,b=b):
                if not self.enabled:return
                import torch
                o=o[0] if isinstance(o,tuple) else o
                self.hidden(b+1,o[0]);d=self.active.pop(b)
                r,att,mlp=(d[k][0].float() for k in ('r','a','m'))
                y=o[0].float()
                self.energy[b]=torch.stack([r.square().mean(-1),att.square().mean(-1),mlp.square().mean(-1),
                    2*(r*att).mean(-1),2*(r*mlp).mean(-1),2*(att*mlp).mean(-1),y.square().mean(-1)]).cpu().numpy()
            self.handles.append(layer.register_forward_hook(after))
            if b in blocks:
                def x(m,a,o,b=b):
                    if self.enabled:self.data[f'L{b}_x']=bits(o[0,self.positions])
                self.handles.append(layer.post_attention_layernorm.register_forward_hook(x))
                if hasattr(layer.mlp,'gate_proj'):
                    for key,mod in [('gate',layer.mlp.gate_proj),('up',layer.mlp.up_proj)]:
                        def factor(m,a,o,b=b,key=key):
                            if self.enabled:self.data[f'L{b}_{key}']=bits(o[0,self.positions])
                        self.handles.append(mod.register_forward_hook(factor))
                else:
                    assert hasattr(layer.mlp,'gate_up_proj')
                    def combined(m,a,o,b=b):
                        if self.enabled:
                            g,u=o.chunk(2,-1)
                            self.data[f'L{b}_gate']=bits(g[0,self.positions]);self.data[f'L{b}_up']=bits(u[0,self.positions])
                    self.handles.append(layer.mlp.gate_up_proj.register_forward_hook(combined))
                def activation(m,a,b=b):
                    if self.enabled:self.data[f'L{b}_activation']=bits(a[0][0,self.positions])
                self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(activation))
    def hidden(self,b,value):
        import torch
        raw=bits(value);self.h[b]=raw[self.positions];self.hashes[b]=identity(raw)
        v=value.float();rms=v.square().mean(-1).sqrt().clamp_min(1e-12)
        u=v/rms[:,None]
        indicator=self.ind.to(v.device)
        self.stats[b]=torch.stack([indicator.T@v,indicator.T@v.square(),indicator.T@u,indicator.T@u.square()]).cpu().numpy()
    def reset(self,positions,ind):
        self.positions=positions;self.ind=ind;self.data={};self.h={};self.hashes={};self.stats={};self.energy={};self.active={}
    def close(self):
        for h in self.handles:h.remove()


def native_weights(key,b):
    from safetensors import safe_open
    path=ROOT/'models/hf'/MODELS[key];ix=read(path/'model.safetensors.index.json')['weight_map']
    def tensor(name):
        with safe_open(str(path/ix[name]),framework='pt',device='cpu',backend='pread') as f:return f.get_tensor(name).float()
    base=f'model.layers.{b}.mlp.'
    if base+'gate_proj.weight' in ix:
        return {'g':tensor(base+'gate_proj.weight'),'u':tensor(base+'up_proj.weight'),'d':tensor(base+'down_proj.weight')}
    both=tensor(base+'gate_up_proj.weight');g,u=both.chunk(2,0)
    return {'g':g,'u':u,'d':tensor(base+'down_proj.weight')}


def read_data(out,records,b):
    pieces=defaultdict(list);meta=[]
    for r in records:
        m=read(out/'rows'/f'{r["sample_id"]}.json');meta.extend(m['anchor_meta'])
        with np.load(out/'fields'/f'{r["sample_id"]}.npz') as z:
            for name in ('x','gate','up','mlp'):pieces[name].append(unbits(z[f'L{b}_{name}']))
    return {k:np.concatenate(v).astype(np.float32) for k,v in pieces.items()},meta


def fit(data,meta):
    from rdc_native_conditional_operator import silu_np
    ix=np.array([i for i,r in enumerate(meta) if r['split']=='train'])
    phi=silu_np(data['gate']);bank={}
    for group in ('global','piece','cue','token','position'):
        centers={};labels=[label(meta[i],group) for i in ix]
        for key in sorted(set(labels)) if group in ('global','piece','cue') else []:
            jj=ix[np.asarray(labels)==key]
            centers[key]={'count':len(jj),**{k:data[k][jj].astype(float).mean(0).astype(np.float32) for k in ('x','up','mlp')},
                'phi':phi[jj].astype(float).mean(0).astype(np.float32)}
        bank[group]=centers
    bank['diagonal']={'slope':np.zeros(data['x'].shape[-1],np.float32),'intercept':bank['global']['all']['mlp']}
    return bank


def main(key):
    import torch
    out=BASE/'scale'/key
    if (out/'result.json').exists():return
    start=time.monotonic();material=selected();guard(450*1024**2)
    protocol={'timestamp':stamp(),'source':snapshot(Path(__file__)),'model':key,'source_ids':[r['sample_id'] for r in material],
        'sources':128,'train_sources':64,'validation_sources':32,'confirmation_sources':32,'anchors_per_source':2,'quantization':False,
        'all_token_coverage':'Every token/all native coordinates at embedding and all block outputs streamed to original-order moments and SHA; all-layer/all-coordinate2anchor raw fields and all-layer all-token energy/cross terms retained.',
        'block_selection':'floor(depth/6),floor(depth/2),depth-2 in own model; indices do not imply matched functions.',
        'fresh_rule':'Capture96train/validation, unload model, fit CPU full-native operators/select validation and freeze; then load same model to capture32confirmation. Confirmation output not used for selection.',
        'alignment':'Keep Q4 native anchor endpoint; other tokenizer last nonempty piece ending no later than that character. Actual IDs/offsets/causal decoded prefix retained; equal char endpoint is not identical token information.',
        'scope':'Independent native-coordinate small-scale replication, not coordinate isomorphism or a full semantic benchmark. Local MLP prediction receives actual current post-attention-normalized x.'}
    if not (out/'protocol.json').exists():immutable(out/'protocol.json',protocol)
    assert read(out/'protocol.json')['source_ids']==protocol['source_ids']
    save(out/'execution_source.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'original_protocol_sha':sha(out/'protocol.json'),
        'refinement':'Original scientific material/selection unchanged. Reduced-residency loader, per-source cache release, verified native replay and persistent original protocol on resume.'})
    names=['constant_global','frozen_gate_global','frozen_gate_piece','frozen_gate_cue','tangent_global','quadratic_global']
    reports=[];choices={};blocks=None;checks=[]
    for stage in ('main','confirmation'):
        model,tok=load(key,out/(stage+'_commit_safe'));device=model.get_input_embeddings().weight.device
        depth,width=len(model.model.layers),model.config.hidden_size;blocks=[depth//6,depth//2,depth-2]
        observer=Observer(model,blocks);summaries={};count={}
        stage_rows=[r for r in material if (r['split']=='confirmation')==(stage=='confirmation')]
        try:
          with torch.inference_mode():
            for i,r in enumerate(stage_rows):
                cp=out/'rows'/f'{r["sample_id"]}.json'
                enc=tok(r['text'],add_special_tokens=False,return_offsets_mapping=True)
                ids=enc['input_ids'];positions=[];exact=[]
                for p in r['anchors']:
                    end=r['token_offsets'][p][1]
                    q=max(j for j,(a,b) in enumerate(enc['offset_mapping']) if b>a and b<=end)
                    positions.append(q);exact.append(enc['offset_mapping'][q][1]==end)
                mm=causal_meta(ids,r['language'],tok)
                cats=np.array([m['piece'] for m in mm]);ind=torch.as_tensor(np.eye(7,dtype=np.float32)[cats],device=device)
                observer.reset(positions,ind);observer.enabled=True
                x=torch.tensor([ids],device=device);post=model.model(input_ids=x,use_cache=False).last_hidden_state
                observer.enabled=False
                packet={**observer.data,'H':np.stack([observer.h[b] for b in range(depth+1)]),'postnorm':bits(post[0,positions]),
                    'block_energy_terms':np.stack([observer.energy[b] for b in range(depth)])}
                from rdc_operator_capture import persist_arrays
                persist_arrays(out/'fields'/f'{r["sample_id"]}.npz',packet)
                moments=np.stack([observer.stats[b] for b in range(depth+1)]).astype(np.float64)
                group=r['split']+'_'+r['language'];summaries[group]=summaries.get(group,0)+moments
                count[group]=count.get(group,0)+np.bincount(cats,minlength=7)
                score=model.lm_head(post[0,positions]).float().log_softmax(-1)
                anchor_meta=[{**mm[q],'sample_id':r['sample_id'],'source_group':r['source_group'],'split':r['split'],'anchor':j} for j,q in enumerate(positions)]
                output={k:r[k] for k in ('sample_id','source_group','split','language','text')}
                output.update(prompt_ids=ids,token_offsets=enc['offset_mapping'],positions=positions,exact_char_endpoint=exact,anchor_meta=anchor_meta,
                    all_token_layer_identity=observer.hashes,field_sha=sha(out/'fields'/f'{r["sample_id"]}.npz'),
                    native_next_NLL=[float(-score[j,ids[q+1]]) for j,q in enumerate(positions)])
                save(cp,output)
                if i<2 or (i+1)%16==0:print('OPERATOR_SCALE',key,stage,i+1,len(stage_rows),'seconds',round(time.monotonic()-start,1),flush=True)
                del packet,post,x,score,ind,moments;observer.reset([],None)
                torch.cuda.empty_cache()
                if (i+1)%8==0:
                    save(out/'resource_progress.json',{'timestamp':stamp(),'stage':stage,'sources_processed':i+1,**memory(),
                        'GPU_allocated':torch.cuda.memory_allocated(),'GPU_reserved':torch.cuda.memory_reserved()})
                guard();assert time.monotonic()-start<7200
        finally:
            observer.close();del observer,model;gc.collect();torch.cuda.empty_cache()
        for g,value in summaries.items():npz(out/'moments'/f'{g}.npz',H_sums=value,counts=count[g])
        torch.set_num_threads(2)
        for b in blocks:
            data,meta=read_data(out,stage_rows,b)
            bank_path=out/'operators'/f'L{b}_bank'
            if stage=='main' and not bank_path.with_suffix('.npz').exists():
                bank=fit(data,meta);save_bank(bank_path,bank)
            else:
                bank=load_bank(bank_path)
                if (out/'operators/frozen.json').exists():
                    saved=read(out/'operators/frozen.json')
                    for p in (bank_path.with_suffix('.json'),bank_path.with_suffix('.npz')):
                        assert sha(p)==saved['banks'][p.name]
            w=native_weights(key,b);ii=[i for i,m in enumerate(meta) if m['split']!='train'];mm=[meta[i] for i in ii]
            xx=torch.as_tensor(data['x'][ii]);preds={}
            with torch.inference_mode():
                for name in names:preds[name]=apply_operator(name,xx,mm,bank,w).numpy()
                preds['native32_oracle']=((torch.nn.functional.silu(xx@w['g'].T)*(xx@w['u'].T))@w['d'].T).numpy()
            target=data['mlp'][ii];energy=np.maximum(np.mean(target.astype(float)**2,1),1e-20)
            npz(out/'operators'/f'{stage}_L{b}_predictions.npz',**preds,native=target)
            block_reports=[]
            for name,pred in preds.items():
                assert np.isfinite(pred).all()
                loss=np.mean((pred.astype(float)-target)**2,1)/energy
                result={'model':key,'stage':stage,'block':b,'name':name,'anchors':len(ii),'native_width':width,'native_units':int(w['u'].shape[0]),
                    'relative_MSE':float(loss.mean()),'relative_MSE_cluster':clustered(loss,[m['source_group'] for m in mm]),
                    'raw_MSE':float(np.mean(loss*energy))}
                reports.append(result);block_reports.append(result)
            if stage=='main':choices[str(b)]=min([r for r in block_reports if r['name'] in names],key=lambda r:(r['relative_MSE'],r['name']))['name']
            checks.append({'stage':stage,'block':b,'all_predictions_finite':True,'full_coordinate_count':width,'full_unit_count':int(w['u'].shape[0]),'architecture':'GLM combined gate/up split gate first' if key=='glm4' else 'Qwen separate gate/up'})
            del bank,w,xx,preds,data,target;gc.collect()
        if stage=='main':
            frozen_path=out/'operators/frozen.json'
            if not frozen_path.exists():
                immutable(frozen_path,{'timestamp':stamp(),'choices':choices,'confirmation_capture_exists':any((out/'rows'/f'{r["sample_id"]}.json').exists() for r in material if r['split']=='confirmation'),
                    'banks':{p.name:sha(p) for p in (out/'operators').glob('*bank.*')}})
            saved=read(frozen_path)
            assert not saved['confirmation_capture_exists'] and saved['choices']==choices
        else:
            assert choices==read(out/'operators/frozen.json')['choices']
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'model':key,'sources':len(material),'own_blocks':blocks,'native_width':width,'choices':choices,'reports':reports,'checks':checks,
        'tokens':sum(len(read(out/'rows'/f'{r["sample_id"]}.json')['prompt_ids']) for r in material),
        'exact_endpoint_anchors':sum(sum(read(out/'rows'/f'{r["sample_id"]}.json')['exact_char_endpoint']) for r in material),
        'limits':'128training anchors and64validation/64confirmation anchors per block are targeted independent replication, much smaller than4B main study. Parameter count differs between condition groups. Native own widths/depth/tokenizers are retained; no cross-model scalar-coordinate identification. Semantic behavior is evaluated in separate native QA runs.'}
    save(out/'result.json',result);ledger('full_native_operator_scale_'+key,time.monotonic()-start,sources=128);guard()
    print('OPERATOR_SCALE_COMPLETE',key,result['choices'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen14','glm4'],required=True)
    main(p.parse_args().model)
