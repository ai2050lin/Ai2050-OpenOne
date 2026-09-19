"""All-coordinate natural forward atlas and exact frozen-prefix training interface.

Every position/layer is observed. Full H12 source matrices, all-layer anchor fields,
final-MLP inputs and residuals are retained. No gold relationship becomes input.
"""
import argparse
import gc
import sys
from collections import defaultdict
from rdc_law_common import *

BLOCKS = (6,16,35)


def persist(path, packet):
    if path.exists():
        with np.load(path) as z:
            assert set(z.files)==set(packet)
            for k,a in packet.items():
                assert np.array_equal(a,z[k],equal_nan=True),('Native repeat differs',path,k)
    else:
        npz(path,**packet)


class Observer:
    def __init__(self,model):
        self.model=model
        self.handles=[]
        self.enabled=False
        self.reset([],False)
        def emb(m,a,o):
            if self.enabled:self.hidden(0,o[0])
        self.handles.append(model.get_input_embeddings().register_forward_hook(emb))
        for b,layer in enumerate(model.model.layers):
            def before(m,a,b=b):
                if self.enabled:self.active[b]={'input':a[0]}
            self.handles.append(layer.register_forward_pre_hook(before))
            def attention(m,a,o,b=b):
                if self.enabled:
                    self.active[b]['attention']=o[0]
                    if b in (*BLOCKS,11):
                        assert o[1] is not None,'Native eager attention weights required'
                        self.attention[b]=bits(o[1][0,:,self.positions])
            self.handles.append(layer.self_attn.register_forward_hook(attention))
            def mlp(m,a,o,b=b):
                if self.enabled:self.active[b]['mlp']=o
            self.handles.append(layer.mlp.register_forward_hook(mlp))
            if b in BLOCKS:
                for key,module in [('x',layer.post_attention_layernorm),('gate',layer.mlp.gate_proj),('up',layer.mlp.up_proj)]:
                    def factor(m,a,o,b=b,key=key):
                        if self.enabled:self.active[b][key]=o
                    self.handles.append(module.register_forward_hook(factor))
                def activation(m,a,b=b):
                    if self.enabled:self.active[b]['activation']=a[0]
                self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(activation))
            def after(m,a,o,b=b):
                if not self.enabled:return
                import torch
                y=o[0] if isinstance(o,tuple) else o
                self.hidden(b+1,y[0])
                d=self.active.pop(b)
                r,at,ml=(d[k][0].float() for k in ('input','attention','mlp'))
                self.energy[b]=torch.stack([r.square().mean(-1),at.square().mean(-1),ml.square().mean(-1),
                    2*(r*at).mean(-1),2*(r*ml).mean(-1),2*(at*ml).mean(-1),y[0].float().square().mean(-1),
                    (y[0].float()-r-at-ml).square().mean(-1)]).cpu().numpy()
                if b in BLOCKS:
                    d['output']=y
                    for key in ('input','attention','x','gate','up','activation','mlp','output'):
                        self.factors[f'L{b}_{key}']=bits(d[key][0,self.positions])
                    g,u,act=(d[k][0].float() for k in ('gate','up','activation'))
                    phi=torch.nn.functional.silu(d['gate'])[0].float()
                    self.unit_moments[b]=torch.stack([g.sum(0),u.sum(0),phi.sum(0),act.sum(0),
                        g.square().sum(0),u.square().sum(0),phi.square().sum(0),act.square().sum(0),(phi*u).sum(0)]).cpu().numpy()
                    if b==35:
                        # Final MLP is pointwise. Changing it does not change upstream attention K/V.
                        self.training={'x':bits(d['x'][0]),'residual':bits((d['input']+d['attention'])[0]),'native_output':bits(y[0])}
                    if self.check:
                        residual=d['input']+d['attention']
                        self.enabled=False
                        try:
                            assert torch.equal(layer_ref(b).post_attention_layernorm(residual),d['x'])
                            assert torch.equal(torch.nn.functional.silu(d['gate'])*d['up'],d['activation'])
                            assert torch.equal(layer_ref(b).mlp.down_proj(d['activation']),d['mlp'])
                            assert torch.equal(residual+d['mlp'],y)
                        finally:
                            self.enabled=True
                        self.checks[f'L{b}_all_coordinate_native_norm_product_writeback_residual']=True
            def layer_ref(index):return self.model.model.layers[index]
            self.handles.append(layer.register_forward_hook(after))

    def reset(self,positions,full,check=False):
        self.positions=positions;self.full=full;self.check=check
        self.H={};self.full_H={};self.active={};self.factors={};self.training={}
        self.attention={};self.energy={};self.moments={};self.unit_moments={};self.H_hashes={};self.checks={}
        self.source_H12=None

    def hidden(self,index,value):
        import torch
        a=bits(value)
        self.H[index]=a[self.positions]
        self.H_hashes[index]=identity(a)
        if self.full:self.full_H[index]=a
        if index==12:self.source_H12=a
        f=value.float();rms=f.square().mean(-1).sqrt().clamp_min(1e-12)
        unit=f/rms[:,None]
        self.moments[index]=torch.stack([f.sum(0),f.square().sum(0),unit.sum(0),unit.square().sum(0)]).cpu().numpy()

    def close(self):
        for h in self.handles:h.remove()


def main(mode):
    import torch
    import psutil
    from phase2662_symmetric_mapping_contract import load_native
    source=snapshot(Path(__file__))
    out=BASE/'capture'/mode
    if (out/'result.json').exists():
        print('LAW_CAPTURE_ALREADY_COMPLETE',mode,flush=True);return
    material=gzread(BASE/('confirmation_material.json.gz' if mode=='confirmation' else 'material.json.gz'))
    if mode=='pilot':
        selection=[r for cohort in sorted({r['cohort'] for r in material}) for r in [s for s in material if s['cohort']==cohort and s['split']=='train'][:2]]
    elif mode=='main':
        assert read(BASE/'capture/pilot/result.json')['pilot_passed'];selection=material
    else:
        assert (BASE/'prediction/frozen.json').exists(),'No confirmation outputs before candidate freeze'
        selection=material
    assert len({r['sample_id'] for r in selection})==len(selection)
    fixtures=set(read(BASE/'material_audit.json')['representative_full_fields'])
    start=time.monotonic();guard(256*1024**2)
    model,tok=load_native('qwen4');model.eval();torch.set_num_threads(2)
    assert len(model.model.layers)==36 and model.config.hidden_size==2560 and model.dtype==torch.bfloat16
    assert not getattr(model,'is_quantized',False)
    observer=Observer(model);device=model.get_input_embeddings().weight.device
    runtime={'timestamp':stamp(),'source':source,'model':'qwen3-4b','quantized':False,'dtype':str(model.dtype),
        'blocks':BLOCKS,'torch':torch.__version__,'model_code_sha':sha(Path(sys.modules[model.model.__class__.__module__].__file__)),
        'execution':'Unpadded batch1 native eager CUDA BF16, use_cache=False. Exact frozen text/input IDs; natural raw text, QA explicit no-think user chat template.',
        'full_coverage':'Every token,embedding+36residual boundaries,all2560coordinates;every token/all9728units at3blocks.',
        'storage':'All-layer complete anchors, every H12 source vector, all final-MLP inputs/residuals/native outputs and full source attention at anchors. Predeclared representative all-layer/all-token fixtures.',
        'prediction_boundary':'Only H12 and its causal sources may enter early predictors; final x/g/u/m are targets/audit. Gold graph/answer not online feature.',
        'training_interface':'Full fixed-prefix native last-MLP inputs/residuals. Actual teacher-forced next corpus tokens are TRAINING labels only, never future predictor features.'}
    save(out/'runtime.json',runtime)
    records=[];hstats={};ustats={};counts=CounterLocal()
    try:
      with torch.inference_mode():
        for index,row in enumerate(selection):
            tick=time.monotonic();sid=row['sample_id'];ids_list=row['prompt_ids'];n=len(ids_list);positions=row['anchors']
            assert tok(row['text'],add_special_tokens=False)['input_ids']==ids_list
            observer.reset(positions,mode=='pilot' or sid in fixtures,index<2)
            observer.enabled=True
            ids=torch.tensor([ids_list],device=device)
            post=model.model(input_ids=ids,use_cache=False).last_hidden_state
            observer.enabled=False
            assert len(observer.H)==37 and len(observer.energy)==36
            H=np.stack([observer.H[k] for k in range(37)])
            h12=unbits(observer.source_H12)
            history=[];weighted=[]
            ap=unbits(observer.attention[11]).mean(0)
            assert ap.shape==(len(positions),n)
            for ai,p in enumerate(positions):
                assert not np.any(ap[ai,p+1:]),('Future source attention',sid,p)
                source_vectors=h12[:p+1]
                source_vectors=source_vectors/np.maximum(np.sqrt(np.mean(source_vectors**2,axis=-1,keepdims=True)),1e-12)
                history.append(source_vectors.mean(0))
                weights_=ap[ai,:p+1].astype(float)
                weights_/=weights_.sum()
                weighted.append(weights_@source_vectors)
            fields={'H':H,'postnorm':bits(post[0,positions]),'positions':np.array(positions),
                'history_mean_H12_RMS':np.stack(history).astype(np.float32),'attention_weighted_H12_RMS':np.stack(weighted).astype(np.float32)}
            fields.update(observer.factors)
            for b,a in observer.attention.items():fields[f'L{b}_attention_sources']=a
            fullsource={'H12_sources':observer.source_H12,'token_ids':np.array(ids_list,dtype=np.int32),**observer.training}
            outputstats=defaultdict(list)
            for at in range(0,n,16):
                logits=model.lm_head(post[0,at:at+16]).float()
                lp=logits.log_softmax(-1);valid=min(len(logits),n-at-1)
                nll=-lp[torch.arange(valid,device=device),ids[0,at+1:at+1+valid]]
                outputstats['next_nll'].append(np.r_[nll.cpu().numpy(),[np.nan] if at+16>=n else []])
                outputstats['argmax'].append(logits.argmax(-1).cpu().numpy())
                outputstats['entropy'].append((-(lp.exp()*lp).sum(-1)).cpu().numpy())
                del logits,lp,nll
            energy={'block_terms':np.stack([observer.energy[b] for b in range(36)]),
                **{k:np.concatenate(v) for k,v in outputstats.items()}}
            for a in (fields['H'],fullsource['H12_sources'],fullsource['x'],fullsource['residual']):assert np.isfinite(unbits(a)).all()
            checks=dict(observer.checks)
            if index==0:
                plain=model.model(input_ids=ids,use_cache=False).last_hidden_state
                assert torch.equal(plain,post),'Observer changes forward'
                checks['observer_noop_same_shape_bitwise']=True
                p=positions[0]
                if p+1<n:
                    changed=ids.clone();changed[:,p+1:]=tok.eos_token_id
                    alt=model.model(input_ids=changed,use_cache=False).last_hidden_state
                    assert torch.equal(alt[:,:p+1],post[:,:p+1])
                    checks['same_shape_future_suffix_causal_invariance']=True
                    del alt,changed
                emb=bits(model.get_input_embeddings().weight[ids[0,positions]])
                assert np.array_equal(emb,H[0]);checks['embedding_row_identity']=True
                del plain,emb
            persist(out/'fields'/f'{sid}.npz',fields)
            persist(out/'sources'/f'{sid}.npz',fullsource)
            persist(out/'energies'/f'{sid}.npz',energy)
            if observer.full:
                persist(out/'full_fields'/f'{sid}.npz',{'H':np.stack([observer.full_H[k] for k in range(37)]),'postnorm':bits(post[0])})
            key=row['split']+'_'+row['cohort']
            hs=np.stack([observer.moments[k] for k in range(37)]).astype(float)
            us=np.stack([observer.unit_moments[b] for b in BLOCKS]).astype(float)
            if key not in hstats:hstats[key]=np.zeros_like(hs);ustats[key]=np.zeros_like(us)
            hstats[key]+=hs;ustats[key]+=us;counts[key]+=n
            record={'sample_id':sid,'source_group':row['source_group'],'tokens':n,'anchors':positions,'checks':checks,
                'full_layer_all_token_identities':observer.H_hashes,'field_sha':sha(out/'fields'/f'{sid}.npz'),
                'source_sha':sha(out/'sources'/f'{sid}.npz'),'energy_sha':sha(out/'energies'/f'{sid}.npz'),
                'full_field':observer.full,'mean_native_next_nll':float(np.mean(energy['next_nll'][:-1])),
                'native_next_argmax_match':float(np.mean(energy['argmax'][:-1]==np.array(ids_list[1:]))),
                'seconds':time.monotonic()-tick}
            cp=out/'commits'/f'{sid}.json'
            if cp.exists():
                old=read(cp)
                assert all(old[k]==record[k] for k in ('field_sha','source_sha','energy_sha'))
            else:save(cp,record)
            # Main must replay pilot exactly, including full source matrices, not only summary statistics.
            pilot=BASE/'capture/pilot/commits'/f'{sid}.json'
            if mode=='main' and pilot.exists():
                old=read(pilot)
                assert all(old[k]==record[k] for k in ('field_sha','source_sha','energy_sha')),('Pilot/main divergence',sid)
            records.append(record)
            del post,ids,H,h12,fields,fullsource,energy,hs,us
            observer.reset([],False)
            guard(256*1024**2)
            assert psutil.virtual_memory().available>2*1024**3
            assert time.monotonic()-start<read(BASE/'resources.json')['per_process_ceiling_seconds']
            if index<2 or (index+1)%16==0:
                print('LAW_CAPTURE',mode,index+1,len(selection),'elapsed',round(time.monotonic()-start,1),'bytes',usage(),flush=True)
        for key in hstats:
            npz(out/'moments'/f'{key}.npz',H_sums=hstats[key],unit_sums=ustats[key],tokens=np.array(counts[key]))
    except Exception as exc:
        import traceback
        save(out/('failure_'+str(int(time.time()))+'.json'),{'timestamp':stamp(),'source':source,'type':type(exc).__name__,
            'message':str(exc),'traceback':traceback.format_exc(),'seconds':time.monotonic()-start,'completed_rows':len(records)})
        ledger('failed_native_qwen4_capture_'+mode,time.monotonic()-start,completed_rows=len(records))
        raise
    finally:
        observer.close();del observer,model;gc.collect();torch.cuda.empty_cache()
    seconds=time.monotonic()-start
    result={'timestamp':stamp(),'source':source,'mode':mode,'rows':len(records),'tokens':sum(r['tokens'] for r in records),
        'anchors':sum(len(r['anchors']) for r in records),'seconds':seconds,'all_coordinate_full_token_coverage':True,
        'moments':{'H':['sum','sum_square','sum_RMS','sum_RMS_square'],'units':['g','u','phi','activation','g2','u2','phi2','activation2','phi_times_u']},
        'undefined_values':'Exactly final token next_nll per row has no given next corpus token and is NaN. It is not a missing internal state.',
        'checks':[r['checks'] for r in records if r['checks']], 'raw_fixture_ids':[r['sample_id'] for r in records if r['full_field']]}
    if mode=='pilot':
        estimated=sum(r['seconds'] for r in records)/sum(r['tokens'] for r in records)*sum(len(r['prompt_ids']) for r in material)
        # Additional allocation includes training parameter snapshots and future prediction/behavior work.
        by_id={r['sample_id']:r for r in selection}
        estimated_bytes=3*1024**3
        for kind in ('natural','QA'):
            rr=[r for r in records if by_id[r['sample_id']]['kind']==kind]
            mean_bytes=np.mean([sum((out/folder/f'{r["sample_id"]}.npz').stat().st_size for folder in ('fields','sources','energies')) for r in rr])
            estimated_bytes+=mean_bytes*sum(r['kind']==kind for r in material)
        estimated_bytes+=sum(len(r['prompt_ids'])*38*2560*2 for r in material if r['sample_id'] in fixtures)
        result.update(pilot_estimated_main_seconds=float(estimated),pilot_conservative_estimated_bytes=float(estimated_bytes),
            pilot_passed=bool(estimated<5400 and estimated_bytes<read(BASE/'resources.json')['result_ceiling_bytes']))
    save(out/'result.json',result);ledger('native_qwen4_capture_'+mode,seconds,rows=len(records),tokens=result['tokens'])
    print('LAW_CAPTURE_COMPLETE',result,flush=True)


class CounterLocal(defaultdict):
    def __init__(self):super().__init__(int)


def recover_pilot():
    """Recover only a serialization failure after all native per-row commits exist."""
    out=BASE/'capture/pilot'
    assert not (out/'result.json').exists()
    material=gzread(BASE/'material.json.gz')
    selection=[r for cohort in sorted({r['cohort'] for r in material}) for r in [s for s in material if s['cohort']==cohort and s['split']=='train'][:2]]
    records=[]
    for row in selection:
        r=read(out/'commits'/f'{row["sample_id"]}.json')
        for folder,key in [('fields','field_sha'),('sources','source_sha'),('energies','energy_sha')]:
            assert sha(out/folder/f'{row["sample_id"]}.npz')==r[key]
        assert (out/'full_fields'/f'{row["sample_id"]}.npz').exists()
        records.append(r)
    assert len(records)==12 and records[0]['checks']['observer_noop_same_shape_bitwise']
    fixtures=set(read(BASE/'material_audit.json')['representative_full_fields'])
    by_id={r['sample_id']:r for r in selection}
    estimated_bytes=3*1024**3
    for kind in ('natural','QA'):
        rr=[r for r in records if by_id[r['sample_id']]['kind']==kind]
        mean_bytes=np.mean([sum((out/f/f'{r["sample_id"]}.npz').stat().st_size for f in ('fields','sources','energies')) for r in rr])
        estimated_bytes+=mean_bytes*sum(r['kind']==kind for r in material)
    estimated_bytes+=sum(len(r['prompt_ids'])*38*2560*2 for r in material if r['sample_id'] in fixtures)
    recorded=sum(r['seconds'] for r in records)
    estimated=recorded/sum(r['tokens'] for r in records)*sum(len(r['prompt_ids']) for r in material)
    result={'timestamp':stamp(),'source':read(out/'runtime.json')['source'],'recovery_source':snapshot(Path(__file__)),
        'mode':'pilot','rows':len(records),'tokens':sum(r['tokens'] for r in records),'anchors':sum(len(r['anchors']) for r in records),
        'seconds':recorded+30,'timing_scope':'Sum actual per-row timers plus30second conservative allowance for loading/final bookkeeping; complete failed process timer was not persisted.',
        'recovery':'All12native captures completed; only final numpy.bool JSON serialization failed. All committed field/source/energy SHA rechecked; no model output fabricated or recomputed by recovery.',
        'all_coordinate_full_token_coverage':True,'checks':[r['checks'] for r in records if r['checks']],
        'raw_fixture_ids':[r['sample_id'] for r in records if r['full_field']],
        'pilot_estimated_main_seconds':float(estimated),'pilot_conservative_estimated_bytes':float(estimated_bytes),
        'pilot_passed':bool(estimated<5400 and estimated_bytes<read(BASE/'resources.json')['result_ceiling_bytes'])}
    save(out/'serialization_failure_recovery.json',result)
    save(out/'result.json',result)
    ledger('native_qwen4_capture_pilot_recovered',recorded+30,actual_row_seconds=recorded,overhead_allowance_seconds=30)
    print('LAW_PILOT_RECOVERED',result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['pilot','main','confirmation'],default='pilot');parser.add_argument('--recover-pilot',action='store_true')
    args=parser.parse_args()
    recover_pilot() if args.recover_pilot else main(args.mode)
