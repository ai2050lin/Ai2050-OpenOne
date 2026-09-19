"""Matched-source own-coordinate native replication, sequential and nonquantized."""
import argparse,gc
from collections import defaultdict
from rdc_law_common import *
from rdc_law_predict import normalized_features,kernel_torch,ridge_lambda,decode,widths
from phase2729_rdc_law_prediction import training_weights
from phase2730_rdc_law_protocol import protocol
from rdc_operator_model import load,memory
from phase2726_rdc_operator_scale import native_weights
from rdc_operator_qa import QueryTrace,prompt,evaluate,repeated_ngrams


class Trace:
    def __init__(self,model):
        self.model=model;self.depth=len(model.model.layers);self.early=self.depth//3;self.block=self.depth-1
        self.enabled=False;self.handles=[];self.reset([])
        def emb(m,a,o):
            if self.enabled:self.hidden(0,o[0])
        self.handles.append(model.get_input_embeddings().register_forward_hook(emb))
        for b,layer in enumerate(model.model.layers):
            def h(m,a,o,b=b):
                if self.enabled:self.hidden(b+1,(o[0] if isinstance(o,tuple) else o)[0])
            self.handles.append(layer.register_forward_hook(h))
        layer=model.model.layers[self.block]
        def before(m,a):
            if self.enabled:self.residual_input=a[0]
        def attention(m,a,o):
            if self.enabled:self.fields['residual']=bits((self.residual_input+o[0])[0,self.positions])
        self.handles.extend([layer.register_forward_pre_hook(before),layer.self_attn.register_forward_hook(attention)])
        for name,module in [('x',layer.post_attention_layernorm),('mlp',layer.mlp)]:
            def factor(m,a,o,name=name):
                if self.enabled:self.fields[name]=bits(o[0,self.positions])
            self.handles.append(module.register_forward_hook(factor))
        if hasattr(layer.mlp,'gate_proj'):
            for name,module in [('gate',layer.mlp.gate_proj),('up',layer.mlp.up_proj)]:
                def factor(m,a,o,name=name):
                    if self.enabled:self.fields[name]=bits(o[0,self.positions])
                self.handles.append(module.register_forward_hook(factor))
        else:
            def factors(m,a,o):
                if self.enabled:
                    g,u=o.chunk(2,-1);self.fields['gate']=bits(g[0,self.positions]);self.fields['up']=bits(u[0,self.positions])
            self.handles.append(layer.mlp.gate_up_proj.register_forward_hook(factors))
        def product(m,a):
            if self.enabled:self.fields['activation']=bits(a[0][0,self.positions])
        self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(product))

    def hidden(self,b,h):
        import torch
        a=bits(h);self.H[b]=a[self.positions];self.hashes[b]=identity(a)
        v=h.float();rms=v.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-12);u=v/rms
        self.moments[b]=torch.stack([v.sum(0),v.square().sum(0),u.sum(0),u.square().sum(0)]).cpu().numpy()
        if b==self.early:
            self.fields['history']=np.stack([u[:p+1].double().mean(0).cpu().numpy() for p in self.positions]).astype(np.float32)

    def reset(self,positions):
        self.positions=positions;self.H={};self.hashes={};self.moments={};self.fields={};self.residual_input=None

    def close(self):
        for h in self.handles:h.remove()


def pack(out,records,early):
    pieces=defaultdict(list);raw=defaultdict(list);meta=[]
    for r in records:
        sid=r['sample_id'];m=read(out/'rows'/f'{sid}.json')
        with np.load(out/'fields'/f'{sid}.npz') as z:
            h=unbits(z['H']);raw['q'].append(h[early]);raw['embedding'].append(h[0])
            raw['history'].append(z['history']);raw['routed'].append(z['history'])
            raw['position'].extend(z['positions']);raw['class'].extend([0 if r['language']=='en' else 2]*3)
            for k in ('x','gate','up','activation','mlp','postnorm','residual'):pieces[k].append(unbits(z[k]))
        for i,pos in enumerate(m['positions']):
            meta.append({k:r[k] for k in ('sample_id','source_group','cohort','split','language')}|{'position':pos,'anchor':i,
                'target_id':m['prompt_ids'][pos+1],'held_relation_combinations':r.get('held_relation_combinations',[])})
    return meta,{k:np.asarray(v) if k in ('position','class') else np.concatenate(v) for k,v in raw.items()},{k:np.concatenate(v) for k,v in pieces.items()}


def main(key):
    import torch
    p=protocol();out=BASE/'scale'/key
    if (out/'result.json').exists():return
    start=time.monotonic();guard(650*1024**2)
    material={r['sample_id']:r for r in gzread(BASE/'material.json.gz')+gzread(BASE/'confirmation_material.json.gz')}
    records=[material[s] for s in p['scale_ids']];qas=[material[s] for s in p['scale_QA_ids']]
    immutable(out/'protocol.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'common_protocol_sha':sha(BASE/'deployment/protocol.json'),
        'rows':p['scale_ids'],'QA':p['scale_QA_ids'],'model':key,'new_model_test':'Same frozen material, all own coordinates/units. Early=floor(depth/3), finalMLP=depth-1; coordinates not aligned across models.',
        'prediction':'All432 natural anchors.72train24validation48confirmation windows. Five full kernels x2df x4decoders. DF32/128. Validation mean relativeMLP MSE selects, excluding shuffled control. Own confirmation capture follows bank freeze.',
        'storage':'Everytoken/alllayer raw/RMS moments and array identities, all3anchor/alllayer raw fields plus every finalMLPunit. All causal sources participate in mean but fullsource matrices not permanently archived here; recomputation input/config retained.',
        'QA':'24matchedquestions, nativegreedymax32, no gold fed; fullanswer and stop scoring. OriginalQA context retained in material, differentnativechattemplates/positions recorded.',
        'formation':'Q4 controlled native training evidence links thisPhase; no claim that larger checkpoints were retrained.'})
    model,tok=load(key,out/'residency',cpu_gib=11) if key=='qwen14' else load(key,out/'residency')
    torch.set_num_threads(2);device=model.get_input_embeddings().weight.device
    trace=Trace(model);early=trace.early;b=trace.block;D=model.config.hidden_size;units=model.config.intermediate_size
    stats={};capture_times=[];checks=[]
    def capture(row):
        t0=time.monotonic();sid=row['sample_id'];enc=tok(row['text'],add_special_tokens=False,return_offsets_mapping=True)
        if key=='qwen4':positions=row['anchors']
        else:
            positions=[]
            for anchor in row['anchors']:
                end=row['token_offsets'][anchor][1]
                positions.append(max(i for i,(a,z) in enumerate(enc['offset_mapping']) if z>a and z<=end))
        assert len(positions)==3 and max(positions)+1<len(enc['input_ids'])
        trace.reset(positions);trace.enabled=True;ids=torch.tensor([enc['input_ids']],device=device)
        post=model.model(input_ids=ids,use_cache=False).last_hidden_state;trace.enabled=False
        field=dict(trace.fields,H=np.stack([trace.H[i] for i in range(trace.depth+1)]),postnorm=bits(post[0,positions]),positions=np.array(positions))
        if not capture_times:
            repeat=model.model(input_ids=ids,use_cache=False).last_hidden_state
            assert torch.equal(post,repeat);checks.append({'same_shape_no_observer_postnorm_exact':True,'sample_id':sid})
        if key=='qwen4':
            mode='confirmation' if row['split']=='confirmation' else 'main'
            with np.load(BASE/'capture'/mode/'fields'/f'{sid}.npz') as z:
                assert np.array_equal(z['H'],field['H'])
                for k in ('x','gate','up','activation','mlp'):assert np.array_equal(z[f'L35_{k}'],field[k])
        npz(out/'fields'/f'{sid}.npz',**field)
        save(out/'rows'/f'{sid}.json',{'sample_id':sid,'prompt_ids':enc['input_ids'],'positions':positions,'offsets':enc['offset_mapping'],
            'all_token_H_identities':trace.hashes,'exact_char_endpoint':[enc['offset_mapping'][p][1]==row['token_offsets'][a][1] for p,a in zip(positions,row['anchors'])],
            'alignment_limit':'Same character endpoint can cover different partial-token information; Q4 original indices retained.'})
        group=row['split']+'_'+row['cohort'];mom=np.stack([trace.moments[i] for i in range(trace.depth+1)])
        if group not in stats:stats[group]=[0,np.zeros_like(mom,dtype=float)]
        stats[group][0]+=len(enc['input_ids']);stats[group][1]+=mom
        capture_times.append(time.monotonic()-t0);trace.reset([])
        del ids,post,field;gc.collect();torch.cuda.empty_cache()
        assert memory()['host_available_bytes']>2*1024**3
        if len(capture_times)<=2 or len(capture_times)%16==0:
            print('LAW_SCALE_CAPTURE',key,len(capture_times),144,'elapsed',round(time.monotonic()-start,1),flush=True)
        guard();assert time.monotonic()-start<read(BASE/'resources.json')['per_process_ceiling_seconds']
    qaresults=[]
    def qa(row):
        t0=time.monotonic();sid=row['sample_id'];user=prompt(row)
        actual=tok.apply_chat_template([{'role':'user','content':user}],tokenize=False,add_generation_prompt=True,**({'enable_thinking':False} if key.startswith('qwen') else {}))
        ids=tok(actual,add_special_tokens=False,return_tensors='pt')['input_ids'].to(device)
        qt=QueryTrace(model,[b]);qt.enabled=True
        post=model.model(input_ids=ids,use_cache=False).last_hidden_state;qt.enabled=False
        npz(out/'qa/fields'/f'{sid}.npz',H=np.stack([qt.layers[i] for i in range(trace.depth+1)]),postnorm=bits(post[0,-1]),**qt.data);qt.close()
        eos=model.generation_config.eos_token_id or tok.eos_token_id;stop=set(eos if isinstance(eos,list) else [eos])
        generated=model.generate(input_ids=ids,do_sample=False,max_new_tokens=32,use_cache=True,eos_token_id=eos,pad_token_id=tok.pad_token_id or tok.eos_token_id)[0,ids.shape[1]:].tolist()
        text=tok.decode(generated,skip_special_tokens=True)
        result={k:row[k] for k in ('sample_id','source_group','cohort','language','question','answers','question_type')}
        result.update(actual_prompt=actual,prompt_ids=ids[0].tolist(),generated_ids=generated,generated_text=text,stopped_by_native_EOS=any(i in stop for i in generated),
            repeated_4gram_fraction=repeated_ngrams(generated),seconds=time.monotonic()-t0,**evaluate(text,row['answers'],row['language']))
        save(out/'qa/commits'/f'{sid}.json',result);qaresults.append(result);del ids,post,qt;gc.collect();torch.cuda.empty_cache()
        print('LAW_SCALE_QA',key,len(qaresults),24,'elapsed',round(time.monotonic()-start,1),flush=True)
        assert time.monotonic()-start<read(BASE/'resources.json')['per_process_ceiling_seconds']
    with torch.inference_mode():
      try:
        trainval=[r for r in records if r['split']!='confirmation'];fresh=[r for r in records if r['split']=='confirmation']
        for row in trainval[:2]:capture(row)
        qa(qas[0])
        estimate=float(np.mean(capture_times)*142+qaresults[0]['seconds']*23+300)
        save(out/'pilot.json',{'timestamp':stamp(),'natural_pilots':2,'QA_pilots':1,'actual_natural_seconds':capture_times,
            'actual_QA_seconds':qaresults[0]['seconds'],'estimated_remaining_seconds':estimate,'elapsed':time.monotonic()-start,
            'method':'Average two natural source captures x142 plus one actual QA x23 plus300seconds analysis reserve. Timings contain original load/offloadconditions; subsequent monitoring enforces processcap.',
            'memory':memory(),'passed_cost_preflight':estimate+time.monotonic()-start<read(BASE/'resources.json')['per_process_ceiling_seconds']})
        assert read(out/'pilot.json')['passed_cost_preflight'],('Model pilot exceeds finite envelope',key,estimate)
        for row in trainval[2:]:capture(row)
        meta,raw,data=pack(out,trainval,early);train=np.array([i for i,r in enumerate(meta) if r['split']=='train']);val=np.array([i for i,r in enumerate(meta) if r['split']=='validation'])
        weights=training_weights(meta,train);f,rulers=normalized_features(raw,train,weights)
        rng=np.random.default_rng(2730);donors=np.empty(len(meta),int)
        for cls in (0,2):
            ii=train[np.array([f['class'][i]==cls for i in train])];donors[ii]=rng.permutation(ii)
            for i in val[f['class'][val]==cls]:donors[i]=rng.choice(ii)
        f['shuffled_history_0']=f['history'][donors]
        ft={k:torch.as_tensor(v,dtype=torch.int64 if k=='class' else torch.float64) for k,v in f.items()};tt={k:v[train] for k,v in ft.items()}
        yy=np.concatenate([data['x'],torch.nn.functional.silu(torch.as_tensor(data['gate'])).numpy(),data['up'],data['activation'],data['mlp']],1)
        y=torch.as_tensor(yy);sw=torch.as_tensor(weights,dtype=torch.float64);root=sw.sqrt();mean=(y[train].double()*sw[:,None]).sum(0)/sw.sum()
        centered=(y[train].double()-mean)*root[:,None];w=native_weights(key,b);slices=widths(D,units)
        decoder_names=('direct_mlp','predicted_x_native','product_of_predicted_factors','predicted_joint_product')
        keys={'direct_mlp':['mlp'],'predicted_x_native':['x'],'product_of_predicted_factors':['phi','u'],'predicted_joint_product':['activation']}
        best={k:(float('inf'),None) for k in decoder_names};grid=[]
        for kernelname in ('early_linear','additive_history','multiplicative_history','task_conditioned','shuffled_history_0'):
            k=kernel_torch(kernelname,tt,tt);e,V=torch.linalg.eigh(k*root[:,None]*root[None,:]);projected=V.T@centered;cross=kernel_torch(kernelname,ft,tt)
            for df in (32,128):
                lam,actualdf=ridge_lambda(e,df);coef=(V@(projected/(e.clamp_min(0)[:,None]+lam)))*root[:,None]
                pred=cross.float()@coef.float()+mean.float()
                for decoder in decoder_names:
                    m=decode(pred,decoder,w,slices).numpy();rel=np.mean((m-data['mlp'])**2,1)/np.maximum(np.mean(data['mlp']**2,1),1e-20)
                    score=float(np.mean([rel[[i for i in val if meta[i]['cohort']==c]].mean() for c in ('gum','ewt','cmrc')]))
                    info={'kernel':kernelname,'decoder':decoder,'df':df,'effective_df':actualdf,'lambda':lam,'validation_MSE':score}
                    grid.append(info)
                    if not kernelname.startswith('shuffled') and score<best[decoder][0]:
                        ii=np.concatenate([np.arange(slices[k].start,slices[k].stop) for k in keys[decoder]])
                        arrays={'coefficients':coef[:,ii].float().numpy(),'target_center':mean[ii].float().numpy(),'indices':ii}
                        npz(out/'banks'/f'{decoder}.npz',**arrays);save(out/'banks'/f'{decoder}.json',info);best[decoder]=(score,info)
            print('LAW_SCALE_FIT',key,kernelname,flush=True)
        immutable(out/'frozen.json',{'timestamp':stamp(),'choices':{k:v[1] for k,v in best.items()},'grid':grid,
            'own_confirmation_captured':False,'training_sources':72,'validation_sources':24,'source':snapshot(Path(__file__))})
        npz(out/'feature_rulers.npz',**{key+'_'+k:np.asarray(v) for key,d in rulers.items() for k,v in d.items()})
        npz(out/'training_features.npz',**{k:v.numpy() for k,v in tt.items()})
        del y,yy,mean,centered,coef,pred,k,V,projected;gc.collect()
        for row in fresh:capture(row)
        fm,fr,fd=pack(out,fresh,early);ff,_=normalized_features(fr,rulers=rulers)
        fft={k:torch.as_tensor(v,dtype=torch.int64 if k=='class' else torch.float64) for k,v in ff.items()}
        comparisons=[]
        for decoder in decoder_names:
            info=best[decoder][1]
            with np.load(out/'banks'/f'{decoder}.npz') as z:
                pp=kernel_torch(info['kernel'],fft,tt).float()@torch.as_tensor(z['coefficients'])+torch.as_tensor(z['target_center'])
                indexes=z['indices']
            pred=torch.zeros((len(fm),sum((D,D,units,units,units))),dtype=torch.float32);pred[:,indexes]=pp
            m=decode(pred,decoder,w,slices).numpy();rel=np.mean((m-fd['mlp'])**2,1)/np.maximum(np.mean(fd['mlp']**2,1),1e-20)
            npz(out/'confirmation_predictions'/f'{decoder}.npz',mlp=m,relative_MSE=rel)
            for group in ('all','gum','ewt','cmrc','held_relation_pair'):
                ix=np.array([i for i,r in enumerate(fm) if group=='all' or r['cohort']==group or (group=='held_relation_pair' and bool(r['held_relation_combinations']))])
                comparisons.append(dict(info,group=group,queries=len(ix),relative_MSE=float(rel[ix].mean()),source_cluster=clustered(rel[ix],[fm[i]['source_group'] for i in ix])))
        # Complete native per-unit phi/up covariance remains a descriptive cross-model observation.
        unitreports=[]
        for cohort in ('gum','ewt','cmrc'):
            ix=np.array([i for i,r in enumerate(meta) if r['split']=='train' and r['cohort']==cohort])
            phi=torch.nn.functional.silu(torch.as_tensor(data['gate'][ix])).numpy().astype(float);u=data['up'][ix].astype(float)
            cov=(phi*u).mean(0)-phi.mean(0)*u.mean(0)
            npz(out/'all_unit_relations'/f'{cohort}.npz',phi_mean=phi.mean(0),up_mean=u.mean(0),product_mean=(phi*u).mean(0),phi_up_covariance=cov)
            unitreports.append({'cohort':cohort,'all_units':len(cov),'covariance_L2':float(np.linalg.norm(cov)),'covariance_mean_absolute':float(np.mean(abs(cov)))})
        del w,data,fd,raw,fr,ft,tt,fft;gc.collect()
        for row in qas[1:]:qa(row)
      finally:
        trace.close();del trace,model;gc.collect();torch.cuda.empty_cache()
    for group,(n,mom) in stats.items():npz(out/'moments'/f'{group}.npz',token_count=np.array(n),sums=mom)
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'model':key,'natural_windows':144,'QA_questions':24,'natural_anchors':432,
        'native_width':D,'native_units':units,'early_boundary':early,'last_MLP_block':b,'checks':checks,'frozen_choices':{k:v[1] for k,v in best.items()},
        'confirmation':comparisons,'unit_relations':unitreports,
        'QA_summary':{c:{'questions':sum(r['cohort']==c for r in qaresults),'EM':float(np.mean([r['normalized_full_EM'] for r in qaresults if r['cohort']==c])),
            'F1':float(np.mean([r['answer_F1'] for r in qaresults if r['cohort']==c])),'EOS':float(np.mean([r['stopped_by_native_EOS'] for r in qaresults if r['cohort']==c]))} for c in ('squad_qa','cmrc_qa','hotpot_qa')},
        'seconds':time.monotonic()-start,'scope':'Native width/units and natural-source pilot-controlled replication. No cross-model coordinate isomorphism, no model-size causal effect, no larger-model training-formation experiment.'}
    save(out/'result.json',result);ledger('native_scale_'+key,result['seconds'],sources=168)
    print('LAW_SCALE_COMPLETE',key,result['QA_summary'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['qwen4','qwen14','glm4'],required=True);a=ap.parse_args();main(a.model)
