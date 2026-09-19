"""Matched128-source4B re-analysis, capturing only the missing proportional middle block."""
from collections import defaultdict
from rdc_operator_common import *
from phase2726_rdc_operator_scale import selected,fit,native_weights
from phase2726_rdc_operator_compile import causal_meta
from rdc_native_conditional_operator import apply_operator,save_bank,load_bank


def run(model,tok):
    import torch
    out=BASE/'scale/qwen4'
    if (out/'result.json').exists():return read(out/'result.json')
    start=time.monotonic();material=selected();device=model.get_input_embeddings().weight.device;positions=[];data={};handles=[]
    if not (out/'protocol.json').exists():immutable(out/'protocol.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'sources':128,'source_ids':[r['sample_id'] for r in material],
        'split':'64train/32validation/32confirmation,2anchors each, identical source IDs and fit counts to larger-model replication.',
        'blocks':[6,18,34],'existing_reuse':'Reuse original complete fields and factors at6/34. Only missing native middle block18 factors are newly captured; all-layer two-anchor identity verifies the replay.',
        'scope':'Matched-subset re-analysis of4B already observed sources, not an untouched new semantic holdout. Block18 follows proportional layer rule and differs from primary4B block16.',
        'selection':'Same six candidates and train shrinkage as larger-model fit;64validation anchors select before reading64confirmation target factors.'})
    for i,layer in enumerate(model.model.layers):
        def h(m,a,o,i=i):
            o=o[0] if isinstance(o,tuple) else o;data[f'H{i+1}']=bits(o[0,positions])
        handles.append(layer.register_forward_hook(h))
    def emb(m,a,o):data['H0']=bits(o[0,positions])
    handles.append(model.get_input_embeddings().register_forward_hook(emb))
    layer=model.model.layers[18]
    for key,module in [('x',layer.post_attention_layernorm),('gate',layer.mlp.gate_proj),('up',layer.mlp.up_proj),('mlp',layer.mlp)]:
        def capture(m,a,o,key=key):data['L18_'+key]=bits(o[0,positions])
        handles.append(module.register_forward_hook(capture))
    def act(m,a):data['L18_activation']=bits(a[0][0,positions])
    handles.append(layer.mlp.down_proj.register_forward_pre_hook(act))
    metadata={};checks=[]
    try:
        for i,r in enumerate(material):
            positions=r['anchors'];data={};scope='confirmation' if r['split']=='confirmation' else 'main'
            model.model(input_ids=torch.tensor([r['prompt_ids']],device=device),use_cache=False)
            with np.load(BASE/'capture'/scope/'fields'/f'{r["sample_id"]}.npz') as z:assert np.array_equal(z['H'],np.stack([data[f'H{l}'] for l in range(37)]))
            from rdc_operator_capture import persist_arrays
            persist_arrays(out/'fields'/f'{r["sample_id"]}.npz',{k:v for k,v in data.items() if k.startswith('L18')})
            mm=causal_meta(r['prompt_ids'],r['language'],tok)
            metadata[r['sample_id']]=[{**mm[p],'sample_id':r['sample_id'],'source_group':r['source_group'],'split':r['split'],'anchor':j} for j,p in enumerate(positions)]
            save(out/'rows'/f'{r["sample_id"]}.json',{'sample_id':r['sample_id'],'anchors':positions,'anchor_meta':metadata[r['sample_id']],
                'primary_source_field':f'capture/{scope}/fields/{r["sample_id"]}.npz','all37layer_two_anchor_bit_identity':True})
            checks.append({'sample_id':r['sample_id'],'all37layer_two_anchor_bit_identity':True})
            if i<2 or (i+1)%32==0:print('Q4_MATCHED_MIDDLE',i+1,128,flush=True)
    finally:
        for h in handles:h.remove()
    names=['constant_global','frozen_gate_global','frozen_gate_piece','frozen_gate_cue','tangent_global','quadratic_global'];reports=[];choices={}
    for stage in ('main','confirmation'):
        records=[r for r in material if (r['split']=='confirmation')==(stage=='confirmation')]
        meta=[m for r in records for m in metadata[r['sample_id']]]
        for b in (6,18,34):
            pieces=defaultdict(list)
            for r in records:
                scope='confirmation' if r['split']=='confirmation' else 'main'
                path=(out/'fields'/f'{r["sample_id"]}.npz') if b==18 else BASE/'capture'/scope/'factors'/f'{r["sample_id"]}.npz'
                with np.load(path) as z:
                    for k in ('x','gate','up','mlp'):pieces[k].append(unbits(z[f'L{b}_{k}']))
            d={k:np.concatenate(v).astype(np.float32) for k,v in pieces.items()}
            bank=fit(d,meta) if stage=='main' else load_bank(out/'operators'/f'L{b}_bank')
            if stage=='main':save_bank(out/'operators'/f'L{b}_bank',bank)
            ii=[i for i,m in enumerate(meta) if m['split']!='train'];mm=[meta[i] for i in ii]
            x=torch.as_tensor(d['x'][ii]);w=native_weights('qwen4',b);preds={}
            for name in names:preds[name]=apply_operator(name,x,mm,bank,w).numpy()
            preds['native32_oracle']=((torch.nn.functional.silu(x@w['g'].T)*(x@w['u'].T))@w['d'].T).numpy()
            target=d['mlp'][ii];energy=np.maximum(np.mean(target.astype(float)**2,1),1e-20)
            npz(out/'operators'/f'{stage}_L{b}_predictions.npz',**preds,native=target)
            br=[]
            for name,pred in preds.items():
                loss=np.mean((pred.astype(float)-target)**2,1)/energy;assert np.isfinite(pred).all()
                r={'model':'qwen4','stage':stage,'block':b,'name':name,'anchors':len(ii),'native_width':2560,'native_units':9728,
                    'relative_MSE':float(loss.mean()),'raw_MSE':float(np.mean(loss*energy)),'relative_MSE_cluster':clustered(loss,[m['source_group'] for m in mm])}
                reports.append(r);br.append(r)
            if stage=='main':choices[str(b)]=min([r for r in br if r['name'] in names],key=lambda r:(r['relative_MSE'],r['name']))['name']
            del w,x,bank,preds,d,target,pieces
        if stage=='main':save(out/'operators/frozen.json',{'timestamp':stamp(),'choices':choices,'scope':'Matched re-analysis; freeze before reading confirmation targets in this fit, but their earlier baseline observations are not erased.'})
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'model':'qwen4','sources':128,'tokens':sum(len(r['prompt_ids']) for r in material),
        'native_width':2560,'own_blocks':[6,18,34],'choices':choices,'reports':reports,'checks':checks,'seconds':time.monotonic()-start,
        'limits':'Same128source/128train-anchor sample count as larger models; coordinate systems and tokenizations still differ. Additional block18 differs from main operator block16. Re-analysis, not a new untouched4B generalization cohort.'}
    save(out/'result.json',result);guard();return result
