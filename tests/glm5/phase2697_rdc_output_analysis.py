"""Conditional final-block native source/unit accounting and frozen earlier-layer prediction."""
import argparse
from rdc_mechanism_common import *
from rdc_feature_extractors import fit_predict,metrics
OUT=CAMPAIGN/'c_generation'

def protocol():
    if (OUT/'analysis_protocol.json').exists():
        old=read(OUT/'analysis_protocol.json')
        if old['source_sha']!=sha(Path(__file__)):
            immutable(OUT/'analysis_implementation_v2.json',{'old_source_sha':old['source_sha'],'new_source_sha':sha(Path(__file__)),
                'reason':'Actual checkpoint ties lm_head to embed_tokens; resolve real shared tensor via config instead of assuming standalone lm_head key.',
                'incident':'Checkpoint key error before any output ledger calculation; capture untouched.'})
        return
    immutable(OUT/'analysis_protocol.json',{'source_sha':sha(Path(__file__)),'native_account_layer':35,
        'target':'actual model Yes-minus-No native logit margin at each observed decoding step',
        'all_coordinates':'Final block all2560 residual coordinates, all9728 MLP units, all32 heads and all actual source tokens',
        'conditional_normalizer':'v=gamma*(Wyes-Wno)/sqrt(mean(H36^2)+eps); rounding residual retained; v uses observed H36 and is accounting, NOT a predictive input',
        'prediction':'Separate C-only H12 linear readout fit B train256 and validation128, heldout B128; frozen evaluate C base6 at each natural generated step. Never use H36 or future token in prediction.',
        'limits':['Exact composition identity is not causal necessity or sufficient explanation.',
            'Native complete source decomposition only final block35; earlier layers11/23 source fields remain accessible, not claimed as output causes.',
            'After first answer, Yes/No margin is a diagnostic output contrast, not content correctness or EOS score.',
            'Cached shape and prefill shapes differ legitimately; record rather than silently pool them.']})

def main():
    protocol();rows=read(OUT/'material.json');assert len(list((OUT/'prefix_commits').glob('*.json')))==128
    wd=checkpoint('model.layers.35.mlp.down_proj.weight').float().numpy().astype(np.float64)
    wo=checkpoint('model.layers.35.self_attn.o_proj.weight').float().numpy().astype(np.float64)
    gamma=checkpoint('model.norm.weight').float().numpy().astype(np.float64)
    # Only the four language-specific output rows are read, not a duplicate full vocabulary matrix.
    from safetensors import safe_open
    model=ROOT/'models/hf/qwen3-4b';idx=read(model/'model.safetensors.index.json')['weight_map'];dw={}
    key='lm_head.weight' if 'lm_head.weight' in idx else 'model.embed_tokens.weight'
    if key!='lm_head.weight':assert read(model/'config.json')['tie_word_embeddings']
    save(OUT/'unembedding_source.json',{'key':key,'checkpoint_file':idx[key],'tied_embeddings':key=='model.embed_tokens.weight'})
    with safe_open(str(model/idx[key]),framework='pt',device='cpu') as stream:
        weight=stream.get_slice(key)
        for lang in ('en','zh'):
            first=next(r for r in rows if r['language']==lang);ids=read(OUT/f'behavior/{first["sample_id"]}.json')['answer_ids']
            dw[lang]=np.stack([weight[i:i+1,:].float().numpy()[0] for i in ids]).astype(np.float64)
    accounts=[];features=[];margins=[]
    eps=read(model/'config.json')['rms_norm_eps']
    for ri,r in enumerate(rows):
        with np.load(OUT/f'fields/{r["sample_id"]}.npz') as z:
            h=unbits(z['h']).astype(np.float64)[:,0];norm=unbits(z['postnorm']).astype(np.float64)[0]
            native={k:unbits(z[f'L35_{k}']).astype(np.float64) for k in ('a','down','attention_out','head_output','p','v')}
            behavior=read(OUT/f'behavior/{r["sample_id"]}.json');actual=float(behavior['yes_no_logits'][0]-behavior['yes_no_logits'][1])
        delta=dw[r['language']][0]-dw[r['language']][1];scale=gamma/np.sqrt(np.mean(h[36]**2)+eps);effective=delta*scale
        a=native['a'][0];down=native['down'][0];attention=native['attention_out'][0];head=native['head_output'][0]
        beta=wd.T@effective;unit=a*beta;obeta=(wo.T@effective).reshape(32,128)
        p=native['p'];value=native['v'];expanded=value[np.arange(32)//4]
        source=p*np.einsum('hsd,hd->hs',expanded,obeta)
        account={'sample_id':r['sample_id'],'step':r['generation_step'],'family':r['family'],'language':r['language'],
            'actual_margin':actual,'input_residual_term':float(h[35]@effective),'native_mlp_unit_sum':float(unit.sum()),
            'native_attention_source_sum':float(source.sum()),
            'head_matmul_rounding':float(head@obeta.reshape(-1)-source.sum()),
            'o_projection_rounding':float(attention@effective-head@obeta.reshape(-1)),
            'down_projection_rounding':float(down@effective-unit.sum()),
            'residual_add_rounding':float((h[36]-h[35]-attention-down)@effective),
            'normalization_rounding':float((norm-h[36]*scale)@delta),'lm_head_rounding':float(actual-norm@delta),
            'source_tokens':p.shape[1],'unit_positive_sum':float(np.maximum(unit,0).sum()),'unit_negative_sum':float(np.minimum(unit,0).sum())}
        terms=('input_residual_term','native_mlp_unit_sum','native_attention_source_sum','head_matmul_rounding','o_projection_rounding','down_projection_rounding','residual_add_rounding','normalization_rounding','lm_head_rounding')
        account['account_error']=float(sum(account[k] for k in terms)-actual);assert abs(account['account_error'])<1e-8
        account['largest_abs_rounding_term']=max(abs(account[k]) for k in terms if 'rounding' in k)
        npz(OUT/f'ledgers/{r["sample_id"]}.npz',effective=effective,delta_unembedding=delta,
            native_a=a,beta=beta,unit_contribution=unit,attention_source_contribution=source,attention_p=p,
            postnorm=norm,logit_coordinate_contribution=norm*delta)
        save(OUT/f'accounts/{r["sample_id"]}.json',account);accounts.append(account);features.append(h[12]);margins.append([actual])
        if ri%64==0:print('OUTPUT_ACCOUNT',ri,len(rows),flush=True)
    save(OUT/'account_result.json',{'cases':len(rows),'prefixes':128,'accounts':accounts,
        'max_abs_account_error':max(abs(a['account_error']) for a in accounts),
        'max_abs_rounding_term':max(a['largest_abs_rounding_term'] for a in accounts)})
    b=CAMPAIGN/'b_relations';br=read(b/'material.json')
    with np.load(b/'features/all_samples.npz') as z:x=z['H12_c'].astype(np.float64)
    y=np.array([[bb['yes_no_logprob'][0]-bb['yes_no_logprob'][1]] for r in br if (bb:=read(b/f'behavior/{r["sample_id"]}.json'))])
    tr,va,te=[[i for i,r in enumerate(br) if r['word_split']==s] for s in ('train','validation','test')]
    baseline,pred,param=fit_predict([x],y,tr,va,te,'A1_linear',False)
    xp=np.asarray(features);sc=param['scales'][0];prediction=(1+(xp/sc)@(x[tr]/sc).T)@param['alpha'];target=np.array(margins)
    npz(OUT/'predictions/frozen_H12_margin.npz',prediction=prediction,target=target,input_H12=xp,train_x=x[tr],alpha=param['alpha'],scale=np.array(sc))
    results=[dict(split='base_heldout',target='actual_margin',representation='B_prefill_H12C',algorithm='A1_linear',**baseline)]
    for step in sorted({r['generation_step'] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r['unit']==6 and r['generation_step']==step]
        score=metrics(target[ids],prediction[ids],False);score['sign_agreement']=float(np.mean((prediction[ids]>0)==(target[ids]>0)))
        results.append(dict(split='base6_subset_cached',target='actual_margin',representation=f'C_step{step}_H12C',algorithm='frozen_B_A1',**score))
        mean=np.repeat(y[tr].mean(0)[None],len(ids),axis=0)
        results.append(dict(split='base6_subset_cached',target='actual_margin',representation=f'C_step{step}_H12C',algorithm='B_train_mean',**metrics(target[ids],mean,False)))
    save(OUT/'result.json',{'timestamp':stamp(),'case_count':len(rows),'prefix_count':128,'results':results,
        'max_abs_account_error':max(abs(a['account_error']) for a in accounts),
        'first_pair_correct':sum(read(OUT/f'behavior/{r["sample_id"]}.json')['first_pair_correct'] for r in rows if r['generation_step']==0),
        'eos_prefixes':sum(read(p)['eos'] for p in (OUT/'prefix_commits').glob('*.json')),
        'limits':read(OUT/'analysis_protocol.json')['limits']})
    announce('c_generation',state='analysis_complete',completed=len(rows),total=len(rows),prefixes=128)
    events('c_generation','analysis_complete',steps=len(rows));print('OUTPUT_DONE',len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args();protocol() if a.prepare else main()
