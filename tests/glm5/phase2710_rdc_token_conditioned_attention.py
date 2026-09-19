"""Exploratory token-conditioned full-coordinate kernels and native H23 compilation.

O has already been evaluated: this repartition is explicitly exploratory, not a new
untouched confirmation. Raw O and the frozen N predictors are never modified.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,gc,shutil
from rdc_conditional_common import *
from rdc_conditional_estimators import RIDGES
from phase2708_rdc_cached_attention_prediction import fit_kernel,df_ridge
from phase2709_rdc_attention_transfer_analysis import objects_for,factor_output,summary
OUT=CAMPAIGN/'p_token_conditioned';O=CAMPAIGN/'o_generalization'


def prepare():
    assert (O/'result.json').exists()
    protocol={'phase':2710,'source_sha':sha(Path(__file__)),'source_O_result_sha':sha(O/'result.json'),
      'status':'Exploratory follow-up chosen after seeing O frozen-domain transfer results; not an untouched independent confirmation.',
      'material':'All1536 actual O analysis states, original O records remain all-test and immutable. Derived P split by entity0..7train768/8..11validation384/12..15test384; translations/forms/allsteps stay grouped. No filtering on correctness or error.',
      'inputs':'Actual current-token embeddingH0, currentH12, and ALL already-available pastL23 K/V coordinates/positions. Current token ID is prompt_ids[-1], not emitted next_token_id. FutureH23, currentL23Q/K/V and P are targets only.',
      'kernels':'H0linear, H12linear, full(H12+pastK+pastV)linear/quadratic, additiveH0+full, and product(1+H0dot)*(1+fullcontextdot). Each rawblock RMS scaled byP train only. Every native coordinate used; product is a standard tensor-product kernel, not proof of new mathematics. Logically512paddedsource positions; entirely absent source-position blocks skipped only because all padding is zero.',
      'routes':'Each sameGram predicts Q/Kself/Vself6144 -> actualcacheattention; or futureH23(2560) -> real inputRMSNorm/Wq/Wk/Wv/qkheadRMSNorm -> actualcacheattention; or direct head+Kself+Vself6144 -> WO. All ridge choices use validation attention-output MSE. Full_linear/token_product additionally use fixeddf128 across all3routes. Equal head count is not identical effective nonlinear capacity.',
      'native_compiler':'H23 is block22 residual output, before block23 inputnorm. Predicted H23 is normalized using its OWN RMS and real gamma/eps1e-6; never use actualfutureRMS. Q/K perhead norms and all native projection weights are fixed checkpoint parameters. Same known-position RoPE convention asN/O. Native actualH23 oracle reports extra FP64/BF16 arithmetic floor separately.',
      'controls':'Train current-token-ID mean QKV/H23/head routes, unseen current-tokenIDs fall back toP globaltrainingmean. Check every currentH0 against its fixed embedding table row, all native coordinates. No language/family/answer labels enter the kernels.',
      'resources':{'maximum_seconds':1800,'maximum_new_bytes':768*1024**2,'campaign_ceiling':30*1024**3,'free_floor':8*1024**3,'new_CUDA_models':0},
      'limits':['O outcomes were already inspected, so P entity grouping does not turn this into an independent confirmation.','Only4testentitygroups and correlatedgeneratedtoken states.','Embedding may help a restricted learner without adding information absent fromH12 in an information-theoretic sense.','Using real weights/normalization supplies native architectural prior, not independently learned semantic rules.','PastKV retains earlier full-model computation; no end-to-end closure or native answer-correctness claim.']}
    immutable(OUT/'protocol.json',protocol)
    total=sum(p.stat().st_size for p in CAMPAIGN.rglob('*') if p.is_file())
    assert total+protocol['resources']['maximum_new_bytes']<protocol['resources']['campaign_ceiling'] and shutil.disk_usage(OUT).free>protocol['resources']['free_floor']
    return protocol


def full_cache_grams(objects,tr):
    n=len(objects);end=max(o['position'] for o in objects);result={};scales={}
    for part in ('past_k','past_v'):
        gram=np.zeros((n,n),np.float64)
        for start in range(0,end,16):
            block=np.zeros((n,16,1024),np.float64)
            for i,o in enumerate(objects):
                x=o[part][:,start:start+16].transpose(1,0,2).reshape(-1,1024);block[i,:len(x)]=x
            x=block.reshape(n,-1);gram+=x@x.T
        scale=max(float(np.sqrt(np.diag(gram)[tr].mean())),1e-12);result[part]=gram/scale**2;scales[part]=scale
        print('P_FULL_CACHE_GRAM',part,n,flush=True)
    return result,scales


class Compiler:
    def __init__(self):
        self.weights={key:checkpoint('model.layers.23.'+key).float().numpy().astype(np.float64) for key in (
          'input_layernorm.weight','self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight',
          'self_attn.q_norm.weight','self_attn.k_norm.weight','self_attn.o_proj.weight')}
        self.wo=self.weights['self_attn.o_proj.weight'];self.eps=read(ROOT/'models/hf/qwen3-4b/config.json')['rms_norm_eps']
    def qkv(self,h):
        x=h/np.sqrt(np.mean(h*h,axis=1,keepdims=True)+self.eps)*self.weights['input_layernorm.weight']
        q=(x@self.weights['self_attn.q_proj.weight'].T).reshape(-1,32,128)
        k=(x@self.weights['self_attn.k_proj.weight'].T).reshape(-1,8,128)
        q=q/np.sqrt(np.mean(q*q,axis=2,keepdims=True)+self.eps)*self.weights['self_attn.q_norm.weight']
        k=k/np.sqrt(np.mean(k*k,axis=2,keepdims=True)+self.eps)*self.weights['self_attn.k_norm.weight']
        v=x@self.weights['self_attn.v_proj.weight'].T
        return np.concatenate((q.reshape(-1,4096),k.reshape(-1,1024),v),1)


def embedding_check(rows,h0):
    from safetensors import safe_open
    ids=np.array([r['prompt_ids'][-1] for r in rows],int);index=read(ROOT/'models/hf/qwen3-4b/model.safetensors.index.json')['weight_map'];key='model.embed_tokens.weight';records=[]
    with safe_open(str(ROOT/'models/hf/qwen3-4b'/index[key]),framework='pt',device='cpu') as f:
        embedding=f.get_slice(key)
        for token in np.unique(ids):
            value=embedding[int(token):int(token)+1].float().numpy()[0];mask=ids==token;assert np.array_equal(h0[mask],np.broadcast_to(value,h0[mask].shape))
            records.append({'current_input_token_id':int(token),'occurrences':int(mask.sum()),'all2560coordinates_equal_fixed_parameter_row':True})
    save(OUT/'embedding_identity_audit.json',{'passed':True,'states':len(rows),'unique_current_tokens':len(records),'records':records,
      'scope':'Fixed embedding-table identity verified, not semantic identity or causal mechanism.'})
    return ids


def main():
    protocol=prepare();started=time.monotonic();original=read(O/'features/selected_rows.json');rows=[]
    for r in original:rows.append(dict(r,origin_word_split=r['word_split'],word_split='train' if r['unit']<8 else 'validation' if r['unit']<12 else 'test',evidence_status='post_O_result_exploratory_repartition'))
    tr,va,te=splits(rows);assert tuple(map(len,(tr,va,te)))==(768,384,384)
    objects,h12,target,head,qkv,_=objects_for(rows,O);h0=[];h23=[]
    for r in rows:
        with np.load(O/f'fields/{r["sample_id"]}.npz') as z:h=unbits(z['h_c']);h0.append(h[0]);h23.append(h[23])
    h0=np.stack(h0).astype(np.float64);h23=np.stack(h23).astype(np.float64);ids=embedding_check(rows,h0);compiler=Compiler()
    save(OUT/'selected_rows.json',rows);npz(OUT/'features.npz',h0=h0.astype(np.float32),h12=h12.astype(np.float32),h23=h23.astype(np.float32),qkv=qkv,head=head.astype(np.float32),attention=target.astype(np.float32),current_input_token_ids=ids)
    gs={};scales={}
    for name,x in (('H0',h0),('H12',h12)):
        scale=max(float(np.sqrt(np.sum(x[tr]**2,1).mean())),1e-12);scales[name]=scale;gs[name]=x@x.T/scale**2
    cache,cs=full_cache_grams(objects,tr);full=(gs['H12']+cache['past_k']+cache['past_v'])/3
    npz(OUT/'input_grams.npz',H0=gs['H0'],H12=gs['H12'],past_K=cache['past_k'],past_V=cache['past_v'],train=tr,validation=va,test=te,
      H0_scale=np.array(scales['H0']),H12_scale=np.array(scales['H12']),K_scale=np.array(cs['past_k']),V_scale=np.array(cs['past_v']))
    kernels={'H0_linear':1+gs['H0'],'H12_linear':1+gs['H12'],'full_linear':1+full,'full_quadratic':(1+full)**2,
      'token_add':1+(gs['H0']+3*full)/4,'token_product':(1+gs['H0'])*(1+full)}
    energy=np.mean(target[tr]**2,0);rtest=[rows[i] for i in te];otest=[objects[i] for i in te];oval=[objects[i] for i in va];reports=[]
    native_qkv,_=factor_output(qkv[te],otest,compiler.wo);native_h23,_=factor_output(compiler.qkv(h23[te]),otest,compiler.wo)
    floors={}
    for name,pred in (('native_QKV',native_qkv),('native_H23_compiler',native_h23)):
        report,arr=summary(target[te],pred,energy,rtest);floors[name]=report;npz(OUT/f'predictions/{name}_floor.npz',prediction=pred.astype(np.float32),test=te,**arr)
    save(OUT/'native_arithmetic_audit.json',{'scope':'Native future-target arithmetic oracles, not prediction inputs','floors':floors})
    def output(route,p,oo):
        if route=='hidden23':return factor_output(compiler.qkv(p),oo,compiler.wo)
        if route=='factors':return factor_output(p,oo,compiler.wo)
        return p[:,:4096]@compiler.wo.T,{}
    def record(mid,pred,extra,meta):
        report,arr=summary(target[te],pred,energy,rtest)
        if extra:report['all_head_attention_KL']=float(extra['attention_KL_by_head'].mean())
        reports.append({'model':mid,**meta,**report});npz(OUT/f'predictions/{mid}.npz',prediction=pred.astype(np.float32),test=te,**arr,**extra)
        print('P_TOKEN_CONDITION',mid,report['mse'],flush=True);assert time.monotonic()-started<protocol['resources']['maximum_seconds']
    targets={'factors':qkv,'hidden23':h23,'direct_head':np.concatenate((head,qkv[:,4096:]),1)}
    train_ids=set(ids[tr].tolist());unseen=sum(int(ids[i]) not in train_ids for i in te)
    for route,y in targets.items():
        means={token:y[tr[ids[tr]==token]].mean(0) for token in train_ids};p=np.stack([means.get(int(ids[i]),y[tr].mean(0)) for i in te]);pred,extra=output(route,p,otest)
        record('token_mean_'+route,pred,extra,{'route':route,'baseline':'Training current-token-ID means, global fallback','unseen_current_token_test_states':unseen,'evidence_status':'exploratory'})
    for name,gram in kernels.items():
        e,q,vq,tq=fit_kernel(gram,tr,va,te)
        for route,y in targets.items():
            qty=q.T@y[tr];loss=[]
            for ridge in RIDGES:
                p=vq@(qty/(e[:,None]+ridge));pred,_=output(route,p,oval);loss.append(float(np.mean((pred-target[va])**2)))
            best=min(range(len(RIDGES)),key=lambda i:(loss[i],-RIDGES[i]));choices=[('validation',RIDGES[best])]
            if name in ('full_linear','token_product'):choices.append(('fixed_df128',df_ridge(e,128.)))
            for selection,ridge in choices:
                spectral=qty/(e[:,None]+ridge);p=tq@spectral;pred,extra=output(route,p,otest);mid=f'{name}_{route}_{selection}'
                npz(OUT/f'models/{mid}.npz',alpha=(q@spectral).astype(np.float32),train=tr,test=te,ridge=np.array(ridge))
                record(mid,pred,extra,{'kernel':name,'route':route,'selection':selection,'ridge':ridge,'effective_degrees_of_freedom':float(np.sum(e/(e+ridge))),
                  'validation_attention_grid':dict(zip(map(str,RIDGES),loss)),'fitted_output_dimensions':y.shape[1],'evidence_status':'post_O_result_exploratory_repartition'})
                if route=='hidden23':
                    npz(OUT/f'predictions/{mid}_hidden_target.npz',prediction=p.astype(np.float32),target=h23[te].astype(np.float32),test=te)
        print('P_KERNEL_DONE',name,flush=True);gc.collect()
    total=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());campaign=sum(p.stat().st_size for p in CAMPAIGN.rglob('*') if p.is_file())
    assert total<protocol['resources']['maximum_new_bytes'] and campaign<protocol['resources']['campaign_ceiling'] and shutil.disk_usage(OUT).free>protocol['resources']['free_floor'],(total,campaign)
    result={'phase':2710,'timestamp':stamp(),'states':1536,'train':768,'validation':384,'test':384,'train_entities':8,'validation_entities':4,'test_entities':4,
      'reports':reports,'native_arithmetic_floors':floors,'unseen_current_token_test_states':unseen,'new_bytes':total,'campaign_bytes':campaign,
      'elapsed_seconds':time.monotonic()-started,'evidence_status':'Exploratory; original O frozen-generalization result remains unchanged','limits':protocol['limits']}
    save(OUT/'result.json',result);announce('p_token_conditioned',state='analysis_complete',completed=1536,total=1536,test=384);print('TOKEN_CONDITIONED_COMPLETE',len(reports),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args();prepare() if a.prepare else main()
