"""Native H23 compilation and all-unit MLP accounting on the shared prefix library."""
import argparse
from scipy.special import softmax,expit
from rdc_prefix_estimators import *
from phase2712_rdc_prefix_native_probability import checkpoint


def rotate(x,position):
    theta=read(ROOT/'models/hf/qwen3-4b/config.json')['rope_theta']
    angle=position/(theta**(np.arange(0,128,2)/128.));c=np.cos(angle);s=np.sin(angle)
    a,b=x[...,:64],x[...,64:]
    return np.concatenate([a*c-b*s,a*s+b*c],-1)


class Compiler:
    def __init__(self):
        keys=('input_layernorm.weight','self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight',
          'self_attn.q_norm.weight','self_attn.k_norm.weight','self_attn.o_proj.weight','mlp.down_proj.weight')
        self.w={k:checkpoint('model.layers.23.'+k).float().numpy().astype(np.float64) for k in keys}
        self.eps=read(ROOT/'models/hf/qwen3-4b/config.json')['rms_norm_eps']

    def factors(self,h):
        h=np.asarray(h,np.float64);x=h/np.sqrt((h*h).mean(1,keepdims=True)+self.eps)*self.w['input_layernorm.weight']
        q=(x@self.w['self_attn.q_proj.weight'].T).reshape(-1,32,128)
        k=(x@self.w['self_attn.k_proj.weight'].T).reshape(-1,8,128)
        v=(x@self.w['self_attn.v_proj.weight'].T).reshape(-1,8,128)
        q=q/np.sqrt((q*q).mean(2,keepdims=True)+self.eps)*self.w['self_attn.q_norm.weight']
        k=k/np.sqrt((k*k).mean(2,keepdims=True)+self.eps)*self.w['self_attn.k_norm.weight']
        return q,k,v

    def compose(self,h,objects):
        q,k,v=self.factors(h);head=[];allkl=[]
        for i,o in enumerate(objects):
            position=o['position'];qq=rotate(q[i],position);kk=rotate(k[i],position)
            keys=np.concatenate([o['past_k'],kk[:,None]],1);values=np.concatenate([o['past_v'],v[i,:,None]],1)
            ek=keys[np.arange(32)//4];ev=values[np.arange(32)//4]
            p=softmax(np.einsum('hd,hsd->hs',qq,ek)/np.sqrt(128),axis=1)
            head.append(np.einsum('hs,hsd->hd',p,ev).reshape(-1))
            reference=o['p']/o['p'].sum(1,keepdims=True)
            allkl.append(np.sum(reference*np.log(np.maximum(reference,1e-30)/np.maximum(p,1e-30)),1))
        return np.stack(head)@self.w['self_attn.o_proj.weight'].T,np.stack(allkl)


def main(confirmation):
    src=CAMPAIGN/('confirmation' if confirmation else 'shared_rules');out=src/'native_path'
    source=CAMPAIGN/('qwen4_confirmation' if confirmation else 'qwen4')
    allrows=read(CAMPAIGN/'shared_rules'/source.name/'rows.json');ii=np.arange(len(allrows)) if confirmation else splits(allrows)[2]
    rows=[allrows[i] for i in ii];compiler=Compiler();objects=[];actual=[];h23=[];heads=[];mlp=[];mlp_ideal=[];factor_mse=[];energy=[];unital=[]
    for r in rows:
        k=r['anchor_array_index'];position=r['position']
        with np.load(CAMPAIGN/r['field_path']) as z:
            K,V,P=[unbits(z['L23_'+part]).astype(np.float64) for part in ('k','v','p')]
            assert K.shape==V.shape and K.shape[0]==8 and K.shape[-1]==128
            assert P.shape[0]==32 and P.shape[1]==6
            assert np.max(np.abs(P[:,k,position+1:]),initial=0)==0
            objects.append({'position':position,'past_k':K[:,:position],'past_v':V[:,:position],'p':P[:,k,:position+1]})
            actual.append(unbits(z['L23_attention_out'][k]));h23.append(unbits(z['h'][23,k]));heads.append(unbits(z['L23_head_output'][k]))
            g,u,a,d=[unbits(z['L23_'+part][k]).astype(np.float64) for part in ('gate','up','a','down')]
            ideal=g*expit(g)*u;mlp.append(d);mlp_ideal.append(a@compiler.w['mlp.down_proj.weight'].T)
            factor_mse.append((a-ideal)**2);energy.append(a*a);unital.append(a)
    actual=np.stack(actual);h23=np.stack(h23);heads=np.stack(heads);trainrows=read(CAMPAIGN/'shared_rules/qwen4/rows.json');tr=splits(trainrows)[0]
    with np.load(CAMPAIGN/'shared_rules/qwen4/features.npz') as z:trainA=z['native_attention'][tr];trainH=z['h23'][tr]
    methods=[('actual_head_FP64_oracle',heads@compiler.w['self_attn.o_proj.weight'].T,None)]
    pred,pkl=compiler.compose(h23,objects);methods.append(('actual_H23_FP64_oracle',pred,pkl))
    pred,pkl=compiler.compose(np.broadcast_to(trainH.mean(0),h23.shape),objects);methods.append(('train_mean_H23_plus_actual_past',pred,pkl))
    for name in ('early_linear','full_linear','full_quadratic','graph_interaction','hash_interaction'):
        with np.load(src/f'predictions/{name}.npz') as z:predH=z['prediction'][:,:2560]
        pred,pkl=compiler.compose(predH,objects);methods.append((name+'_H23_plus_actual_past',pred,pkl))
    reports=[]
    for name,pred,pkl in methods:
        report,arr=errors(actual,pred,trainA,rows)
        if pkl is not None:report['mean_native_attention_KL']=float(pkl.mean());arr['all_head_attention_KL']=pkl.astype(np.float32)
        reports.append({'model':name,**report});npz(out/f'predictions/{name}.npz',prediction=pred.astype(np.float32),**arr)
        print('PREFIX_NATIVE_PATH',name,report['mse'],flush=True)
    mlp=np.stack(mlp);mlp_ideal=np.stack(mlp_ideal);factor_mse=np.stack(factor_mse);energy=np.stack(energy)
    npz(out/'all_unit_accounting.npz',activation_factor_mse=factor_mse.mean(0).astype(np.float32),activation_energy=energy.mean(0).astype(np.float32),
      down_projection_coordinate_mse=np.mean((mlp-mlp_ideal)**2,0).astype(np.float32))
    save(out/'result.json',{'timestamp':stamp(),'phase':2713 if confirmation else 2712,'rows':len(rows),'reports':reports,
      'MLP':{'units':9728,'output_coordinates':2560,'full_unit_ideal_factor_mse':float(factor_mse.mean()),
        'saved_a_real_Wdown_FP64_mse':float(np.mean((mlp-mlp_ideal)**2)),
        'scope':'Observed-factor mathematical accounting, not prediction or proof of semantic gate roles. BF16 arithmetic residual retained.'},
      'available_input_warning':'H23 predictions use currentH12/prefix features; native compilation additionally uses every actual past L23 K/V source coordinate. This is deeper past model computation, not a decoder replacement and not an equal-information comparison to a direct H12-only regressor.',
      'future_scope':'Current q/k/v, H23, P and MLP factors are not prediction inputs except explicitly named arithmetic oracles. Slice past cache strictly before current position; all future attention entries checked zero.',
      'formulas':'Predicted H23 -> its own input RMSNorm -> fixed Wq/Wk/Wv -> head Q/K RMSNorm -> known-position RoPE -> all-source softmax/PV -> fixed Wo.',
      'mechanism_closed':False,'new_mathematical_theorem':False})
    guard();print('NATIVE_PATH_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--confirmation',action='store_true');a=p.parse_args();main(a.confirmation)
