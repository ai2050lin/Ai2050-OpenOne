"""Fixed raw-coordinate rulers, native conditional factors, and genuine earlier-state forecasts."""
from rdc_continuity_common import *
from rdc_feature_extractors import fit_predict
from phase2699_rdc_confirmation_analysis import reader
SRC=CAMPAIGN/'e_confirmation';OUT=CAMPAIGN/'f_continuity'

def main():
    import torch
    torch.set_num_threads(2)
    immutable(OUT/'protocol.json',{'phase':2700,'source_sha':sha(Path(__file__)),'material_sha':sha(SRC/'material.json'),
      'rulers':'Frozen old B H24 C-only t/y binary difference. Identical weights/bias across all37 native checkpoints.',
      'native_layers':[11,23,35],'units':'all9728; real gate/up/down weights; no neuron ranking selection',
      'factor_tests':'Observed g,u and full sigmoid(g) identity versus training global/lang/family-lang mean sigmoid. Mean gates use no truth or answer labels; native projection replay distinguished from earlier-state forecasting.',
      'forecasts':'New train512 val256 test256: H12 C or H12 UVC predicts all H24 C coordinates and all L23 a units; A0mean/A1linear/A2quadratic; H24 state also compare identity and mean update.',
      'full_input_dot_audit':'32 cases: everyfamily x bothtruths x bothlanguages atunit0 q0; actual Wgate/Wup against captured normalized input, all2560 input coordinates andall9728 outputs. FP64 products versus observedBF16 are rounding errors, not semantic effects.',
      'no_leakage':'Future activations used only as training targets/evaluation. Input has no future state, final norm, ground-truth support or answer.',
      'limits':['Fixed probe is external, not native unembedding.','Full low-rank kernel prediction is a statistical approximation, not extracted native dynamics.','Residual update minus measured MLP includes attention AND residual rounding.']})
    rows=read(SRC/'material.json');tr,va,te=[[i for i,r in enumerate(rows) if r['word_split']==s] for s in ('train','validation','test')]
    with np.load(SRC/'features/roles.npz') as z:h=z['matched']
    with np.load(SRC/'features/native.npz') as z:native={k:z[k] for k in z.files}
    matrices=[];weights=[];biases=[]
    for t in ('positive_support','requested_answer'):
        w,b=reader(HISTORY/f'b_relations/models/H24__{t}__C_only.npz');weights.append(w[:,1]-w[:,0]);biases.append(b[1]-b[0])
    w=np.stack(weights,axis=1);b=np.array(biases);scores=h[:,:,2].astype(np.float64)@w+b
    labels=np.array([[int(r['fact_truth']),int(r['expected_yes'])] for r in rows])
    ruler=[{'H':l,'support_correct':int(np.sum((scores[:,l,0]>0)==labels[:,0])),'answer_correct':int(np.sum((scores[:,l,1]>0)==labels[:,1])),'n':1024} for l in range(37)]
    npz(OUT/'fixed_ruler.npz',weight=w,bias=b,score=scores,coordinate_step=h[:,:,2][:,1:]-h[:,:,2][:,:-1])
    for l in (11,23,35):
        g=native[f'L{l}_gate'].astype(np.float64);u=native[f'L{l}_up'].astype(np.float64);a=native[f'L{l}_a'].astype(np.float64)
        sig=1/(1+np.exp(-g));ideal=g*sig*u
        ids=[i for i,r in enumerate(rows) if r['unit']==0 and not r['negative_query']]
        input_audit={}
        for part in ('gate','up'):
            weight=checkpoint(f'model.layers.{l}.mlp.{part}_proj.weight').float().numpy().astype(np.float64)
            full=native[f'L{l}_mlp_x'][ids].astype(np.float64)@weight.T
            error=native[f'L{l}_{part}'][ids]-full
            input_audit[part]={'cases':len(ids),'scalar_outputs':int(full.size),'full_input_dimension':2560,'rounding_max':float(np.max(np.abs(error))),'rounding_rms':float(np.sqrt(np.mean(error**2)))}
            npz(OUT/f'L{l}_{part}_input_audit.npz',sample_indices=np.array(ids),full_dot=full,observed=native[f'L{l}_{part}'][ids],rounding=error)
            del weight,full,error
        wd=checkpoint(f'model.layers.{l}.mlp.down_proj.weight').float().numpy().astype(np.float64);beta=wd.T@w
        observed_write=native[f'L{l}_down'].astype(np.float64)@w;unit_write=a@beta
        delta=scores[:,l+1]-scores[:,l];other=delta-observed_write
        means={'global':np.repeat(sig[tr].mean(0)[None],len(rows),axis=0)}
        for mode in ('language','family_language'):
            key=lambda r:r['language'] if mode=='language' else r['family']+'_'+r['language']
            lookup={v:sig[[i for i in tr if key(rows[i])==v]].mean(0) for v in {key(r) for r in rows}}
            means[mode]=np.stack([lookup[key(r)] for r in rows])
        factors=[];products={}
        for name,s in [('half',np.full_like(sig,.5)),*means.items(),('observed_sigmoid_identity',sig)]:
            pred=g*u*s;err=pred[te]-a[te]
            factors.append({'mode':name,'n':len(te),'all_unit_mse':float(np.mean(err**2)),'fixed_score_mse':float(np.mean(((pred[te]-a[te])@beta)**2))})
            if name in ('global','family_language'):products[name+'_gate']=s.astype(np.float32);products[name+'_a']=pred.astype(np.float32)
        products.update(beta=beta,observed_gate=g.astype(np.float32),observed_up=u.astype(np.float32),observed_a=a.astype(np.float32),ideal_a=ideal.astype(np.float32),unit_contribution=(a[:,:,None]*beta[None]).astype(np.float32),observed_write=observed_write,unit_write=unit_write,residual_update=delta,attention_and_rounding=other,down_rounding=observed_write-unit_write)
        npz(OUT/f'L{l}_factors.npz',**products)
        matrices.append({'layer':l,'factor_comparisons':factors,'full_input_dot_audit':input_audit,'down_rounding_max':float(np.max(np.abs(observed_write-unit_write))),'unit_real_arithmetic_max_error':float(np.max(np.abs(a-ideal)))})
        print('NATIVE_FACTORS',l,flush=True)
    forecasts=[];targets={'H24_C':h[:,24,2],'L23_a':native['L23_a']}
    for target,y in targets.items():
      for mode,blocks in [('C',[h[:,12,2]]),('UVC',[h[:,12,j] for j in range(3)])]:
       for algo in ('A0_mean','A1_linear','A2_quadratic'):
        metric,p,params=fit_predict(blocks,y,tr,va,te,algo,False)
        mid=f'{target}_{mode}_{algo}';forecasts.append({'target':target,'input':'H12_'+mode,'algorithm':algo,**metric})
        npz(OUT/f'predictions/{mid}.npz',prediction=p.astype(np.float32),target=y[te],test=np.asarray(te),coordinate_mse=np.mean((p-y[te])**2,axis=0))
        if algo!='A0_mean':npz(OUT/f'models/{mid}.npz',**{k:v for k,v in params.items() if isinstance(v,np.ndarray)})
      if target=='H24_C':
       for name,p in [('identity',h[te,12,2]),('mean_update',h[te,12,2]+(h[tr,24,2]-h[tr,12,2]).mean(0))]:forecasts.append({'target':target,'input':'H12_C','algorithm':name,'n':len(te),'mse':float(np.mean((p-y[te])**2))})
      print('FORECAST',target,flush=True)
    save(OUT/'result.json',{'phase':2700,'timestamp':stamp(),'ruler':ruler,'native':matrices,'forecasts':forecasts,'limits':read(OUT/'protocol.json')['limits']})
    announce('f_continuity',state='analysis_complete',completed=1024,total=1024)
    save(SRC/'extension_result.json',{'continuity':read(OUT/'result.json')})

if __name__=='__main__':main()
