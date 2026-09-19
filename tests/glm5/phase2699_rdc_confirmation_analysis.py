"""Old frozen readers on new cases; new grouped fits separated from prospective confirmation."""
from rdc_continuity_common import *
from rdc_feature_extractors import fit_predict,metrics
OUT=CAMPAIGN/'e_confirmation'

def reader(path):
    with np.load(path) as z:return (z['z_train'].T@z['alpha'])/z['raw_scale_vector'][:,None],z['alpha'].sum(0)

def extract():
    rows=read(OUT/'material.json');assert len(list((OUT/'commits').glob('*.json')))==1024
    if (OUT/'features/roles.npz').exists():return
    h=np.empty((1024,37,3,2560),np.float32);nat=np.empty_like(h);native={};pairs=[];pending={};shape=[]
    for i,r in enumerate(rows):
      with np.load(OUT/f'fields/{r["sample_id"]}.npz') as z:
        a=z['h'];pos=z['native_positions'].tolist();c=pos.index(len(r['prompt_ids'])-1)
        for j,ix in enumerate([r['spans']['u']['positions'],r['spans']['v']['positions'],[len(r['prompt_ids'])-1]]):h[i,:,j]=unbits(a[:,ix]).mean(1)
        nat[i]=z['natural_roles'];shape.append(z['shape_error_coordinate_max'])
        for l in (11,23,35):
         for key in ('gate','up','a','down','mlp_x','attention_x'):
          native.setdefault(f'L{l}_{key}',[]).append(unbits(z[f'L{l}_{key}'][c]))
        key=(r['family'],r['unit'],r['fact_truth'],r['language'])
        if not r['negative_query']:pending[key]=(r,a.copy())
        else:
          r0,h0=pending.pop(key);shared=next((j for j,(x,y) in enumerate(zip(r0['prompt_ids'],r['prompt_ids'])) if x!=y),min(len(r0['prompt_ids']),len(r['prompt_ids'])))
          diff=unbits(a[:,:shared]).astype(np.float64)-unbits(h0[:,:shared]).astype(np.float64)
          pairs.append({'sample_ids':[r0['sample_id'],r['sample_id']],'shared_tokens':shared,'nonzero':int(np.count_nonzero(diff)),'max_abs':float(np.abs(diff).max())})
      if i%64==0:print('EXTRACT',i,flush=True)
    npz(OUT/'features/roles.npz',matched=h,natural=nat)
    npz(OUT/'features/native.npz',**{k:np.stack(v) for k,v in native.items()})
    npz(OUT/'features/numerical_shape.npz',max_abs=np.stack(shape))
    save(OUT/'prefix_audit.json',{'pairs':pairs,'nonzero_pairs':sum(p['nonzero']>0 for p in pairs),'all_coordinate_count_per_position':37*2560})

def main():
    immutable(OUT/'analysis_protocol.json',{'source_sha':sha(Path(__file__)),'old_confirmation':'Frozen B A1 UVC and C-only for t/y atH12/24/36, separately natural and matched; no selection on new cases.',
      'new_fits':'A1 fullC at all37 checkpoints, t/y; H12/24/36 U/UV/norm/direction/length controls. Train512 validation256 test256.',
      'frozen_crosslayer':'Old H24 C-only reader unchanged at all37 checkpoints; observed-score evolution, not inferred semantic creation.',
      'no_topk':True})
    extract();rows=read(OUT/'material.json');b=[read(OUT/f'behavior/{r["sample_id"]}.json') for r in rows]
    with np.load(OUT/'features/roles.npz') as z:h=z['matched'];nat=z['natural']
    ys={k:np.eye(2)[[int(r[field]) for r in rows]] for k,field in [('positive_support','fact_truth'),('requested_answer','expected_yes')]}
    old=[];new=[];fixed={};train,val,test=[[i for i,r in enumerate(rows) if r['word_split']==s] for s in ('train','validation','test')]
    def report(y,p,indices):
        score=metrics(y[indices],p,True);score['correct']=int(np.sum(y[indices].argmax(1)==p.argmax(1)))
        score['by_family']={f:metrics(y[indices][mask],p[mask],True) for f in sorted({r['family'] for r in rows}) if (mask:=np.array([rows[i]['family']==f for i in indices])).any()}
        return score
    for target,y in ys.items():
      for l in (12,24,36):
       for mode,suffix in [('UVC','A1_linear'),('C','C_only')]:
        path=HISTORY/f'b_relations/models/H{l}__{target}__{suffix}.npz';w,bias=reader(path)
        for track,a in [('natural',nat),('matched',h)]:
          x=a[:,l].reshape(1024,-1) if mode=='UVC' else a[:,l,2];p=x@w+bias
          old.append(dict(representation=f'H{l}_{mode}',target=target,track=track,**report(y,p,list(range(1024)))))
          npz(OUT/f'predictions/frozen_H{l}_{mode}_{target}_{track}.npz',prediction=p,target=y)
      w,bias=reader(HISTORY/f'b_relations/models/H24__{target}__C_only.npz')
      fixed[target]=h[:,:,2]@w+bias;fixed[target+'_weight']=w;fixed[target+'_bias']=bias
      for l in range(37):
        score,p,params=fit_predict([h[:,l,2]],y,train,val,test,'A1_linear',True)
        new.append(dict(representation=f'H{l}_C',target=target,algorithm='A1_linear',**report(y,p,test),ridge=score['ridge']))
        npz(OUT/f'models/H{l}_{target}.npz',**{k:v for k,v in params.items() if isinstance(v,np.ndarray)})
      for l in (12,24,36):
       blocks=h[:,l];norm=np.linalg.norm(blocks,axis=-1)
       controls={'U':[blocks[:,0]],'UV':[blocks[:,0],blocks[:,1]],'norm':[norm],'directionC':[blocks[:,2]/np.maximum(norm[:,2,None],1e-12)],'length':[np.array([[len(r['prompt_ids']),r['spans']['u']['positions'][0],r['spans']['v']['positions'][0]] for r in rows])]}
       for label,x in controls.items():
        score,p,_=fit_predict(x,y,train,val,test,'A1_linear',True)
        new.append(dict(representation=f'H{l}_{label}',target=target,algorithm='A1_control',**report(y,p,test)))
      print('READERS',target,len(new),flush=True)
    npz(OUT/'features/fixed_ruler.npz',**fixed)
    behavior={f:{'n':sum(r['family']==f for r in rows),**{t+'_correct':sum(bb[t+'_correct'] for r,bb in zip(rows,b) if r['family']==f) for t in ('natural','matched')}} for f in sorted({r['family'] for r in rows})}
    result={'timestamp':stamp(),'case_count':1024,'frozen_confirmation':old,'new_grouped_results':new,'behavior_by_family':behavior,
      'argmax_changed':sum(bb['natural_argmax']!=bb['matched_argmax'] for bb in b),'shape_max':max(bb['shape_max'] for bb in b),'prefix_audit':read(OUT/'prefix_audit.json'),
      'limits':read(OUT/'protocol.json')['limits'],'results':[dict(r,split='prospective',algorithm='old_A1_'+r['track']) for r in old]+[dict(r,split='new_base_heldout') for r in new]}
    save(OUT/'result.json',result);announce('e_confirmation',state='analysis_complete',completed=1024,total=1024);events('e_confirmation','analysis_complete',comparisons=len(old)+len(new))

if __name__=='__main__':main()
