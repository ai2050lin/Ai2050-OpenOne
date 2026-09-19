"""Coordinatewise shared cross-layer operator identification with genuine rolled predictions."""
from rdc_prefix_estimators import *
OUT=CAMPAIGN/'layer_operators'


def layer_features(graph,layer):
    n=len(graph);depth=layer/35.
    return np.column_stack([np.ones(n),graph,np.full(n,depth),np.full(n,depth*depth)])


def predict(coef,x,g,layer):
    d=layer/35.
    return x*(coef[:,0]+d*coef[:,1])+layer_features(g,layer)@coef[:,2:].T


def main():
    immutable(OUT/'protocol.json',{'phase':2712,'source_sha':sha(Path(__file__)),'material_sha':sha(CAMPAIGN/'material_stratified.json'),
      'object':'One coordinate-specific affine/conditional update shared across all36 transitions, versus independent-layer coordinate affine baselines and copies.',
      'equation':'z_next,i = (a_i + d*b_i)*z_i + [1, visible_cues, d, d^2] c_i. z_l = H_l/s_l; each s_l is training-only full-coordinate RMS.',
      'inputs':'Observed current layer H and prefix-only cue counts/open state plus known depth. No future H/norm, gold syntax, answer or later token.',
      'fit':'All640 training anchors *36 layers. One ridge selected on192 validation anchors, all layers/coordinates normalized. Coordinate indices are native; the coordinatewise correspondence is a deliberately restricted testable assumption.',
      'checks':'Observed-input single-layer forecasts versus H12-to-H36 rollout using only its own predicted intermediate states. Residual-increment relative error prevents high background fit being called explained new computation.',
      'limits':['Single-token H plus cue graph omits source positions and detailed history; this is not assumed a complete state.','Shared fitted coefficients are an extraction hypothesis, not proof native layers share their parameter values.','Failure does not invalidate all shared nonlinear/trans-position rules.']})
    rows=read(CAMPAIGN/'shared_rules/qwen4/rows.json');tr,va,te=splits(rows)
    with np.load(CAMPAIGN/'shared_rules/qwen4/features.npz') as z:graph=z['graph'][:,:21].astype(np.float64)
    mean=graph[tr].mean(0);std=np.maximum(graph[tr].std(0),.1);g=(graph-mean)/std
    h=np.empty((len(rows),37,2560),np.float32)
    cached_id=None;cached=None
    for i,r in enumerate(rows):
        if r['sample_id']!=cached_id:
            with np.load(CAMPAIGN/r['field_path']) as z:cached=unbits(z['h'])
            cached_id=r['sample_id']
        h[i]=cached[:,r['anchor_array_index']]
    scale=np.sqrt(np.mean(h[tr].astype(np.float64)**2,axis=(0,2)));scale=np.maximum(scale,1e-12)
    width=h.shape[-1];q=24;f=q+2
    hh=np.zeros((width,2,2));hg=np.zeros((width,2,q));gy=np.zeros((width,q));hy=np.zeros((width,2));gg=np.zeros((q,q))
    independent=[]
    for l in range(36):
        x=h[tr,l].astype(np.float64)/scale[l];y=h[tr,l+1].astype(np.float64)/scale[l+1]
        d=l/35.;G=layer_features(g[tr],l);xx=(x*x).sum(0);xy=(x*y).sum(0);xg=x.T@G
        hh[:,0,0]+=xx;hh[:,0,1]+=d*xx;hh[:,1,0]+=d*xx;hh[:,1,1]+=d*d*xx
        hg[:,0]+=xg;hg[:,1]+=d*xg;gy+=y.T@G;hy[:,0]+=xy;hy[:,1]+=d*xy;gg+=G.T@G
        xm=x.mean(0);ym=y.mean(0);slope=((x-xm)*(y-ym)).sum(0)/np.maximum(((x-xm)**2).sum(0),1e-12)
        independent.append(np.stack([slope,ym-slope*xm]))
    A=np.zeros((width,f,f));B=np.zeros((width,f));A[:,:2,:2]=hh;A[:,:2,2:]=hg;A[:,2:,:2]=hg.transpose(0,2,1);A[:,2:,2:]=gg
    B[:,:2]=hy;B[:,2:]=gy;losses=[];candidates=[]
    for ridge in (.001,.1,10.,1000.):
        coef=np.linalg.solve(A+np.eye(f)[None]*ridge,B[...,None])[...,0];loss=0.
        for l in range(36):
            p=predict(coef,h[va,l]/scale[l],g[va],l);loss+=np.mean((p-h[va,l+1]/scale[l+1])**2)
        candidates.append(coef);losses.append(float(loss/36))
    best=min(range(len(losses)),key=losses.__getitem__);coef=candidates[best];independent=np.stack(independent)
    npz(OUT/'models.npz',shared=coef.astype(np.float32),independent=independent.astype(np.float32),layer_rms=scale,
      graph_mean=mean,graph_std=std,train=tr,validation=va,test=te,ridge=np.array((.001,.1,10.,1000.)[best]))
    reports=[];arrays={}
    for name in ('shared_conditional','independent_affine','copy_raw','copy_training_RMS'):
        coordinate=[];perrow=[];residual=[]
        for l in range(36):
            x=h[te,l].astype(np.float64);target=h[te,l+1].astype(np.float64)
            if name=='shared_conditional':p=predict(coef,x/scale[l],g[te],l)*scale[l+1]
            elif name=='independent_affine':p=(x/scale[l]*independent[l,0]+independent[l,1])*scale[l+1]
            elif name=='copy_training_RMS':p=x/scale[l]*scale[l+1]
            else:p=x
            err=(p-target)**2;coordinate.append(err.mean(0));perrow.append(err.mean(1));residual.append(float(np.mean((target-x)**2)))
        c=np.stack(coordinate);r=np.stack(perrow)
        arrays[name+'_coordinate_mse']=c.astype(np.float32);arrays[name+'_row_mse']=r.astype(np.float32)
        reports.append({'model':name,'single_layer_raw_mse':float(c.mean()),'by_transition_mse':c.mean(1).tolist(),
          'residual_increment_energy_by_transition':residual,'error_over_residual_increment_by_transition':(c.mean(1)/np.maximum(residual,1e-30)).tolist(),
          'full_coordinates':width,'transitions':36,'test_source_units':len({rows[i]['sample_id'] for i in te})})
    rollout_reports=[]
    for name in ('shared_conditional','independent_affine'):
        p=h[te,12].astype(np.float64)/scale[12];trajectory=[]
        for l in range(12,36):
            p=predict(coef,p,g[te],l) if name=='shared_conditional' else p*independent[l,0]+independent[l,1]
            trajectory.append(float(np.mean((p*scale[l+1]-h[te,l+1])**2)))
        pred=p*scale[36];assert np.isfinite(pred).all()
        report,arr=errors(h[te,36],pred,h[tr,36],[rows[i] for i in te]);rollout_reports.append({'model':name,'from_H12':True,'H36':report,'trajectory_mse':trajectory})
        npz(OUT/f'predictions/{name}_rollout.npz',prediction=pred.astype(np.float32),test=te,**arr)
    npz(OUT/'coordinate_errors.npz',**arrays)
    result={'phase':2712,'timestamp':stamp(),'shared_parameters':int(coef.size),'independent_parameters':int(independent.size),
      'validation_ridge_grid':dict(zip(map(str,(.001,.1,10.,1000.)),losses)),'selected_ridge':(.001,.1,10.,1000.)[best],
      'single_layer_reports':reports,'rollout_reports':rollout_reports,'mechanism_closed':False,
      'interpretation':'A single observed-layer fit may exploit residual copying. Rollout tests accumulated missing computation. Coordinatewise failure is scoped to this operator class, not all possible shared mechanisms.'}
    save(OUT/'result.json',result)
    immutable(OUT/'frozen.json',{'timestamp':stamp(),'model_sha':sha(OUT/'models.npz'),'protocol_sha':sha(OUT/'protocol.json'),'result_sha':sha(OUT/'result.json')})
    status('layer_operators',state='complete',transitions=36);guard();print('LAYER_OPERATORS_COMPLETE',[(r['model'],r['H36']['mse']) for r in rollout_reports],flush=True)


if __name__=='__main__':main()
