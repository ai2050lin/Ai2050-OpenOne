"""Matched natural coherent/order current-gradient spans; finite positive/reverse steps."""
import gc
from rdc_binding_common import *
from rdc_binding_gradients import *

def main():
    import torch
    from rdc_law_native import Tail
    out=BASE/'alpha_natural'
    if (out/'result.json').exists():return
    start=time.monotonic();views=read(LAW/'formation/protocol.json')['controlled_views']
    arrays={'x':[],'r':[],'targets':[]};meta=[]
    for condition in ('coherent','order_control'):
      for row in views:
        pos=row['training_positions'][-1]
        if condition=='coherent':
            with np.load(LAW/'capture/main/sources'/f'{row["sample_id"]}.npz') as z:
                arrays['x'].append(unbits(z['x'])[pos]);arrays['r'].append(unbits(z['residual'])[pos])
        else:
            with np.load(LAW/'formation/controlled_capture/fields'/f'{row["sample_id"]}.npz') as z:
                arrays['x'].append(unbits(z['x'])[-1]);arrays['r'].append(unbits(z['residual'])[-1])
        arrays['targets'].append(row['target_ids'][-1]);meta.append({k:row[k] for k in ('sample_id','source_group','cohort','split')}|{'condition':condition})
    compressed(out/'material.json.gz',meta);tail=Tail()
    x=torch.tensor(np.array(arrays['x']),device='cuda');r=torch.tensor(np.array(arrays['r']),device='cuda');targets=torch.tensor(arrays['targets'],device='cuda')
    f,initial=collect(tail,x,r,targets);gram=factor_gram(f)['total'];norm=gram.diag().clamp_min(1e-30).sqrt();cos=gram/norm[:,None]/norm[None,:]
    npz(out/'full_current_gradient_factors.npz',**{k:v.float().cpu().numpy() for k,v in f.items()},gram=gram.cpu().numpy(),cosine=cos.cpu().numpy(),**initial)
    ci=np.array([i for i,v in enumerate(meta) if v['split']=='train' and v['condition']=='coherent'])
    oi=np.array([i for i,v in enumerate(meta) if v['split']=='train' and v['condition']=='order_control'])
    assert len(ci)==len(oi)==96
    average=torch.zeros(len(meta),device='cuda',dtype=torch.float64);average[ci]=1/len(ci)
    oc,oinfo=projection_coefficients(gram[oi][:,oi],(gram@average)[oi]);ocfull=torch.zeros_like(average);ocfull[oi]=oc
    # How much heldout gradient norm each complete training span explains, normalized per query.
    spanreports=[]
    for name,ix in [('coherent_span',ci),('order_span',oi)]:
        g=gram[ix][:,ix];eig,vec=torch.linalg.eigh((g+g.T)*.5);keep=eig>eig.max()*1e-9
        cross=gram[ix];projection=(vec[:,keep].T@cross)/eig[keep].sqrt()[:,None]
        fraction=projection.square().sum(0)/gram.diag().clamp_min(1e-30)
        for condition in ('coherent','order_control'):
          for split in ('validation','test'):
            ii=np.array([i for i,v in enumerate(meta) if v['condition']==condition and v['split']==split])
            spanreports.append({'span':name,'condition':condition,'split':split,'n':len(ii),'fraction_mean':float(fraction[ii].mean()),
              'cluster':clustered(fraction[ii].cpu().numpy(),[meta[i]['source_group'] for i in ii]),'numerical_rank':int(keep.sum())})
        npz(out/f'{name}_fractions.npz',fraction=fraction.cpu().numpy())
    candidates={'coherent_mean':average,'coherent_projected_order_span':ocfull,'coherent_residual_order_span':average-ocfull,'reverse_coherent_mean':-average}
    original={k:v.detach().clone() for k,v in tail.w.items()};reports=[]
    for name,c in candidates.items():
        dense={k:v.float() for k,v in combination(f,c).items()};unit,dnorm=normalized(dense);del dense
        for step in (.02,.10):
            with torch.no_grad():
                for k,v in tail.w.items():v.copy_(original[k]);v.add_(unit[k],alpha=-step)
            _,now=collect(tail,x,r,targets);delta=now['loss']-initial['loss']
            predicted=(-step*(gram@c)/dnorm).cpu().numpy()
            npz(out/'updates'/f'{name}_{step}.npz',**now,actual_loss_delta=delta,predicted_first_order_loss_delta=predicted)
            for condition in ('coherent','order_control'):
              for split in ('validation','test'):
                ix=np.array([i for i,v in enumerate(meta) if v['condition']==condition and v['split']==split])
                reports.append({'direction':name,'step_norm':step,'condition':condition,'split':split,'n':len(ix),
                  'loss_delta':float(delta[ix].mean()),'first_order_error_RMSE':float(np.sqrt(np.mean((predicted[ix]-delta[ix])**2))),
                  'cluster':clustered(delta[ix],[meta[i]['source_group'] for i in ix])})
        print('NATURAL_ALPHA_UPDATE',name,flush=True)
        del unit
    with torch.no_grad():
        for k,v in tail.w.items():v.copy_(original[k])
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'queries':len(meta),'source_pairs':len(views),
      'order_projection':oinfo,'spans':spanreports,'finite_updates':reports,'seconds':time.monotonic()-start,
      'scope':'Previously frozen natural coherent/order prefixes, same lastquery target/suffix/token frequency/length. Current supervised gradient spans, full g/u/d matrices; not original training history.',
      'limits':['Historical discovery data, not newly independent language coverage.','Ordering changes difficulty despite matched targets; report actual loss and norm separately.','Reverse CE increase is not selective knowledge erasure.','Finite sample span overlap is not proof of a universal task subspace.']}
    save(out/'result.json',result);ledger('alpha_natural_gradient_spans',time.monotonic()-start)
    del tail,original;gc.collect();torch.cuda.empty_cache()
    print('NATURAL_ALPHA_COMPLETE',spanreports,flush=True)

if __name__=='__main__':main()
