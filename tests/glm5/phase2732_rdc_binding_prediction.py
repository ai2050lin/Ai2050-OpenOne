"""Retrospective natural binding discovery, frozen before new confirmation inference."""
from rdc_binding_common import *
from rdc_binding_kernels import *

def main():
    import torch
    from rdc_law_predict import ridge_lambda
    from rdc_law_native import parameter
    out=BASE/'prediction'
    if (out/'frozen.json').exists():print('BINDING_PREDICTION_EXISTS',flush=True);return
    start=time.monotonic();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    rows=gzread(BASE/'natural_discovery.json.gz')
    sources,q,e,labels,targets,lengths=source_arrays(rows)
    roles=learn_roles(rows,sources,labels,out)
    pack=feature_pack(sources,q,e,roles)
    tick=time.monotonic();pilot=pair_kernels({k:v[:16] for k,v in pack.items()})
    pilot_sec=time.monotonic()-tick
    estimate=pilot_sec*(len(rows)/16)**2
    save(out/'pilot.json',{'seconds':pilot_sec,'rows':16,'quadratic_estimate_seconds':estimate,
        'device':'CUDA, fullFP32, noTF32','peak_cuda_bytes':torch.cuda.max_memory_allocated()})
    assert estimate<3600,'Review exact kernel cost before expanding.'
    ks=pair_kernels(pack)
    npz(out/'kernels.npz',**{k:v.cpu().numpy() for k,v in ks.items()})
    compressed(out/'source_features.json.gz',[{'sample_id':r['sample_id'],'length':int(n)} for r,n in zip(rows,lengths)])
    # Persist all source-role values in original token order; original H12 remains linked, not copied.
    npz(out/'source_roles.npz',lengths=lengths,roles=np.concatenate(roles))
    train=np.array([i for i,r in enumerate(rows) if r['split']=='train']);val=np.array([i for i,r in enumerate(rows) if r['split']=='validation'])
    test=np.array([i for i,r in enumerate(rows) if r['split']=='test'])
    ti=torch.tensor(train,device='cuda');group_counts=__import__('collections').Counter(rows[i]['source_group'] for i in train)
    wt=torch.tensor([1/group_counts[rows[i]['source_group']] for i in train],device='cuda')
    wt=wt/wt.sum();rootw=wt.sqrt();records=[];frozen={}
    for b,target in targets.items():
      yy=torch.tensor(target,device='cuda');center=(wt[:,None]*yy[ti]).sum(0);yc=yy[ti]-center
      w={key:parameter(f'model.layers.{b}.mlp.{name}_proj.weight') for key,name in [('g','gate'),('u','up'),('d','down')]}
      means=yy[ti,:2560].mean(0);base_error=(yy[:,:2560]-means).square().sum(1)
      block_records=[]
      for name,k in ks.items():
        # Same trace normalization and effective degrees-of-freedom budget for every candidate.
        scale=k[ti,ti].mean();kn=k/scale
        kt=kn[ti][:,ti]*rootw[:,None]*rootw[None,:]
        eig,vec=torch.linalg.eigh(kt.double())
        for df in (32,128):
            lam,actual=ridge_lambda(eig,df)
            inv=(vec/(eig+lam)[None,:])@vec.T
            coeff=rootw[:,None]*(inv.float()@(rootw[:,None]*yc))
            pred=kn[:,ti]@coeff+center
            xx=pred[:,2560:5120]
            decoded={'direct_mlp':pred[:,:2560],
              'predicted_x_native':torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(xx,w['g']))*torch.nn.functional.linear(xx,w['u']),w['d']),
              'predicted_joint_native':torch.nn.functional.linear(pred[:,5120:],w['d'])}
            for decoder,yhat in decoded.items():
                err=(yhat-yy[:,:2560]).square().sum(1)
                cosine=torch.nn.functional.cosine_similarity(yhat,yy[:,:2560])
                per=(err/base_error.clamp_min(1e-8)).cpu().numpy()
                rec={'block':b,'kernel':name,'df':df,'actual_df':actual,'lambda':lam,'decoder':decoder,
                  'validation_relative_mse':float(err[val].sum()/base_error[val].sum()),
                  'test_relative_mse':float(err[test].sum()/base_error[test].sum()),
                  'test_cosine':float(cosine[test].mean()),
                  'test_cluster_relative_error':clustered(per[test],[rows[i]['source_group'] for i in test])}
                block_records.append(rec)
                npz(out/'predictions'/f'b{b}_{name}_{df}_{decoder}.npz',prediction=yhat.cpu().numpy(),relative_error=per,cosine=cosine.cpu().numpy())
            # Full coefficients preserved for every kernel/df, no outcome-based rank or coordinate filtering.
            npz(out/'banks'/f'b{b}_{name}_{df}.npz',coefficients=coeff.cpu().numpy(),center=center.cpu().numpy(),scale=np.array(float(scale)),train_indices=train)
      winner=min(block_records,key=lambda r:r['validation_relative_mse'])
      frozen[str(b)]=winner
      records.extend(block_records);del yy,w
    save(out/'results.json',records)
    selected={'timestamp':stamp(),'source':snapshot(Path(__file__)),'selected':frozen,
      'material_sha':sha(BASE/'natural_discovery.json.gz'),'kernels_sha':sha(out/'kernels.npz'),
      'selection':'Validation relative MSE only; test is old public discovery data, not new independent confirmation.',
      'online_inputs':'Current prefix allH12source coordinates, currentH12, currenttoken embedding, source positions, frozen H12-to-coarse-role scores.',
      'excluded_inputs':'Gold UD roles/head edges, future source tokens, target block states, future answer.',
      'limitations':'Finite pair moments may still lose information. Soft coarse role scores are not a full semantic binding graph. State gain alone is not output/behavior gain.',
      'seconds':time.monotonic()-start,'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
    save(out/'frozen.json',selected);ledger('binding_prediction',time.monotonic()-start)
    print('BINDING_PREDICTION_FROZEN',selected,flush=True)

if __name__=='__main__':main()
