"""Evaluate frozen binding rules on newly captured connected-graph holdouts."""
from collections import Counter
from rdc_binding_common import *
from rdc_binding_kernels import *

def main():
    import torch
    from rdc_law_native import parameter
    out=BASE/'confirmation'
    if (out/'result.json').exists():return
    assert (BASE/'capture/result.json').exists()
    start=time.monotonic();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    training=gzread(BASE/'natural_discovery.json.gz');rows=gzread(BASE/'natural_confirmation.json.gz')
    olds,oldq,olde,_,_,_=source_arrays(training);sources,q,e,labels,targets,lengths=source_arrays(rows)
    with np.load(BASE/'prediction/role_probe.npz') as z:coef=z['coefficients'];counts=z['training_class_counts'];majority=int(counts.argmax())
    oldroles=apply_roles(olds,coef);roles=apply_roles(sources,coef)
    left=feature_pack(sources,q,e,roles);right=feature_pack(olds,oldq,olde,oldroles)
    ks=pair_kernels(left,right);npz(out/'cross_kernels.npz',**{k:v.cpu().numpy() for k,v in ks.items()})
    frozen=read(BASE/'prediction/frozen.json');reports=[]
    for b,target in targets.items():
        yy=torch.tensor(target[:,:2560],device='cuda')
        w={k:parameter(f'model.layers.{b}.mlp.{name}_proj.weight') for k,name in [('g','gate'),('u','up'),('d','down')]}
        winner=frozen['selected'][str(b)]
        # Frozen winner plus all predeclared simple/control kernels at same df and decoder.
        for name in KERNELS:
            df=winner['df'];decoder=winner['decoder']
            with np.load(BASE/'prediction/banks'/f'b{b}_{name}_{df}.npz') as z:
                c=torch.tensor(z['coefficients'],device='cuda');center=torch.tensor(z['center'],device='cuda');scale=float(z['scale']);ix=z['train_indices']
            pred=ks[name][:,ix]/scale@c+center
            if decoder=='direct_mlp':yh=pred[:,:2560]
            elif decoder=='predicted_joint_native':yh=torch.nn.functional.linear(pred[:,5120:],w['d'])
            else:
                x=pred[:,2560:5120];yh=torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(x,w['g']))*torch.nn.functional.linear(x,w['u']),w['d'])
            err=(yh-yy).square().sum(1);den=(yy-center[:2560]).square().sum(1)
            cosine=torch.nn.functional.cosine_similarity(yh,yy)
            npz(out/f'b{b}_{name}.npz',prediction=yh.cpu().numpy(),squared_error=err.cpu().numpy(),baseline_squared_error=den.cpu().numpy(),cosine=cosine.cpu().numpy())
            for split in ('connected_test','matched_test'):
              for cohort in ('gum','ewt'):
                use=np.array([i for i,r in enumerate(rows) if r['split']==split and r['cohort']==cohort]);ratio=(err/den.clamp_min(1e-8)).cpu().numpy()
                reports.append({'block':b,'kernel':name,'df':df,'decoder':decoder,'frozen_winner':name==winner['kernel'],
                  'split':split,'cohort':cohort,'rows':len(use),'relative_mse':float(err[use].sum()/den[use].sum()),'cosine':float(cosine[use].mean()),
                  'cluster_ratio':clustered(ratio[use],[rows[i]['source_group'] for i in use])})
    role_eval=[]
    for cohort in ('gum','ewt'):
        y=np.concatenate([l[l>=0] for r,l in zip(rows,labels) if r['cohort']==cohort])
        yh=np.concatenate([p.argmax(1)[l>=0] for r,l,p in zip(rows,labels,roles) if r['cohort']==cohort])
        role_eval.append({'cohort':cohort,'tokens':len(y),'accuracy':float(np.mean(y==yh)),'training_majority_accuracy':float(np.mean(y==majority))})
    save(out/'result.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'rows':len(rows),'reports':reports,'role_evaluation':role_eval,
      'selected_before_capture':True,'prediction_freeze_sha':sha(BASE/'prediction/frozen.json'),
      'scope':'Head-sharing UD relation combinations were absent from all new strict fit rows. QA and CMRC excluded from this fit; semantic-program depth evaluated separately.',
      'seconds':time.monotonic()-start})
    ledger('connected_confirmation',time.monotonic()-start)
    print('BINDING_CONNECTED_CONFIRMATION',[r for r in reports if r['frozen_winner']],flush=True)

if __name__=='__main__':main()

