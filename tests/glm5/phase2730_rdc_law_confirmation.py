"""Frozen early predictors and actual trained matrices on untouched natural combinations."""
import gc
from rdc_law_common import *
from rdc_law_predict import Bank,normalized_features,kernel_torch
from rdc_law_native import Tail
from phase2729_rdc_law_prediction import load_arrays
from phase2729_rdc_law_training import evaluate


def main():
    import torch
    out=BASE/'confirmation';start=time.monotonic()
    if (out/'result.json').exists():return
    assert (BASE/'capture/confirmation/result.json').exists()
    frozen=read(BASE/'prediction/frozen.json');guard(450*1024**2)
    meta,raw,target,residual,post=load_arrays('confirmation')
    with np.load(BASE/'prediction/feature_rulers.npz') as z:
        rulers={k:{t:z[k+'_'+t] for t in ('mean','scale')} for k in ('q','embedding','history','routed')}
    f,_=normalized_features(raw,rulers=rulers);device='cuda:0'
    ft={k:torch.as_tensor(v,device=device,dtype=torch.int64 if k=='class' else torch.float64) for k,v in f.items()}
    with np.load(BASE/'prediction/training_features.npz') as z:
        tt={k:torch.as_tensor(z[k],device=device,dtype=torch.int64 if k=='class' else torch.float64) for k in f}
    tail=Tail();rr=torch.as_tensor(residual,device=device);post=torch.as_tensor(post,device=device);lp=[]
    with torch.no_grad():
        for at in range(0,len(meta),16):lp.append((post[at:at+16]@tail.head.T).log_softmax(-1))
    lp=torch.cat(lp);reports=[];predictions={}
    group_masks={'all':np.ones(len(meta),bool)}
    for c in ('gum','ewt','cmrc'):group_masks[c]=np.array([r['cohort']==c for r in meta])
    group_masks['held_relation_pair']=np.array([bool(r['held_relation_combinations']) for r in meta])
    group_masks['nonpair_English']=np.array([r['language']=='en' and not r['held_relation_combinations'] for r in meta])
    from rdc_law_native import parameter,readout_actions
    for b in (16,35):
        w=tail.w if b==35 else {k:parameter(f'model.layers.{b}.mlp.{v}_proj.weight') for k,v in [('g','gate'),('u','up'),('d','down')]}
        true=torch.as_tensor(target[b]['mlp'],device=device)
        for decoder in ('fixed_early','direct_mlp','predicted_x_native','product_of_predicted_factors','predicted_joint_product'):
            bank=Bank(BASE/'prediction/banks'/f'L{b}_{decoder}')
            with torch.no_grad():m=bank.predict(kernel_torch(bank.kernel,ft,tt),w)
            rel=((m-true).square().mean(-1)/true.square().mean(-1).clamp_min(1e-20)).cpu().numpy()
            packet={'mlp':m.cpu().numpy(),'relative_MSE':rel}
            if b==35:
                values={k:[] for k in ('KL','argmax_agreement','Fisher_endpoint_half_variance','postnorm_relative_MSE')}
                for at in range(0,len(meta),16):
                    r=rr[at:at+16]+m[at:at+16];n=tail.norm*r*torch.rsqrt(r.square().mean(-1,keepdim=True)+tail.eps)
                    predlp=(n@tail.head.T).log_softmax(-1);ref=lp[at:at+16]
                    vals={'KL':(ref.exp()*(ref-predlp)).sum(-1),'argmax_agreement':(ref.argmax(-1)==predlp.argmax(-1)).float(),
                        'postnorm_relative_MSE':(n-post[at:at+16]).square().mean(-1)/post[at:at+16].square().mean(-1).clamp_min(1e-20)}
                    _,var=readout_actions(rr[at:at+16]+true[at:at+16],m[at:at+16]-true[at:at+16],tail.norm,tail.head,tail.eps,ref.exp())
                    vals['Fisher_endpoint_half_variance']=var*.5
                    for k,v in vals.items():values[k].append(v.cpu().numpy())
                packet.update({k:np.concatenate(v) for k,v in values.items()})
            npz(out/'predictions'/f'L{b}_{decoder}.npz',**packet)
            predictions[(b,decoder)]=packet
            for name,mask in group_masks.items():
                ix=np.flatnonzero(mask)
                reports.append({'block':b,'decoder':decoder,'kernel':bank.kernel,'df':bank.info['df'],'group':name,'queries':len(ix),
                    'metrics':{k:float(v[ix].mean()) for k,v in packet.items() if k!='mlp'},
                    'relative_MSE_cluster':clustered(rel[ix],[meta[i]['source_group'] for i in ix])})
            del bank,m
        if b!=35:del w
    gains=[]
    for b in (16,35):
        win=frozen['winners'][str(b)]['decoder'];baseline=predictions[(b,'fixed_early')];actual=predictions[(b,win)]
        for name,mask in group_masks.items():
            ix=np.flatnonzero(mask)
            for metric in ('relative_MSE','KL'):
                if metric not in actual:continue
                gains.append({'block':b,'winner':win,'metric':metric,'group':name,
                    'gain':clustered(baseline[metric][ix]-actual[metric][ix],[meta[i]['source_group'] for i in ix])})
    # No confirmation target or outcome chooses an update or predictor.
    x=torch.as_tensor(target[35]['x'],device=device);targets=torch.tensor([r['target_id'] for r in meta],device=device)
    initial_lp=torch.cat([tail.forward(x[at:at+16],rr[at:at+16])['logprobs'] for at in range(0,len(x),16)])
    cohorts=[r['cohort'] for r in meta];baseline=evaluate(tail,x,rr,targets,initial_lp,cohorts)
    npz(out/'training/initial_full_panel.npz',**baseline);training=[]
    original={k:v.clone() for k,v in tail.w.items()}
    # Prospective directional loss predictions use known final parameter deltas and original gradients.
    factors={k:[] for k in ('x','a','s','bg','bu')}
    for at in range(0,len(x),16):
        z=tail.forward(x[at:at+16],rr[at:at+16],targets[at:at+16],True)
        for k,v in z['factors'].items():factors[k].append(v)
    factors={k:torch.cat(v) for k,v in factors.items()};del z
    for run in read(BASE/'formation/trajectories/result.json')['runs']:
        with np.load(BASE/run['delta_path']) as z:delta={k:torch.as_tensor(z[k],device=device) for k in original}
        prospective=((factors['bg']*(x@delta['g'].T)).sum(-1)+(factors['bu']*(x@delta['u'].T)).sum(-1)+(factors['s']*(factors['a']@delta['d'].T)).sum(-1)).cpu().numpy()
        npz(out/'training/prospective'/f"{run['name']}.npz",initial_gradient_linearized_loss_delta=prospective)
        with torch.no_grad():
            for k in tail.w:tail.w[k].copy_(original[k]+delta[k])
        packet=evaluate(tail,x,rr,targets,initial_lp,cohorts);packet['linearized_loss_delta']=prospective
        npz(out/'training'/f"{run['name']}_full_panel.npz",**packet)
        lossdelta=packet['loss']-baseline['loss']
        for name,mask in group_masks.items():
            ix=np.flatnonzero(mask)
            training.append({'run':run['name'],'group':name,'queries':len(ix),'initial_loss':float(baseline['loss'][ix].mean()),
                'current_loss':float(packet['loss'][ix].mean()),'loss_delta':clustered(lossdelta[ix],[meta[i]['source_group'] for i in ix]),
                'linearized_relative_SSE':float(np.mean((lossdelta[ix]-prospective[ix])**2)/max(np.mean(lossdelta[ix]**2),1e-30)),
                'scope':'Actual fixed-upstream full-native-MLP FP32 continuation. Prospective initial-gradient response to known64step delta, not original pretraining or a gradient-free answer forecast.'})
        del delta,packet
    with torch.no_grad():
        for k in tail.w:tail.w[k].copy_(original[k])
    compressed(out/'query_catalog.json.gz',meta)
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'frozen_predictors_sha':sha(BASE/'prediction/frozen.json'),
        'source_queries':len(meta),'unique_sources':len({r['source_group'] for r in meta}),'predictions':reports,'frozen_winner_gains':gains,'training':training,
        'new_selection_on_confirmation':False,'seconds':time.monotonic()-start,
        'limits':['Relation co-occurrence is retrospective annotation, not established semantic operator composition.','Final-block KL includes actual residual for evaluation only; it is a local compiler score.','Endpoint Fisher term is an approximation and native BF16 arithmetic adds a separately measured floor.','Training targets are observed labels; this is not future target prediction without labels.']}
    save(out/'result.json',result);ledger('frozen_confirmation_and_training',result['seconds'])
    del tail,lp,initial_lp,original;gc.collect();torch.cuda.empty_cache()
    print('LAW_CONFIRMATION_COMPLETE',len(meta),gains,flush=True)


if __name__=='__main__':main()
