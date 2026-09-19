"""Exact full-vocabulary content/format gradient decomposition and prospective tests."""
import gc
from rdc_binding_common import *
from rdc_binding_gradients import *

def arrays(rows,future=False):
    if not future:return program_arrays(rows)
    x=[];r=[]
    for row in rows:
        with np.load(BASE/'format_content/prospective_fields'/f'{row["sample_id"]}.npz') as z:
            x.append(unbits(z['last_x'])[-1]);r.append(unbits(z['last_residual'])[-1])
    return np.array(x),np.array(r),np.array([r['target_ids'][0] for r in rows])

def collect_parts(tail,x,r,targets,digits,factors=True):
    import torch
    records={k:[] for k in ('full_loss','content_loss','format_loss','digit_mass','argmax','conditional_digit_argmax')}
    ff={part:{k:[] for k in ('x','a','s','bg','bu')} for part in ('content','format','full')}
    with torch.no_grad():
      for i in range(0,len(x),8):
        xx=x[i:i+8];tt=targets[i:i+8];z=tail.forward(xx,r[i:i+8],tt,False)
        # The model/logit forward stays FP32. Normalize probabilities in FP64 and
        # compute conditional CE directly: subtracting two FP32 losses caused a
        # reproducible native-autograd cancellation failure on confident rows.
        lp=z['logits'].double().log_softmax(-1);dlp=z['logits'][:,digits].double().log_softmax(-1)
        logmass=torch.logsumexp(lp[:,digits],1);mass=logmass.exp();conditional=dlp.exp()
        digit_ids=torch.tensor(digits,device=x.device);local=(tt[:,None]==digit_ids).long().argmax(-1)
        assert torch.all(digit_ids[local]==tt)
        selected=digit_ids[conditional.argmax(-1)];batch_ids=torch.arange(len(xx),device=x.device)
        values={'full_loss':-lp[batch_ids,tt],'content_loss':-dlp[batch_ids,local],'format_loss':-logmass,
          'digit_mass':mass,'argmax':z['argmax'],'conditional_digit_argmax':selected}
        for key,v in values.items():records[key].append(v.cpu().numpy())
        if factors:
            pc=torch.zeros_like(lp);pc[:,digits]=conditional
            ec=pc.clone();ec[torch.arange(len(xx),device=x.device),tt]-=1
            ef=lp.exp()-pc
            for name,error in [('content',ec),('format',ef),('full',ec+ef)]:
                f=error_factors(tail,z,xx,error.float())
                for k,v in f.items():ff[name][k].append(v.double())
    stats={k:np.concatenate(v) for k,v in records.items()}
    return ({part:{k:torch.cat(v) for k,v in f.items()} for part,f in ff.items()} if factors else None),stats

def describe(rows,stats,initial=None):
    reports=[]
    for split,rep in sorted({(r['split'],r['representation']) for r in rows}):
        ix=np.array([i for i,r in enumerate(rows) if r['split']==split and r['representation']==rep]);targets=np.array([r['target_ids'][0] for r in rows])
        rr={'split':split,'representation':rep,'rows':len(ix)}
        for key in ('full_loss','content_loss','format_loss','digit_mass'):rr[key]=float(stats[key][ix].mean())
        rr['first_token_accuracy']=float(np.mean(stats['argmax'][ix]==targets[ix]))
        rr['conditional_digit_accuracy']=float(np.mean(stats['conditional_digit_argmax'][ix]==targets[ix]))
        if initial:
            for key in ('full_loss','content_loss','format_loss'):
                delta=stats[key][ix]-initial[key][ix];rr[key+'_delta']=float(delta.mean())
                rr[key+'_cluster']=clustered(delta,[rows[i]['source_group'] for i in ix])
        reports.append(rr)
    return reports

def paired(rows,cos):
    result=[];lookup={(r['source_group'],r['representation']):i for i,r in enumerate(rows)}
    for rep in ('zh','python','en_reordered'):
        delta=[];groups=[];matches=[]
        for i,r in enumerate(rows):
            if r['representation']!='en':continue
            j=lookup[(r['source_group'],rep)];matches.append(float(cos[i,j]))
            other=[k for k,s in enumerate(rows) if s['representation']==rep and s['source_group']!=r['source_group'] and s['target']==r['target'] and s['family']==r['family'] and s['split']==r['split']]
            if other:delta.append(float(cos[i,j]-cos[i,other].mean()));groups.append(r['source_group'])
        result.append({'representation':rep,'pairs':len(matches),'mean_cosine':float(np.mean(matches)),
          'controlled_pairs':len(delta),'same_target_family_split_advantage':clustered(delta,groups)})
    return result

def main():
    import torch
    from rdc_law_native import Tail
    from transformers import AutoTokenizer
    out=BASE/'format_content'
    if (out/'decomposition_result.json').exists():return
    assert (out/'native_capture_result.json').exists()
    protocol=read(out/'protocol.json');start=time.monotonic();guard(950*1024**2)
    rows=gzread(BASE/'program_material.json.gz');future=gzread(out/'prospective_material.json.gz')
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True)
    digits=[tok(str(i),add_special_tokens=False)['input_ids'][0] for i in range(1,9)]
    tail=Tail();xx,rr,tt=arrays(rows);x=torch.tensor(xx,device='cuda');r=torch.tensor(rr,device='cuda');targets=torch.tensor(tt,device='cuda')
    f,base=collect_parts(tail,x,r,targets,digits);npz(out/'initial_decomposition.npz',**base)
    gram={part:factor_gram(factor)['total'] for part,factor in f.items()}
    cross=factor_gram(f['content'],f['format'])['total']
    reconstruction=gram['content']+gram['format']+cross+cross.T
    residual=(reconstruction-gram['full']).abs().max()/gram['full'].abs().max().clamp_min(1e-30)
    assert float(residual)<1e-6
    loss_error=np.max(abs(base['full_loss']-base['content_loss']-base['format_loss']))
    assert loss_error<1e-5
    # Same execution shape as collected factors: first full batch of eight actual
    # examples, including confident examples that exposed the failed formulation.
    audits={}
    for part in ('content','format'):
        for v in tail.w.values():v.requires_grad_(True)
        z=tail.forward(x[:8],r[:8],targets[:8]);lp=z['logits'].double().log_softmax(-1)
        dlp=z['logits'][:,digits].double().log_softmax(-1);logmass=torch.logsumexp(lp[:,digits],1)
        digit_ids=torch.tensor(digits,device=x.device);local=(targets[:8,None]==digit_ids).long().argmax(-1)
        loss=-dlp[torch.arange(8,device=x.device),local].mean() if part=='content' else -logmass.mean();loss.backward()
        expected=dense_gradient({k:v[:8].float() for k,v in f[part].items()})
        audits[part]={k:float((v.grad-expected[k]).abs().max()/v.grad.abs().max().clamp_min(1e-12)) for k,v in tail.w.items()}
        assert max(audits[part].values())<5e-5,audits
        for v in tail.w.values():v.grad=None;v.requires_grad_(False)
        del z,loss,expected,lp,dlp,logmass
    npz(out/'all_parameter_gram_decomposition.npz',**{k:v.cpu().numpy() for k,v in gram.items()},content_format_cross=cross.cpu().numpy())
    for part in ('content','format'):
        npz(out/f'{part}_gradient_factors.npz',**{k:v.float().cpu().numpy() for k,v in f[part].items()})
    comparisons={}
    for part,g in gram.items():
        n=g.diag().clamp_min(1e-30).sqrt();cos=g/n[:,None]/n[None,:];comparisons[part]=paired(rows,cos)
    energy=[]
    for split,rep in sorted({(r['split'],r['representation']) for r in rows}):
        ix=[i for i,r in enumerate(rows) if r['split']==split and r['representation']==rep]
        den=gram['full'].diag()[ix].sum().clamp_min(1e-30)
        energy.append({'split':split,'representation':rep,'rows':len(ix),
          'content_squared_norm_ratio':float(gram['content'].diag()[ix].sum()/den),
          'format_squared_norm_ratio':float(gram['format'].diag()[ix].sum()/den),
          'signed_cross_ratio':float(2*cross.diag()[ix].sum()/den),
          'scope':'Three signed/nonorthogonal terms sum to1. Not independent semantic percentages.'})
    train_en=np.array([i for i,r in enumerate(rows) if r['split']=='train' and r['representation']=='en'])
    train_code=np.array([i for i,r in enumerate(rows) if r['split']=='train' and r['representation']=='python'])
    mean=torch.zeros(len(rows),device='cuda',dtype=torch.float64);mean[train_en]=1/len(train_en)
    coefficients,projection=projection_coefficients(gram['content'][train_code][:,train_code],(gram['content']@mean)[train_code])
    projected=torch.zeros_like(mean);projected[train_code]=coefficients
    npz(out/'direction_coefficients.npz',content_projection=projected.cpu().numpy(),format_mean=mean.cpu().numpy())
    xa,ra,ta=arrays(future,True);fx=torch.tensor(xa,device='cuda');fr=torch.tensor(ra,device='cuda');ft=torch.tensor(ta,device='cuda')
    _,fb=collect_parts(tail,fx,fr,ft,digits,False);npz(out/'prospective_initial.npz',**fb)
    original={k:v.detach().clone() for k,v in tail.w.items()};updates=[];directions=[]
    for name,part,coef in [('content_EN_projected_to_code_content_span','content',projected),('mean_EN_format','format',mean),('random_control',None,None)]:
        if part:dense={k:v.float() for k,v in combination(f[part],coef).items()}
        else:
            gen=torch.Generator(device='cuda').manual_seed(2735)
            dense={k:torch.randint(0,2,v.shape,device='cuda',generator=gen,dtype=torch.int8).float().mul_(2).sub_(1) for k,v in tail.w.items()}
        unit,norm=normalized(dense);del dense;directions.append({'name':name,'unnormalized_norm':norm})
        if part:npz(out/'directions'/f'{name}.npz',**{k:v.cpu().numpy() for k,v in unit.items()})
        for step in protocol['step_norms']:
            with torch.no_grad():
                for k,v in tail.w.items():v.copy_(original[k]);v.add_(unit[k],alpha=-step)
            _,oldstats=collect_parts(tail,x,r,targets,digits,False);_,newstats=collect_parts(tail,fx,fr,ft,digits,False)
            npz(out/'updates'/f'{name}_{step}_old.npz',**oldstats);npz(out/'updates'/f'{name}_{step}_prospective.npz',**newstats)
            updates.append({'direction':name,'parameter_norm':step,'old':describe(rows,oldstats,base),'prospective':describe(future,newstats,fb)})
            print('CONTENT_FORMAT_UPDATE',name,step,flush=True)
        del unit
    with torch.no_grad():
        for k,v in tail.w.items():v.copy_(original[k])
    with np.load(out/'numerical_recovery/failed_FP32_initial_decomposition.npz') as prior:
        precision_score_change={k:float(np.max(np.abs(base[k]-prior[k]))) for k in ('full_loss','content_loss','format_loss','digit_mass')}
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'initial':describe(rows,base),'prospective_initial':describe(future,fb),
      'loss_identity_max_error':float(loss_error),'full_parameter_gram_identity_relative_max_error':float(residual),
      'autograd_checks':audits,'gradient_pair_controls':comparisons,'gradient_nonorthogonal_terms':energy,
      'autograd_audit_shape':{'batch':8,'samples':'First8 fixed original program rows, all three full matrices for each objective'},
      'numerical_precision':'Native-valued FP32 MLP/logit arithmetic; FP64 probability normalization and direct conditional CE, with FP32 logit adjoints/native parameter gradients. Not an all-FP64 model or BF16 forward.',
      'maximum_score_change_from_failed_FP32_evaluation':precision_score_change,
      'numerical_recovery':'numerical_recovery/precision_diagnostic.json; failed outputs and unchanged frozen protocol remain preserved.',
      'projection':projection,'directions':directions,'updates':updates,'seconds':time.monotonic()-start,
      'scope':'Exact loss/gradient identity and train-only finite-span parameter directions. Conditional candidate accuracy is separately labeled and is not natural first-token behavior.'}
    save(out/'decomposition_result.json',result);ledger('format_content_decomposition',result['seconds'])
    del tail,original,f,gram;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
