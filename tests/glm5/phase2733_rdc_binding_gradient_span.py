"""Attachment Alpha/Gamma: finite current-gradient spans and real norm-matched updates."""
import gc
from collections import defaultdict
from rdc_binding_common import *
from rdc_binding_gradients import *

def main():
    import torch
    from rdc_law_native import Tail
    out=BASE/'gradient_span'
    if (out/'result.json').exists():return
    assert (BASE/'capture/result.json').exists()
    start=time.monotonic();rows=gzread(BASE/'program_material.json.gz');xa,ra,ta=program_arrays(rows)
    tail=Tail();x=torch.tensor(xa,device='cuda');residual=torch.tensor(ra,device='cuda');targets=torch.tensor(ta,device='cuda')
    f,base=collect(tail,x,residual,targets);gram=factor_gram(f)['total']
    norm=gram.diag().clamp_min(1e-30).sqrt();cos=gram/norm[:,None]/norm[None,:]
    npz(out/'all_program_gradient_factors.npz',**{k:v.float().cpu().numpy() for k,v in f.items()},gram=gram.cpu().numpy(),cosine=cos.cpu().numpy(),**base)
    # Whole training-case spans, not high-eigenvalue subspaces; all three parameter matrices participate.
    train={rep:np.array([i for i,r in enumerate(rows) if r['split']=='train' and r['representation']==rep]) for rep in ('en','zh','python','en_reordered')}
    p=len(rows);directions={};meta={}
    for rep,ix in train.items():
        c=torch.zeros(p,device='cuda',dtype=torch.float64);c[ix]=1/len(ix);directions['mean_'+rep]=c
    g_en=gram@directions['mean_en'];ix=train['python']
    pc,info=projection_coefficients(gram[ix][:,ix],g_en[ix])
    c=torch.zeros(p,device='cuda',dtype=torch.float64);c[ix]=pc
    directions['en_projected_to_code_span']=c;directions['en_residual_from_code_span']=directions['mean_en']-c
    directions['negative_projected_control']=-c
    npz(out/'direction_coefficients.npz',**{k:v.cpu().numpy() for k,v in directions.items()})
    meta['code_projection']=info
    residual_dot=gram[ix]@(directions['mean_en']-c)
    meta['projection_residual_inner_product_max']=float(residual_dot.abs().max())
    # Control semantic-case matching by target digit and family, not unrelated arbitrary pairs.
    paired=[]
    lookup={(r['source_group'],r['representation']):i for i,r in enumerate(rows)}
    for rep in ('zh','python','en_reordered'):
        matches=[];controls=[]
        for i,r in enumerate(rows):
            if r['representation']!='en':continue
            j=lookup[(r['source_group'],rep)];matches.append(float(cos[i,j]))
            other=[k for k,s in enumerate(rows) if s['representation']==rep and s['source_group']!=r['source_group'] and s['target']==r['target'] and s['family']==r['family'] and s['split']==r['split']]
            if other:controls.append({'group':r['source_group'],'matched':float(cos[i,j]),'same_target_family_mismatched':float(cos[i,other].mean())})
        paired.append({'representation':rep,'pairs':len(matches),'mean_semantic_pair_cosine':float(np.mean(matches)),
          'controlled_pairs':len(controls),'paired_advantage':clustered([a['matched']-a['same_target_family_mismatched'] for a in controls],[a['group'] for a in controls])})
    original={k:v.detach().clone() for k,v in tail.w.items()};reports=[];direction_manifest=[]
    # Natural frozen corpus collateral does not share the controlled program labels.
    natural=gzread(BASE/'natural_discovery.json.gz')
    collateral=sorted([r for r in natural if r['split']=='test'],key=lambda r:ranked('collateral/'+r['sample_id']))[:48]
    cx=[];cr=[];ct=[]
    for r in collateral:
        pos=r['anchors'][-1]
        with np.load(source_path(r)) as z:cx.append(unbits(z['x'])[pos]);cr.append(unbits(z['residual'])[pos])
        ct.append(r['prompt_ids'][pos+1])
    cx=torch.tensor(np.array(cx),device='cuda');cr=torch.tensor(np.array(cr),device='cuda');ct=torch.tensor(ct,device='cuda')
    _,natural_base=collect(tail,cx,cr,ct)
    for name in list(directions)+['random_control']:
        if name=='random_control':
            gen=torch.Generator(device='cuda').manual_seed(2733)
            dense={k:torch.randint(0,2,v.shape,device='cuda',generator=gen,dtype=torch.int8).float().mul_(2).sub_(1) for k,v in tail.w.items()}
        else:dense={k:v.float() for k,v in combination(f,directions[name]).items()}
        unit,dnorm=normalized(dense);del dense
        direction_manifest.append({'name':name,'unnormalized_norm':dnorm})
        if name in ('en_projected_to_code_span','mean_en'):
            npz(out/'directions'/f'{name}.npz',**{k:v.cpu().numpy() for k,v in unit.items()})
        for step_norm in (.02,.10):
            with torch.no_grad():
                for k,v in tail.w.items():v.copy_(original[k]);v.add_(unit[k],alpha=-step_norm)
            _,current=collect(tail,x,residual,targets)
            _,collateral_stats=collect(tail,cx,cr,ct)
            delta=current['loss']-base['loss'];cdelta=collateral_stats['loss']-natural_base['loss']
            stats=[]
            for split in ('train','validation','test','depth_test'):
              for rep in train:
                selected=np.array([i for i,r in enumerate(rows) if r['split']==split and r['representation']==rep])
                stats.append({'split':split,'representation':rep,'n':len(selected),'loss_delta':float(delta[selected].mean()),
                  'initial_first_argmax_accuracy':float(np.mean(base['argmax'][selected]==ta[selected])),
                  'current_first_argmax_accuracy':float(np.mean(current['argmax'][selected]==ta[selected])),
                  'cluster':clustered(delta[selected],[rows[i]['source_group'] for i in selected])})
            report={'direction':name,'step_parameter_norm':step_norm,'stats':stats,
              'natural_collateral_loss_delta':float(cdelta.mean()),'natural_collateral_cluster':clustered(cdelta,[r['source_group'] for r in collateral])}
            reports.append(report);npz(out/'updates'/f'{name}_{step_norm}.npz',**current,natural_loss_delta=cdelta)
            save(out/'progress.json',{'completed':reports,'directions':direction_manifest})
            print('GRADIENT_SPAN_UPDATE',name,step_norm,'natural_delta',report['natural_collateral_loss_delta'],flush=True)
        del unit
        guard(400*1024**2)
    with torch.no_grad():
        for k,v in tail.w.items():v.copy_(original[k])
    save(out/'result.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'program_rows':len(rows),'semantic_cases':len(lookup)//4,
      'projection':meta,'paired_gradient_comparisons':paired,'directions':direction_manifest,'updates':reports,'seconds':time.monotonic()-start,
      'scope':'Current supervised first-answer-token CE gradients of original full lastMLP, on cached causal native prefixes; real transient parameter steps in FP32.',
      'limits':['Not original training-history reconstruction.','EN/ZH/Python are text token representations in one model, not three physical modalities.','Finite training-sample gradient spans need not equal a task/knowledge subspace.','First-token cached-prefix scoring is not autonomous generation; separate deployment required.','Same-target family control is unavailable in some sparse strata and counts are explicitly reported.']})
    ledger('alpha_gamma_gradient_span',time.monotonic()-start)
    del tail,original,f,gram,cos;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
