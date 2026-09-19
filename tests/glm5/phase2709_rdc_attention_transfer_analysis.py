"""Apply every frozen N predictor to new words, operations, forms and decode steps."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,gc,shutil
from scipy.special import softmax,logsumexp
from rdc_conditional_common import *
from phase2708_rdc_cached_attention_prediction import rotate,GROUPS,MAX_PAST
from phase2704_rdc_predictive_gates import errors
OUT=CAMPAIGN/'o_generalization';N=CAMPAIGN/'n_cached_attention';M=CAMPAIGN/'m_order'


def prepare():
    frozen=read(OUT/'frozen_predictors.json');assert frozen
    for rel,digest in frozen['files'].items():assert sha(N/rel)==digest,(rel,'frozen predictor artifact changed')
    immutable(OUT/'analysis_protocol.json',{'phase':2709,'source_sha':sha(Path(__file__)),'frozen_predictors_sha':sha(OUT/'frozen_predictors.json'),
      'prediction':'Apply all24 declared N routes/controls without any O fitting, calibration, ridge choice, headnorm choice or outcome filtering. N mean family-language-order has no matching requested order in O and falls back to N global mean. Add2 explicitly frozen-prior family-language means across N orders; unseen operation families use global mean. All hyperparameters and prior sufficient data are fixed before any O forecast result.',
      'inputs':'Actual current H12 and pastL23 K/V0..s-1 only; current token Q/K/V, P, head/output, emitted next token and external constraints are heldout targets. Early EOS gives fewer actual selected states; no generated placeholders. All native past cache coordinates, position convention and512zero-padding identical to N.',
      'metrics':'Every native2560 attention-output coordinate: MSE, training-N-energy bins, percoordinate/perstate errors, byfamily/language/form/entity/generation_step. Every head and every source probability used for factor-route attention KL/MSE; nativeBF16 P normalized perhead solely for KL, log floor1e-300. Full-vocab native entropy computed from all savedlogits; no predicted-token or whole-answer-ability claim.',
      'serialization':'Before O scoring, compare four prior N test states through representative frozen exported FP32-alpha models with original N saved predictions. Record differences and require MSE<1e-5 so export drift cannot silently dominate. No refitting on O.',
      'retention':'All native O fields retained for client, full attention-output prediction vectors and percoordinate errors for all26routes retained. Forecast P computed in full for metrics but not duplicated across every model; originalK/V plus frozenmodels allow exact recomputation. This is no hidden-coordinate reduction.',
      'resources':'CPU only, maximum1200analysisseconds, O total<=3.2GiB andcampaign<=30GiB; retain8GiBfree floor.',
      'limits':['New entity strings are heldout from current fit materials, not guaranteed absent from pretraining.','Crossedforms change wording and source/instruction order together; all new states are from free generation, not Result-label boundaries.','Native cache includes prior full-model computation; not a standalone decoder.','Tokens within prefixes and relatedforms/entities are not independent samples.','Generalization of numericalattention does not establish semantic correctness or a unique native algorithm.']})


def objects_for(rows,root):
    objects=[];h=[];target=[];native_heads=[];entropy=[];qkv=[]
    for row in rows:
        with np.load(root/f'fields/{row["sample_id"]}.npz') as z:
            q,k,v,p,head,a=[unbits(z['L23_'+part]).astype(np.float64) for part in ('q','k','v','p','head_output','attention_out')]
            h.append(unbits(z['h_c'][12]));target.append(a);native_heads.append(head)
            if 'logits' in z:
                logits=unbits(z['logits']).astype(np.float64);lp=logits-logsumexp(logits);entropy.append(float(-np.exp(lp)@lp))
            else:entropy.append(None)
        pos=row['query_position'];assert row['generation_step']>=1 and pos<=MAX_PAST and k.shape==v.shape==(8,pos+1,128)
        qkv.append(np.concatenate((rotate(q,pos,True).reshape(-1),rotate(k[:,-1],pos,True).reshape(-1),v[:,-1].reshape(-1))))
        objects.append({'past_k':k[:,:-1].astype(np.float32),'past_v':v[:,:-1].astype(np.float32),'position':pos,'native_p':p,
          'groups':np.array([GROUPS.index(g) for g in row['source_labels']],int)})
    return objects,np.stack(h).astype(np.float64),np.stack(target),np.stack(native_heads),np.stack(qkv),entropy


def cross_cache(objects,train_objects,scales):
    grams=[]
    for part,scale in zip(('past_k','past_v'),scales):
        gram=np.zeros((len(objects),len(train_objects)),np.float64)
        for start in range(0,MAX_PAST,32):
            a=np.zeros((len(objects),32,1024),np.float64);b=np.zeros((len(train_objects),32,1024),np.float64)
            for dest,seq in ((a,objects),(b,train_objects)):
                for i,o in enumerate(seq):
                    x=o[part][:,start:start+32].transpose(1,0,2).reshape(-1,1024);dest[i,:len(x)]=x
            gram+=a.reshape(len(a),-1)@b.reshape(len(b),-1).T
        grams.append(gram/scale**2);print('O_CROSS_CACHE',part,len(objects),flush=True)
    return grams


def factor_output(factors,objects,wo,norms=None):
    heads=[];kl=[];pmse=[]
    for row,o in zip(factors,objects):
        q=row[:4096].reshape(32,128).astype(np.float64);k=row[4096:5120].reshape(8,128).astype(np.float64);v=row[5120:].reshape(8,128).astype(np.float64)
        if norms is not None:
            q=q*norms[0][:,None]/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-12)
            k=k*norms[1][:,None]/np.maximum(np.linalg.norm(k,axis=1,keepdims=True),1e-12)
        q=rotate(q,o['position']);k=rotate(k,o['position'])
        kk=np.concatenate((o['past_k'].astype(np.float64),k[:,None]),1)[np.arange(32)//4]
        vv=np.concatenate((o['past_v'].astype(np.float64),v[:,None]),1)[np.arange(32)//4]
        p=softmax(np.einsum('hd,hsd->hs',q,kk)/np.sqrt(128),axis=1)
        heads.append(np.einsum('hs,hsd->hd',p,vv).reshape(-1))
        native=o['native_p'];native=native/native.sum(1,keepdims=True)
        divergence=np.sum(native*(np.log(np.maximum(native,1e-300))-np.log(np.maximum(p,1e-300))),1)
        kl.append(divergence);pmse.append(np.mean((p-native)**2,1))
    return np.stack(heads)@wo.T,{'attention_KL_by_head':np.stack(kl).astype(np.float32),'attention_probability_MSE_by_head':np.stack(pmse).astype(np.float32)}


def summary(target,prediction,energy,rows):
    report,arrays=errors(target,prediction,energy);per=np.mean((target-prediction)**2,1)
    for key in ('family','language','form','unit','generation_step'):
        report['by_'+key]={str(v):{'n':int(mask.sum()),'mse':float(per[mask].mean()),
          'error_energy_ratio':float(np.sum((target[mask]-prediction[mask])**2)/max(np.sum(target[mask]**2),1e-30))}
          for v in sorted({r[key] for r in rows},key=str) if (mask:=np.array([r[key]==v for r in rows])).any()}
    return report,arrays


def main():
    prepare();started=time.monotonic();assert len(list((OUT/'prefix_commits').glob('*.json')))==512
    allrows=read(OUT/'material.json');rows=[r for r in allrows if r['analysis_selected']];assert rows
    objects,h,target,heads,qkv,entropy=objects_for(rows,OUT)
    nr=read(N/'selected_rows.json')
    with np.load(N/'features.npz') as z:nh=z['h12'].astype(np.float64);nqkv=z['qkv'];nheads=z['head'];nattention=z['attention']
    with np.load(N/'input_grams.npz') as z:tr=z['train'];oldte=z['test'];hscale=float(z['H12_scale']);scales=(float(z['K_scale']),float(z['V_scale']))
    trainrows=[nr[i] for i in tr];trainobjects,_,_,_,_,_=objects_for(trainrows,M)
    wo=checkpoint('model.layers.23.self_attn.o_proj.weight').float().numpy().astype(np.float64)
    hg=h@nh[tr].T/hscale**2;kg,vg=cross_cache(objects,trainobjects,scales);grams={'H12':hg,'H12_fullpastKV':(hg+kg+vg)/3}
    npz(OUT/'features/transfer_features.npz',h12=h.astype(np.float32),attention=target.astype(np.float32),native_head=heads.astype(np.float32))
    npz(OUT/'features/cross_grams.npz',H12=hg,past_K=kg,past_V=vg,training_N_indices=tr)
    save(OUT/'features/selected_rows.json',rows)
    # Serialization audit uses only already-evaluated N heldout states, before O scores.
    check_indices=oldte[:4];checkrows=[nr[i] for i in check_indices];checkobjects,ch,_,_,_,_=objects_for(checkrows,M)
    chk_h=ch@nh[tr].T/hscale**2;chk_k,chk_v=cross_cache(checkobjects,trainobjects,scales);checks=[]
    for mid in ('H12_linear_factors_validation','H12_linear_direct_head_equal6144_validation','H12_fullpastKV_quadratic_factors_validation'):
        with np.load(N/f'models/{mid}.npz') as z:alpha=z['alpha'].astype(np.float64)
        g=(chk_h+chk_k+chk_v)/3 if 'fullpastKV' in mid else chk_h;kernel=1+g
        if 'quadratic' in mid:kernel=kernel*kernel
        p=kernel@alpha;reconstructed=factor_output(p,checkobjects,wo)[0] if '_factors_' in mid else p[:,:4096]@wo.T
        with np.load(N/f'predictions/{mid}.npz') as z:original=z['prediction'][:4].astype(np.float64)
        delta=reconstructed-original;item={'model':mid,'states':4,'mse':float(np.mean(delta**2)),'max_abs':float(np.max(np.abs(delta)))};checks.append(item);assert item['mse']<1e-5,item
    save(OUT/'serialization_audit.json',{'passed':True,'timestamp':stamp(),'checks':checks,'future_O_targets_used':False})
    energy=np.mean(nattention[tr]**2,0);reports=[]
    def record(mid,pred,extra,metadata):
        report,arr=summary(target,pred,energy,rows)
        if 'attention_KL_by_head' in extra:report.update(mean_all_head_attention_KL=float(extra['attention_KL_by_head'].mean()),mean_all_head_probability_MSE=float(extra['attention_probability_MSE_by_head'].mean()))
        reports.append({'model':mid,**metadata,**report});npz(OUT/f'predictions/{mid}.npz',prediction=pred.astype(np.float32),**arr,**extra)
        print('O_FROZEN_PREDICTION',mid,report['mse'],flush=True)
        assert time.monotonic()-started<1200
    native,native_p=factor_output(qkv,objects,wo);report,arr=summary(target,native,energy,rows)
    npz(OUT/'predictions/native_arithmetic_floor.npz',prediction=native.astype(np.float32),**arr,**native_p)
    save(OUT/'native_arithmetic_audit.json',{'scope':'Actual currentQ/K/V, allpastKV and realWO inFP64; native-future oracle, not forecast','metrics':report,'mean_attention_KL':float(native_p['attention_KL_by_head'].mean())})
    for previous in read(N/'result.json')['reports']:
        mid=previous['model']
        if mid.startswith('mean_'):
            # O requests no Result/Trace/Neutral order: original18group reader has no
            # matching group, therefore its original global-fallback rule applies.
            y=nqkv if mid.endswith('_factors') else nheads;predicted=np.repeat(y[tr].mean(0)[None],len(rows),axis=0)
            pred,extra=factor_output(predicted,objects,wo) if mid.endswith('_factors') else (predicted@wo.T,{})
            record(mid,pred,extra,{'frozen_N_route':True,'unavailable_order_group_fallback':'N training global mean','calibrated_on_O':False});continue
        filename=mid.removesuffix('_headnorm')
        with np.load(N/f'models/{filename}.npz') as z:alpha=z['alpha'].astype(np.float64);norms=(z['query_norm'],z['key_norm'])
        inputname=previous['input'];kernel=1+grams[inputname]
        if previous['kernel']=='quadratic':kernel=kernel*kernel
        p=kernel@alpha
        if previous['route']=='factors':pred,extra=factor_output(p,objects,wo,norms=norms if mid.endswith('_headnorm') else None)
        else:pred,extra=(p[:,:4096]@wo.T,{}) if previous['route']=='direct_head_equal6144' else (p,{})
        record(mid,pred,extra,{'frozen_N_route':True,'input_encoder':inputname,'route':previous['route'],'kernel':previous['kernel'],
          'frozen_ridge':previous['ridge'],'selection_in_N':previous['selection'],'calibrated_on_O':False})
    for route,y in (('factors',nqkv),('head',nheads)):
        means={key:y[[i for i in tr if (nr[i]['family'],nr[i]['language'])==key]].mean(0) for key in {(nr[i]['family'],nr[i]['language']) for i in tr}}
        p=np.stack([means.get((r['family'],r['language']),y[tr].mean(0)) for r in rows]);pred,extra=factor_output(p,objects,wo) if route=='factors' else (p@wo.T,{})
        record('mean_supported_family_language_'+route,pred,extra,{'prior_only_additional_control':True,'unknown_family_fallback':'N global training mean','calibrated_on_O':False})
    prefixes=read(OUT/'prefixes.json');bb=[read(OUT/f'behavior/{r["sample_id"]}.json') for r in prefixes]
    total=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());campaign=sum(p.stat().st_size for p in CAMPAIGN.rglob('*') if p.is_file());limits=read(OUT/'protocol.json')['resources']
    assert total<limits['maximum_phase_bytes'] and campaign<limits['campaign_ceiling'] and shutil.disk_usage(OUT).free>limits['free_floor'],(total,campaign)
    result={'phase':2709,'timestamp':stamp(),'prefixes':512,'states':len(allrows),'selected_states':len(rows),'entity_groups':16,'frozen_predictor_routes':26,
      'native_output_observation':{'eos_prefixes':sum(b['eos'] for b in bb),'capped_without_eos':sum(b['cap_without_eos'] for b in bb),'maximum_steps':12,'complete_answer_scoring':False,
        'analysis_step_counts':{str(s):sum(r['generation_step']==s for r in rows) for s in (1,4,8)},'native_full_vocabulary_entropy_mean':float(np.mean(entropy))},
      'forecasts':reports,'native_arithmetic_floor':report,'elapsed_seconds':time.monotonic()-started,'new_bytes':total,'campaign_bytes':campaign,
      'limits':read(OUT/'analysis_protocol.json')['limits']}
    # Use separately persisted floor: avoid any ambiguity from loop-local report names.
    result['native_arithmetic_floor']=read(OUT/'native_arithmetic_audit.json')['metrics']
    save(OUT/'native_entropy.json',{'sample_ids':[r['sample_id'] for r in rows],'all_vocabulary_entropy':entropy});save(OUT/'result.json',result)
    announce('o_generalization',state='analysis_complete',completed=512,total=512,states=len(allrows),analysis_states=len(rows));print('O_GENERALIZATION_COMPLETE',len(rows),len(reports),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args();prepare() if a.prepare else main()
