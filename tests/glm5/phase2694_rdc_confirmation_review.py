"""Post-result scientific review: independent frozen predictions, all-coordinate ledger, answer/format separation."""
import re
from rdc_feature_common import *
from rdc_feature_extractors import kernel
from phase2693_rdc_delivery_audit import get

OUT=CAMPAIGN/'s2pilot';OLD=CAMPAIGN/'s1'

def main():
    rows=read(OUT/'material.json');protocol=read(OUT/'protocol.json');checks={};numerical=[]
    for mid,d in protocol['frozen_models'].items():assert sha(OLD/f'models/{mid}.npz')==d
    with np.load(OUT/'features/all_samples.npz') as z:f={k:z[k] for k in z.files}
    for l in (0,12,24,36):
        new=np.concatenate([f[f'H{l}_{b}'] for b in ('u','v','c')],axis=1).astype(np.float64)
        for algo in ('A1_linear','A2_quadratic','A2_cubic','A3_ordered_pair','A4_conditional'):
            mid=f'word__family__H{l}__{algo}'
            with np.load(OLD/f'models/{mid}.npz') as z:
                old=z['z_train'];normalized_new=new/z['raw_scale_vector'];n=len(old)
                blocks=[np.concatenate([old[:,i*2560:(i+1)*2560],normalized_new[:,i*2560:(i+1)*2560]],axis=0)*np.sqrt(3) for i in range(3)]
                reference=kernel(blocks,np.arange(n,n+512),np.arange(n),algo)@z['alpha']
            with np.load(OUT/f'predictions/{mid}.npz') as z:err=float(np.max(np.abs(reference-z['prediction'])))
            assert err<1e-8;numerical.append(dict(model_id=mid,max_error=err))
        mid=f'word__family__H{l}__A1_linear'
        with np.load(OUT/f'coordinate_ledgers/{mid}.npz') as z:w=z['weights'];bias=z['bias']
        # Every input coordinate and every readout class retained. No ranked coordinate subset.
        stats={'raw_coordinate':np.arange(7680),'block':np.repeat(np.arange(3),2560),'within_block':np.tile(np.arange(2560),3),
               'mean_abs_input':np.abs(new).mean(0),'readout_weight':w,
               'mean_signed_contribution':new.mean(0)[:,None]*w,
               'mean_abs_contribution':np.abs(new).mean(0)[:,None]*np.abs(w)}
        npz(OUT/f'coordinate_ledgers/{mid}_all_coordinate_statistics.npz',**stats)
        selected=rows[0]['sample_id']
        reply=get('/linear_contribution',sample=selected,layer=l,class_index=0,coordinate=7616,width=64)
        truth=np.stack([new[0],w[:,0],new[0]*w[:,0]])[:,None,7616:7680]
        assert np.allclose(reply['values'],truth,rtol=1e-12,atol=1e-12)
        assert abs(reply['full_sum']-(new[0]@w[:,0]+bias[0]))<1e-10
    checks['frozen_model_sha_unchanged']=40;checks['independent_kernel_reproduction']=20
    checks['full_7680_coordinate_ledgers']=4;checks['http_coordinate_contribution_exact']=4
    # The new material was frozen before forward; relation/answer balanced independently of lexical block.
    balance=[]
    for lang in ('en','zh'):
        for fam in sorted({r['family'] for r in rows}):
            for form in (0,1):
                group=[r for r in rows if (r['language'],r['family'],r['form'])==(lang,fam,form)]
                assert sum(r['expected_yes'] for r in group)==len(group)//2
                balance.append(dict(language=lang,family=fam,form=form,n=len(group),yes=sum(r['expected_yes'] for r in group)))
    checks['balanced_language_family_form_groups']=len(balance)
    behavior={};examples=[]
    for run in ('s1','s2pilot'):
        rr=read(CAMPAIGN/run/'material.json')
        for lang in ('en','zh'):
            records=[]
            for row in rr:
                if row['language']!=lang:continue
                b=read(CAMPAIGN/run/f'behavior/{row["sample_id"]}.json')
                match=re.match(r'^(Yes|No)\b' if lang=='en' else r'^(是|否)',b['generated'],re.I)
                first_correct=bool(match and match[1].casefold()==row['target'].casefold())
                records.append(dict(exact_format=b['correct'],content_prefix=first_correct,eos=b['eos'],parsed=bool(match),
                    argmax_agreement=bool(b['generated_ids'] and b['generated_ids'][0]==b['first_argmax'])))
                if run=='s2pilot' and len(examples)<8 and (not b['correct'] or b['generated_ids'][0]!=b['first_argmax']):
                    examples.append(dict(sample_id=row['sample_id'],u=row['u'],v=row['v'],target=row['target'],generated=b['generated'],
                        first_answer_correct=first_correct,format_correct=b['correct'],eos=b['eos']))
            behavior[f'{run}/{lang}']={'n':len(records),**{k:sum(x[k] for x in records) for k in records[0]}}
    checks['content_format_stop_reported_separately']=True
    save(OUT/'confirmation_review.json',{'timestamp':stamp(),'status':'review_complete','checks':checks,'numerical':numerical,
        'balance':balance,'generation_audit':behavior,'actual_examples':examples,
        'limits':['Content-prefix analysis is a post-result diagnostic, not replacement of the frozen whole-answer metric.',
          'All-coordinate reader contributions exactly reconstruct the learned class score, not model logits or causal contributions.',
          'Cross-family output classes are external task conventions; strong family readability alone does not identify relation logic.',
          'Per-coordinate statistics average512 related realizations; individual native arrays remain queryable.'],
        'source_sha':sha(Path(__file__))})
    print('REVIEW_DONE',checks,behavior,flush=True)

if __name__=='__main__':main()
