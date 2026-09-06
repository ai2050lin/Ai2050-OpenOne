"""Post-observation full-coordinate amplitude envelopes, signed transfer and caveats."""
import numpy as np
from phase2620_native_coordinate_contract import *
from phase2647_matched_operation_delivery import OUT,MAPS

def run():
    sums={};rows=[]
    for p in sorted((MAPS['initial']/'field').glob('*_fullcoordinate_maps.npz')):
        with np.load(p) as a,np.load(MAPS['confirmation']/'field'/p.name) as b:
            for metric in ('h','mlp'):
                arrays={}
                for label,z in [('initial',a),('confirmation',b)]:
                    t=z[f'target__{metric}__rms'];o=z[f'order__{metric}__rms'];f=z[f'form__{metric}__rms']
                    arrays[label]=(t>0)&(t>o)&(t>f)
                both=arrays['initial']&arrays['confirmation']
                ma=a[f'target__{metric}__mean'];mb=b[f'target__{metric}__mean']
                signed=both&(ma!=0)&(mb!=0)&(np.sign(ma)==np.sign(mb))
                for key,mask in [('initial_dominant',arrays['initial']),('confirmation_dominant',arrays['confirmation']),('both_dominant',both),('both_dominant_same_sign',signed)]:
                    kk=metric+'__'+key
                    if kk not in sums:sums[kk]=np.zeros(mask.shape,dtype='int16')
                    sums[kk]+=mask
                rows.append({'group':p.stem.removesuffix('_fullcoordinate_maps'),'metric':metric,
                    'both_dominant_fraction_by_layer_position':both.mean(-1).tolist(),'both_dominant_same_sign_fraction_by_layer_position':signed.mean(-1).tolist()})
    np.savez(OUT/'field/allcoordinate_response_envelopes.npz',**sums);save(OUT/'analysis/response_envelope_groups.json',rows)
    summary={}
    for metric,layer in [('h',36),('mlp',35)]:
        n=sums[metric+'__both_dominant'][layer,2];s=sums[metric+'__both_dominant_same_sign'][layer,2]
        summary[metric]={'coordinate_count':len(n),'all16_groups_amplitude_dominance_coordinates':int((n==16).sum()),
            'all16_groups_same_signed_dominance_coordinates':int((s==16).sum()),
            'amplitude_group_coverage_histogram':np.bincount(n,minlength=17).tolist(),'signed_group_coverage_histogram':np.bincount(s,minlength=17).tolist()}
    natural=read(OUT/'analysis/natural_fullcoordinate_audit.json')
    ratios=[]
    for group,r in natural.items():
        rr=r['response'];t=rr['target__h']['rms_by_layer_position'][36][2];o=rr['order__h']['rms_by_layer_position'][36][2];f=rr['form__h']['rms_by_layer_position'][36][2]
        ratios.append({'group':group,'target_over_order':t/o,'target_over_form':t/f,'bf16_fp32_target_amplitude_cosine':rr['target__h']['bf_fp_response_rms_cosine'][36][2]})
    report={'summary':summary,'natural_boundary_ratios':ratios,'all16_natural_target_exceeds_order_and_form':all(r['target_over_order']>1 and r['target_over_form']>1 for r in ratios),
        'selection_timing':'Exploratory rule added AFTER seeing initial and confirmation group-level responses. The two sample sets were collected independently, but this envelope rule has NOT received a prospectively frozen third-set validation.',
        'definition':'Each physical coordinate gets a0..16 coverage count: target RMS strictly exceeds BOTH order and form RMS separately in initial and confirmation. Signed version additionally requires nonzero group target means with equal sign. Every coordinate retained, no ranking orTopK.',
        'boundary':'Amplitude dominance is task/answer-response discrimination, NOT semantic specificity, unique core, necessity or stable sign across all language. Same signed means initial/confirmation agreement WITHIN EACH group, not the same sign across language families. Named-person answer selection and output-head rows remain confounds. Canonical A-minus-B means compare different lexical entities across sets.'}
    save(OUT/'analysis/response_envelopes.json',report);print(json.dumps(report,ensure_ascii=True),flush=True)
    return report

if __name__=='__main__':run()
