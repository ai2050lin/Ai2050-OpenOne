"""Simple mean/scale calibration control for prefill-to-content cross-model shift."""
from rdc_conditional_common import *
from phase2704_rdc_predictive_gates import errors
OUT=CAMPAIGN/'l_aligned'


def main():
    rows=read(OUT/'qwen4/prefixes.json');tr,va,te=splits(rows)
    with np.load(OUT/'qwen4/features.npz') as z:source=unbits(z['content_h'][:,24]).astype(np.float64)
    immutable(OUT/'stage_calibration_protocol.json',{'source_sha':sha(Path(__file__)),'status':'Sensitivity analysis motivated by observed GLM preceding-format-token pattern; not new untouched confirmation',
      'input':'Keep the frozen prefill-fit sourceQ4H24->targetH27 mapping; targetmodel first-content H on64training records supplies only offset and optionally one global scalar. No heldout target state is a prediction input.',
      'controls':'Training mean targetcontent-minus-prefill offset; or one sharedscalar and per-coordinate intercept least squares using trainingcontent states. No per-coordinate slope or fullcontent mapping refit.',
      'capacity':'Offset adds D fitted values; scalar+offset addsD+1. Does not equate all effective capacities. Compare originalfrozenpremap and fullcontentfit on same128heldout records.',
      'limits':'Observed content boundary is assigned aftergeneration, so this remains a conditional observational map. ForQwen prefill/content maycoincide. Onlyone-format-step contrast and2heldoutentitygroups; not general decoding dynamics.'})
    reports=[]
    for key in ('qwen14','glm4'):
        with np.load(OUT/key/'features.npz') as z:
            pre=unbits(z['prefill_h'][:,27]).astype(np.float64);content=unbits(z['content_h'][:,27]).astype(np.float64)
        for kind in ('linear','quadratic'):
            mid=f'{key}_prefill_to_content_{kind}'
            with np.load(OUT/f'predictions/{mid}.npz') as z:p=z['prediction'].astype(np.float64)
            with np.load(OUT/f'models/{mid}.npz') as z:alpha=z['alpha'].astype(np.float64);scale=float(z['scale'])
            x=source/scale;k=1+x[tr]@x[tr].T
            if kind=='quadratic':k=k*k
            predtrain=k@alpha;delta=(content[tr]-pre[tr]).mean(0)
            pm=predtrain.mean(0);ym=content[tr].mean(0);center=predtrain-pm
            slope=float(np.sum(center*(content[tr]-ym))/max(np.sum(center*center),1e-30));intercept=ym-slope*pm
            energy=np.mean(content[tr]**2,0)
            for name,pred in [('frozen_prefill',p),('training_mean_offset',p+delta),('training_scalar_and_offset',p*slope+intercept)]:
                report,arr=errors(content[te],pred,energy);reports.append({'target_model':key,'algorithm':kind,'calibration':name,**report,
                  'scalar':slope if name=='training_scalar_and_offset' else 1.,'extra_parameters':content.shape[1]+int(name=='training_scalar_and_offset') if name!='frozen_prefill' else 0,
                  'by_entity':{str(unit):float(np.mean((pred[[rows[i]['unit']==unit for i in te]]-content[te][[rows[i]['unit']==unit for i in te]])**2)) for unit in (12,13)}})
                npz(OUT/f'predictions/calibration_{key}_{kind}_{name}.npz',prediction=pred.astype(np.float32),target=content[te].astype(np.float32),test=te,
                  offset=delta if name=='training_mean_offset' else intercept if name=='training_scalar_and_offset' else np.zeros(content.shape[1]),scalar=np.array(slope if name=='training_scalar_and_offset' else 1.),**arr)
        print('STAGE_CALIBRATION',key,flush=True)
    save(OUT/'stage_calibration_audit.json',{'timestamp':stamp(),'reports':reports,'scope':read(OUT/'stage_calibration_protocol.json')})


if __name__=='__main__':main()
