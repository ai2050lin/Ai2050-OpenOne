"""All-unit write-error accounting and known linear-ridge commutation check."""
from rdc_conditional_common import *
OUT=CAMPAIGN/'j_predictive_gates'


def main():
    import torch
    torch.set_num_threads(2)
    rows=read(CAMPAIGN/'i_factorial/material.json');tr,va,te=splits(rows)
    immutable(OUT/'write_error_audit_protocol.json',{'source_sha':sha(Path(__file__)),'result_sha':sha(OUT/'result.json'),
      'design':'Post-result arithmetic/error-orientation audit, not additional independent language confirmation.',
      'identity':'For identical inputkernel andridge, P*A*WdT = P*(A*WdT); direct-a-through-Wd and direct-down are not independent mechanisms. BF16native rounding and FP32persistedprediction rounding break bitwise equality slightly.',
      'full_unit_accounting':'delta_a WdT squared norm decomposed into all-unit diagonal columnnorm term and cross term. Compute fullmatrixproduct; do not select units or build a compressedhiddenbasis.',
      'paired_scope':'Fourglobaltestentitygroups andeightfamilies; report pairederrors descriptively, not broadpopulationconfidence.'})
    with np.load(OUT/'features/L23.npz') as z:a=unbits(z['a'][te]).astype(np.float64);down=unbits(z['down'][te]).astype(np.float64)
    wd=checkpoint('model.layers.23.mlp.down_proj.weight').float().numpy().astype(np.float64);colnorm=np.square(wd).sum(0);d=wd.shape[0]
    actual_ideal=a@wd.T;rounding=down-actual_ideal;results=[];pairs=[]
    forecasts=read(OUT/'result.json')['forecasts']
    for mode in ('C','UVC'):
      for kind in ('linear','quadratic'):
        stem=f'H12_{mode}_{kind}';case_errors={};projected={}
        for target in ('a','derived_from_g_up','equal_head_budget_direct_a'):
            with np.load(OUT/f'predictions/{stem}_{target}.npz') as z:p=z['prediction'].astype(np.float64)
            error=p-a;write_error=error@wd.T
            diagonal=(np.mean(error*error,0)*colnorm)/d
            ideal_per=np.mean(write_error*write_error,1);observed_per=np.mean((write_error-rounding)**2,1)
            diag=float(diagonal.sum());ideal=float(ideal_per.mean());observed=float(observed_per.mean())
            cross=ideal-diag;round_cross=float(np.mean(-2*write_error*rounding));round_sq=float(np.mean(rounding*rounding))
            assert abs(observed-(diag+cross+round_cross+round_sq))<1e-10
            results.append({'input':'H12_'+mode,'kernel':kind,'a_prediction':target,'unit_mse':float(np.mean(error*error)),
              'write_error_diagonal':diag,'write_error_cross':cross,'ideal_write_mse':ideal,
              'rounding_square':round_sq,'prediction_rounding_cross':round_cross,'observed_down_mse':observed})
            npz(OUT/f'write_error/{stem}_{target}.npz',unit_diagonal_contribution=diagonal,ideal_write_error_by_sample=ideal_per,observed_write_error_by_sample=observed_per,test=te)
            case_errors[target]=observed_per;projected[target]=p@wd.T
        with np.load(OUT/f'predictions/{stem}_down.npz') as z:direct_down=z['prediction'].astype(np.float64)
        commutation_error=projected['a']-direct_down
        direct_a=next(r for r in forecasts if r.get('input')=='H12_'+mode and r.get('algorithm')==kind and r.get('target')=='a' and r.get('kind')=='direct')
        dd=next(r for r in forecasts if r.get('input')=='H12_'+mode and r.get('algorithm')==kind and r.get('target')=='down' and r.get('kind')=='direct')
        delta=case_errors['derived_from_g_up']-case_errors['a']
        pairs.append({'input':'H12_'+mode,'kernel':kind,'ridge_a':direct_a['ridge'],'ridge_down':dd['ridge'],
          'same_ridge':direct_a['ridge']==dd['ridge'],'direct_a_write_minus_direct_down_max':float(np.abs(commutation_error).max()),
          'direct_a_write_minus_direct_down_rms':float(np.sqrt(np.mean(commutation_error**2))),
          'factor_minus_direct_a_write_mse':float(delta.mean()),
          'by_entity_group':{str(g):float(delta[[rows[i]['unit']==g for i in te]].mean()) for g in (12,13,14,15)},
          'by_family':{f:float(delta[[rows[i]['family']==f for i in te]].mean()) for f in sorted({r['family'] for r in rows})}})
    save(OUT/'write_error_audit.json',{'timestamp':stamp(),'unit_write_decomposition':results,'paired_and_commutation':pairs,
      'interpretation':'Cancellation is an arithmetic relation among prediction errors and fixedcolumns, not proof of linguistic cooperation. Near-equal directreadouts can be a known fitting identity.'})
    print('WRITE_ERROR_AUDIT',len(results),len(pairs),flush=True)


if __name__=='__main__':main()
