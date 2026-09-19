"""Check simple residual-state baselines and preserve all-unit tails, not just totalMSE."""
from rdc_conditional_common import *
OUT=CAMPAIGN/'j_predictive_gates'


def main():
    rows=read(CAMPAIGN/'i_factorial/material.json');tr,va,te=splits(rows)
    with np.load(CAMPAIGN/'i_factorial/features/state.npz') as z:h12=unbits(z['h_c'][:,12]);h24=unbits(z['h_c'][:,24])
    baselines=[]
    for label,p in [('H12_identity',h12[te]),('H12_train_mean_update',h12[te]+(h24[tr]-h12[tr]).mean(0))]:
        baselines.append({'target':'H24','model':label,'n':len(te),'mse':float(np.mean((p.astype(np.float64)-h24[te])**2))})
    summaries=[];result=read(OUT/'result.json')
    for l in (11,23,35):
        with np.load(OUT/f'unit_errors/L{l}_comparison.npz') as z:wins=int(z['family_improves'].sum())
        for mode in ('ordinary_global','ordinary_family_language','weighted_global','weighted_family_language'):
            r=next(r for r in result['reconstruction'] if r['layer']==l and r['model']==mode)
            summaries.append({'layer':l,'model':mode,'mse':r['mse'],'relative_mse_quantiles':r['relative_mse_quantiles'],
              'zero_test_energy':r['zero_test_energy'],'zero_training_denominators':r['zero_training_denominators'],
              'weighted_family_beats_weighted_global_units':wins})
    save(OUT/'summary_audit.json',{'timestamp':stamp(),'H24_simple_baselines':baselines,'all_unit_tails':summaries,
      'model_files':len(list((OUT/'models').glob('*.npz'))),'prediction_files':len(list((OUT/'predictions').glob('*.npz'))),
      'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),'main_comparisons':{'reconstruction':len(result['reconstruction']),'forecast_and_baselines':len(result['forecasts'])},
      'scope':'Descriptive supplementary baselines on existing frozen testset, not a fresh confirmation.'})
    print(json.dumps({'baselines':baselines,'tails':summaries},ensure_ascii=True,indent=2),flush=True)


if __name__=='__main__':main()
