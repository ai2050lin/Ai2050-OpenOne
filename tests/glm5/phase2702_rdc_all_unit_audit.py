"""Check every unit, so aggregate MSE is not mistaken for a universal per-unit improvement."""
from rdc_continuity_common import *

def main():
    out=CAMPAIGN/'f_continuity';rows=read(CAMPAIGN/'e_confirmation/material.json')
    te=np.array([i for i,r in enumerate(rows) if r['word_split']=='test']);results=[]
    for layer in (11,23,35):
        with np.load(out/f'L{layer}_factors.npz') as z:
            actual=z['observed_a'][te].astype(np.float64)
            errors={mode:np.mean((z[mode+'_a'][te].astype(np.float64)-actual)**2,axis=0) for mode in ('global','family_language')}
        energy=np.mean(actual**2,axis=0);assert len(energy)==9728
        result={'layer':layer,'units':9728,'n':len(te),'zero_energy_units':int(np.sum(energy==0)),
          'family_language_better_than_global_units':int(np.sum(errors['family_language']<errors['global'])),
          'equal_error_units':int(np.sum(errors['family_language']==errors['global'])),
          'per_unit_relative_mse_quantiles':{mode:np.quantile(e[energy>0]/energy[energy>0],[0,.25,.5,.75,1]).tolist() for mode,e in errors.items()}}
        npz(out/f'unit_errors/L{layer}.npz',energy=energy,**errors);results.append(result)
    save(out/'all_unit_audit.json',{'timestamp':stamp(),'results':results,
      'scope':'Post-result diagnostic on all9728 units; no selected units, no new fit. All-unit aggregate MSE and equal-unit normalized error are different metrics. Zero-energy units excluded only from ratio quantiles, counted and retained in arrays. Ratios of near-zero units may be unstable.'})
    print(results,flush=True)

if __name__=='__main__':main()
