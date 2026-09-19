"""Preserve failed discrete norm matching before a weights-only control amendment."""
from rdc_update_common import *

def main():
    out=BASE/'learning/norm_recovery'
    if (out/'failed_attempt.json').exists():return
    files=list((BASE/'learning/finite_updates').glob('*.npz'))+list((BASE/'learning/native_deltas').glob('*.npz'))
    result={'timestamp':stamp(),'failed_source':snapshot(Path(__file__).with_name('phase2737_rdc_update_finite.py')),
      'failure':'Global scalar rescaling of a constant-magnitude Rademacher direction could not hit native BF16 norm0.02 within0.5%. Quantized scalar weights cross shared rounding thresholds together.',
      'target':.02,'closest_reported_actual_norm':.017217157408595085,'reported_pre_round_scale':.06593750000000001,
      'relative_error':.0027828425914049153/.02,
      'measured_failed_duration_seconds':None,'timing_scope':'Initial failing script did not persist a start/end duration; no fabricated runtime is booked.',
      'preserved_artifacts':{str(p.relative_to(BASE)):sha(p) for p in files},
      'amendment':'Keep all15 frozen FP32 finite tests and original Rademacher direction unchanged. Replace ONLY the infeasible native matched-random deployment with full-coordinate Gaussian seed273700, before observing any Gaussian loss or generation outcome. Record different control identity explicitly.',
      'native_gaussian_purpose':'Continuous magnitudes break simultaneous equal-amplitude rounding ties; actual weights-only norm tolerance remains unchanged. No loss-based calibration or outcome-based direction selection.'}
    save(out/'failed_attempt.json',result)
    print('NORM_RECOVERY_FROZEN',len(files),flush=True)

if __name__=='__main__':main()
