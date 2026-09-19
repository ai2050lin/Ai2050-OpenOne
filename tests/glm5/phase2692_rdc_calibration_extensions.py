"""Prospective diagnostic after cubic full-input finite-sample failure; preserve S0 result."""
from rdc_feature_common import *
from rdc_feature_extractors import *

def main():
    out=CAMPAIGN/'s0';old=read(out/'result.json')
    protocol={'source_result_sha':sha(out/'result.json'),'source_code_sha':sha(Path(__file__)),
              'reason':'Cubic full u/v/c fit did not recover at Ntrain128; isolate irrelevant blocks without dropping any coordinate inside u.',
              'scope':'Post-result diagnostic, not blinded confirmation; original failed result retained.'}
    immutable(out/'extension_protocol.json',protocol)
    tr=np.arange(128);va=np.arange(128,192);te=np.arange(192,256)
    with np.load(out/'fields/cubic.npz') as z:
        u=z['u'];y=z['y']
    results={}
    for name in ('A1_linear','A2_quadratic','A2_cubic'):
        score,pred,_=fit_predict([u],y,tr,va,te,name)
        results[name]=score
    with np.load(out/'fields/layer_reencoding.npz') as z:
        before=z['u'];y=z['y']
    rotation=np.linalg.qr(np.random.default_rng(269201).normal(size=(48,48)))[0]
    after=before@rotation
    score,pred,params=fit_predict([before],y,tr,va,te,'A1_linear')
    normalized,scales=normalize_blocks([before],tr)
    transferred=(1+(after[te]/scales[0])@(before[tr]/scales[0]).T)@params['alpha']
    refit,_,_=fit_predict([after],y,tr,va,te,'A1_linear')
    results['reencoding']={'before_fit_mse':score['mse'],'unchanged_readout_after_rotation_mse':metrics(y[te],transferred,False)['mse'],
                           'refitted_after_rotation_mse':refit['mse'],'inverse_reconstruction_max_abs':float(np.max(np.abs(after@rotation.T-before)))}
    npz(out/'fields/reencoding_pair.npz',before=before,after=after,rotation=rotation,y=y)
    result={'timestamp':stamp(),'results':results,'checks':{'restricted_cubic_recovers':results['A2_cubic']['mse']<1e-6,
                'reencoding_inverse':results['reencoding']['inverse_reconstruction_max_abs']<1e-10,
                'same_information_refit_recovers':refit['mse']<1e-8},
            'conclusion':'Correct polynomial degree is insufficient at fixed sample budget with nuisance blocks; scope and inductive bias matter. Different linear readout can recover a changed basis.'}
    save(out/'extension_result.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
