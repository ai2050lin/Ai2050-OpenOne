"""S0: continuous synthetic inputs; algorithm calibration is NOT an LLM result."""
from rdc_feature_common import *
from rdc_feature_extractors import *

RUN='s0'; OUT=CAMPAIGN/RUN
NAMES=('distributed_linear','quadratic','cubic','ordered_role','context_gate','layer_reencoding')

def main():
    rng=np.random.default_rng(269200)
    contract={'version':1,'seed':269200,'instances_per_structure':256,'structures':list(NAMES),
              'native_coordinates_per_block':48,'continuous_latents':4,'partition':[128,64,64],
              'algorithms':list(ALGORITHMS)+['A5_known_function_positive_control'], 'lambdas':list(LAMBDAS),
              'scope':'Continuous latent realizations held out within each fixed encoding; unseen encoding generalization NOT claimed.',
              'source_sha':{p.name:sha(p) for p in [Path(__file__),ROOT/'tests/glm5/rdc_feature_extractors.py']}}
    immutable(OUT/'protocol.json',contract)
    status(RUN,state='running',completed=0,total=1536,source_mode='synthetic',protocol_sha=sha(OUT/'protocol.json'))
    all_results=[]; material=[]; checks={}
    for m,name in enumerate(NAMES):
        lat=[rng.normal(size=(256,4)) for _ in range(3)]
        enc=[np.linalg.qr(rng.normal(size=(48,4)))[0].T for _ in range(3)]
        blocks=[z@w*(1e-3 if name=='distributed_linear' else 1) for z,w in zip(lat,enc)]
        a,b,c=lat
        values={'distributed_linear':a[:,0], 'quadratic':a[:,0]*a[:,1], 'cubic':a[:,0]*a[:,1]*a[:,2],
                'ordered_role':a[:,0]*b[:,1]-a[:,1]*b[:,0],
                'context_gate':(a[:,0]*b[:,1]-a[:,1]*b[:,0])*c[:,0], 'layer_reencoding':a[:,0]}
        y=values[name][:,None]; tr=np.arange(128);va=np.arange(128,192);te=np.arange(192,256)
        npz(OUT/f'fields/{name}.npz',u=blocks[0],v=blocks[1],c=blocks[2],y=y,latent_a=a,latent_b=b,latent_c=c,encoding_u=enc[0])
        for algo in ALGORITHMS:
            score,pred,params=fit_predict(blocks,y,tr,va,te,algo)
            all_results.append(dict(structure=name,algorithm=algo,**score))
            npz(OUT/f'models/{name}__{algo}.npz',prediction=pred,target=y[te],**{k:v for k,v in params.items() if isinstance(v,np.ndarray)})
            if algo=='A2_quadratic':
                z=params['z_train'];alpha=params['alpha'];scale=params['raw_scale_vector'];x=np.concatenate(blocks,axis=1)[te[0]]
                bias=float(alpha.sum()); w=2*(alpha[:,0]@z)/scale
                full=np.stack([quadratic_row(z,alpha,scale,j,0,len(x))[2] for j in range(len(x))])
                reconstructed=bias+w@x+x@full@x
                checks[name+'_coordinate_expansion']=bool(np.isclose(reconstructed,pred[0,0],rtol=1e-7,atol=1e-8))
        # Known generating function is a declared oracle positive control, not a fair semantic extractor.
        all_results.append(dict(structure=name,algorithm='A5_known_function_positive_control',mse=0.,n=64,oracle=True))
        material += [dict(sample_id=f'{name}-{i:03d}',structure=name,index=i,split='train' if i<128 else 'validation' if i<192 else 'test',
                          source_mode='synthetic',field=f'fields/{name}.npz') for i in range(256)]
        status(RUN,state='running',completed=(m+1)*256,total=1536,source_mode='synthetic',protocol_sha=sha(OUT/'protocol.json'))
        event(RUN,'structure_complete',structure=name,completed=(m+1)*256,total=1536,protocol_sha=sha(OUT/'protocol.json'))
    expected={'distributed_linear':'A1_linear','quadratic':'A2_quadratic','cubic':'A2_cubic','ordered_role':'A3_ordered_pair','context_gate':'A4_conditional','layer_reencoding':'A1_linear'}
    for name,algo in expected.items():
        chosen=next(r for r in all_results if r['structure']==name and r['algorithm']==algo)
        baseline=next(r for r in all_results if r['structure']==name and r['algorithm']=='A0_mean')
        checks[name+'_known_signal_recovered']=chosen['mse']<.01*baseline['mse']
    save(OUT/'material.json',material)
    result={'phase':2692,'timestamp':stamp(),'source_mode':'synthetic','instances':1536,'results':all_results,'checks':checks,
            'measurement_checks_passed':all(v for k,v in checks.items() if 'expansion' in k),
            'algorithm_calibration_passed':all(checks.values()),'language_mechanism_closed':False,
            'limits':['Not 1536 independent mechanisms.','New latent realizations; not unseen encodings.',
                      'A5 is a known-function oracle, never interpreted as learned semantic success.',
                      'No conclusion that an LLM implements any synthetic mechanism.']}
    save(OUT/'result.json',result)
    status(RUN,state='complete',completed=1536,total=1536,source_mode='synthetic',protocol_sha=sha(OUT/'protocol.json'))
    print(json.dumps(result,ensure_ascii=True),flush=True)

if __name__=='__main__':main()
