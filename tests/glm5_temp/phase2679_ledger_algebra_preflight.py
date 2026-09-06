"""Synthetic arithmetic tests only: no trained model evidence or Phase append."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import RESULT,save
from phase2679_source_coordinate_ledger import *


def test():
    rng=np.random.default_rng(2679);records=[]
    for heads,kv,dim,ns,d in ((4,2,3,7,11),(3,3,4,5,9),(8,1,2,13,17)):
        p=rng.uniform(size=(2,heads,ns));p/=p.sum(axis=-1,keepdims=True)
        v=rng.normal(size=(ns,kv,dim));w=rng.normal(size=(d,heads*dim));bias=rng.normal(size=d)
        # Explicit, intentionally slow, independent scalar reference.
        brute=np.zeros((2,ns,d))
        for q in range(2):
            for s in range(ns):
                for i in range(d):
                    for h in range(heads):
                        for k in range(dim):brute[q,s,i]+=p[q,h,s]*v[s,h//(heads//kv),k]*w[i,h*dim+k]
        av=np.einsum('qhs,shk->qhk',p,np.repeat(v,heads//kv,axis=1)).reshape(2,-1)
        native_z=av+rng.normal(scale=1e-4,size=av.shape)
        native_out=native_z@w.T+bias+rng.normal(scale=1e-4,size=(2,d))
        seen=[]
        ledger=attention_ledger(p,v,w,native_out,native_z,bias,lambda h,t:seen.append((h,t.shape,t.flags.writeable)))
        h=rng.normal(size=(2,d));pre=h+native_out+rng.normal(scale=1e-5,size=(2,d));gamma=rng.normal(size=d)
        x=pre*gamma/np.sqrt(np.mean(pre*pre,axis=-1,keepdims=True)+1e-6)+rng.normal(scale=1e-5,size=(2,d))
        norm=conditional_norm_ledger(h,ledger,gamma,1e-6,pre,x)
        wg=rng.normal(size=d);g=x@wg+.125+rng.normal(scale=1e-5,size=2)
        gate=input_weight_ledger(norm,wg,g,.125)
        errors={'independent_source_loop':float(np.max(np.abs(brute-ledger['source_terms']))),
                'attention_reconstruction':float(np.abs(ledger['reconstruction_error']).max()),
                'norm_reconstruction':float(np.abs(norm['reconstruction_error']).max()),
                'gate_reconstruction':float(np.abs(gate['reconstruction_error']).max())}
        assert max(errors.values())<1e-11 and len(seen)==heads and all(not r[2] for r in seen)
        assert np.allclose(gate['positive_source_sum']+gate['negative_source_sum'],gate['source_signed_sum'],atol=1e-12,rtol=0)
        # Reorder sources and their probability columns together: exact sum must
        # remain unchanged. This is not a language-permutation experiment.
        perm=rng.permutation(ns);other=attention_ledger(p[:,:,perm],v[perm],w,native_out,native_z,bias)
        assert np.allclose(other['source_terms'],ledger['source_terms'][:,perm],atol=1e-12,rtol=0)
        records.append({'heads':heads,'kv_heads':kv,'head_dim':dim,'source_tokens':ns,'physical_output_dimensions':d,'errors':errors})
    out=RESULT/'phase2676_native_mlp_delivery/analysis/next_ledger_algebra_preflight.json'
    save(out,{'all_checks_passed':True,'records':records,
              'scope':'Synthetic NumPy-only GQA/attention/RMSNorm/scalar multiplication identities; not model execution, not evidence of semantics, not completed Phase2679.'})
    print('SYNTHETIC LEDGER PREFLIGHT',records)


if __name__=='__main__':test()
