"""Independent small explicit-coordinate checks for source-binding kernels."""
from rdc_binding_common import *
from rdc_binding_kernels import feature_pack,pair_kernels,KERNELS

def main():
    import torch
    rng=np.random.default_rng(2732);sources=[rng.normal(size=(n,4)).astype(np.float32) for n in (3,5,4)]
    q=rng.normal(size=(3,4)).astype(np.float32);e=rng.normal(size=(3,4)).astype(np.float32)
    roles=[rng.uniform(size=(len(s),6)).astype(np.float32) for s in sources]
    roles=[r/r.sum(1,keepdims=True) for r in roles]
    pack=feature_pack(sources,q,e,roles);got=pair_kernels(pack,block=2);errors={}
    cpu={k:v.cpu().numpy().astype(float) for k,v in pack.items()}
    for name in KERNELS[2:]:
        explicit=[]
        for i,s in enumerate(sources):
            feats=[]
            for j,h in enumerate(s):
                f=np.outer(h,h).ravel()/4
                if name!='source_pair':
                    pkey='shuffledpos' if name=='shuffled_position_pair' else 'pos'
                    f=np.outer(f,cpu[pkey][i,j]).ravel()
                if name in ('role_position_pair','shuffled_role_position_pair'):
                    rkey='shuffledrole' if name=='shuffled_role_position_pair' else 'role'
                    f=np.outer(f,np.r_[1,cpu[rkey][i,j]]).ravel()
                feats.append(f)
            explicit.append(np.mean(feats,axis=0))
        f=np.array(explicit);s=f@f.T;base=(cpu['q']@cpu['q'].T+cpu['e']@cpu['e'].T)/8
        expected=1+base+s+base*s
        err=np.max(np.abs(expected-got[name].cpu().numpy()));errors[name]=float(err)
        assert err<2e-5,(name,err)
    eig={k:float(torch.linalg.eigvalsh(v.double()).min()) for k,v in got.items()}
    assert min(eig.values())>-1e-5
    save(BASE/'verification/kernel_math.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'explicit_all_coordinate_feature_max_errors':errors,
      'minimum_eigenvalues':eig,'scope':'Small exact feature expansion versus operational source-pair kernel; no semantic theorem.'})
    print('BINDING_KERNEL_MATH_PASS',errors,flush=True)

if __name__=='__main__':main()

