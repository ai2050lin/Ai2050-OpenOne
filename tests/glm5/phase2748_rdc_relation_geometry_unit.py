"""All permutation and all vocabulary reference calculations on syntheticdata."""
from rdc_question_common import *
from phase2748_rdc_relation_geometry import gram,token_gram,relations,PERMUTATIONS,CENTER


def main():
    rng=np.random.default_rng(2748012)
    values=[rng.normal(size=(4,w))for w in [7,11,23]]
    kernels=np.stack([gram(x)for x in values]);r=relations(kernels);checks=[]
    assert all(np.allclose(k,CENTER@k@CENTER,atol=1e-14)for k in kernels)
    assert np.allclose(np.diag(r['similarity']),1,atol=1e-14)
    for i,a in enumerate(kernels):
        for j,b in enumerate(kernels):
            expected=np.mean([np.sum(a*b[np.ix_(p,p)])/(np.linalg.norm(a)*np.linalg.norm(b))for p in PERMUTATIONS])
            assert abs(expected-r['permutation_mean'][i,j])<1e-13
    checks.append('All9pair permutation means agree with explicit enumeration of24permutations')
    shifted=[x+np.arange(x.shape[1])[None,:]for x in values]
    assert np.allclose(kernels,np.stack([gram(x)for x in shifted]),atol=1e-13)
    checks.append('Context-common coordinate translation cancels exactly within floatingtolerance')
    tokens=[[1,1,9],[9,10],[1,10,10],[4]];dense=np.zeros((4,11))
    for i,t in enumerate(tokens):
        for k in t:dense[i,k]+=1/len(t)
    assert np.allclose(token_gram(tokens),CENTER @ dense @ dense.T @ CENTER,atol=1e-15)
    checks.append('Full-vocabulary sparse histogram contraction equals dense fullvocabulary reference')
    iso=relations(np.stack([CENTER,2*CENTER,np.zeros((4,4))]))
    assert np.allclose(iso['excess'][:2,:2],0,atol=1e-15)
    assert not iso['valid'][2].any()and not iso['valid'][:,2].any()
    checks.append('Isotropic Gram identity has highrawsimilarity butzeroexcess; zeroenergy explicitlyundefined')
    value={'timestamp':stamp(),'all_passed':True,'analysis':snapshot(Path(__file__).with_name('phase2748_rdc_relation_geometry.py')),'checks':checks}
    save(OUT/'unit/relation_geometry_current.json',value);print('NATURAL_RELATION_GEOMETRY_UNIT_PASS',len(checks),flush=True)


if __name__=='__main__':main()
