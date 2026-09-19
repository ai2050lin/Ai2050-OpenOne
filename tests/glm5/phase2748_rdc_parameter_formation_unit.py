"""Chunked every-word parameter moments agree with full synthetic computation."""
from rdc_question_common import *
from phase2748_rdc_parameter_formation import pair_moments,cosine


def main():
    original32=(np.arange(63,dtype=np.float32).reshape(7,9)-31)/8
    original=(original32.view(np.uint32)>>16).astype(np.uint16)
    assert np.array_equal(unbits(original),original32)
    left32=original32+np.arange(63,dtype=np.float32).reshape(7,9)/1024
    right32=original32-np.arange(63,dtype=np.float32).reshape(7,9)/2048
    left16=(left32.view(np.uint32)>>16).astype(np.uint16)
    right16=(right32.view(np.uint32)>>16).astype(np.uint16)
    old=original32.astype(float);dl=left32.astype(float)-old;dr=right32.astype(float)-old
    bl=unbits(left16).astype(float)-old;br=unbits(right16).astype(float)-old
    expected={'FP32_delta_dot':float(np.sum(dl*dr)),'BF16_delta_dot':float(np.sum(bl*br)),
        'left_FP32_BF16_delta_dot':float(np.sum(dl*bl)),
        'left_cast_error_squared':float(np.sum((left32.astype(float)-unbits(left16))**2)),
        'original_norm_squared':float(np.sum(old**2)),
        'left_FP32_changed_words':int(np.count_nonzero(left32.view(np.uint32)!=original32.view(np.uint32))),
        'left_BF16_changed_words':int(np.count_nonzero(left16!=original)),
        'left_FP32_maximum_change':float(np.max(np.abs(dl))),'left_BF16_maximum_change':float(np.max(np.abs(bl)))}
    for size in [1,5,32,1048576]:
        actual=pair_moments(original,(left32,left16),(right32,right16),size)
        assert actual==expected,(size,actual,expected)
    gram=np.array([[1.,-2.,0.],[-2.,4.,0.],[0.,0.,0.]])
    assert cosine(gram)==[[1.,-1.,None],[-1.,1.,None],[None,None,None]]
    value={'timestamp':stamp(),'source':snapshot(__file__),'analysis':snapshot(Path(__file__).with_name('phase2748_rdc_parameter_formation.py')),
        'all_passed':True,'checks':['Every declared moment andwordcount equal explicit full synthetic calculation at4chunkwidths',
            'Zero update direction returnsundefinednullcosine, notorthogonality'],
        'scope':'Synthetic CPU arithmetic only. SyntheticBF16values truncateforfixtureconstruction; no claim about actualCUDAcast behavior, which is verified from actualpersistedwords inrealruns.'}
    immutable(OUT/'unit'/('parameter_formation_'+str(time.time_ns())+'.json'),value)
    save(OUT/'unit/parameter_formation_current.json',value)
    print('NATURAL_PARAMETER_FORMATION_UNIT_PASS',5,flush=True)


if __name__=='__main__':main()
