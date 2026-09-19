"""Synthetic in-memory storage tests; not native model or learned endpoints."""
import io
from rdc_question_common import *
from phase2748_rdc_integrity import inspect_archive


def packet(**arrays):
    stream=io.BytesIO();np.savez_compressed(stream,**arrays);stream.seek(0);return stream


def rejects(stream,name):
    try:inspect_archive(stream,name)
    except AssertionError:return True
    return False


def main():
    records=[]
    actual=inspect_archive(packet(native_BF16=np.array([0,0x3f80,0xbf80],np.uint16),
        values=np.asfortranarray(np.arange(12,dtype=float).reshape(3,4)),flag=np.array(True)),
        'phase2748/field_store/unit/native.npz')
    assert [r['scalars']for r in actual]==[3,12,1] and actual[1]['fortran']
    records.append('All scalar counts, shape/dtype, Boolean scalar and Fortran layout retained')
    assert rejects(packet(native_BF16=np.array([0x7f80],np.uint16)),'phase2748/field_store/unit/nonfinite.npz')
    assert rejects(packet(values=np.array([np.nan])),'phase2748/field_store/unit/nonfinite.npz')
    records.append('BF16infinity and FP64NaN rejected')
    encoded=inspect_archive(packet(FP32_XOR_words=np.array([0xffffffff],np.uint32),
        BF16_XOR_words=np.array([0x7f80,0xffff],np.uint16)),
        'phase2748/field_store/training/synthetic_not_a_real_run/checkpoint96/gate.npz')
    assert all(r['interpretation'].startswith('Lossless XOR')for r in encoded)
    records.append('Encoded XOR words with apparent float nonfinite bitpatterns are correctly treated as bitstrings')
    assert rejects(packet(BF16_XOR_words=np.array([0xffff],np.uint16)),'phase2748/field_store/unit/wrong_scope.npz')
    assert rejects(packet(FP32_XOR_words=np.array([3],np.uint16)),
        'phase2748/field_store/training/synthetic_not_a_real_run/checkpoint96/wrong_dtype.npz')
    records.append('XOR outside registered checkpoint scope and wrong XOR storage dtype rejected')
    assert rejects(packet(values=np.array([{'unsafe':'object'}],dtype=object)),'phase2748/field_store/unit/object.npz')
    records.append('Object array rejected without object loading')
    value={'timestamp':stamp(),'all_passed':True,'source':snapshot(__file__),
        'analysis':snapshot(Path(__file__).with_name('phase2748_rdc_integrity.py')),
        'checks':records,'scope':'Five synthetic storage cases entirely in memory. No actual archive, checkpoint or scientific endpoint claimed audited by this unit.'}
    save(OUT/'unit/integrity_current.json',value)
    print('NATURAL_INTEGRITY_UNIT_PASS',len(records),flush=True)


if __name__=='__main__':main()
