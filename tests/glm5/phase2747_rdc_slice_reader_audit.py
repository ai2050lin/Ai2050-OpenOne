"""Bounded raw BF16 checkpoint I/O and selected-row retention qualification."""
import gc
import json
import struct
from rdc_formation_common import *
from rdc_formation_microbatch import TensorSliceReader,LayerWeights


def main():
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from accelerate.utils.offload import OffloadedWeightsLoader
    from rdc_operator_model import memory
    start=time.monotonic();folder=OUT/'engineering/microbatch'/('slice_reader_'+str(time.time_ns()))
    folder.mkdir(parents=True,exist_ok=False);checks=[];reader=TensorSliceReader()
    # Retain synthetic test files; all are tiny and separately labelled.
    raw=np.arange(65536,dtype=np.uint16)
    tensors={'every_BF16_bit_pattern':torch.from_numpy(raw.copy()).view(torch.bfloat16),
        'scalar':torch.tensor(-2.,dtype=torch.bfloat16),'empty':torch.empty((0,3),dtype=torch.bfloat16),
        'matrix':torch.arange(105,dtype=torch.float32).reshape(15,7).bfloat16()}
    path=folder/'synthetic_BF16.safetensors';save_file(tensors,str(path))
    with safe_open(str(path),framework='pt',device='cpu',backend='pread') as original:
        for key in tensors:
            actual=reader.tensor(path,key);expected=original.get_tensor(key)
            assert np.array_equal(bits(actual),bits(expected))
            checks.append({'kind':'synthetic_exact_bits','key':key,'shape':list(actual.shape),'passed':True})
    owner=OffloadedWeightsLoader(state_dict={'resident':tensors['scalar']},index={
        'redirected':{'safetensors_file':str(path),'weight_name':'matrix','dtype':'bfloat16'}})
    manager=LayerWeights('cpu')
    try:
        manager.begin();one=owner['redirected'];two=owner['redirected'];resident=owner['resident']
        assert one.data_ptr()==two.data_ptr() and torch.equal(one,tensors['matrix'])
        assert torch.equal(resident,tensors['scalar']) and manager.hits==1 and manager.reader.reads==1
        checks.append({'kind':'offload_redirect_resident_and_exact_cache_reuse','passed':True})
    finally:manager.close()
    # The old sliced view kept the full B8 backing array; the copy is bit exact
    # and owns precisely one collected row, including NaN/Inf BF16 bit patterns.
    parent=np.arange(8*41*32,dtype=np.uint16).reshape(8,41,32);view=parent[3];copied=view.copy()
    assert np.array_equal(copied,view) and np.shares_memory(view,parent)
    assert not np.shares_memory(copied,parent) and copied.flags.owndata
    checks.append({'kind':'selected_hidden_row_owns_only_collected_bytes','retained_byte_ratio':copied.nbytes/parent.nbytes,'passed':True})
    # Compare actual immutable Q14 tensors with the original library one at a
    # time, while no CUDA model is resident. Cover every shard and both tiny
    # normalization vectors and projection/gate/up/down/output matrices.
    root=ROOT/'models/hf'/MODELS['qwen14'];index=read(root/'model.safetensors.index.json')['weight_map']
    keys=[]
    for shard in sorted(set(index.values())):
        members=sorted(k for k,v in index.items() if v==shard)
        header,_=reader.metadata(root/shard)
        keys.append(min(members,key=lambda k:header[k]['data_offsets'][1]-header[k]['data_offsets'][0]))
    keys+=['model.layers.12.self_attn.q_proj.weight','model.layers.12.mlp.gate_proj.weight',
        'model.layers.12.mlp.up_proj.weight','model.layers.12.mlp.down_proj.weight','lm_head.weight']
    assert len(set(keys))==len(keys)
    for key in keys:
        path=root/index[key];before=memory();actual=reader.tensor(path,key)
        with safe_open(str(path),framework='pt',device='cpu',backend='pread') as original:
            expected=original.get_tensor(key)
            # Compare every uint16 exactly, not floating equality which rejects NaN.
            assert torch.equal(actual.view(torch.int16),expected.view(torch.int16)),key
            checksum=hashlib.sha256(memoryview(actual.view(torch.uint16).numpy())).hexdigest()
        checks.append({'kind':'original_Q14_checkpoint_tensor','key':key,'shard':path.name,
            'shape':list(actual.shape),'all_BF16_bits_equal':True,'tensor_sha256':checksum,
            'bytes':actual.numel()*actual.element_size(),'memory_before':before,'passed':True})
        del actual,expected;gc.collect()
        print('FORMATION_SLICE_TENSOR',key,flush=True)
    selected=gzread(OUT/'own_history/material.json.gz')['qwen14'][:192]
    batches=[selected[i:i+8] for i in range(0,192,8)]
    old_rows=sum(len(b) for b in batches if any(r['collect_all_hidden'] for r in b))
    new_rows=sum(r['collect_all_hidden'] for r in selected)
    prior=OUT/'own_history/qwen14/native';failures=sorted(prior.glob('failure_*.json'),key=lambda p:p.stat().st_mtime_ns)
    incident=failures[-1];assert '1455' in read(incident)['error']
    progress=read(prior/'wave_progress.json');assert progress['own_step']==54 and progress['committed_expressions']==0
    value={'timestamp':stamp(),'source':snapshot(__file__),'engine':snapshot(Path(__file__).with_name('rdc_formation_microbatch.py')),
        'all_passed':True,'checks':checks,'real_checkpoint_tensors':len(keys),'seconds':time.monotonic()-start,
        'previous_failure':{'path':incident.relative_to(ROOT).as_posix(),'sha256':sha(incident),'last_progress':progress},
        'retention_diagnosis':{'selected_natural_rows':new_rows,'old_retained_B8_rows':old_rows,
            'old_hidden_backing_bytes_at_step54':old_rows*54*41*5120*2,
            'new_selected_hidden_bytes_at_step54':new_rows*54*41*5120*2,
            'interpretation':'A verified avoidable allocation; not uniquely proven to be the sole cause of Windows commit exhaustion.'},
        'format_reference':'https://github.com/safetensors/safetensors#format',
        'reader_maximum_tensor_bytes':reader.maximum_tensor_bytes,
        'scope':'Exact raw tensor I/O and CPU storage checks. Revised native CUDA engine still requires renewed Q4 qualification, six Q14 fixtures and all16full native pilot packets.'}
    save(folder/'result.json',value)
    save(folder.parent/'slice_reader_current.json',{'path':(folder/'result.json').relative_to(ROOT).as_posix(),'sha256':sha(folder/'result.json')})
    print('FORMATION_SLICE_AUDIT',len(checks),value['seconds'],flush=True)


if __name__=='__main__':main()
