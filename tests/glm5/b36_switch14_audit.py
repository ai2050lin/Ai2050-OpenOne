"""Independent CPU identity/position/logit-array checks, separate from GPU execution."""
import json,sys
import numpy as np
import b36_switch14 as M
S=M.S


def run():
    S.configure();p=S.read(S.OUT/'calibration-plan.json');M.admit(p);a=S.read(S.OUT/'calibration.json')
    assert a['plan_sha256']==S.file_sha(S.OUT/'calibration-plan.json')
    assert a['snapshot_sha256']==S.file_sha(S.OUT/'cache-snapshot.json') and a['logits_sha256']==S.file_sha(S.OUT/'cache-first-logits.npz')
    snap=S.read(S.OUT/'cache-snapshot.json');prefix=np.asarray(snap['prefix_ids']);mask=np.asarray(snap['prepared_inputs']['attention_mask']);positions=np.asarray(snap['prepared_inputs']['position_ids']);pending=np.asarray(snap['prepared_inputs']['input_ids'])
    original=S.read(S.OUT/'cache-original.json')['results'];replay=S.read(S.OUT/'cache-replay.json')['results']
    assert len(p['rows'])==len(original)==len(replay)==96
    assert np.array_equal(pending,prefix[:,-1:]) and np.array_equal(positions,mask.sum(1)[:,None]-1)
    assert prefix.shape==mask.shape and mask.dtype.kind in 'biu'
    for row,x,y,t,m in zip(p['rows'],original,replay,prefix,mask):
        assert row['id']==x['id']==y['id']
        assert list(t[m.astype(bool)])==row['tokens']+x['tokens'][:8]
        assert x['tokens']==y['tokens']
    cfg=S.read(S.MODEL/'config.json')
    expected=[96,cfg['num_key_value_heads'],prefix.shape[1]-1,cfg['head_dim']]
    assert len(snap['layers'])==cfg['num_hidden_layers']==40
    assert all(r['keys_shape']==r['values_shape']==expected and r['dtype']=='torch.bfloat16' for r in snap['layers'])
    with np.load(S.OUT/'cache-first-logits.npz') as z:
        assert z['original'].dtype==z['replay'].dtype==np.dtype('uint16')
        assert z['original'].shape==z['replay'].shape==(96,cfg['vocab_size'])
        assert np.array_equal(z['original'],z['replay'])
    S.save('calibration-independent-audit.json',dict(passed=True,n=96,n_layers=40,kv_shape=expected,first_logits_shape=[96,cfg['vocab_size']],pending_token_equal=True,positions_from_actual_mask_equal=True,prefixes_equal=True,raw_logits_arrays_bitwise=True,
        scope='CPU independent reconstruction of saved token prefixes, masks, position IDs and logits arrays. KV tensor bytes were transient; their unchanged hashes are execution attestation, not an independent later rehash of the discarded full tensors.',source_sha256=S.file_sha(__file__)))
    print('CPU cache audit passed:96 prefixes, pending tokens, positions,40 shapes and complete logits',flush=True)


if __name__=='__main__':run()
