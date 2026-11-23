"""CPU-only storage-contract checks using tiny deterministic synthetic BF16 arrays."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from deploy.storage_policy import check_write
from shared.dist_protocol.mechanism_trace import TraceStore, digest, file_sha, unbits
from server import mechanism_trace_service as service


class NativeHeadTest(unittest.TestCase):
    def setUp(self):
        parent=check_write(ROOT/'tests/glm5_temp',2*1024**2,operation='Native field contract fixture')
        parent.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.store=TraceStore(self.root/'index',self.root)
        mock=patch.object(service,'store',return_value=self.store)
        mock.start();self.addCleanup(mock.stop)

    def save(self,deep=True,cached=False,corrupt_axis=False):
        # Numbers supply predictable but distinct channel, head and token values.
        shape=(4,1 if cached else 4,3)
        def bits(shape,offset):
            a=np.arange(np.prod(shape),dtype=np.float32).reshape(shape)+offset
            return (a.view(np.uint32)>>16).astype(np.uint16)
        self.arrays={'head_z':bits((1,4,3),50),'activation':bits((1,9),7)}
        if deep:
            self.arrays.update(q=bits(shape,1),k=bits((2,3 if corrupt_axis else 4,3),30),v=bits((2,4,3),60))
        f=self.root/'block_00.npz';np.savez(f,**self.arrays)
        manifest={'schema':'mechanism_trace.v1','kind':'real','run_id':'isolated-synthetic-software-fixture',
                  'n_heads':4,'n_kv_heads':2,'head_dim':3,'n_layers':1,'n_tokens':4,
                  'model_snapshot':'c'*64,'readout_positions':[3],'deep':deep,
                  'generation_step':int(cached),'parent_trace_id':'prefix' if cached else None,
                  'files':[{'path':f.name,'bytes':f.stat().st_size,'sha256':file_sha(f)}]}
        self.identity=digest(manifest)
        manifest['trace_id']=self.identity
        p=self.root/'manifest.json';p.write_text(json.dumps(manifest),encoding='utf8')
        self.store.register(p)

    def field(self,measure,position=3,head=None):
        return service.field(self.identity,block=0,position=position,boundary=measure,head=head)

    def expect_error(self,status,*args,**kwargs):
        with self.assertRaises(HTTPException) as raised:self.field(*args,**kwargs)
        self.assertEqual(raised.exception.status_code,status)

    def test_q_and_shared_kv(self):
        self.save()
        for measure,key,h,source_head in [('head_q','q',2,2),('head_k','k',2,1),('head_v','v',3,1)]:
            with self.subTest(key=key):
                d=self.field(measure,1,h)
                self.assertEqual((d['trace_id'],d['block'],d['head'],d['kv_group'],d['position'],d['shape']),
                                 (self.identity,0,h,1,1,[3]))
                self.assertEqual(d['values'],unbits(self.arrays[key][source_head,1]).tolist())

    def test_old_protocol_full_matrix_and_ffn(self):
        self.save(deep=False)
        old=self.field('head_z');self.assertEqual(old['shape'],[4,3])
        new=self.field('head_z',head=2)
        self.assertEqual(new['values'],old['values'][6:9])
        self.assertEqual(self.field('mlp_activation')['shape'],[9])
        self.expect_error(409,'head_q',head=1)
        self.expect_error(409,'head_z',position=0,head=1)
        self.expect_error(422,'mlp_activation',head=0)
        self.expect_error(422,'head_z',head=4)
        self.expect_error(422,'head_z',position=4,head=0)

    def test_cached_suffix_q_and_full_prefix_kv(self):
        self.save(cached=True)
        self.assertEqual(self.field('head_q',3,3)['values'],unbits(self.arrays['q'][3,0]).tolist())
        self.expect_error(409,'head_q',1,3)
        self.assertEqual(self.field('head_k',1,1)['values'],unbits(self.arrays['k'][0,1]).tolist())
        self.assertEqual(self.field('head_v',0,2)['values'],unbits(self.arrays['v'][1,0]).tolist())

    def test_incompatible_field_axis_rejected(self):
        self.save(cached=True,corrupt_axis=True)
        self.expect_error(422,'head_k',2,1)


if __name__=='__main__':unittest.main()
