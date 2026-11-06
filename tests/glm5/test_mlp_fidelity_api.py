"""Node-local evidence guards, using CPU filesystem fixtures only."""
import sys,tempfile,json,gc,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from shared.dist_protocol.mechanism_trace import file_sha
from fastapi import HTTPException
class EvidenceGuards(unittest.TestCase):
    def setUp(self):
        from server import mlp_fidelity_service as S
        self.S=S;self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'tests/glm5_temp');self.root=Path(self.tmp.name);self.patch=patch.object(S,'OUT',self.root);self.patch.start()
    def tearDown(self):self.patch.stop();gc.collect();self.tmp.cleanup()
    def test_missing_research_step_is_not_replaced(self):
        with self.assertRaises(HTTPException) as e:self.S.layer(35)
        self.assertEqual(e.exception.status_code,409)
    def test_native_dtype_and_shape_are_verified_after_hash(self):
        p=self.root/'f.npy';np.save(p,np.zeros((2,3,4),np.float32));meta={'sha256':file_sha(p),'shape':[2,3,4]}
        with self.assertRaises(HTTPException) as e:self.S.native(p,meta)
        self.assertEqual(e.exception.status_code,409)
        np.save(p,np.zeros((2,3,4),np.uint16));meta={'sha256':file_sha(p),'shape':[2,3,5]}
        with self.assertRaises(HTTPException):self.S.native(p,meta)
    def test_corrupted_bytes_are_rejected(self):
        p=self.root/'f.npy';np.save(p,np.zeros((2,3,4),np.uint16))
        with self.assertRaises(HTTPException):self.S.native(p,{'sha256':'0'*64,'shape':[2,3,4]})
    def test_index_and_task_type_bounds_precede_native_read(self):
        for call in [lambda:self.S.natural(512,0,35),lambda:self.S.natural(0,8,35),lambda:self.S.tail('../../private',0,0),lambda:self.S.unit(35,9728)]:
            with self.assertRaises(HTTPException) as e:call()
            self.assertEqual(e.exception.status_code,422)
if __name__=='__main__':unittest.main()
