"""CPU engineering fixtures only; fail closed on wrong atlas identity and damaged files."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from server import native_mlp_service as S
from deploy import native_mlp_pack as P
from fastapi import HTTPException
from deploy.storage_policy import check_write


class NativeServiceChecks(unittest.TestCase):
    def setUp(self):
        root=check_write(ROOT/'tests/glm5_temp/native-service-tests');root.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=root);self.base=Path(self.tmp.name)
        self.env=patch.dict(os.environ,{'AI2050_NATIVE_MLP_ROOT':str(self.base)});self.env.start()
        self.key='qwen4';self.spec=S.SPECS[self.key];self.run=self.spec['run_id'];self.folder=self.base/self.key;self.folder.mkdir()
        self.raw=b'{"observation":true}';(self.folder/'data.json').write_bytes(self.raw)
        self.doc=dict(schema='native_mlp_manifest.v1',state='complete',kind='real',run_id=self.run,
            model_snapshot=self.spec['snapshot'],width=self.spec['width'],hidden_size=self.spec['dim'],blocks=list(range(18,29)),
            files=[dict(path='data.json',bytes=len(self.raw),sha256=hashlib.sha256(self.raw).hexdigest())])
        self.save();S._checked.clear();S._read.cache_clear()

    def save(self):(self.folder/'manifest.json').write_text(json.dumps(self.doc),encoding='utf-8')
    def tearDown(self):self.env.stop();self.tmp.cleanup();S._checked.clear();S._read.cache_clear()
    def rejects(self,code,fn,*args,**kw):
        with self.assertRaises(HTTPException) as c:fn(*args,**kw)
        self.assertEqual(c.exception.status_code,code)

    def test_both_known_model_shapes(self):
        self.assertEqual(S.spec(self.run)[1]['width'],9728)
        self.assertEqual(S.spec(S.SPECS['qwen14']['run_id'])[1]['dim'],5120)
        self.rejects(404,S.spec,'unknown')

    def test_identity_and_dimensions(self):
        for k,v in [('hidden_size',5120),('width',17408),('state','pending'),('kind','synthetic'),('model_snapshot','wrong')]:
            original=self.doc[k];self.doc[k]=v;self.save();self.rejects(409,S.manifest,self.run)
            self.doc[k]=original
        self.save();self.assertEqual(S.manifest(self.run)['run_id'],self.run)

    def test_bounds_and_snapshot(self):
        self.rejects(422,S.check,17,0,run=self.run)
        self.rejects(422,S.check,28,9728,run=self.run)
        self.rejects(409,S.check,18,0,'wrong',self.run)
        self.assertEqual(S.check(28,9727,run=self.run)['width'],9728)

    def test_integrity_damage_and_repair(self):
        self.assertEqual(S.read('data.json',self.run),{'observation':True})
        (self.folder/'data.json').write_bytes(b'{"observation":fals}')
        self.rejects(409,S.read,'data.json',self.run)
        (self.folder/'data.json').write_bytes(self.raw)
        self.assertEqual(S.read('data.json',self.run),{'observation':True})

    def test_duplicate_path_and_missing_artifact(self):
        self.rejects(404,S.path,'absent.json',self.run)
        self.doc['files']*=2;self.save();self.rejects(409,S.manifest,self.run)

    def test_path_traversal_rejected(self):
        self.doc['files'][0]['path']='../data.json';self.save()
        self.rejects(409,S.path,'../data.json',self.run)
        with self.assertRaisesRegex(ValueError,'Unsafe'):P.validate(self.doc)

    def test_package_requires_all_query_dependencies(self):
        with self.assertRaisesRegex(ValueError,'Missing'):P.validate(self.doc)
        self.doc['width']=1
        with self.assertRaisesRegex(ValueError,'Wrong'):P.validate(self.doc)

    def test_json_run_mismatch(self):
        raw=json.dumps(dict(run_id='another-run')).encode();(self.folder/'data.json').write_bytes(raw)
        self.doc['files'][0].update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());self.save()
        self.rejects(409,S.read,'data.json',self.run)

    def test_transport_roundtrip_retry_and_damage(self):
        # Intentionally tiny engineering fixtures; never registered as research.
        names={'report.json','audit.json','documents.json','tokenizer.json','prefix-extrema.npz','QWEN-LICENSE.txt','WEIGHT-NOTICE.txt',
               'natural_exact-effects.json','capability_exact-effects.json','capability-inputs.json'}
        names.update(f'B{b}-{s}.npy' for b in range(18,29) for s in ['G','V','A','input','W-gate','W-up','W-down'])
        names.update(f'B{b}-{s}.json' for b in range(18,29) for s in ['profile','cues','readout'])
        names.update(f'B{b}-{s}.npz' for b in range(18,29) for s in ['regimes','prediction'])
        names.update(f'B{b}-B{b+1}-prediction.json' for b in range(18,28))
        records=[]
        for name in sorted(names):
            raw=('fixture-only:'+name).encode();(self.folder/name).write_bytes(raw)
            records.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        self.doc['files']=records;self.save();archive=self.base/'transport.tar.gz';dest=self.base/'installed'
        with patch.object(P.transport,'validate',P.validate):
            P.transport.build(self.folder,archive)
            self.assertEqual(P.transport.install(archive,dest)['state'],'installed')
            self.assertEqual(P.transport.install(archive,dest)['state'],'already_installed')
            installed=dest/'native_atlases'/self.run/'report.json';installed.write_text('damaged')
            with self.assertRaisesRegex(ValueError,'damaged'):P.transport.install(archive,dest)
            installed.write_bytes((self.folder/'report.json').read_bytes())
            self.assertEqual(P.transport.install(archive,dest)['state'],'already_installed')


if __name__=='__main__':unittest.main(verbosity=2)
