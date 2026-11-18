"""Tiny engineering-only archives; never ingested into the research library."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from deploy import qwen14_mlp_pack as P


class PackChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).parents[1]/'glm5_temp')
        self.root=Path(self.tmp.name);self.source=self.root/'source';self.source.mkdir()
        names={'report.json','documents.json','tokenizer.json','prefix-extrema.npz','QWEN-LICENSE.txt','WEIGHT-NOTICE.txt'}
        names.update(f'B{b}-{s}.npy' for b in range(28,36) for s in ['G','V','A','input','W-gate','W-up','W-down'])
        names.update(f'B{b}-{s}.json' for b in range(28,36) for s in ['profile','cues','prediction','readout'])
        data=b'isolated engineering fixture'
        self.manifest={'schema':'qwen14_mlp_manifest.v1','state':'complete','kind':'real','run_id':P.RUN,
            'model_snapshot':P.SNAPSHOT,'blocks':list(range(28,36)),'width':17408,
            'files':[{'path':n,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for n in sorted(names)]}
        for n in names:(self.source/n).write_bytes(data)
        (self.source/'manifest.json').write_text(json.dumps(self.manifest),encoding='utf-8')
    def tearDown(self):self.tmp.cleanup()

    def test_install_verify_and_idempotence(self):
        out=self.root/'pack.tar.gz';P.build(self.source,out)
        self.assertEqual(P.install(out,self.root/'data')['state'],'installed')
        self.assertEqual(P.install(out,self.root/'data')['state'],'already_installed')
        (self.root/'data/native_atlases'/P.RUN/'report.json').write_text('damaged')
        with self.assertRaises(ValueError):P.install(out,self.root/'data')

    def test_corruption_rejected_without_published_directory(self):
        out=self.root/'corrupt.tar.gz'
        with tarfile.open(out,'w:gz') as tar:
            tar.add(self.source/'manifest.json',arcname='manifest.json')
            record=self.manifest['files'][0];m=tarfile.TarInfo(record['path']);m.size=record['bytes']
            tar.addfile(m,io.BytesIO(b'x'*m.size))
        with self.assertRaises(ValueError):P.install(out,self.root/'data')
        self.assertEqual(list((self.root/'data/native_atlases').iterdir()),[])

    def test_traversal_and_foreign_model_rejected(self):
        for key,value in [('width',9728),('model_snapshot','foreign')]:
            with self.assertRaises(ValueError):P.validate({**self.manifest,key:value})
        for name in ['../escape','x/y','x\\y','C:escape','manifest.json']:
            bad=json.loads(json.dumps(self.manifest));bad['files'][0]['path']=name
            with self.assertRaises(ValueError):P.validate(bad)

if __name__=='__main__':unittest.main()
