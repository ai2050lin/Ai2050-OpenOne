import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts.apply_client_deployment_data import apply, digest


class DeploymentDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / 'center'
        self.data.mkdir()
        (self.data / 'dist.sqlite3').write_bytes(b'existing account database must stay unchanged')
        self.state = {'schema':'research_mainline.v1','kind':'real','sequence':4}
        self.revision = digest(self.state)

    def bundle(self, corrupt=False, unsafe=False):
        extras = {'research-library':{'version':'v1','created_at':1}}
        entries = {
            'mainline_release.json': json.dumps({'state':self.state}).encode(),
            'datapack/datapack.json.gz': gzip.compress(json.dumps({'mainline_revision':self.revision}).encode()),
            'datapack/datapack.meta.json': b'{"version":"v1"}',
            'datapack/extras.json': json.dumps(extras).encode(),
            'datapack/extra/research-library.json.gz': gzip.compress(b'{"schema":"datapack_extra.v1"}'),
        }
        if unsafe:
            entries['../outside.txt'] = b'unsafe'
        meta = {'schema':'client_deployment_data.v1','mainline_revision':self.revision,
                'files':{k:{'sha256':hashlib.sha256(v).hexdigest(),'bytes':len(v)} for k,v in entries.items()}}
        if corrupt:
            meta['files']['datapack/datapack.meta.json']['sha256'] = '0' * 64
        entries['DATA_MANIFEST.json'] = json.dumps(meta).encode()
        file = self.root / 'data.tar'
        with tarfile.open(file,'w') as tar:
            for name,raw in entries.items():
                info = tarfile.TarInfo(name); info.size = len(raw)
                tar.addfile(info,io.BytesIO(raw))
        return file

    def test_verify_only_does_not_write(self):
        result = apply(self.bundle(),self.data)
        self.assertTrue(result['verified']); self.assertFalse(result['applied'])
        self.assertFalse((self.data/'datapack').exists())

    def test_apply_duplicate_and_account_preservation(self):
        bundle = self.bundle()
        first = apply(bundle,self.data,write=True)
        second = apply(bundle,self.data,write=True)
        self.assertEqual(first['files'],4)
        self.assertEqual(second['files'],0)
        self.assertEqual((self.data/'dist.sqlite3').read_bytes(),b'existing account database must stay unchanged')
        self.assertFalse(list(self.data.glob('.client-data-stage-*')))

    def test_corruption_and_traversal_fail_before_mutation(self):
        for opts in ({'corrupt':True},{'unsafe':True}):
            with self.assertRaises(ValueError):apply(self.bundle(**opts),self.data,write=True)
        self.assertFalse((self.data/'datapack').exists())
        self.assertFalse((self.root/'outside.txt').exists())

    def test_newer_extra_is_preserved(self):
        bundle = self.bundle()
        apply(bundle,self.data,write=True)
        p=self.data/'datapack/extra/research-library.json.gz'; p.write_bytes(b'newer existing pack')
        (self.data/'datapack/extras.json').write_text(json.dumps({'research-library':{'version':'v2','created_at':100},'server-only':{'version':'x'}}))
        result = apply(bundle,self.data,write=True)
        self.assertEqual(result['preserved_newer_extras'],1)
        self.assertEqual(p.read_bytes(),b'newer existing pack')
        extras=json.loads((self.data/'datapack/extras.json').read_text())
        self.assertEqual(extras['research-library']['version'],'v2');self.assertIn('server-only',extras)

    def test_incomplete_newer_extra_is_repaired(self):
        bundle=self.bundle();apply(bundle,self.data,write=True)
        (self.data/'datapack/extra/research-library.json.gz').unlink()
        (self.data/'datapack/extras.json').write_text(json.dumps({'research-library':{'version':'v2','created_at':100}}))
        apply(bundle,self.data,write=True)
        self.assertTrue((self.data/'datapack/extra/research-library.json.gz').is_file())
        self.assertEqual(json.loads((self.data/'datapack/extras.json').read_text())['research-library']['version'],'v1')

    def test_write_failure_restores_replaced_file(self):
        bundle=self.bundle()
        old=self.data/'datapack/datapack.json.gz';old.parent.mkdir();old.write_bytes(b'old cache')
        import os
        original=os.replace
        def fail_once(src,dst):
            if '.client-data-stage-' in str(src) and str(dst).endswith('datapack.json.gz'):
                raise OSError('injected disk failure')
            return original(src,dst)
        with patch('scripts.apply_client_deployment_data.os.replace',side_effect=fail_once):
            with self.assertRaises(OSError):apply(bundle,self.data,write=True)
        self.assertEqual(old.read_bytes(),b'old cache')
        self.assertFalse(list(self.data.glob('.client-data-stage-*')))

    def test_future_center_state_blocks_cache_downgrade(self):
        lib=self.data/'research_library';lib.mkdir()
        newer=copy.deepcopy(self.state);newer['sequence']=5;sha=digest(newer)
        obj=lib/'objects'/sha[:2]/(sha+'.json');obj.parent.mkdir(parents=True);obj.write_text(json.dumps(newer))
        with sqlite3.connect(lib/'library.sqlite3') as db:
            db.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)')
            db.execute('INSERT INTO metadata VALUES(?,?)',('research_mainline.current',sha))
        db.close()
        with self.assertRaisesRegex(ValueError,'downgrade refused'):apply(self.bundle(),self.data,write=True)
        self.assertFalse((self.data/'datapack').exists())


if __name__ == '__main__':unittest.main()
