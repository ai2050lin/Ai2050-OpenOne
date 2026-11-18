"""Engineering-only version isolation and delta-install recovery checks."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
from test_qwen14_pack import PackChecks
from deploy import qwen14_mlp_pack as P
from fastapi import HTTPException
from server import qwen14_mlp_service as S


class VersionChecks(unittest.TestCase):
    def test_versions_and_bounds_are_independent(self):
        for run,last in [(P.RUN,35),(P.RUN_V2,39)]:
            with patch.object(S,'manifest',return_value={'run_id':run,'blocks':list(range(28,last+1))}):
                self.assertEqual(S.check(last,17407,run=run)['run_id'],run)
                with self.assertRaises(HTTPException):S.check(last+1,0,run=run)
                with self.assertRaises(HTTPException):S.check(last,17408,run=run)

    def test_source_mapping_keeps_old_effects_and_extrema(self):
        for b in [28,35,36,39]:
            for n in ['prefix-extrema.npz','natural_exact-effects.json','capability-inputs.json']:
                self.assertEqual(S.legacy_name({'run_id':P.RUN_V2},b,n),('legacy-' if b<36 else '')+n)
                self.assertEqual(S.legacy_name({'run_id':P.RUN},b,n),n)

    def test_foreign_run_and_snapshot_fail(self):
        with self.assertRaises(HTTPException):S.folder('untrusted-path/../x')
        with self.assertRaises(HTTPException):S.check(snapshot='wrong')


class DeltaChecks(PackChecks):
    def delta(self):
        doc=copy.deepcopy(self.manifest);doc.update(schema='qwen14_mlp_manifest.v2',run_id=P.RUN_V2,blocks=list(range(28,40)))
        for r in doc['files']:r.update(source_run=P.RUN,source_path=r['path'])
        extra={f'B{b}-{s}.npy' for b in range(36,40) for s in ['G','V','A','input','W-gate','W-up','W-down']}
        extra.update(f'B{b}-{s}.json' for b in range(36,40) for s in ['profile','cues','prediction','readout'])
        extra.update(f'B{b}-{s}.npz' for b in range(28,40) for s in ['regimes','scaled-prediction'])
        extra.update({'legacy-prefix-extrema.npz','natural_exact-effects.json','capability_exact-effects.json','legacy-natural_exact-effects.json','legacy-capability_exact-effects.json','capability-inputs.json','legacy-capability-inputs.json'})
        data=b'isolated v2 engineering fixture'
        doc['files'] += [{'path':n,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for n in sorted(extra)]
        archive=self.root/'delta.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            raw=json.dumps(doc).encode();m=tarfile.TarInfo('manifest.json');m.size=len(raw);tar.addfile(m,io.BytesIO(raw))
            for r in doc['files']:
                if 'source_run' in r:continue
                m=tarfile.TarInfo(r['path']);m.size=len(data);tar.addfile(m,io.BytesIO(data))
        return archive,doc

    def test_delta_missing_base_recovers_after_install(self):
        archive,doc=self.delta();data=self.root/'data'
        with self.assertRaisesRegex(ValueError,'base atlas first'):P.install(archive,data)
        self.assertFalse((data/'native_atlases'/P.RUN_V2).exists())
        self.assertEqual(list((data/'native_atlases').iterdir()),[])
        base=self.root/'base.tar.gz';P.build(self.source,base);P.install(base,data)
        self.assertEqual(P.install(archive,data)['state'],'installed')
        self.assertEqual(P.install(archive,data)['state'],'already_installed')
        self.assertEqual((data/'native_atlases'/P.RUN_V2/'report.json').stat().st_ino,(data/'native_atlases'/P.RUN/'report.json').stat().st_ino)

    def test_delta_base_tamper_rejected(self):
        archive,doc=self.delta();data=self.root/'data';base=self.root/'base.tar.gz'
        P.build(self.source,base);P.install(base,data)
        (data/'native_atlases'/P.RUN/doc['files'][0]['path']).write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'base atlas first'):P.install(archive,data)
        self.assertFalse((data/'native_atlases'/P.RUN_V2).exists())

    def test_reuse_path_cannot_escape_base(self):
        _,doc=self.delta()
        for name in ['../x','x/y','x\\y','D:x']:
            altered=copy.deepcopy(doc);altered['files'][0]['source_path']=name
            with self.assertRaises(ValueError):P.validate(altered)


if __name__=='__main__':unittest.main()
