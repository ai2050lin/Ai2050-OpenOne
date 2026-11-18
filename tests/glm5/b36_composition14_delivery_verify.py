"""Read-only final cross-artifact check; no model inference or result mutation."""
import hashlib,json,zipfile
from pathlib import Path
import b36_composition14 as B
import b36_composition14_collect as C
import b36_composition14_tail as T


def sha(data):return hashlib.sha256(data).hexdigest()


def run():
    C.admit();T.admit()
    manifest=B.read(B.OUT/'manifest.json')
    for name,f in manifest['files'].items():
        p=B.OUT/name
        assert p.stat().st_size==f['bytes'] and B.file_sha(p)==f['sha256'],name
    delivery=B.read(B.OUT/'delivery-source.json')
    assert B.file_sha(B.OUT/delivery['archive'])==delivery['sha256']
    with zipfile.ZipFile(B.OUT/delivery['archive']) as z:
        assert z.testzip() is None
        assert set(z.namelist())=={f['path'] for f in delivery['files']}
        for f in delivery['files']:
            data=z.read(f['path']);assert len(data)==f['bytes'] and sha(data)==f['sha256']
            p=B.ROOT/f['path']
            if p.exists():assert B.file_sha(p)==f['sha256'],f['path']
            else:assert f['path'].startswith('tests/glm5_temp/'),f['path']
    for name in ['report-build.json','memo-append.json']:
        d=B.read(B.OUT/name)
        for file,expected in d['inputs'].items():assert B.file_sha(B.OUT/file)==expected,(name,file)
    for r in B.read(B.OUT/'memo-append.json')['receipts']:
        data=(B.ROOT/r['path']).read_bytes();n=r['old_bytes'];m=r['appended_bytes']
        assert sha(data[:n])==r['before_sha256']
        assert sha(data[n:n+m])==r['append_sha256']
        assert sha(data[:n+m])==r['prefix_after_sha256']
        assert f'## Phase {r["phase"]}:' in data[n:n+m].decode('utf8')
    browser=B.read(B.OUT/'browser-check.json');audit=B.read(B.OUT/'audit.json')
    assert browser['passed'] and audit['passed']
    assert browser['report_sha256']==B.file_sha(B.OUT/'report.html')==B.read(B.OUT/'report-build.json')['report_sha256']
    for name,expected in browser['screenshots'].items():assert B.file_sha(B.OUT/name)==expected
    cards=B.read(B.OUT/'feature-cards.json')
    for name,expected in cards['evidence'].items():assert B.file_sha(B.OUT/name)==expected
    for r in B.read(B.OUT/'cleanup.json')['removed']:
        p=Path(r['path']);assert not p.exists(),p
    assert 'All planned research computation complete' in (B.OUT/'pipeline.log').read_text(encoding='utf8')
    print(json.dumps(dict(passed=True,manifest_files=len(manifest['files']),archived_sources=len(delivery['files']),
        append_only_memos=2,source_freezes_intact=True,report_and_evidence_consistent=True,owned_scratch_absent=True),ensure_ascii=False),flush=True)


if __name__=='__main__':run()
