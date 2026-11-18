"""Read-only preflight and final integrity verification; never recreates scratch."""
import ast,hashlib,json,re,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tests/glm5/result/b36_transition14_20261114_v1'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*2**20),b''):h.update(chunk)
    return h.hexdigest()


def run(final=False):
    for p in (ROOT/'tests/glm5').glob('b36_transition14*.py'):
        ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    text=(OUT/'report.html').read_text(encoding='utf8')
    links=re.findall(r'(?:href|src)="([^"]+)"',text)
    for name in links:
        path=(OUT/name).resolve();assert path.is_relative_to(OUT.resolve()),name
        assert path.is_file() or (not final and name in ['source.zip','manifest.json']),name
    if final:
        manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf8'))
        for entry in manifest['files']:
            path=OUT/entry['path'];assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],entry['path']
        with zipfile.ZipFile(OUT/'source.zip') as z:assert z.testzip() is None
        receipts=json.loads((OUT/'memo-append-receipt.json').read_text(encoding='utf8'))['records']
        for r in receipts:
            data=(ROOT/r['path']).read_bytes()
            assert hashlib.sha256(data[:r['original_bytes']]).hexdigest()==r['original_sha256']
            assert hashlib.sha256(data[:r['original_bytes']+r['appended_bytes']]).hexdigest()==r['after_sha256']
        cleanup=json.loads((OUT/'cleanup.json').read_text(encoding='utf8'))
        assert not Path(cleanup['removed_directory']).exists()
        decision=json.loads((OUT/'stage-decision.json').read_text(encoding='utf8'))
        assert decision['status']=='behavior_passed_native_completed'
        print(json.dumps(dict(verified_manifest_files=len(manifest['files']),memo_phases=[r['phase'] for r in receipts],original_memo_prefixes_unchanged=True,scratch_removed=True,final_status=decision['status']),ensure_ascii=False))
    else:print('Preflight passed: all transition source syntax and',len(links),'report links')


if __name__=='__main__':run(len(sys.argv)>1 and sys.argv[1]=='verify')
