"""Acceptance of the real release cache: hashes, dry run, apply and idempotent replay."""
from pathlib import Path
import gzip
import json
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.apply_client_deployment_data import apply

out = ROOT / 'tests/glm5/result/deployment_bundle_20261108'
archive = ROOT / 'release/deployment-20261108/ai2050-data-20261108-mainline4.tar'
target = out / 'data-acceptance'
target.mkdir(exist_ok=True)
sentinel = target / 'dist.sqlite3'
sentinel.write_bytes(b'nonproduction account-state sentinel')
dry = apply(archive, target)
first = apply(archive, target, write=True)
again = apply(archive, target, write=True)
assert dry['verified'] and not dry['applied']
assert first['applied'] and again['files'] == 0
assert sentinel.read_bytes() == b'nonproduction account-state sentinel'
with gzip.open(target / 'datapack/datapack.json.gz', 'rt', encoding='utf-8') as stream:
    pack = json.load(stream)
entries = pack['entries']
revision = pack['mainline_revision']
assert entries['/api/research/mainline']['revision'] == revision
assert entries['/api/traces/research-space']['mainline_revision'] == revision
assert entries['/api/research-library/workpackages']['mainline_revision'] == revision
cards = entries['/api/traces/research-space']['objects']
assert len([c for c in cards if c['body']['run_id'] == 'MA-H3b-v1']) == 70
extras = json.loads((target / 'datapack/extras.json').read_text(encoding='utf-8'))
for name, desc in extras.items():
    import re
    path = target / 'datapack/extra' / (re.sub(r'[^A-Za-z0-9._-]', '_', name) + '.json.gz')
    assert path.is_file() and path.stat().st_size == desc['bytes'], name
    # Stream the complete gzip to verify its CRC without allocating GB-sized JSON.
    with gzip.open(path, 'rb') as stream:
        while stream.read(4 * 2**20):
            pass
    print('gzip verified', name, flush=True)
assert len(extras) == 15
report = {'dry_run': dry, 'first_apply': first, 'replay': again, 'extra_packs': len(extras),
          'gzip_crc_all_passed': True, 'account_state_preserved': True,
          'mainline_revision': revision, 'h3b_candidate_heads': 70,
          'source_cache': pack['source_cache'], 'new_scientific_results': False}
(out / 'data-acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print('Release data acceptance passed', flush=True)
