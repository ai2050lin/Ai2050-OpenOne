"""Read Phase 2803 through the public production API and verify the full archive."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tarfile
import time
from zoneinfo import ZoneInfo

from deploy.research_ingest_bundle import read_member, verified_objects
from deploy.research_library import canonical
from mlp_factor_steer_production_check import ARTIFACT, BASE, get


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tests/glm5/result/b35_recursive_v2'
PACKAGE = ROOT / 'release/ai2050-research-qwen3-4b-b35.tar.gz'


def wait_for_binding(study, wait):
    deadline = time.monotonic() + wait
    while True:
        index = get('/api/traces/research-space')
        assert index['schema'] == 'research_space_index.v1'
        published = next((row for row in index['studies'] if row['run_id'] == study['run_id']
                          and row['model_snapshot'] == study['model_snapshot']), None)
        if published and all(published.get(key) == study[key] for key in
                             ('report_sha256', 'binding_sha256', 'workpackage_sha256')):
            return published
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SystemExit('Phase 2803 is not present in the production study index')
        time.sleep(min(15, remaining))


def verify_object(sha):
    doc = get(ARTIFACT + sha)
    assert hashlib.sha256(canonical(doc)).hexdigest() == sha, sha
    return sha, doc


def check(wait):
    with PACKAGE.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert digest == Path(str(PACKAGE)+'.sha256').read_text(encoding='ascii').split()[0]
    with tarfile.open(PACKAGE) as tar:
        manifest = json.loads(read_member(tar, 'MANIFEST.json', 4 * 2**20))
        hashes = [sha for sha, _ in verified_objects(tar, manifest)]
    (study,) = manifest['studies']
    assert len(hashes) == 100
    health = get('/api/health')
    assert health['version'] == '1.0.27', health
    published = wait_for_binding(study, wait)
    print('Production study indexed; checking 100 objects.', flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        objects = dict(pool.map(verify_object, hashes))
    report = objects[study['report_sha256']]
    graph = objects[study['graph_sha256']]
    feature_sha = report['feature_index_sha256']
    assert published['stages']['b35_features']['index_sha256'] == feature_sha
    overview = objects[feature_sha]
    assert overview['unit_count'] == 9728 and len(overview['chunks']) == 76
    assert all(len(values) == 9728 for values in overview['plots'].values())
    all_units, candidate_count, replicated = [], 0, 0
    for sha in overview['chunks']:
        chunk = objects[sha]
        assert chunk['start'] == len(all_units) and chunk['block'] == 35
        assert chunk['model_snapshot'] == study['model_snapshot']
        for row in chunk['units']:
            all_units.append(row['unit'])
            assert all(k in row for k in ('mean', 'std', 'min', 'max', 'poles', 'readout'))
            assert row['confirm']['n'] == 160 and row['lockbox']['n'] == 224
            replicated += int(row['behavior_eligible'])
            for kind in ('gate', 'value'):
                candidates = row['upstream'][kind]
                assert len(candidates) == 3 and len({r['unit'] for r in candidates}) == 3
                assert all(0 <= r['unit'] < 9728 and abs(r['cosine']) <= 1.001 for r in candidates)
                candidate_count += len(candidates)
    assert all_units == list(range(9728)) and replicated == report['eligible_units']
    assert candidate_count == 9728*6
    failures = {}
    for part, info in report['case_indexes'].items():
        rows = [row for sha in info['chunks'] for row in objects[sha]['cases']]
        assert len(rows) == info['n']
        assert all(len(row['input_ids']) == 1024 for row in rows)
        bad = [i for i, row in enumerate(rows) if not all(s['correct'] for s in row['outcome']['free'])]
        if part in report['error_cases']:
            assert bad == report['error_cases'][part]
        failures[part] = {'n':len(rows),'failed':len(bad),'example':rows[bad[0]]['id'] if bad else None}
    assert failures['lockbox']['failed'] == 89
    proof = {
        'timestamp':datetime.now(ZoneInfo('America/Chicago')).isoformat(timespec='seconds'),
        'center':BASE,'health':health,'package_sha256':digest,**study,
        'verified_objects':len(objects),'unit_count':len(all_units),'unit_shards':len(overview['chunks']),
        'b34_candidates':candidate_count,'eligible_units':replicated,'cases':failures,
        'graph_cards':len(graph['feature_cards']),'graph_edges':len(graph['graph_edges']),
    }
    (OUT/'production-api-proof.json').write_bytes(canonical(proof)+b'\n')
    print(json.dumps(proof,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait',type=int,default=0)
    check(parser.parse_args().wait)
