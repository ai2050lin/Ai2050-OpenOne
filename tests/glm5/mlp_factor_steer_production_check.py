"""Verify that each Phase 2801 object is served by the production center."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from deploy.research_ingest_bundle import read_member, verified_objects
from deploy.research_library import canonical


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tests/glm5/result/mlp_factor_steer_v1'
BASE = 'https://server.ai2050.com'
PACKAGE = ROOT / 'release/ai2050-research-factor-steer.tar.gz'
ARTIFACT = '/api/research-library/artifacts/'


def get(path):
    request = Request(BASE + path, headers={'Accept-Encoding': 'gzip', 'Cache-Control': 'no-cache'})
    with urlopen(request, timeout=40) as response:
        raw = response.read()
        if response.headers.get('Content-Encoding') == 'gzip':
            raw = gzip.decompress(raw)
        return json.loads(raw)


def verify_remote(sha):
    try:
        doc = get(ARTIFACT + sha)
    except HTTPError as exc:
        if exc.code == 404:
            return sha, 'missing'
        raise
    actual = hashlib.sha256(canonical(doc)).hexdigest()
    return sha, 'verified' if actual == sha else 'hash_mismatch'


def main():
    with PACKAGE.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    assert digest == Path(str(PACKAGE) + '.sha256').read_text(encoding='ascii').split()[0]
    with tarfile.open(PACKAGE) as tar:
        manifest = json.loads(read_member(tar, 'MANIFEST.json', 4 * 2**20))
        hashes = [sha for sha, _ in verified_objects(tar, manifest)]
    (study,) = manifest['studies']
    health = get('/api/health')
    assert health['version'] == '1.0.26', health
    index = get('/api/traces/research-space')
    assert index['schema'] == 'research_space_index.v1'
    published = next((s for s in index['studies'] if s['run_id'] == study['run_id'] and
                      s['model_snapshot'] == study['model_snapshot']), None)
    with ThreadPoolExecutor(max_workers=4) as pool:
        checked = dict(pool.map(verify_remote, hashes))
    missing = {sha: status for sha, status in checked.items() if status != 'verified'}
    if missing:
        raise SystemExit(json.dumps({'not_verified': missing, 'verified': len(checked)-len(missing)}))
    assert published and all(published[k] == study[k] for k in
                             ('report_sha256', 'binding_sha256', 'workpackage_sha256'))
    assert published['localization'] == 'native_bank'
    graph = get(ARTIFACT + study['graph_sha256'])
    report = get(ARTIFACT + study['report_sha256'])
    model = next(m for m in index['models'] if m['snapshot'] == study['model_snapshot'])
    assert [model[k] for k in ('n_layers', 'n_heads', 'd_model', 'd_ff')] == [36, 32, 2560, 9728]
    tasks = {}
    for task in ('lookup', 'dispatch'):
        case = get(ARTIFACT + report['cases'][task]['chunks'][0])['cases'][0]
        native = get(ARTIFACT + report['native'][task]['chunks'][0])['cases'][0]
        assert native['id'] == case['id']
        for values in ('clean', 'edited'):
            for role in ('attribute', 'selector', 'tail_a', 'tail_b'):
                for field in ('keys', 'values'):
                    groups = native[values][role][field]
                    assert len(groups) == 8 and all(len(g) == 128 for g in groups)
        tasks[task] = {'n': report['cases'][task]['n'], 'native_cases': report['native'][task]['n'],
                       'sample_verified': native['id'], 'kv_heads': 8, 'head_dim': 128}
    assert len(graph['feature_cards']) == 33 and len(graph['graph_edges']) == 60
    assert len(report['counterexamples']) == 208
    proof = {
        'timestamp': datetime.now(ZoneInfo('America/Chicago')).isoformat(timespec='seconds'),
        'center': BASE, 'health': health, 'package_sha256': digest, **study,
        'verified_object_count': len(checked), 'index_counts': index['counts'],
        'feature_cards': len(graph['feature_cards']), 'graph_edges': len(graph['graph_edges']),
        'case_shortcuts': len(report['counterexamples']),
        'non_targeted_cases': sum(not c['target_reached'] for c in report['counterexamples']),
        'tasks': tasks, 'capability': report['capability']['summary']['targeted']['n'],
    }
    (OUT / 'production-api-proof.json').write_bytes(canonical(proof) + b'\n')
    print(json.dumps(proof, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
