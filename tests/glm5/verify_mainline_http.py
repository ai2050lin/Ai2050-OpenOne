"""Read-only acceptance of the actual CPU center and content-addressed evidence."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen


def get(base, path):
    with urlopen(base.rstrip('/') + path, timeout=20) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='http://127.0.0.1:5031')
    args = parser.parse_args()
    envelope = get(args.base, '/api/research/mainline')
    revision = hashlib.sha256(envelope['record'].encode('utf-8')).hexdigest()
    assert revision == envelope['revision']
    state = json.loads(envelope['record'])
    pages = {name: get(args.base, path) for name, path in {
        'workpackages': '/api/research-library/workpackages', 'fragments': '/api/research-library/fragments',
        'space': '/api/traces/research-space'}.items()}
    assert all(p['mainline_revision'] == revision for p in pages.values())
    cards = [o for o in pages['space']['objects'] if o['body']['run_id'] == 'MA-H3b-v1']
    assert len(cards) == 70
    assert all(o['type'] == 'feature' and not o['body']['scope']['individual_causality_established'] for o in cards)
    for record in state['records']:
        proof = get(args.base, '/api/research-library/artifacts/' + record['report_sha256'])
        encoded = json.dumps(proof, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        assert hashlib.sha256(encoded).hexdigest() == record['report_sha256']
    official = {}
    for path in ('/api/health', '/api/research/mainline'):
        try:
            doc = get('https://server.ai2050.com', path)
            official[path] = {'available': True, 'version': doc.get('version'), 'schema': doc.get('schema')}
        except (OSError, HTTPError, ValueError) as exc:
            official[path] = {'available': False, 'reason': str(exc)}
    report = {'base': args.base, 'revision': revision, 'sequence': state['sequence'],
              'model_snapshot': state['model_snapshot'], 'custody': state['custody'],
              'records': state['records'], 'space': pages['space']['counts'],
              'h3b_candidate_heads': len(cards), 'same_revision_all_pages': True,
              'proofs_verified': True, 'visual_acceptance': 'IAB: custody/H3b workpackage/3D/dose/raw candidate JSON verified',
              'official_read_only_check': official, 'formal_server_modified': False}
    out = Path(__file__).parent / 'result/mainline_sync_20261108/http-acceptance.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('verified revision', revision, 'H3b heads', len(cards), 'official', official)


if __name__ == '__main__':
    main()
