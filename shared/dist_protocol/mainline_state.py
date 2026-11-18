"""Versioned, content-addressed research status. Publication never promotes evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = 'research_mainline.v1'
KEY = 'research_mainline.current'
RELEASE = Path(__file__).with_name('mainline_release.json')


def canonical(doc):
    return json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(doc):
    return hashlib.sha256(canonical(doc)).hexdigest()


def validate(bundle, previous=None):
    if not isinstance(bundle, dict) or bundle.get('schema') != 'mainline_release.v1':
        raise ValueError('invalid mainline release')
    state, artifacts = bundle.get('state'), bundle.get('artifacts')
    if not isinstance(state, dict) or state.get('schema') != SCHEMA or state.get('kind') != 'real':
        raise ValueError('real research state required')
    if type(state.get('sequence')) is not int or state['sequence'] < 1:
        raise ValueError('positive publication sequence required')
    snapshot = state.get('model_snapshot', '')
    if len(snapshot) != 64 or any(c not in '0123456789abcdef' for c in snapshot):
        raise ValueError('full model snapshot required')
    if not isinstance(artifacts, dict) or not artifacts or len(canonical(bundle)) > 8 * 1024 * 1024:
        raise ValueError('bounded proof artifacts required')
    for sha, doc in artifacts.items():
        if digest(doc) != sha or not isinstance(doc, dict):
            raise ValueError('proof artifact checksum mismatch')
        if any(str(doc.get(key, '')).lower() == 'synthetic' for key in ('kind', 'mode')) or doc.get('synthetic') is True or doc.get('sandbox_synthetic') is True:
            raise ValueError('synthetic proof is not accepted')
    view = state.get('view', {})
    for key in ('STAGES', 'CLAIMS', 'LOCKBOXES', 'RUNS', 'FINDINGS', 'SCORES', 'TESTS', 'CAPABILITIES'):
        if not isinstance(view.get(key), list):
            raise ValueError('missing view collection: ' + key)
    custody = state.get('custody', [])
    boxes = {b['id']: b for b in view['LOCKBOXES']}
    if len({b['id'] for b in custody}) != len(custody):
        raise ValueError('duplicate custody identity')
    for item in custody:
        if item.get('state') not in ('sealed', 'opened') or boxes.get(item['id'], {}).get('state') != item['state']:
            raise ValueError('custody and displayed state differ')
        if item['state'] == 'opened' and not item.get('opened_at'):
            raise ValueError('opening event is required')
        if item.get('proof_sha256') not in artifacts:
            raise ValueError('custody proof is missing')
        proof = artifacts[item['proof_sha256']]
        if proof.get('schema') != 'mainline_custody.v1' or proof.get('model_snapshot') != snapshot:
            raise ValueError('custody proof identity differs')
        events = [e for e in proof.get('events', []) if e.get('split') == item.get('split')]
        if events and (item['state'] != 'opened' or item.get('opened_at') != events[-1].get('when')):
            raise ValueError('custody opening event and status differ')
        if item['state'] == 'opened' and not events:
            raise ValueError('custody opening event is missing')
    for record in state.get('records', []):
        if record.get('report_sha256') not in artifacts:
            raise ValueError('run proof is missing')
        if record.get('model_snapshot') != snapshot:
            raise ValueError('run model snapshot differs')
        proof = artifacts[record['report_sha256']]
        if proof.get('model_snapshot') != snapshot or proof.get('run_id') != record['run_id']:
            raise ValueError('run proof identity differs')
        displayed = [r for r in view['RUNS'] if r.get('id') == record['run_id']]
        if len(displayed) != 1 or displayed[0].get('report_sha256') != record['report_sha256']:
            raise ValueError('run status and proof differ')
    for result in view.get('METHOD_RESULTS', []):
        if result.get('artifact_sha256') not in artifacts:
            raise ValueError('method result proof is missing')
    for ref in state.get('graph_sha256', []):
        graph = artifacts.get(ref, {})
        if graph.get('schema') != 'research_graph.v1' or graph.get('model_snapshot') != snapshot:
            raise ValueError('graph identity differs')
        for obj in graph.get('feature_cards', []) + graph.get('graph_edges', []):
            if obj.get('evidence', {}).get('artifact_sha256') not in artifacts:
                raise ValueError('graph proof is missing')
    if previous:
        if state['sequence'] <= previous['sequence']:
            raise ValueError('publication sequence must increase')
        if state.get('parent_revision') != digest(previous):
            raise ValueError('publication parent differs; refresh before publishing')
        current = {b['id']: b for b in custody}
        for old in previous.get('custody', []):
            if old['state'] == 'opened' and current.get(old['id'], {}).get('state') != 'opened':
                raise ValueError('an opened split cannot be resealed or removed')
    return state


class MainlineStore:
    def __init__(self, library):
        self.lib = library

    def stored(self):
        with self.lib.db() as db:
            row = db.execute('SELECT value FROM metadata WHERE key=?', (KEY,)).fetchone()
        return self.lib.artifact(row['value']) if row else None

    def publish(self, bundle, *, source='reviewed-mainline'):
        # Share the library lock with ingestion and the 3D index, so a reader cannot
        # observe a half-published set of proofs, graph cards and status.
        from deploy.research_library import _lock
        with _lock:
            previous = self.stored()
            if previous and digest(previous) == digest(bundle.get('state')):
                validate(bundle)
                return self.envelope(previous)
            state = validate(bundle, previous)
            refs = [*bundle['artifacts'], digest(state)]
            # Incomplete publications remain archived for recovery, never visible
            # as current graph cards or workpackages. Commit visibility and the
            # status pointer in the same SQLite transaction.
            with self.lib.db() as db:
                db.executemany('INSERT OR IGNORE INTO mainline_publications VALUES(?,0)',
                               [(sha,) for sha in refs])
            docs = list(bundle['artifacts'].values())
            order = {'research_graph.v1': 1, 'research_workpackage.v1': 2, 'research_space_binding.v1': 3}
            for doc in sorted(docs, key=lambda d: order.get(d.get('schema'), 0)):
                self.lib.ingest(doc, source, kind='mainline_proof')
            ref = self.lib.ingest(state, source, kind='mainline_state')['sha256']
            with self.lib.db() as db:
                db.executemany('UPDATE mainline_publications SET published=1 WHERE sha=?',
                               [(sha,) for sha in refs])
                db.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)', (KEY, ref))
            return self.envelope(state)

    def ensure(self, release=RELEASE):
        from deploy.research_library import _lock
        with _lock:
            current = self.stored()
            seed = json.loads(Path(release).read_text(encoding='utf-8'))
            if current and current['sequence'] >= seed['state']['sequence']:
                return self.envelope(current)
            return self.publish(seed, source='release-mainline')

    @staticmethod
    def envelope(state):
        return {'schema': 'mainline_snapshot.v1', 'revision': digest(state),
                'sequence': state['sequence'], 'record': canonical(state).decode('utf-8'),
                'evidence_promoted': False}
