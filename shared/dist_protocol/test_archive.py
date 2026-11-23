"""Private research run ledger and resumable CAS; registration is not evidence promotion."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import sqlite3
from contextlib import contextmanager
import time

from deploy.storage_policy import check_write

CHUNK_BYTES = 2 * 2**20
META_BYTES = 512 * 1024
CORE_ROLES = {'script', 'questions', 'protocol', 'case_results', 'results', 'log', 'reproducibility'}
ROLES = CORE_ROLES | {'native_field', 'evidence', 'dictionary', 'derived'}
TERMINAL = {'completed', 'failed', 'interrupted'}
RUN_FILTERS = ('q', 'domain', 'model', 'state', 'family', 'snapshot', 'target', 'method', 'archive_state', 'artifact_role', 'decision')
NATIVE_ADDRESS = re.compile(r'(?:B(?:0|[1-9][0-9]{0,5})/[HU]|L(?:0|[1-9][0-9]{0,5})/D)(?:0|[1-9][0-9]{0,5})')


class Conflict(ValueError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('full lowercase sha256 required')
    return value


def text(value, name, maximum=4000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError('invalid ' + name)
    return value


def identity(value):
    if not isinstance(value, str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,159}', value):
        raise ValueError('invalid record identity')
    return value


def file_sha(path):
    value, count, last = hashlib.sha256(), 0, time.monotonic()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 2**20):
            value.update(chunk)
            count += len(chunk)
            if time.monotonic() - last > 60:
                print(f'[test-archive] checksum {count}/{path.stat().st_size} bytes', flush=True)
                last = time.monotonic()
    return value.hexdigest()


def validate_plan(plan):
    if not isinstance(plan, dict) or plan.get('schema') != 'research_test_plan.v1' or plan.get('kind') != 'real':
        raise ValueError('research_test_plan.v1 and kind=real required; sandbox fixtures stay isolated')
    if len(canonical(plan)) > META_BYTES:
        raise ValueError('plan exceeds metadata limit')
    for k in ('title', 'objective', 'principle', 'hypothesis', 'alternatives', 'criteria'):
        text(plan.get(k), k, 16000)
    for k in ('domains', 'family_ids'):
        v = plan.get(k)
        if not isinstance(v, list) or not v or len(v) > 100 or any(not isinstance(x, str) or not x or len(x) > 200 for x in v):
            raise ValueError('bounded nonempty ' + k + ' required')
    if not set(plan['domains']) <= {'KN', 'RS', 'GR'}:
        raise ValueError('domains must use KN/RS/GR')
    ds = plan.get('dataset')
    if not isinstance(ds, dict):
        raise ValueError('dataset revision required')
    for k in ('id', 'revision', 'exposure'):
        text(ds.get(k), 'dataset.' + k)
    for k in ('content_sha256', 'split_manifest_sha256'):
        sha(ds.get(k))
    if type(ds.get('case_count')) is not int or ds['case_count'] < 1:
        raise ValueError('positive dataset.case_count required')
    acq = plan.get('acquisition')
    if not isinstance(acq, dict):
        raise ValueError('acquisition specification required')
    for k in ('model_snapshot', 'tokenizer_sha256', 'inputs_sha256', 'code_sha256', 'environment_sha256'):
        sha(acq.get(k))
    for k in ('model_id', 'precision'):
        text(acq.get(k), 'acquisition.' + k)
    if type(acq.get('seed')) is not int or not isinstance(acq.get('generation'), dict):
        raise ValueError('acquisition seed and full generation configuration required')
    capture = acq.get('capture')
    if not isinstance(capture, dict) or not capture.get('layers') or not capture.get('readouts') or not capture.get('boundary'):
        raise ValueError('capture layers, readouts and boundary required')
    targets = capture.get('targets', [])
    if not isinstance(targets, list) or len(targets) > 5000 or any(
        not isinstance(target, str) or NATIVE_ADDRESS.fullmatch(target) is None for target in targets
    ):
        raise ValueError('capture.targets must contain canonical B/H, B/U or L/D addresses')
    analysis = plan.get('analysis')
    if not isinstance(analysis, dict):
        raise ValueError('analysis specification required')
    text(analysis.get('method_version'), 'method_version')
    for k in ('code_sha256', 'scoring_sha256'):
        sha(analysis.get(k))
    if not isinstance(analysis.get('selection'), dict):
        raise ValueError('analysis selection/freeze specification required')
    roles = analysis.get('required_artifact_roles')
    if not isinstance(roles, list) or not roles or not set(roles) <= ROLES:
        raise ValueError('analysis required_artifact_roles required')
    return digest(acq), digest({'acquisition': digest(acq), 'analysis': analysis,
                               'dataset': ds, 'criteria': plan['criteria']})


class Archive:
    def __init__(self, root):
        self.root = check_write(root, 2**20, operation='研究测试档案')
        self.root.mkdir(parents=True, exist_ok=True)
        for d in ('objects', 'uploads'):
            (self.root / d).mkdir(exist_ok=True)
        self.maximum = int(os.environ.get('AI2050_TEST_ARCHIVE_MAX_OBJECT_BYTES', 64 * 2**30))
        with self.db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS preflights(id TEXT PRIMARY KEY,plan_sha TEXT,acquisition_key TEXT,
                  analysis_key TEXT,plan TEXT,created REAL,actor TEXT);
                CREATE TABLE IF NOT EXISTS preflight_receipts(id TEXT PRIMARY KEY,receipt TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,preflight_id TEXT,plan_sha TEXT,
                  acquisition_key TEXT,analysis_key TEXT,decision TEXT,source_run_id TEXT,reason TEXT,
                  state TEXT,archive_state TEXT,created REAL,actor TEXT,final TEXT,receipt TEXT);
                CREATE INDEX IF NOT EXISTS run_acquisition ON runs(acquisition_key);
                CREATE INDEX IF NOT EXISTS run_analysis ON runs(analysis_key);
                CREATE TABLE IF NOT EXISTS objects(sha TEXT PRIMARY KEY,size INTEGER,verified REAL);
                CREATE TABLE IF NOT EXISTS uploads(id TEXT PRIMARY KEY,sha TEXT UNIQUE,size INTEGER,
                  offset INTEGER,created REAL,actor TEXT);
                CREATE TABLE IF NOT EXISTS run_revisions(run_id TEXT,manifest_sha TEXT,parent_sha TEXT,
                  created REAL,final TEXT,receipt TEXT,PRIMARY KEY(run_id,manifest_sha));
            ''')

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.root / 'archive.sqlite3', timeout=60)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA journal_mode=WAL')
        try:
            with db:
                yield db
        finally:
            db.close()

    def object_path(self, value):
        return self.root / 'objects' / sha(value)

    def _present(self, db, value, size=None, verify=False):
        row = db.execute('SELECT * FROM objects WHERE sha=?', (sha(value),)).fetchone()
        path = self.object_path(value)
        if not row or not path.is_file() or path.is_symlink() or path.stat().st_size != row['size']:
            return False
        return (size is None or size == row['size']) and (not verify or file_sha(path) == value)

    def _run(self, db, run_id):
        row = db.execute('SELECT r.*,p.plan FROM runs r JOIN preflights p ON p.id=r.preflight_id WHERE r.id=?',
                         (identity(run_id),)).fetchone()
        if not row:
            raise FileNotFoundError('run not found')
        doc = dict(row)
        for k in ('plan', 'final', 'receipt'):
            doc[k] = json.loads(doc[k]) if doc[k] else None
        return doc

    def get(self, run_id):
        with self.db() as db:
            return self._run(db, run_id)

    def search_inventory(self):
        """Metadata needed by the authorized evidence index, from one database snapshot."""
        with self.db() as db:
            return [{**dict(row), **{key:json.loads(row[key]) if row[key] else None
                                     for key in ('plan','receipt','final')}}
                    for row in db.execute('''SELECT r.id,r.state,r.archive_state,r.reason,r.decision,r.source_run_id,
                        r.receipt,r.final,p.plan FROM runs r JOIN preflights p ON r.preflight_id=p.id
                        ORDER BY r.created DESC,r.id''')]

    def query(self, *, q='', domain='', model='', state='', family='', snapshot='', target='', method='',
              archive_state='', artifact_role='', decision='', limit=50, offset=0):
        # AND across filters; correlated plan fields never borrow identity from other runs.
        filters = {key: value for key, value in locals().items() if key in RUN_FILTERS and value}
        clauses, args = [], []
        if q:
            clauses.append('(instr(lower(p.plan),lower(?))>0 OR instr(lower(r.id),lower(?))>0)')
            args += [q, q]
        if domain:
            clauses.append("EXISTS(SELECT 1 FROM json_each(p.plan,'$.domains') WHERE value=?)")
            args.append(domain)
        if family:
            clauses.append("EXISTS(SELECT 1 FROM json_each(p.plan,'$.family_ids') WHERE value=?)")
            args.append(family)
        if model:
            clauses.append("json_extract(p.plan,'$.acquisition.model_id')=?")
            args.append(model)
        if state:
            clauses.append('r.state=?')
            args.append(state)
        for column, value in (
            ("json_extract(p.plan,'$.acquisition.model_snapshot')", snapshot),
            ("json_extract(p.plan,'$.analysis.method_version')", method),
            ('r.archive_state', archive_state), ('r.decision', decision),
        ):
            if value:
                clauses.append(column + '=?')
                args.append(value)
        if target:
            if NATIVE_ADDRESS.fullmatch(target) is None:
                raise ValueError('target must be a canonical native address')
            clauses.append("EXISTS(SELECT 1 FROM json_each(p.plan,'$.acquisition.capture.targets') WHERE value=?)")
            args.append(target)
        if artifact_role:
            if artifact_role not in ROLES:
                raise ValueError('unknown artifact role')
            clauses.append("""EXISTS(SELECT 1 FROM json_each(r.receipt,'$.artifacts') a
                JOIN objects o ON o.sha=json_extract(a.value,'$.sha256') AND o.size=json_extract(a.value,'$.size')
                WHERE json_extract(a.value,'$.role')=? AND json_extract(a.value,'$.availability')='uploaded')""")
            args.append(artifact_role)
        where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
        base = ' FROM runs r JOIN preflights p ON p.id=r.preflight_id'
        with self.db() as db:
            total = db.execute('SELECT count(*)' + base + where, args).fetchone()[0]
            ids = db.execute('SELECT r.id' + base + where + ' ORDER BY r.created DESC,r.id LIMIT ? OFFSET ?',
                             [*args, min(max(limit, 1), 200), max(offset, 0)]).fetchall()
            return {'schema': 'research_test_query.v1', 'total': total, 'applied_filters': filters,
                    'items': [self._run(db, r['id']) for r in ids], 'coverage': 'registered_records_only',
                    'note': 'No matches does not prove no historical test; metadata similarity is advisory.'}

    def _matches(self, db, plan, acq, analysis, limit=50):
        rows = db.execute('''SELECT r.*,p.plan FROM runs r JOIN preflights p ON p.id=r.preflight_id
          WHERE r.acquisition_key=? OR r.analysis_key=? OR
            EXISTS(SELECT 1 FROM json_each(p.plan,'$.family_ids') WHERE value IN
              (SELECT value FROM json_each(?,'$.family_ids'))) OR
            EXISTS(SELECT 1 FROM json_each(p.plan,'$.domains') WHERE value IN
              (SELECT value FROM json_each(?,'$.domains')))
          ORDER BY (r.analysis_key=?) DESC,(r.acquisition_key=?) DESC,r.created DESC LIMIT ?''',
          (acq, analysis, canonical(plan).decode(), canonical(plan).decode(), analysis, acq, limit)).fetchall()
        result = []
        for row in rows:
            prior = json.loads(row['plan'])
            reasons = []
            if row['analysis_key'] == analysis:
                reasons.append('same_analysis_key')
            if row['acquisition_key'] == acq:
                reasons.append('same_acquisition_key')
            if set(prior['family_ids']) & set(plan['family_ids']):
                reasons.append('shared_language_family')
            if set(prior['domains']) & set(plan['domains']):
                reasons.append('shared_domain')
            action = 'review_similar'
            if row['acquisition_key'] == acq:
                if row['state'] == 'running':
                    action = 'already_running'
                elif row['state'] == 'completed' and row['archive_state'] == 'server_verified':
                    action = 'reuse_results' if row['analysis_key'] == analysis else 'reuse_acquisition'
                    if action == 'reuse_acquisition' and not self._reusable(db, row, plan):
                        action = 'recapture_required'
                else:
                    action = 'review_incomplete'
            result.append({'run_id': row['id'], 'title': prior['title'], 'state': row['state'],
                           'archive_state': row['archive_state'], 'reasons': reasons, 'suggestion': action,
                           'new_validation': row['reason'] if row['decision']!='new_acquisition' else None})
        return result

    def _reusable(self, db, row, plan):
        final = json.loads(row['final']) if isinstance(row['final'], str) and row['final'] else row['final']
        artifacts = (final or {}).get('artifacts', [])
        return all(any(a['role'] == role and a['availability'] == 'uploaded' and
                       self._present(db, a['sha256'], a['size']) for a in artifacts)
                   for role in plan['analysis']['required_artifact_roles'])

    def preflight(self, plan, actor, history_search=None):
        acq, analysis = validate_plan(plan)
        historical = history_search(plan) if history_search else []
        with self.db() as db:
            matches = self._matches(db, plan, acq, analysis)
            pid = 'check-' + secrets.token_hex(16)
            db.execute('INSERT INTO preflights VALUES(?,?,?,?,?,?,?)',
                       (pid, digest(plan), acq, analysis, canonical(plan).decode(), time.time(), actor))
            receipt = {'schema': 'research_test_preflight.v1', 'preflight_id': pid, 'plan_sha256': digest(plan),
                       'acquisition_key': acq, 'analysis_key': analysis, 'matches': matches,
                       'historical_matches': historical, 'coverage': 'registered_records_only', 'decision_required': True,
                       'note': 'Review similarity, missing data and exposed splits; no automatic model execution.'}
            db.execute('INSERT INTO preflight_receipts VALUES(?,?)', (pid, canonical(receipt).decode()))
        return receipt

    def preflight_receipt(self, pid):
        with self.db() as db:
            row = db.execute('SELECT receipt FROM preflight_receipts WHERE id=?',(identity(pid),)).fetchone()
        if not row:
            raise FileNotFoundError('此版本没有保留原始候选快照，请查本次运行保存的 preflight.json')
        return json.loads(row[0])

    def start(self, doc, actor):
        rid, pid = identity(doc.get('run_id')), identity(doc.get('preflight_id'))
        decision = doc.get('decision')
        if decision not in {'new_acquisition', 'analysis_only', 'independent_replication', 'recapture'}:
            raise ValueError('explicit reuse/new/replication/recapture decision required')
        reason = text(doc.get('reason'), 'novelty or replication reason')
        source = doc.get('source_run_id', '')
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            p = db.execute('SELECT * FROM preflights WHERE id=?', (pid,)).fetchone()
            if not p or p['actor'] != actor:
                raise ValueError('owned preflight required before starting')
            old = db.execute('SELECT * FROM runs WHERE id=?', (rid,)).fetchone()
            if old:
                if any(old[k] != v for k, v in [('preflight_id', pid), ('decision', decision),
                                                ('source_run_id', source), ('reason', reason), ('actor', actor)]):
                    raise Conflict('run identity collision; do not overwrite')
                return self._run(db, rid)
            if time.time() - p['created'] > 24 * 3600:
                raise Conflict('preflight expired; query again')
            plan = json.loads(p['plan'])
            matches = self._matches(db, plan, p['acquisition_key'], p['analysis_key'], 200)
            if decision == 'new_acquisition' and any(m['suggestion'] in
               {'reuse_results', 'reuse_acquisition', 'already_running'} for m in matches):
                raise Conflict('existing acquisition found; reuse it or explicitly justify replication/recapture')
            if decision == 'analysis_only':
                row = db.execute('SELECT * FROM runs WHERE id=?', (identity(source),)).fetchone()
                if not row or row['acquisition_key'] != p['acquisition_key'] or row['archive_state'] != 'server_verified' or not self._reusable(db, row, plan):
                    raise Conflict('source acquisition or required artifacts unavailable; do not silently recapture')
            db.execute('INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                       (rid, pid, p['plan_sha'], p['acquisition_key'], p['analysis_key'], decision, source,
                        reason, 'running', 'pending', time.time(), actor, None, None))
            return self._run(db, rid)

    def upload_init(self, doc, actor):
        value, size = sha(doc.get('sha256')), doc.get('size')
        if type(size) is not int or not 0 <= size <= self.maximum:
            raise ValueError('object size exceeds configured limit')
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            if self._present(db, value, size):
                return {'available': True, 'sha256': value, 'size': size}
            for expired in db.execute('SELECT id FROM uploads WHERE created<?', (time.time() - 86400,)).fetchall():
                (self.root / 'uploads' / (expired['id'] + '.part')).unlink(missing_ok=True)
                db.execute('DELETE FROM uploads WHERE id=?', (expired['id'],))
            old = db.execute('SELECT * FROM uploads WHERE sha=?', (value,)).fetchone()
            if old:
                if old['size'] != size or old['actor'] != actor:
                    raise Conflict('object upload already reserved under different identity')
                return self._upload(db, old['id'], actor)
            reserved = db.execute('SELECT coalesce(sum(size-offset),0) FROM uploads').fetchone()[0]
            check_write(self.root, reserved + size + META_BYTES, operation='并发上传容量预留')
            uid = secrets.token_hex(24)
            (self.root / 'uploads' / (uid + '.part')).touch(exist_ok=False)
            db.execute('INSERT INTO uploads VALUES(?,?,?,?,?,?)', (uid, value, size, 0, time.time(), actor))
            return {'upload_id': uid, 'offset': 0, 'size': size, 'chunk_bytes': CHUNK_BYTES, 'available': False}

    def _upload(self, db, uid, actor):
        identity(uid)
        row = db.execute('SELECT * FROM uploads WHERE id=?', (uid,)).fetchone()
        if not row or row['actor'] != actor:
            raise FileNotFoundError('upload not found')
        path = self.root / 'uploads' / (uid + '.part')
        if not path.exists() and self._present(db, row['sha'], row['size']):
            return {**dict(row), 'upload_id': uid, 'offset': row['size'],
                    'chunk_bytes': CHUNK_BYTES, 'available': True}
        if path.is_symlink() or not path.is_file():
            raise Conflict('upload staging file unavailable')
        return {**dict(row), 'upload_id': uid, 'offset': path.stat().st_size, 'chunk_bytes': CHUNK_BYTES,
                'available': False}

    def upload_status(self, uid, actor):
        with self.db() as db:
            return self._upload(db, uid, actor)

    def upload_chunk(self, uid, offset, data, actor):
        if not data or len(data) > CHUNK_BYTES or type(offset) is not int or offset < 0:
            raise ValueError('invalid bounded chunk/offset')
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            up = self._upload(db, uid, actor)
            if up['available']:
                raise Conflict('upload already finalized')
            path = self.root / 'uploads' / (uid + '.part')
            if offset < up['offset']:
                with path.open('rb') as stream:
                    stream.seek(offset)
                    prior = stream.read(len(data))
                if prior == data and offset + len(data) <= up['offset']:
                    return {'offset': up['offset'], 'replayed': True}
                raise Conflict('replayed chunk differs')
            if offset != up['offset'] or offset + len(data) > up['size']:
                raise Conflict('unexpected upload offset or excess bytes')
            reserved = db.execute('SELECT coalesce(sum(size-offset),0) FROM uploads').fetchone()[0]
            check_write(path, reserved + META_BYTES, operation='上传继续写入')
            with path.open('ab') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            db.execute('UPDATE uploads SET offset=?,created=? WHERE id=?', (offset + len(data), time.time(), uid))
            return {'offset': offset + len(data), 'replayed': False}

    def upload_finish(self, uid, actor):
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            up = self._upload(db, uid, actor)
            if up['available']:
                return {'sha256': up['sha'], 'size': up['size'], 'verified': True, 'replayed': True}
            path = self.root / 'uploads' / (uid + '.part')
            if up['offset'] != up['size']:
                raise Conflict('upload incomplete; keep chunks for resume')
            if file_sha(path) != up['sha']:
                path.unlink()
                db.execute('DELETE FROM uploads WHERE id=?', (uid,))
                # Commit invalid staging removal, then report outside the transaction.
                invalid = True
            else:
                invalid = False
                dest = self.object_path(up['sha'])
                path.replace(dest)
                db.execute('INSERT OR REPLACE INTO objects VALUES(?,?,?)', (up['sha'], up['size'], time.time()))
                db.execute('UPDATE uploads SET offset=size,created=? WHERE id=?', (time.time(), uid))
        if invalid:
            raise ValueError('object sha256 mismatch; restart upload')
        return {'sha256': up['sha'], 'size': up['size'], 'verified': True}

    def finalize(self, run_id, doc, actor, parent_sha=''):
        if doc.get('status') not in TERMINAL:
            raise ValueError('completed/failed/interrupted status required')
        text(doc.get('summary'), 'summary', 64000)
        artifacts = doc.get('artifacts')
        if not isinstance(artifacts, list) or len(artifacts) > 5000 or len(canonical(doc)) > META_BYTES:
            raise ValueError('bounded explicit artifact manifest required')
        names = set()
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            run = self._run(db, run_id)
            if run['actor'] != actor:
                raise ValueError('run owner mismatch')
            if run['final']:
                if canonical(run['final']) != canonical(doc):
                    if parent_sha != digest(run['final']) or run['state'] != doc['status']:
                        raise Conflict('finalized run is immutable; artifact recovery needs current parent hash and same execution status')
                else:
                    return run
            elif parent_sha:
                raise Conflict('no previous manifest to recover')
            roles = set()
            for a in artifacts:
                if not isinstance(a, dict):
                    raise ValueError('invalid artifact entry')
                name = text(a.get('name'), 'artifact name', 500)
                if '\\' in name or ':' in name or name.startswith('/') or '..' in PurePosixPath(name).parts or name in names:
                    raise ValueError('unique relative artifact names required')
                names.add(name)
                if a.get('role') not in ROLES or a.get('retention') not in {'core', 'important', 'reuse'}:
                    raise ValueError('invalid role/retention')
                available = a.get('availability')
                if available not in {'uploaded', 'retired_not_archived', 'historically_missing'}:
                    raise ValueError('explicit artifact availability required')
                if available == 'uploaded':
                    if type(a.get('size')) is not int or not self._present(db, a.get('sha256'), a['size'], verify=True):
                        raise ValueError('artifact missing or corrupt on server')
                    roles.add(a['role'])
                elif a['retention'] == 'important':
                    raise ValueError('important evidence must be uploaded before finalization')
                elif a.get('reason') is None:
                    raise ValueError('missing/retired artifact needs a reason')
            missing = sorted(CORE_ROLES - roles)
            receipt = {'schema': 'research_test_receipt.v1', 'run_id': run_id,
                       'plan_sha256': run['plan_sha'], 'manifest_sha256': digest(doc),
                       'archive_state': 'server_verified' if not missing else 'missing_artifacts',
                       'missing_core_roles': missing, 'cleanup_allowed': not missing,
                       'verified_at': time.time(), 'evidence_status': 'unreviewed',
                       'artifacts': artifacts}
            receipt['receipt_sha256'] = digest(receipt)
            db.execute('INSERT INTO run_revisions VALUES(?,?,?,?,?,?)',
                       (run_id, digest(doc), parent_sha, time.time(), canonical(doc).decode(), canonical(receipt).decode()))
            db.execute('UPDATE runs SET state=?,archive_state=?,final=?,receipt=? WHERE id=?',
                       (doc['status'], receipt['archive_state'], canonical(doc).decode(), canonical(receipt).decode(), run_id))
            return self._run(db, run_id)

    def revisions(self, run_id):
        with self.db() as db:
            self._run(db, run_id)
            return {'run_id': run_id, 'items': [dict(r) for r in db.execute(
                'SELECT manifest_sha,parent_sha,created FROM run_revisions WHERE run_id=? ORDER BY created', (run_id,))]}

    def receipt(self, run_id, verify=True):
        with self.db() as db:
            run = self._run(db, run_id)
            receipt = run['receipt']
            if not receipt:
                raise Conflict('run not finalized')
            good = all(self._present(db, a['sha256'], a['size'], verify=verify)
                       for a in receipt['artifacts'] if a['availability'] == 'uploaded')
            return {**receipt, 'readback_verified': bool(verify and good),
                    'cleanup_allowed': bool(verify and good and receipt['cleanup_allowed'])}

    def download(self, value):
        with self.db() as db:
            if not self._present(db, value):
                raise FileNotFoundError('object unavailable')
        return self.object_path(value)
