"""Isolated real HTTP center and node for the local Playwright contribution check."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests' / 'dist_protocol'))
os.environ['AI2050_DIST_ADMIN_TOKEN'] = 'current-research-browser-fixture-only'

from test_node_service import SandboxCenterE2ETests, call, start_service
import node_agent


def main():
    fixture = SandboxCenterE2ETests()
    fixture.setUpClass()
    node = None
    try:
        headers = {'X-Admin-Token': os.environ['AI2050_DIST_ADMIN_TOKEN']}
        fixture.http('/api/census/jobs', 'POST', {
            'job_id': 'CJ-browser', 'model_dir': str(fixture.mdir), 'snapshot': {'sha256': 'f' * 64, 'files': {}},
            'corpus_fp': 'fp0', 'tokens_per_slice': 2048, 'max_slices': 1, 'seq_len': 64, 'skip_head': 4,
            'k': 6, 'group_size': 2, 'replica_rate': 0.0, 'batch': 4, 'model_name': 'browser-test',
        }, headers)
        node = start_service(fixture.tmp / 'svc', synthetic=True, server_override=fixture.center)
        # Hold the first real progress event long enough for the two-second browser poll.
        def hold_first_progress(progress):
            if progress.get('total'):
                node_agent.PROGRESS_HOOKS.remove(hold_first_progress)
                time.sleep(4)

        node_agent.PROGRESS_HOOKS.append(hold_first_progress)
        print('BROWSER_FIXTURE ' + json.dumps({'port': node['port'], 'token': node['token'], 'center': fixture.center}), flush=True)
        for line in sys.stdin:
            cmd = json.loads(line)
            if cmd['action'] == 'seed_audit':
                sha = 'a' * 64
                db = fixture.D._db()
                with fixture.D._db_lock:
                    db.execute("INSERT INTO results(sha,node_id,kind) VALUES(?,?,'synthetic')", (sha, cmd['node_id']))
                    db.execute("INSERT INTO tasks(task_id,node_id,status,result_sha,done_at) VALUES('T-BROWSER-AUDIT',?,'done',?,?)",
                               (cmd['node_id'], sha, time.time()))
                    db.commit()
            elif cmd['action'] == 'audit':
                fixture.http('/api/audit/' + 'a' * 64, 'POST',
                             {'status': 'needs_rerun', 'note': '浏览器验收：要求复测'}, headers)
            print('BROWSER_ACK', flush=True)
    finally:
        if node:
            call(node['port'], '/shutdown', 'POST', {}, token=node['token'])
            node['thread'].join(10)
        fixture.tearDownClass()


if __name__ == '__main__':
    main()
