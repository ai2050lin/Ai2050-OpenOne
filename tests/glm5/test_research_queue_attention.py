import os
import sqlite3
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

DEPLOY = Path(__file__).resolve().parents[2] / 'deploy'
with patch.object(sys, 'path', [str(DEPLOY), *sys.path]), patch.dict(
    os.environ, {'AI2050_DIST_ADMIN_TOKEN': 'software-fixture-only'}
):
    from deploy import distributed_service


class ResearchQueueAttentionTest(unittest.TestCase):
    def test_old_failures_and_expired_leases_survive_a_busy_completion_history(self):
        with sqlite3.connect(':memory:') as db:
            db.row_factory = sqlite3.Row
            db.executescript("""
                CREATE TABLE tasks (
                    task_id TEXT, tm_id TEXT, work_unit_id TEXT, research_ref_json TEXT,
                    status TEXT, claimed_at REAL, done_at REAL, note TEXT, result_sha TEXT,
                    node_id TEXT, lease_until REAL, shard TEXT
                );
                CREATE TABLE nodes (node_id TEXT, name TEXT, gpu TEXT, model TEXT, dtype TEXT);
                CREATE TABLE templates (tm_id TEXT, name TEXT, version INTEGER);
            """)
            db.executemany("INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
                (f'done-{i}', 'tm', None, '{}', 'done', i + 10, i + 11, None, None, None, None, None)
                for i in range(35)
            ])
            db.executemany("INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
                (f'queued-{i}', 'tm', None, '{}', 'queued', None, None, None, None, None, None, None)
                for i in range(57)
            ])
            db.execute("INSERT INTO templates VALUES ('tm','实验作业',1)")
            db.execute("INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       ('older-failure', 'tm', 'unit-2', '{"shard_index":2}', 'failed', 1, 2,
                        'worker stopped', None, None, None, '2/4'))
            db.execute("INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       ('expired-lease', 'tm', None, '{}', 'claimed', 3, None, None, None, None, time.time() - 10, None))
            with patch.object(distributed_service, '_db', return_value=db):
                queue = distributed_service.tasks_public()
            self.assertEqual({entry['status'] for entry in queue['recent']}, {'done'})
            self.assertEqual(queue['stats'], {'done': 35, 'expired': 1, 'failed': 1, 'queued': 57})
            self.assertEqual(len(queue['queued']), 50)
            self.assertEqual(queue['by_template'], [{'tm_id': 'tm', 'template_name': '实验作业', 'total': 94,
                                                     'queued': 57, 'active': 0, 'done': 35, 'failed': 1, 'expired': 1}])
            attention = {entry['task_id']: entry for entry in queue['attention']}
            self.assertEqual(set(attention), {'older-failure', 'expired-lease'})
            self.assertEqual(attention['older-failure']['research_ref'], {'shard_index': 2})
            self.assertEqual(attention['older-failure']['shard'], '2/4')
            self.assertEqual(attention['expired-lease']['status'], 'expired')
