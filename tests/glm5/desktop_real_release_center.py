"""Loopback center for installed-client acceptance with real local Qwen weights."""
import json
import os
from pathlib import Path
import sys
import threading
import time

import uvicorn


ROOT = Path(__file__).resolve().parents[2]
JOB = 'CJ-desktop-1-0-7-real'


def main():
    area, model = map(Path, sys.argv[1:3])
    native = Path(os.environ.get('AI2050_ACCEPTANCE_NATIVE', 'D:/ai2050_native'))
    empty_archive = area / 'archive'
    empty_archive.mkdir(parents=True)
    os.environ.update(AI2050_DIST_DIR=str(area / 'center'), AI2050_RESEARCH_DIR=str(empty_archive),
                      AI2050_NATIVE_DIR=str(native), AI2050_SNAPSHOT_DISABLE='1',
                      AI2050_DIST_ADMIN_TOKEN='desktop-release-loopback-only')
    os.environ.pop('AI2050_DIST_ALLOW_SYNTHETIC', None)
    sys.path.insert(0, str(ROOT))
    from deploy.center_app import create_app
    import census_service as census

    app = create_app()
    original = ROOT / 'distributed_data/census/CJ-qwen3-4b-pilot/files/job.json'
    pilot = json.loads(original.read_text(encoding='utf-8'))
    frozen = pilot['model']
    census.create_job(census.JobCreate(
        job_id=JOB, model_dir=str(model), model_name=frozen['name'],
        snapshot={'sha256': frozen['snapshot_sha256'], 'files': frozen['files']},
        corpus_fp=pilot['corpus_fp'], tokens_per_slice=8192, max_slices=1,
        group_size=18, replica_rate=0, k=8, seq_len=512, skip_head=16, batch=2,
    ))
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=0, log_level='warning'))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(200):
            if server.started:
                break
            if not thread.is_alive():
                raise RuntimeError('验收中心启动失败')
            time.sleep(0.05)
        if not server.started:
            raise RuntimeError('验收中心启动超时')
        port = server.servers[0].sockets[0].getsockname()[1]
        print('CENTER_READY ' + json.dumps({'url': f'http://127.0.0.1:{port}', 'job': JOB,
              'snapshot': frozen['snapshot_sha256'], 'corpus': pilot['corpus_fp']}), flush=True)
        for line in sys.stdin:
            if line.strip() != 'inspect':
                continue
            rows = []
            for row in census.db().execute('SELECT * FROM census_results WHERE job_id=? ORDER BY shard', (JOB,)):
                summary = json.loads(row['summary_json'])
                rows.append({**{key: row[key] for key in ('lease_id', 'sha', 'shard', 'kind', 'status', 'n_tokens')},
                             'validation': json.loads(row['validation_json']),
                             'snapshot': summary.get('model_snapshot'), 'env': summary.get('env')})
            print('CENTER_RESULTS ' + json.dumps(rows), flush=True)
    finally:
        server.should_exit = True
        thread.join(10)


if __name__ == '__main__':
    main()
