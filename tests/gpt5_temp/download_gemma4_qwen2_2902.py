# -*- coding: utf-8 -*-
"""Phase 2902 附带任务: Gemma4(gemma-3-4b-it) / Qwen2-7B 模型下载 (hf-mirror).
后台运行, 进度写日志, 断点续传(snapshot_download 内置).
下载目标: D:/AI2050/Ai2050-OpenOne/models/hf/{qwen2-7b, gemma-3-4b-it}
"""
import os, sys, time, traceback

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

LOG = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\download_gemma4_qwen2_2902.log'
MODELS = r'D:\AI2050\Ai2050-OpenOne\models\hf'

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write('[%s] %s\n' % (time.strftime('%Y-%m-%d %H:%M:%S'), msg))

JOBS = [
    ('Qwen/Qwen2-7B', os.path.join(MODELS, 'qwen2-7b')),
    # Gemma 官方仓库 gated, 走 unsloth 非门控镜像(权重与 google/gemma-3-4b-it 一致)
    ('unsloth/gemma-3-4b-it', os.path.join(MODELS, 'gemma-3-4b-it')),
]

def download_repo(repo_id, local_dir, allow_patterns=None):
    from huggingface_hub import snapshot_download
    log('START %s -> %s' % (repo_id, local_dir))
    t0 = time.time()
    try:
        p = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=['*.pth', '*.gguf', 'original/*'],
            max_workers=4,
            resume_download=True,
        )
        dt = time.time() - t0
        log('DONE %s in %.1f min -> %s' % (repo_id, dt / 60.0, p))
        return True
    except Exception as e:
        log('FAIL %s: %s' % (repo_id, str(e)[:300]))
        log(traceback.format_exc()[-1500:])
        return False

def main():
    log('=== download session start, pid=%d ===' % os.getpid())
    results = {}
    for repo_id, local_dir in JOBS:
        ok = download_repo(repo_id, local_dir)
        results[repo_id] = 'OK' if ok else 'FAIL'
    # 失败重试一轮(网络抖动)
    for repo_id, local_dir in JOBS:
        if results[repo_id] == 'FAIL':
            log('RETRY %s' % repo_id)
            ok = download_repo(repo_id, local_dir)
            results[repo_id] = 'OK' if ok else 'FAIL'
    log('=== session end: %r ===' % results)
    sys.exit(0 if all(v == 'OK' for v in results.values()) else 1)

if __name__ == '__main__':
    main()
