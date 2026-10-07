# -*- coding: utf-8 -*-
"""Direct urllib downloader for Qwen/Qwen3-4B-Base
(hf-mirror.com), with resume support."""
import io
import json
import os
import sys
import time
import urllib.request

DEST = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b-base'
LOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
       r'\.workbuddy\tmp_bdl3_log.txt')
MIRROR = 'https://hf-mirror.com'
REPO = 'Qwen/Qwen3-4B-Base'


def get_json(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode('utf-8'))


def need_file(name, size):
    p = os.path.join(DEST, name)
    if os.path.exists(p):
        return os.path.getsize(p) != size
    return True


def download(name, size):
    p = os.path.join(DEST, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    url = '%s/%s/resolve/main/%s' % (MIRROR, REPO, name)
    part = p + '.part'
    done = os.path.getsize(part) if os.path.exists(part) \
        else 0
    if done >= size:
        os.replace(part, p)
        return 'resumed-complete'
    req = urllib.request.Request(url)
    if done:
        req.add_header('Range', 'bytes=%d-' % done)
    with urllib.request.urlopen(req, timeout=120) as r, \
            open(part, 'ab') as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    if os.path.getsize(part) == size:
        os.replace(part, p)
        return 'ok'
    return 'size-mismatch %d != %d' % (
        os.path.getsize(part), size)


def main():
    meta = get_json(
        'https://hf-mirror.com/api/models/%s' % REPO)
    siblings = [s['rfilename'] for s in meta['siblings']]
    lines = ['files: %d' % len(siblings)]
    with open(LOG, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    ok_all = True
    for name in siblings:
        if name.endswith(('.gitattributes',)):
            continue
        # get size via HEAD-ish: use API siblings w/o
        # size; fetch size by ranging first byte
        url = '%s/%s/resolve/main/%s' % (
            MIRROR, REPO, name)
        req = urllib.request.Request(url, method='GET')
        req.add_header('Range', 'bytes=0-0')
        try:
            with urllib.request.urlopen(req,
                                        timeout=60) as r:
                cr = r.headers.get('Content-Range', '')
                total = int(cr.split('/')[-1]) if cr \
                    else int(r.headers.get(
                        'Content-Length', 0) or 0)
        except Exception as e:
            lines.append('%s: HEAD-FAIL %s'
                         % (name, str(e)[:80]))
            ok_all = False
            continue
        if not need_file(name, total):
            lines.append('%s: already-complete (%d)'
                         % (name, total))
            continue
        t0 = time.time()
        try:
            st = download(name, total)
        except Exception as e:
            st = 'FAIL %s: %s' % (type(e).__name__,
                                  str(e)[:120])
            ok_all = False
        dt = time.time() - t0
        lines.append('%s: %s (%d bytes, %.1fs)'
                     % (name, st, total, dt))
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write('\n'.join(lines[-1:]) + '\n')
    lines.append('ALL_OK=%s' % ok_all)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines[-1:]) + '\n')


if __name__ == '__main__':
    main()
