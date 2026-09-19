# -*- coding: utf-8 -*-
"""v2: normalize remaining MEMO phase headings.
exec ts sources: execution.json 'created' key > file mtime;
fallback: body date scan. Idempotent (already-timestamped headings
kept as-is).
"""
import glob
import io
import json
import os
import re
import shutil
import time

MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
BAK = MEMO + '.bak_20260918_v2'
ROOT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')

shutil.copyfile(MEMO, BAK)

exec_map = {}
for ej in glob.glob(os.path.join(ROOT, 'phase*', '*',
                                 'execution.json')):
    m = re.search(r'phase(\d+)', ej)
    if not m:
        continue
    n = m.group(1)
    c = None
    try:
        c = json.load(io.open(ej, encoding='utf-8')).get('created')
    except Exception:
        c = None
    if not c or not re.match(r'\d{4}-\d{2}-\d{2}', str(c)):
        c = time.strftime('%Y-%m-%dT%H:%M:%S',
                          time.localtime(os.path.getmtime(ej)))
    if n not in exec_map or str(c) > exec_map[n]:
        exec_map[n] = str(c)

NUM_RE = re.compile(
    r'^([0-9]{4}(?:\s*\+\s*[0-9]{4}[a-z]?)*(?:\s*-\s*[0-9]{4})?)'
    r'(?:\s*[（(]([^）)]*)[）)])?\s*(.*)$')
TS_RE = re.compile(
    r'\[([0-9]{4}-[0-9]{2}-[0-9]{2})(?:\s+([0-9]{2}:[0-9]{2}))?\]'
    r'\s*$')
DATE_RE = re.compile(r'([0-9]{4}-[0-9]{2}-[0-9]{2})')

lines = io.open(MEMO, encoding='utf-8').read().split('\n')
head_idx = [i for i, l in enumerate(lines)
            if re.match(r'^#{1,4}\s*Phase\s*\d', l)]
out = list(lines)
report = []
unparsed = []

for k, i in enumerate(head_idx):
    line = lines[i]
    m = re.match(r'^(#{1,4})\s*Phase\s*(.+?)\s*$', line)
    hashes, rest = m.group(1), m.group(2)
    ts = None
    m2 = TS_RE.search(rest)
    if m2:
        continue  # already timestamped
    body = lines[i + 1: head_idx[k + 1] if k + 1 < len(head_idx)
                 else len(lines)]
    mnum = NUM_RE.match(rest)
    if not mnum:
        unparsed.append((i + 1, line))
        continue
    numchunk, annot, tail = mnum.group(1), mnum.group(2), \
        mnum.group(3)
    title = re.sub(r'^[\s—\-：:·]+', '', tail).strip()
    date_val, date_src = None, None
    if annot:
        ma = DATE_RE.match(annot.strip())
        if ma:
            date_val = ma.group(1)
            date_src = 'heading-parens'
            annot_rest = annot.strip()[ma.end():].strip('，, ')
            if annot_rest:
                title = (title + '（' + annot_rest + '）'
                         if title else annot_rest)
        else:
            title = (title + '（' + annot + '）'
                     if title else annot)
    m3 = re.search(r'\s*[（(]\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\s*[）)]'
                   r'\s*$', title)
    if m3 and date_val is None:
        date_val = m3.group(1)
        date_src = 'title-trailing'
        title = title[:m3.start()].rstrip()
    date, hm, src = None, None, None
    nfirst = re.match(r'(\d{4})', numchunk).group(1)
    c = exec_map.get(nfirst)
    if c:
        date, hm, src = c[:10], c[11:16], 'exec(%s)' % nfirst
    if date is None and date_val:
        date, src = date_val, date_src or 'heading'
    if date is None:
        for bl in body:
            mb = DATE_RE.search(bl)
            if mb:
                date, src = mb.group(1), 'body'
                break
    if date is None:
        unparsed.append((i + 1, line))
        continue
    bracket = '[%s %s]' % (date, hm) if hm else '[%s]' % date
    new = '%s Phase %s: %s %s' % (hashes, numchunk, title, bracket)
    out[i] = new
    report.append((i + 1, src, line, new))

with io.open(MEMO, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))

with io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
             r'\memo_norm_report2.txt', 'w', encoding='utf-8') as f:
    f.write('changed=%d unparsed=%d exec_map=%d\n'
            % (len(report), len(unparsed), len(exec_map)))
    for ln, src, old, new in report:
        f.write('L%d [%s]\n  OLD: %s\n  NEW: %s\n'
                % (ln, src, old, new))
    for ln, line in unparsed:
        f.write('UNPARSED L%d: %s\n' % (ln, line))
print('done v2: changed=%d unparsed=%d exec_map=%d'
      % (len(report), len(unparsed), len(exec_map)))
