# -*- coding: utf-8 -*-
"""Normalize all MEMO phase headings to:
## Phase {N}: {title} [yyyy-mm-dd hh:mm]
Timestamp priority: heading [ts] > execution.json created (latest
of that phase's products) > body date (time omitted -> [date]).
Backup first: AGI_GPT5_MEMO.md.bak_20260918
"""
import glob
import io
import json
import os
import re
import shutil

MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
BAK = MEMO + '.bak_20260918'
ROOT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')

shutil.copyfile(MEMO, BAK)
print('backup -> %s' % BAK)

# ---- execution.json created map: phaseN -> max created ----
exec_map = {}
for ej in glob.glob(os.path.join(ROOT, 'phase*', '*',
                                 'execution.json')):
    m = re.search(r'phase(\d+)', ej)
    if not m:
        continue
    n = m.group(1)
    try:
        c = json.load(io.open(ej, encoding='utf-8')).get('created')
    except Exception:
        continue
    if c:
        if n not in exec_map or c > exec_map[n]:
            exec_map[n] = c

NUM_RE = re.compile(
    r'^([0-9]{4}(?:\s*\+\s*[0-9]{4}[a-z]?)*(?:\s*-\s*[0-9]{4})?)'
    r'(?:\s*[（(]([^）)]*)[）)])?\s*(.*)$')
TS_RE = re.compile(
    r'\[([0-9]{4}-[0-9]{2}-[0-9]{2})(?:\s+([0-9]{2}:[0-9]{2}))?\]'
    r'\s*$')
DATE_RE = re.compile(r'([0-9]{4}-[0-9]{2}-[0-9]{2})')

lines = io.open(MEMO, encoding='utf-8').read().split('\n')
out = []
report = []
unparsed = []
cur_body = []

for idx, line in enumerate(lines):
    m = re.match(r'^(#{1,4})\s*Phase\s*(.+?)\s*$', line)
    if not m:
        cur_body.append(line)
        out.append(line)
        continue
    hashes, rest = m.group(1), m.group(2)
    ts = None
    m2 = TS_RE.search(rest)
    if m2:
        ts = (m2.group(1), m2.group(2))
        rest = rest[:m2.start()].rstrip()
    mnum = NUM_RE.match(rest)
    if not mnum:
        unparsed.append((idx + 1, line))
        out.append(line)
        cur_body = []
        continue
    numchunk, annot, tail = mnum.group(1), mnum.group(2), \
        mnum.group(3)
    title = re.sub(r'^[\s—\-：:·]+', '', tail).strip()
    date_src = None
    date_val = None
    if annot:
        ma = DATE_RE.match(annot.strip())
        if ma:
            date_val = ma.group(1)
            date_src = 'heading-parens'
            annot_rest = annot.strip()[ma.end():].strip('，, ')
            if annot_rest:
                if title:
                    title = title + '（' + annot_rest + '）'
                else:
                    title = annot_rest
        else:
            if title:
                title = title + '（' + annot + '）'
            else:
                title = annot
    # strip trailing standalone date parens from title
    m3 = re.search(r'\s*[（(]\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\s*[）)]'
                   r'\s*$', title)
    if m3 and date_val is None:
        date_val = m3.group(1)
        date_src = 'title-trailing'
        title = title[:m3.start()].rstrip()
    # timestamp resolution
    if ts:
        date, hm = ts
        src = 'heading-ts'
    else:
        nfirst = re.match(r'(\d{4})', numchunk).group(1)
        c = exec_map.get(nfirst)
        if c:
            date = c[:10]
            hm = c[11:16]
            src = 'execution.json(%s)' % nfirst
        elif date_val:
            date, hm = date_val, None
            src = date_src or 'body'
        else:
            date, hm = None, None
            src = 'none'
    if date is None:
        unparsed.append((idx + 1, line))
        out.append(line)
        cur_body = []
        continue
    bracket = '[%s %s]' % (date, hm) if hm else '[%s]' % date
    new = '%s Phase %s: %s %s' % (hashes, numchunk, title, bracket)
    if new != line:
        report.append((idx + 1, src, line, new))
    out.append(new)
    cur_body = []

with io.open(MEMO, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))

with io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
             r'\memo_norm_report.txt', 'w', encoding='utf-8') as f:
    f.write('changed=%d unparsed=%d exec_map_phases=%d\n'
            % (len(report), len(unparsed), len(exec_map)))
    for ln, src, old, new in report:
        f.write('L%d [%s]\n  OLD: %s\n  NEW: %s\n'
                % (ln, src, old, new))
    for ln, line in unparsed:
        f.write('UNPARSED L%d: %s\n' % (ln, line))
print('done: changed=%d unparsed=%d' % (len(report),
                                        len(unparsed)))
