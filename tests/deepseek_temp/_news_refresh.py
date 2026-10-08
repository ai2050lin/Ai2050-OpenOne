# -*- coding: utf-8 -*-
"""刷新 news_cache.json：按服务端 _fetch_arxiv 同样口径抓取最新 arXiv 论文并写入缓存。"""
import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ARXIV_URL = ('https://export.arxiv.org/api/query?search_query='
             'all:%22mechanistic+interpretability%22'
             '&sortBy=submittedDate&sortOrder=descending&max_results=10')
CACHE = Path(r'D:\AI2050\Ai2050-OpenOne\distributed_data\news_cache.json')

req = urllib.request.Request(ARXIV_URL, headers={'User-Agent': 'ai2050-dist/1.0'})
with urllib.request.urlopen(req, timeout=15) as resp:
    data = resp.read()

ns = {'a': 'http://www.w3.org/2005/Atom'}
root = ET.fromstring(data)
items = []
for e in root.findall('a:entry', ns)[:10]:
    title = (e.findtext('a:title', default='', namespaces=ns) or '').strip().replace('\n', ' ')
    summary = (e.findtext('a:summary', default='', namespaces=ns) or '').strip().replace('\n', ' ')
    published = (e.findtext('a:published', default='', namespaces=ns) or '')[:10]
    link = ''
    for l in e.findall('a:link', ns):
        link = l.attrib.get('href', link)
    items.append({'date': published, 'title': title[:120], 'p': summary[:220],
                  'src': 'arxiv.org', 'tag': '论文', 'url': link})

assert len(items) == 10, f'expect 10 entries, got {len(items)}'
CACHE.write_text(json.dumps({'fetched_at': time.time(), 'source': 'arxiv',
                             'items': items}, ensure_ascii=False), encoding='utf-8')

# 回读验证
check = json.loads(CACHE.read_text(encoding='utf-8'))
lines = [f"source={check['source']} items={len(check['items'])}"]
for it in check['items']:
    lines.append(f"  {it['date']}  {it['title'][:60]}")
out = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\_news_refresh_report.txt'
Path(out).write_text('\n'.join(lines), encoding='utf-8')
print('cache refreshed:', out)
