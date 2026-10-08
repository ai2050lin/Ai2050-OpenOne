# -*- coding: utf-8 -*-
"""诊断 /api/news 回退：复现 distributed_service._fetch_arxiv 的抓取逻辑，定位失败原因。"""
import json
import ssl
import traceback
import urllib.request

ARXIV_URL = ('https://export.arxiv.org/api/query?search_query='
             'all:%22mechanistic+interpretability%22'
             '&sortBy=submittedDate&sortOrder=descending&max_results=10')

report = []
report.append('python urllib republish test')

# 1) 环境代理
import os
for k in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY', 'no_proxy'):
    report.append(f'{k}={os.environ.get(k)}')

# 2) 与服务完全一致的调用（timeout=6, UA 相同）
try:
    req = urllib.request.Request(ARXIV_URL, headers={'User-Agent': 'ai2050-dist/1.0'})
    with urllib.request.urlopen(req, timeout=6) as resp:
        data = resp.read()
    report.append(f'PLAIN urlopen OK, bytes={len(data)}')
except Exception as e:
    report.append('PLAIN urlopen FAIL:')
    report.append(traceback.format_exc())

# 3) 宽松 SSL 再试
try:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(ARXIV_URL, headers={'User-Agent': 'ai2050-dist/1.0'})
    with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
        data = resp.read()
    report.append(f'NOVERIFY urlopen OK, bytes={len(data)}')
except Exception as e:
    report.append('NOVERIFY urlopen FAIL:')
    report.append(traceback.format_exc())

# 4) 成功则解析看首条日期
try:
    import xml.etree.ElementTree as ET
    ns = {'a': 'http://www.w3.org/2005/Atom'}
    root = ET.fromstring(data)
    entries = root.findall('a:entry', ns)[:10]
    report.append(f'entries={len(entries)}')
    for e in entries[:10]:
        t = (e.findtext('a:title', default='', namespaces=ns) or '').strip()[:70]
        pub = (e.findtext('a:published', default='', namespaces=ns) or '')[:10]
        report.append(f'  {pub}  {t}')
except Exception:
    report.append('parse skipped (no data)')

out = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\_news_diag_report.txt'
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))
print('written', out)
