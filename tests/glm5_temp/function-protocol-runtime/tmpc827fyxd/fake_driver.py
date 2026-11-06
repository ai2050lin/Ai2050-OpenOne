
import json, sys, time
mode = sys.argv[1]
def p(step, label, **kw):
    print('@@progress ' + json.dumps({'step': step, 'label': label, **kw}, ensure_ascii=False), flush=True)
p(1, '文法预览')
if mode == 'gate':
    p(2, '试跑与行为门', detail='3/30 格达标'); sys.exit(3)
if mode == 'crash':
    print('Traceback: boom', flush=True); sys.exit(1)
if mode == 'slow':
    p(4, '分片采集'); time.sleep(30)
p(3, '冻结正式模板', tm_id='TG-X')
p(7, '报告')
print('@@result ' + json.dumps({'tm_id': 'TG-X', 'bucket': 'm|r|bf16', 'findings': {'supported': 2}}), flush=True)
