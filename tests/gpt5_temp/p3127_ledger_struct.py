import io, json
led = json.load(io.open(r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
out = []
def brief(v, d=0):
    if isinstance(v, dict):
        return '{%s}' % ', '.join(sorted(v.keys())[:10])
    if isinstance(v, list):
        return 'list[%d]' % len(v)
    return '%s(%s)' % (type(v).__name__, str(v)[:20])
for k, v in led.items():
    out.append('%s = %s' % (k, brief(v)))
    if isinstance(v, list) and len(v) > 0 and isinstance(v[-1], dict) and k != 'axes':
        out.append('  last item keys: %s' % sorted(v[-1].keys()))
        out.append('  last item: %s' % json.dumps(v[-1], ensure_ascii=False)[:600])
io.open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp\p3127_ledger_struct.txt', 'w', encoding='utf-8').write(chr(10).join(out))
print('LEDGER_OK')
