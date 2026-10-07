"""Phase 3128 closeout: idempotent five-write
chain (result -> ledger -> MEMO -> wlogs ->
MEMORY). MEMO title SHORT per user rule
(2026-09-24): narrative goes to body sections.
"""
import datetime
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3128'
        r'\omega_p126_joint_swap_coord_'
        r'inject_interaction_s0match')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOGS = [ROOT + r'\.workbuddy\memory',
         (r'C:\Users\Admin\WorkBuddy'
          r'\2026-09-17-01-30-05'
          r'\.workbuddy\memory')]
MEMW = ROOT + r'\.workbuddy\memory\MEMORY.md'
NOW = datetime.datetime.now()
STAMP = NOW.strftime('%Y-%m-%d %H:%M')
r = json.load(io.open(OUTD + r'\result.json',
                      encoding='utf-8'))
assert r['phase'] == 3128
assert r['smoke'] is False
V = r['verdict']
assert len(V.split('|')) == 10
pb = r['part_b']
pc = r['part_c']
pd_ = r['part_d']

# sha8 of result.json
raw = io.open(OUTD + r'\result.json',
              'rb').read()
sha8 = hashlib.sha256(raw).hexdigest()[:8]

steps = []
if 'meas3128_omega_p126' in io.open(
        LEDGER, encoding='utf-8').read():
    steps.append('ledger: exists, skip')
else:
    led = json.load(io.open(
        LEDGER, encoding='utf-8'))
    claim = (
        'Omega-P126 (3128, T4 eleventh phase: '
        'joint multi-layer swap W5/C5 qwen + '
        'W4/C4 glm4, coordinate-level +-delta '
        'injection qwen L24 capture_b top-K '
        'K in 5/50/200 + rand50 + glm4 L20 '
        'odd-half fit/even-half test top50, '
        'interaction localization dmg x regen '
        'flip labels, s0 probe batch-32 '
        'matched composition, 6617s) - '
        'verdict ' + V)
    entry = {
        'meas_id':
            'meas3128_omega_p126_joint_swap_'
            'coord_inject_interaction_s0match',
        'phase': 3128,
        'claim': claim,
        'verdict': V,
        'artifacts': {
            'result_json':
                'phase3128/omega_p126_.../result'
                '.json sha256_8=' + sha8,
            'npz': 'p126_readout.npz'},
        'hashes': {'result_sha256_8': sha8},
        'anchors': [],
        'note': 'joint swap double-negative '
                '(qwen 0.897 / glm4 0.896) + '
                'coord injection effective '
                'dose-monotonic + s0 bit-exact '
                'batch-matched'}
    led['measurements'].append(entry)
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w',
                 encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False,
                  indent=1)
    steps.append('ledger: appended n=%d '
                 'sha8=%s'
                 % (len(led['measurements']),
                    led['ledger_sha256_8']))

memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3128:' in memo:
    steps.append('memo: exists, skip')
else:
    sec = ''
    sec += ('## Phase 3128: \u03a9-P126 '
            '\u591a\u5c42\u8054\u5408 swap '
            '\u53cc\u6a21\u578b\u9634\u6027 '
            '+ \u5750\u6807\u6ce8\u5165'
            '\u6709\u6548 + s0 \u6279\u5339'
            '\u914d\u4f4d\u7ea7\u590d\u73b0'
            '\uff08T4 \u7b2c11Phase\uff09'
            '[' + STAMP + ']\n\n')
    sec += (
        '**\u6027\u8d28**\uff1aT4 \u7b2c 11 '
        'Phase\uff0c3127 MEMO \u7b2c 5 \u8282 '
        '\u9884\u6ce8\u518c\u56db\u9879'
        '\u6267\u884c\u3002Part A0 offline '
        '\u590d\u523b 3127 \u95e8\u503c'
        '\uff08f32 \u51cf\u6cd5\u2192\u5347'
        '\u4f4d f64\u2192f64 median \u8def'
        '\u5f84\uff0c1e-12\uff09\uff1bPart B '
        'GPU qwen3-4b\uff08W5/C5 \u8054'
        '\u5408\u8df3\u5c42 + L24 \u5750'
        '\u6807\u6ce8\u5165\u636e capture_b '
        '\u76f8\u5173\u5750\u6807\uff0c'
        '672\u00d72\u00d78 \u524d\u5411'
        '\uff09\uff1bPart C GPU glm4-9b'
        '\uff08W4/C4 + L20 \u5947\u534a fit/'
        '\u5076\u534a\u6ce8\u5165 + s0 '
        'batch-32 \u63a2\u9488\uff09\uff1b'
        'Part D \u4ea4\u4e92\u5b9a\u4f4d'
        '\uff08dmg \u573a\u00d7regen flip '
        '\u6807\u7b7e\uff09\u3002SMOKE 3 '
        '\u8f6e\u4fee\u590d\uff08A0 median '
        '\u7cbe\u5ea6\u8def\u5f84\u3001'
        'SCOND3 \u4e22\u5931\u3001\u8bed'
        '\u6cd5\u6b8b\u7559\uff09\u3002'
        '\u8fd0\u884c 6617s\u3002\n\n')
    sec += '### 1. \u4e09\u5927\u53d1'
    sec += '\u73b0\uff08\u91cd\u590d'
    sec += '\u4e09\u904d\uff09\n'
    f1 = (
        '1. **\u591a\u5c42\u8054\u5408 '
        'swap \u53cc\u6a21\u578b\u9634'
        '\u6027\u2014\u2014\u5199\u5165'
        '\u94fe\u6982\u5ff5\u964d\u7ea7'
        '\u4e3a\u7eaf\u76f8\u5173\u63cf'
        '\u8ff0**\u3002Qwen W5 \u8054'
        '\u5408\u8df3\u5c42 ratio 0.897'
        '\uff08write_med 1.2599 vs ctrl '
        '1.4045\uff09\u3001GLM4 W4 ratio '
        '0.896\uff08write 0.6697 vs ctrl '
        '0.7476\uff09\u2014\u2014\u4e24'
        '\u6a21\u578b\u51e0\u4e4e\u540c'
        '\u503c\u4e14\u5747\u4f4e\u4e8e '
        'ctrl\u3002\u5355\u5c42\uff083127 '
        '\u00d71.04/\u00d70.87\uff09+ '
        '\u591a\u5c42\u8054\u5408\uff08'
        '3128 \u00d70.90/\u00d70.90\uff09'
        '\u53cc\u91cd\u9634\u6027\u2192 '
        'margin \u4e0e write \u5e26\u5c42'
        '\u65e0\u56e0\u679c\u4f9d\u8d56'
        '\uff0c\u8c31\u5cf0\u4e0e\u529f'
        '\u80fd\u5b8c\u5168\u89e3\u8026'
        '\u3002**\u91cd\u590d\uff1a\u5199'
        '\u5165\u94fe\u65e0\u529f\u80fd'
        '\u5fc5\u8981\u6027\uff1b\u5c42'
        '\u7ea7\u5e72\u9884\u65cf\u65cf'
        '\u8017\u5c3d\u3002**\n')
    sec += f1 * 3
    f2 = (
        '2. **\u5750\u6807\u7ea7\u6ce8'
        '\u5165\u53cc\u6a21\u578b\u6709'
        '\u6548\u4e14 dose \u5355\u8c03'
        '**\u3002Qwen L24\uff08\u636e '
        'capture_b \u76f8\u5173\u5750'
        '\u6807\uff09\uff1aflip 0.029'
        '\u21920.182\u21920.328\uff08K='
        '5/50/200\uff09vs rand50 0.095'
        '\uff08top200 = 3.4\u00d7 rand'
        '\uff09\uff1bGLM4 L20\uff08\u5947'
        '\u534a 336 fit/\u5076\u534a\u6d4b'
        '\u8bd5\uff09\uff1atop50 flip '
        '0.140\u3002\u5c42\u7ea7\u5e72'
        '\u9884\u65e0\u6548\u800c\u5750'
        '\u6807\u7ea7\u5b9a\u5411\u6270'
        '\u52a8\u53ef\u9760\u7ffb\u8f6c '
        'margin\u2014\u2014\u6548\u5e94'
        '\u5728\u5750\u6807\u65b9\u5411'
        '\u800c\u975e\u5c42\uff0c\u4e0e '
        '3109-3112 \u5168\u606f\u5197'
        '\u4f59\uff08\u4efb\u610f\u5b50'
        '\u7a7a\u95f4\u53ef\u8bfb\u51fa'
        '\uff09\u76f8\u5bb9\u3002**\u91cd'
        '\u590d\uff1a\u5e72\u9884\u7c92'
        '\u5ea6=\u5750\u6807\u65b9\u5411'
        '\uff1breadout \u5bf9\u9f50\u5750'
        '\u6807\u63a8\u62c9\u5373\u53ef'
        '\u64cd\u63a7 margin\u3002**\n')
    sec += f2 * 3
    f3 = (
        '3. **s0 \u63a2\u9488\u6279\u5339'
        '\u914d\u4f4d\u7ea7\u590d\u73b0'
        '+ \u4ea4\u4e92\u5b9a\u4f4d'
        '\u9634\u6027**\u3002s0 \u63a2'
        '\u9488\uff08\u524d 32 prompts '
        '\u5355\u6279 batch-32\uff0c\u4e0e '
        '3126 gen_base \u9996\u6279\u5b8c'
        '\u5168\u540c\u7ec4\u6210\uff09'
        '\u2192 **bit-exact\uff08mism=0'
        '\uff09**\uff1a3127 \u7684 agree '
        '0.9167 \u6f02\u79fb\u5b8c\u5168'
        '\u7531\u6279\u7ec4\u6210\u5dee'
        '\u5f02\u89e3\u91ca\uff0c\u751f'
        '\u6210\u7ba1\u7ebf\u786e\u5b9a'
        '\u6027\u5f97\u8bc1\u3002\u4ea4'
        '\u4e92\u5b9a\u4f4d\uff08\u5168'
        '\u91cf 672\uff09\uff1a\u5cf0\u503c '
        'L09 sep 0.059/r 0.050\u2192 '
        'interaction_not_in_field\u2014'
        '\u2014swap \u5e45\u5ea6\u573a'
        '\u4e0d\u643a\u5e26\u65b9\u5411'
        '\u00d7\u6270\u52a8\u7c7b\u578b'
        '\u4ea4\u4e92\u4fe1\u606f\uff08'
        '\u5e45\u5ea6\u2260\u7b26\u53f7'
        '\uff0c3127 \u5df2\u77e5 swap '
        '\u6548\u5e94\u5c0f\uff09\u3002'
        '**\u91cd\u590d\uff1a\u751f\u6210'
        '\u7ba1\u7ebf\u4f4d\u7ea7\u786e'
        '\u5b9a\uff1bswap \u5e45\u5ea6'
        '\u573a\u4e0e flip \u884c\u4e3a'
        '\u573a\u89e3\u8026\u3002**\n')
    sec += f3 * 3
    sec += '\n### 2. \u5173\u952e'
    sec += '\u6570\u503c\n'
    sec += (
        'Part A0\uff1arepl \u95e8\u503c '
        'q {w 1.0413/p 1.0872} g {w '
        '0.8705/p 1.9011e-1}\uff081e-12 '
        '\u5168\u5bf9\uff09\u3002Part B'
        '\uff1arepro 0.0\u3001ratio 0.8970'
        '\u3001delta_q 10.6946\u3001coord '
        '{top5 0.0294/0.2010, top50 '
        '0.1823/1.0975, top200 0.3281/'
        '2.7088, rand50 0.0952/0.6986}'
        '\u3002Part C\uff1arepro 0.0\u3001'
        'ratio 0.8958\u3001delta_g 0.9954'
        '\u3001top50 0.1399/1.0251\u3001'
        's0 agree 1.0000 bit_mism 0'
        '\u3002Part D\uff1apeak L09 sep '
        '0.0591 r 0.0504\u3002\n\n')
    sec += '### 3. \u786c\u4f24\n'
    sec += (
        '\u2460 \u5750\u6807\u96c6\u4e24'
        '\u6a21\u578b\u4e0d\u540c\u6e90'
        '\uff08Qwen=capture_b \u76f8'
        '\u5173\u5750\u6807\u3001GLM4='
        '\u5947\u534a\u73b0\u573a fit'
        '\uff09\uff0ccross_model \u4ec5'
        '\u63cf\u8ff0\u65e0\u95e8\uff1b'
        '\u2461 \u6ce8\u5165\u4ec5\u6700'
        '\u540e\u4f4d\u7f6e\uff08margin '
        '\u8bfb\u70b9\uff09\uff0c\u5168'
        '\u5e8f\u5217\u6ce8\u5165\u672a'
        '\u6d4b\uff1b\u2462 dose \u5355'
        '\u8c03\u5224\u636e\u5bbd\u677e'
        '\uff08\u5bb9\u5fcd 0.01 \u53cd'
        '\u8f6c\uff09\u4e14 GLM4 \u4ec5 1 '
        '\u6863 K\uff1b\u2463 \u4ea4\u4e92'
        '\u5b9a\u4f4d\u7528 |dm| \u5e45'
        '\u5ea6\u800c flip \u662f\u7b26'
        '\u53f7\u4e8b\u4ef6\uff0c\u529f'
        '\u6548\u6709\u9650\uff0c\u9634'
        '\u6027\u53ef\u80fd\u4e3a\u4f4e'
        '\u529f\u6548\u800c\u975e\u65e0'
        '\u4ea4\u4e92\uff1b\u2464 s0 \u63a2'
        '\u9488\u4ec5 P \u65b9\u5411\u524d '
        '32\uff0cA1 \u4e0e\u540e\u7eed\u6279'
        '\u672a\u6d4b\uff1b\u2465 \u4e24'
        '\u6a21\u578b\u03b4 \u5c3a\u5ea6'
        '\u4e0d\u540c\u6e90\uff0810.69 vs '
        '0.995\uff09\uff0cflip \u7387\u4e0d'
        '\u53ef\u76f4\u63a5\u6bd4\u8f83'
        '\uff1b\u2466 joint swap \u540e '
        'margin \u53d8\u5316\u5c0f\u4f46'
        '\u751f\u6210\u662f\u5426\u53d8'
        '\u672a\u6d4b\uff08margin-\u751f'
        '\u6210\u89e3\u8026\u672a\u9a8c'
        '\u8bc1\uff09\u3002\n\n')
    sec += '### 4. \u673a\u5236\u62fc'
    sec += '\u56fe\u66f4\u65b0\n'
    sec += (
        '\u2460 \u5199\u5165\u94fe\u6982'
        '\u5ff5\u6b63\u5f0f\u964d\u7ea7'
        '\uff1a\u5355\u5c42\uff083127\uff09'
        '+ \u591a\u5c42\u8054\u5408'
        '\uff083128\uff09\u53cc\u91cd'
        '\u9634\u6027\uff0c\u8c31\u5cf0'
        '\u4ec5\u76f8\u5173\uff1b\u2461 '
        '\u6709\u6548\u5e72\u9884\u7c92'
        '\u5ea6\u786e\u7acb\uff1a\u5750'
        '\u6807\u65b9\u5411\uff08readout '
        '\u5bf9\u9f50 top-K\uff09\u53ef'
        '\u9760\u7ffb\u8f6c margin \u800c'
        '\u5c42\u7ea7\u4e0d\u80fd\uff0c'
        '\u4e0e d_min=5 \u5168\u606f'
        '\u5197\u4f59\u76f8\u5bb9\uff1b'
        '\u2462 \u751f\u6210\u7ba1\u7ebf'
        '\u786e\u5b9a\u6027\u5b8c\u5907'
        '\uff08\u6279\u5339\u914d\u4f4d'
        '\u7ea7\uff09\uff1b\u2463 swap '
        '\u5e45\u5ea6\u573a\u4e0e flip '
        '\u884c\u4e3a\u573a\u89e3\u8026'
        '\u2014\u2014\u4e09\u56fe\u8c31'
        '\u5173\u8054\u9700\u7b26\u53f7'
        '\u7ea7/\u751f\u6210\u5b9e\u65f6'
        '\u5de5\u5177\u3002\n\n')
    sec += '### 5. 3129 \u9884\u6ce8'
    sec += '\u518c\uff08T4 \u7ee7\u7eed'
    sec += '\uff0c\u89c2\u6d4b\u524d'
    sec += '\u51bb\u7ed3\uff09\n'
    sec += (
        '\u2460 \u5750\u6807\u6ce8\u5165'
        '\u6df1\u5316\uff1a\u03b4 \u5242'
        '\u91cf\u626b\u63cf\uff08{0.05,'
        '0.1,0.2}\u00d7norm\uff09+ A1 '
        '\u65b9\u5411\u6269\u5c55 + \u591a'
        '\u70b9\uff08\u4e2d\u95f4\u4f4d'
        '\uff09\u6ce8\u5165\uff1b\u2461 '
        '\u4ea4\u4e92\u5b9a\u4f4d\u5347'
        '\u7ea7\uff1aregen \u524d\u5411'
        '\u73b0\u573a\u91c7 margin \u7b26'
        '\u53f7\u573a\u9010\u5c42\u5b9a'
        '\u4f4d\u65b9\u5411\u00d7\u6761'
        '\u4ef6\u4ea4\u4e92\u6d8c\u73b0'
        '\u5c42\uff08GPU\uff09\uff1b\u2462 '
        'margin-\u751f\u6210\u89e3\u8026'
        '\u68c0\u9a8c\uff1ajoint swap \u540e'
        '\u751f\u6210\u5bf9\u7167\uff08'
        'margin \u53d8\u5c0f\u4f46\u751f'
        '\u6210\u53d8\u4e0d\u53d8\uff1f'
        '\uff09\uff1b\u2463 s0 \u63a2\u9488'
        '\u6269\u5c55\u5168 21 \u6279\u4f4d'
        '\u7ea7\uff08\u751f\u6210\u7ba1'
        '\u7ebf\u5b8c\u5168\u786e\u5b9a'
        '\u6027\u9a8c\u8bc1\uff09\u3002\n')
    sec += '\n\u4ea7\u7269\uff1a`tests/'
    sec += 'glm5/result/rdc_query_'
    sec += 'construction_20260913/'
    sec += 'phase3128/omega_p126_joint_'
    sec += 'swap_coord_inject_interaction_'
    sec += 's0match/`\uff08result.json\u3001'
    sec += 'design_seal.json\u3001run_log'
    sec += '.txt\u3001p126_readout.npz\uff09'
    sec += '\uff1b\u811a\u672c `tests/glm5/'
    sec += 'phase3128_omega_p126_joint_swap_'
    sec += 'coord_inject_interaction_s0match'
    sec += '.py`\u3002'
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write(sec)
    steps.append('memo: appended (short '
                 'title)')

WDATE = NOW.strftime('%Y-%m-%d')
wline = ('- Phase 3128 Omega-P126 closeout: '
         'verdict ' + V + '; ledger sha8 '
         + json.load(io.open(
             LEDGER,
             encoding='utf-8'))
         ['ledger_sha256_8']
         + '; MEMO 3128 section (short '
         'title); runtime 6617s.')
for wd in WLOGS:
    wl = wd + '\\' + WDATE + '.md'
    try:
        prev = io.open(wl,
                       encoding='utf-8').read()
    except IOError:
        prev = ''
    if 'Phase 3128 Omega-P126 closeout' \
            not in prev:
        with io.open(wl, 'a',
                     encoding='utf-8') as f:
            f.write(wline + '\n')
        steps.append('wlog: ' + wl[:12])
    else:
        steps.append('wlog: exists ' + wl[:12])

mem = io.open(MEMW, encoding='utf-8').read()
if '3128\uff08T4\uff09' not in mem:
    old = ('- 3127\uff08T4\uff09\uff1a'
           '\u5199\u5165\u94fe\u529f\u80fd'
           '\u5426\u5b9a\uff08Qwen \u00d71.04/'
           'GLM4 \u00d70.87 \u5168 <2 \u95e8'
           '\u3001spec r 0.02/0.05\u3001'
           '\u6df1\u5ea6\u76f8\u5173 '
           '\u22120.36\uff09=\u5355\u5c42'
           '\u65e0\u5fc5\u8981\u5199\u5165'
           '\uff1btrail \u5b9a\u7a3f lag1-3'
           '\uff08lag4-6 z \u22122.8/'
           '\u22125.6 \u4f4e\u4e8e null\u3001'
           '\u8fc1\u79fb\u8d1f\uff09\uff1b'
           '\u4e8c\u9636/PCA/\u7a00\u758f'
           '\u4e09\u5019\u9009\u5168\u7f3a'
           '\u5e2d\uff1b\u5168\u91cf regen '
           '\u524d 96 \u4f4d\u7ea7\u590d'
           '\u73b0\u3001flip \u590d\u73b0'
           '\u3001No 70%\u3001multi_flip 0\u3002')
    new = old + (
        '\n- 3128\uff08T4\uff09\uff1a'
        '\u591a\u5c42\u8054\u5408 swap '
        '\u53cc\u6a21\u578b\u9634\u6027'
        '\uff08qwen 0.897/glm4 0.896\uff09'
        '=\u5199\u5165\u94fe\u964d\u7ea7'
        '\u7eaf\u76f8\u5173\uff1b\u5750'
        '\u6807\u6ce8\u5165\u6709\u6548'
        '\u4e14 dose \u5355\u8c03\uff08qwen '
        '0.029/0.182/0.328 vs rand 0.095'
        '\u3001glm4 0.140\uff09=\u5e72\u9884'
        '\u7c92\u5ea6\u662f\u5750\u6807'
        '\u65b9\u5411\uff1bs0 \u6279\u5339'
        '\u914d\u4f4d\u7ea7\u590d\u73b0'
        '\uff08mism=0\uff09\uff1b\u4ea4'
        '\u4e92\u5b9a\u4f4d\u9634\u6027'
        '\uff08\u5e45\u5ea6\u573a\u89e3'
        '\u8026\uff09\u3002')
    assert old in mem
    mem2 = mem.replace(old, new)
    old2 = ('\u4e0b\u4e00 3128\uff1a**'
            '\u591a\u5c42\u7ec4\u5408 swap'
            '\uff08write \u5e26\u6574\u6bb5'
            '\u8054\u5408\u8df3\u8fc7\uff09'
            '+ \u5750\u6807\u7ea7\u7a00\u758f'
            '\u5e72\u9884 + \u65b9\u5411'
            '\u00d7\u6270\u52a8\u7c7b\u578b'
            '\u4ea4\u4e92\uff08P \u66ff\u6362'
            '\u654f\u611f/A1 \u5220\u9664'
            '\u654f\u611f\uff09\u6df1\u6316'
            '**\u3002')
    new2 = ('\u4e0b\u4e00 3129\uff1a**'
            '\u5750\u6807\u6ce8\u5165\u6df1'
            '\u5316\uff08\u03b4 \u5242\u91cf'
            '\u626b\u63cf + A1 \u6269\u5c55 '
            '+ \u591a\u70b9\u6ce8\u5165\uff09'
            '+ regen \u7b26\u53f7\u573a\u9010'
            '\u5c42\u4ea4\u4e92\u5b9a\u4f4d '
            '+ margin-\u751f\u6210\u89e3\u8026'
            '\u68c0\u9a8c + s0 \u5168 21 \u6279'
            '\u4f4d\u7ea7**\u3002')
    assert old2 in mem2
    mem2 = mem2.replace(old2, new2)
    with io.open(MEMW, 'w',
                 encoding='utf-8') as f:
        f.write(mem2)
    steps.append('memory: updated (%d '
                 'chars)' % len(mem2))

for s in steps:
    print(s)
print('CLOSEOUT_OK (%d steps)' % len(steps))
