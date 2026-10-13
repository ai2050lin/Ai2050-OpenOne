# -*- coding: utf-8 -*-
"""Phase 3224 independent verify (18 checks). Read-only; fixes go to THIS
script only, never to sealed products."""
import hashlib
import json
import os


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


P5 = (r'tests\glm5\result\rdc_query_construction_20260913\phase3224'
      r'\g5g25b_weights_readiness_inventory')
ok = []


def chk(i, name, cond):
    ok.append((i, name, bool(cond)))
    print('#%-2d %-58s %s' % (i, name, 'PASS' if cond else 'FAIL'))


res = json.load(open(os.path.join(P5, 'result.json'), encoding='utf-8'))
doc = json.load(open(os.path.join(P5, 'weights_readiness_inventory_v1.json'),
                     encoding='utf-8'))
c2 = json.loads(open(os.path.join(P5, 'weights_readiness_inventory_v1.json'),
                     'rb').read().decode('utf-8'))
c2.pop('sha8', None)
content_sha = hashlib.sha256(
    json.dumps(c2, ensure_ascii=False, indent=1, sort_keys=True)
    .encode('utf-8')).hexdigest()[:8]

# 1-2 verdict + arm scope
chk(1, 'verdict registered',
    res['verdict'] == 'weights_readiness_inventory_registered')
chk(2, 'arm zero-action: no safetensors downloaded',
    'metadata json fetches only' in doc['arm']
    and 'NO safetensors' in doc['arm'])
# 3-4 disk
chk(3, 'R1 disk free C 71.7 / D 32.2',
    doc['disk']['C:/']['free_gb'] == 71.7
    and doc['disk']['D:/']['free_gb'] == 32.2)
chk(4, 'plans A/B/C disk verdicts all OK',
    doc['download_plans']['A_minimal_qwen_pair']['disk_verdict'] == 'OK'
    and doc['download_plans']['B_full_qwen_pair']['disk_verdict'] == 'OK'
    and doc['download_plans']['C_plus_glm4']['disk_verdict'] == 'OK')
# 5-6 R2 weights-level
chk(5, 'R2 cache_state weights-level no safetensors',
    doc['cache_state']['qwen3-4b'] in ('no_weights_in_snapshot',
                                       'no_snapshot_dir')
    and doc['cache_state']['qwen3-14b'] in ('no_weights_in_snapshot',
                                            'no_snapshot_dir')
    and doc['cache_state']['glm4-9b'] == 'no_snapshot_dir')
# 6 R3 config cross-consistency with 3223 anchors
wm = doc['w_u_shard_map']
chk(6, 'R3 hidden/vocab match 3223 D1 shapes (2560/5120, vocab 151936)',
    wm['qwen3-4b']['hidden_size'] == 2560 and wm['qwen3-4b']['vocab_size'] == 151936
    and wm['qwen3-14b']['hidden_size'] == 5120 and wm['qwen3-14b']['vocab_size'] == 151936
    and len(wm['qwen3-4b']['minimal_files']) == 4
    and len(wm['qwen3-14b']['minimal_files']) == 3)
# 7-8 R3 shard map
chk(7, 'R3 4B tied=True shards 1+3, minimal 3.78GB',
    wm['qwen3-4b']['tied'] is True
    and wm['qwen3-4b']['wu_shard'] == 'model-00001-of-00003.safetensors'
    and wm['qwen3-4b']['norm_shard'] == 'model-00003-of-00003.safetensors'
    and wm['qwen3-4b']['minimal_gb'] == 3.78)
chk(8, 'R3 14B tied=False both in shard8, minimal 1.78GB',
    wm['qwen3-14b']['tied'] is False
    and wm['qwen3-14b']['wu_shard'] == 'model-00008-of-00008.safetensors'
    and wm['qwen3-14b']['norm_shard'] == 'model-00008-of-00008.safetensors'
    and wm['qwen3-14b']['minimal_gb'] == 1.78)
# 9 plans totals
chk(9, 'plan totals A 5.56 / B 35.02 / C 52.53',
    res['plan_a_minimal_gb'] == 5.56 and res['plan_b_full_gb'] == 35.02
    and res['plan_c_plus_glm4_gb'] == 52.53)
# 10 R4 manifests
rm = doc['repo_manifests']
chk(10, 'R4 manifests 4B 7.50/3 + 14B 27.52/8 + glm4 17.51/10',
    rm['Qwen/Qwen3-4B-Base']['total_gb'] == 7.5
    and rm['Qwen/Qwen3-14B']['total_gb'] == 27.52
    and rm['THUDM/glm-4-9b-chat']['total_gb'] == 17.51)
# 11 integrity plan
chk(11, 'integrity plan: LFS sha256 + A1 per-shard sha8 + CPU-only',
    any('sha256' in s for s in doc['integrity_plan'])
    and any('sha8' in s for s in doc['integrity_plan'])
    and any('no GPU' in s for s in doc['integrity_plan']))
# 12 authorization request block
chk(12, 'authorization request present with exact bytes',
    'explicit user authorization' in doc['authorization_request']
    and '5.56' in doc['authorization_request'])
# 13 inventory shas
chk(13, 'inventory content sha 0590e53b (recomputed) + file c7a367a3',
    content_sha == '0590e53b' and doc['sha8'] == '0590e53b'
    and sha8_file(os.path.join(
        P5, 'weights_readiness_inventory_v1.json')) == 'c7a367a3')
# 14 R6 + result ledger untouched
chk(14, 'result: ledger untouched n=375 5111392c',
    doc['ledger_untouched'] == {'n': 375,
                                'post': 'a1075729_pre_3223_closed_at_5111392c'})
# 15 ledger #376
led = json.load(open(r'research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
e = led['measurements'][375]
chk(15, 'ledger #376 phase/name/verdict prefix',
    len(led['measurements']) == 376 and e['phase'] == 3224
    and e['name'] == 'g5g25b_weights_readiness_inventory'
    and e['verdict'].startswith('weights_readiness_inventory_registered')
    and e['chain']['inventory_sha8'] == '0590e53b')
# 16 rollback-equality
bako = json.loads(open(r'research\gpt5\atlas\atlas_ledger.json.bak_3224',
                       'rb').read().decode('utf-8'))
back = dict(led)
back['measurements'] = led['measurements'][:375]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
chk(16, 'rollback-equality bak_3224 vs truncated head',
    eq and len(bako['measurements']) == 375
    and bako['ledger_sha256_8'] == '5111392c')
# 17 MEMO section + honest log + prereg
memo = open(r'research\gpt5\docs\AGI_GPT5_MEMO.md', encoding='utf-8').read()
chk(17, 'MEMO 3224 section + honest log + prereg 3225',
    '## Phase 3224 — G5-G25-B' in memo and 'R2 DIRTY' in memo
    and '### 预注册 Phase 3225（两支候选）' in memo
    and '0590e53b' in memo and '#376' in memo
    and '5.56GB' in memo)
# 18 daily + MEMORY
daily = open(r'.workbuddy\memory\2026-10-11.md', encoding='utf-8').read()
mem = open(r'.workbuddy\memory\MEMORY.md', encoding='utf-8').read()
chk(18, 'daily 3224 guard + MEMORY 3225 prereg + 3224 row',
    '### Phase 3224' in daily and '3225 待授权' in mem
    and '0590e53b' in mem and '0543371d' in mem)

n_fail = sum(1 for _, _, c in ok if not c)
print('RECHECK: %s (%d/18 PASS)' % ('CLEAN' if n_fail == 0 else 'FAIL',
                                    18 - n_fail))
