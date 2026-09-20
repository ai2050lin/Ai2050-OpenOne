# -*- coding: utf-8 -*-
"""Phase 2944 seal + closeout: forensic snapshot, Ledger, MEMO,
workspace log, MEMORY."""
import hashlib
import json
import os
import time

OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913\phase2944'
       r'\switch_localization')
SEAL_REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
               r'\phase2944_seal_report.txt')
LEDGER = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\2026-09-19.md')
WMEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\MEMORY.md')

MEAS_ID = 'meas_2944_switch_localization'
LINK_ID = 'L14_readout_spectrum_cross_model'


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def main():
    lines = []
    res = json.load(open(os.path.join(OUT, 'result.json'),
                         encoding='utf-8'))
    exec_j = json.load(open(os.path.join(OUT, 'execution.json'),
                            encoding='utf-8'))
    verdict = res['final_verdict']
    created = exec_j['created']

    # ---------- seal ----------
    lines.append('phase 2944 seal @ %s'
                 % time.strftime('%Y-%m-%dT%H:%M:%S'))
    lines.append('verdict: %s' % verdict)
    lines.append('execution created: %s' % created)
    lines.append('runtime_s: %s' % res['runtime_s'])
    for fn in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, fn)
        if os.path.isfile(p):
            lines.append('  %-28s %s %d bytes'
                         % (fn, sha8(p), os.path.getsize(p)))
    script = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
              r'\phase2944_switch_localization.py')
    lines.append('script sha8: %s' % sha8(script))
    with open(SEAL_REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    # ---------- Ledger ----------
    led = json.load(open(LEDGER, encoding='utf-8'))
    if MEAS_ID not in [m['meas_id']
                       for m in led['measurements']]:
        led['measurements'].append({
            'meas_id': MEAS_ID,
            'type': 'forward_family_layer_injection',
            'verdict': verdict,
            'source': 'phase2944/switch_localization'})
        for lk in led['linkage']:
            if lk['link_id'] == LINK_ID:
                if MEAS_ID not in lk['connects']:
                    lk['connects'].append(MEAS_ID)
    led.pop('ledger_sha256_8', None)
    canonical = json.dumps(led, sort_keys=True,
                           ensure_ascii=False).encode('utf-8')
    led['ledger_sha256_8'] = \
        hashlib.sha256(canonical).hexdigest()[:8]
    json.dump(led, open(LEDGER, 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    led2 = json.load(open(LEDGER, encoding='utf-8'))
    chk = led2.pop('ledger_sha256_8')
    c2 = json.dumps(led2, sort_keys=True,
                    ensure_ascii=False).encode('utf-8')
    assert hashlib.sha256(c2).hexdigest()[:8] == chk
    n_meas = len(led2['measurements'])
    l14 = [lk for lk in led2['linkage']
           if lk['link_id'] == LINK_ID][0]

    # ---------- MEMO ----------
    hh, mm = created.split('T')[1].split(':')[:2]
    stamp = '%s %s:%s' % (created.split('T')[0], hh, mm)
    hashes = {}
    for fn in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, fn)
        if os.path.isfile(p):
            hashes[fn] = sha8(p)
    memo_block = """
## Phase 2944: regime 开关层定位与剂量浓度判决 [%s]

**原理**：2942 在 L16 单点注入联合 U8 位移模式发现 sep 陡降/过冲（regime 开关）+ s=2 跨 session 不稳定；2940 把 v3 所有权定位到 L14-L18。开放问题：开关可定位于带内单一层，还是需要多层联合驱动？

**方法**：一次前向族。注入向量 2942 verbatim（xdir = sum_{k in {v1,v2,v5}} dcks*Vt8）；层配置 = 单层 L14/15/16/17/18（系数 1.0）+ multi_split（L14-18 各 0.2，总剂量与单层可比）+ multi_full（各 1.0，描述性）；s in {1,2,4}；K=3 同 session 重复取中位。锚 8/8：a1 2.17e-08（**第 18 次连续前向锚定**）、a3 bit 级 0、a7 9.95e-14、**a8 同 session 确定性 2.84e-14（全 21 配置×s×3 重复）**、a4/a5 ~7e-06、a6 185.70。

**判决 `switch_localized`**：
- **T1 单层谱 s=2：L14 92.3 / L15 14.8 / L16 84.8 / L17 −6.4 / L18 24.1 —— 带内全部 5 层单独触发开关**（< 100 阈值；func 185.7、null0 77.3）。开关不是单一层特权，是 L14-L18 带内冗余属性。
- **T2 决定性负结果：multi_split（总剂量相同）s=2 sep = 172.1 不触发**（几乎不动 vs func 185.7），s=4（总剂量 4×）才达 46.6 —— 比单层 L17 s=1（21.5）还弱。**有效量是单层峰值浓度，不是总注入剂量**：同一总剂量集中单层有效、摊薄 5 层无效——强非线性剂量分配效应，否定"总剂量"解释。
- multi_full（5×总剂量）s=2 = −35.3 强反转（描述性，剂量超 saturate）。
- s=1 谱：L15 67.8 / L17 21.5 已触发，L16 176.8 / L14 157.5 未触发——层敏感度分化（L15/L17 最敏感）。

**D4 跨 session 复现修正**：本 session L16@2 sep = 84.81 与 2942 run2 **bit 级一致**；2942 探针观察的 85/33/14 三分支中 33/14 未再现——跨 session 不稳定性比 2942 担心的轻（登记描述性，不下"已消失"结论）。

**结论（机制链第九环：层定位环）**：开关是 L14-L18 带内冗余、单层浓度驱动的非线性阈值现象——任一带内层的强单点 U8 模式注入都可触发读出重编码，而同剂量摊薄则完全失效。与 2941（v3 单方向阻尼）对照：单方向弱、联合模式强、且要求浓度不要求总量——"regime 开关"的完整操作画像。2943 的线性外壳（收缩×截距）+ 本 Phase 的浓度阈值 = null 重编码的两面。

**文件+SHA256-8**：execution %s / result %s / npz %s / script %s。产物 `tests/glm5/result/rdc_query_construction_20260913/phase2944/switch_localization/`。runtime %s s。

**接续（2945 候选）**：A（主选）浓度阈值曲线——L15/L17 两敏感层 s 细扫 {0.5,0.75,1,1.25,1.5,2}，定位各层阈值 s_c 与 2943 gamma 的关系（一次前向）；B 承重带跨模型复现（glm4）；C 开关的剂量分配律系统化（2 层配置 (1,0) vs (0.5,0.5) vs (0.25,0.75)，检验浓度-位置交互）；D 2942 s=2 跨 session 不稳定源追踪（同 session 多进程对比）。
""" % (stamp, hashes.get('execution.json'),
       hashes.get('result.json'),
       hashes.get('switch_localization.npz'),
       sha8(script), res['runtime_s'])
    memo = open(MEMO, encoding='utf-8').read()
    if '## Phase 2944' not in memo:
        with open(MEMO, 'a', encoding='utf-8') as f:
            f.write(memo_block)

    # ---------- workspace log ----------
    wl = open(WLOG, encoding='utf-8').read()
    if 'Phase 2944' not in wl:
        wl += ('\n## Phase 2944 (2026-09-19)\n'
               '- 判决 switch_localized：L14-L18 全部 5 单层触发开关'
               '（s=2 sep 92/15/85/-6/24），multi_split 同总剂量不触'
               '发（172）——有效量=单层峰值浓度而非总剂量；开关为带内'
               '冗余、浓度驱动的非线性阈值。D4：L16@2 与 2942 bit 级'
               '一致（84.81）。\n'
               '- 锚 8/8（a1 第 18 次、a8 同 session 确定性 2.84e-14）。'
               'Ledger 83 条 / L14 connects 51 / ledger %s。\n'
               % (chk,))
        open(WLOG, 'w', encoding='utf-8').write(wl)

    # ---------- MEMORY.md ----------
    mem = open(WMEM, encoding='utf-8').read()
    if 'max=2944' not in mem:
        mem = mem.replace('max=2943，下一个 2944',
                          'max=2944，下一个 2945')
    with open(WMEM, 'w', encoding='utf-8') as f:
        f.write(mem)

    print('\n'.join(lines))
    print('ledger: n_meas=%d L14=%d ledger8=%s'
          % (n_meas, len(l14['connects']), chk))
    memo2 = open(MEMO, encoding='utf-8').read()
    i = memo2.find('## Phase 2944')
    print('memo 2944 title:', memo2[i:i + 75].split('\n')[0])
    print('memo 2945 absent:', '## Phase 2945' not in memo2)
    print('wlog 2944:', 'Phase 2944' in open(
        WLOG, encoding='utf-8').read())
    print('wmem max=2944:', 'max=2944' in open(
        WMEM, encoding='utf-8').read())
    print('OK closeout')


if __name__ == '__main__':
    main()
