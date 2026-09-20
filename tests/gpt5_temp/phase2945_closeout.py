# -*- coding: utf-8 -*-
"""Phase 2945 seal + closeout."""
import hashlib
import json
import os
import time

OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913\phase2945'
       r'\threshold_curves')
SEAL_REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
               r'\phase2945_seal_report.txt')
LEDGER = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\2026-09-19.md')
WMEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\MEMORY.md')

MEAS_ID = 'meas_2945_threshold_curves'
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

    lines.append('phase 2945 seal @ %s'
                 % time.strftime('%Y-%m-%dT%H:%M:%S'))
    lines.append('verdict: %s' % verdict)
    lines.append('execution created: %s' % created)
    lines.append('runtime_s: %s' % res['runtime_s'])
    hashes = {}
    for fn in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, fn)
        if os.path.isfile(p):
            hashes[fn] = sha8(p)
            lines.append('  %-26s %s %d bytes'
                         % (fn, hashes[fn],
                            os.path.getsize(p)))
    script = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
              r'\phase2945_threshold_curves.py')
    script8 = sha8(script)
    lines.append('script sha8: %s' % script8)
    with open(SEAL_REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    # ---------- Ledger ----------
    led = json.load(open(LEDGER, encoding='utf-8'))
    if MEAS_ID not in [m['meas_id']
                       for m in led['measurements']]:
        led['measurements'].append({
            'meas_id': MEAS_ID,
            'type': 'forward_family_threshold_curve',
            'verdict': verdict,
            'source': 'phase2945/threshold_curves'})
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
    memo_block = """
## Phase 2945: 浓度阈值曲线与量级解耦判决 [@STAMP@]

**原理**：2944 判决开关为 L14-L18 带内冗余、单层浓度驱动（multi_split 同总剂量不触发）。开放问题：各层阈值浓度 s_c 在哪？阈值是否=传导达实际 null 位移量级的点（ratio≈0.86，2942 标定）？

**方法**：一次前向族。单层注入 L15/L16/L17（系数 1.0），s 细扫 {0.25,0.5,0.75,1.0,1.25,1.5,2.0}，K=3 同 session 重复取中位。锚 8/8：a1 2.17e-08（**第 19 次连续前向锚定**）、a8 2.84e-14（全 63 前向×3 重复）、a3 bit 级 0、a7 9.95e-14。

**判决 `threshold_curve_nonmonotone`**（run5 权威）：
- **T1 fail（L16 真实非单调）**：sep(s) L15/L17 完美单调陡降（spearman −1.0，最陡降 56.8/73.6）；**L16 spearman −0.8929 且 s=0.25→0.5 真实微升（191.0→195.4，同 session bit 确定非噪声）**——L16 是渐变缓坡型，与 L15/L17 的陡降开关型定性不同。
- **层敏感度分化定量**：s_c(L17)=0.656 < s_c(L15)=0.845 < s_c(L16)=1.843——最强/最钝差 2.8×。
- **T2 fail（阈值量级解耦，主发现）**：阈值处传导比 ratio_c = L15 0.324 / L16 0.457 / L17 0.313，全部远离 0.86（|Δ| 0.40-0.55 > 0.3）——**开关在传导仅 ~31-46% 实际 null 位移量级时就触发**。阈值是浓度域的（每层独立浓度阈值），不是位移量级域的统一阈值；"重现 null 位移"（ratio=1）在开关触发之后才达到。
- 层间 ratio 曲线形态分化：L15/L17 超线性爬升（0.17→0.58/0.39→0.44 跨越 s∈[0.75,1.25]），L16 近线性缓爬（0.05→0.23）——传导效率与敏感度同序（L17>L15>L16）。

**D5 壳带穿越**：三层阈值处 sep 恒 = 100（插值构造恒等式，不作证据，纪律 17）；但穿越点落在 2943 线性壳带上缘（壳带 77-106）——开关把读出压入"null 型"壳带后，后续行为由收缩×截距外壳接管。

**硬伤与勘误**：run1-4 四连败（键管理混乱：元组键 '%s' % key 格式化、写入/读取键元组-str 混用，3 处 KeyError + 1 处 TypeError）——run5 权威；教训：同一下标结构的写入/读取必须一次统一（ndict 键规范）。T1 阈值 −0.9 边界（L16 −0.8929）与真实微升并存，非边界 artifact。

**结论（机制链第十环：阈值曲线环）**：regime 开关的完整刻画 = 带内冗余（2944）+ 每层独立浓度阈值（s_c 0.66/0.85/1.84，敏感度 L17>L15>L16）+ 阈值与位移量级解耦（ratio_c 0.31-0.46 ≠ 0.86）+ 开关型/渐变型层分化（L15/L17 陡降 vs L16 缓坡微升）。**2942 的 s=2 跨 session 不稳定得到机制解释候选：L16 注入点正处于渐变缓坡段（sep 对 s 的局部斜率大且未饱和），而陡降型层在 s>1 后已进入饱和稳定区**。开关不是"传导到位即翻转"，而是独立的浓度门控。

**文件+SHA256-8**：execution @EXE@ / result @RES@ / npz @NPZ@ / script @SCR@。产物 `tests/glm5/result/rdc_query_construction_20260913/phase2945/threshold_curves/`。runtime @RT@ s。

**接续（2946 候选）**：A（主选）阈值浓度-位置交互——2 层配置 (L17,L16) 联合注入 (0.5,0.5)/(0.25,0.75)/(0.75,0.25)，检验浓度分配对联合阈值的作用（一次前向）；B 承重带跨模型复现（glm4）；C L16 渐变型 vs L15/L17 开关型的头级解剖（层内头分解传导，零前向+一次前向）；D 2942 跨 session 不稳定源：同 session 多进程 L16@2 重复（检验 run 级状态差异）。
""".replace('@STAMP@', stamp) \
       .replace('@EXE@', hashes.get('execution.json')) \
       .replace('@RES@', hashes.get('result.json')) \
       .replace('@NPZ@', hashes.get('threshold_curves.npz')) \
       .replace('@SCR@', script8) \
       .replace('@RT@', str(res['runtime_s']))
    memo = open(MEMO, encoding='utf-8').read()
    if '## Phase 2945' not in memo:
        with open(MEMO, 'a', encoding='utf-8') as f:
            f.write(memo_block)

    # ---------- workspace log ----------
    wl = open(WLOG, encoding='utf-8').read()
    if 'Phase 2945' not in wl:
        wl += ('\n## Phase 2945 (2026-09-19)\n'
               '- 判决 threshold_curve_nonmonotone：L15/L17 完美单调'
               '陡降（spearman -1.0），L16 真实非单调微升（191→195.4）'
               '+ 渐变型；s_c L17 0.656 < L15 0.845 < L16 1.843；阈值'
               '量级解耦（ratio_c 0.31-0.46 ≠ 0.86）——开关是浓度门控'
               '而非传导到位即翻转；L16 缓坡段 = 2942 跨 session 不稳'
               '定机制解释候选。\n'
               '- run1-4 键管理四连败（元组/str 键混用），run5 权威。'
               '锚 8/8（a1 第 19 次）。Ledger 84 条 / L14 connects '
               '52 / ledger %s。\n' % chk)
        open(WLOG, 'w', encoding='utf-8').write(wl)

    # ---------- MEMORY.md ----------
    mem = open(WMEM, encoding='utf-8').read()
    if 'max=2945' not in mem:
        mem = mem.replace('max=2944，下一个 2945',
                          'max=2945，下一个 2946')
    if '2945 教训' not in mem:
        mem = mem.replace('\n## 本机环境缺陷与对策',
            '\n18. **2945 延伸（字典键规范）**：同一复合键结构的写入'
            '与读取必须一次统一（本次元组键与 str 键混用致 4 连败）；'
            '科学上：开关阈值与传导量级解耦（ratio_c 0.31-0.46 ≠ '
            '0.86）——判据设计须区分"浓度域阈值"与"位移量级域阈值"'
            '两种机制假设。\n\n## 本机环境缺陷与对策')
    with open(WMEM, 'w', encoding='utf-8') as f:
        f.write(mem)

    print('\n'.join(lines))
    print('ledger: n_meas=%d L14=%d ledger8=%s'
          % (n_meas, len(l14['connects']), chk))
    memo2 = open(MEMO, encoding='utf-8').read()
    i = memo2.find('## Phase 2945')
    print('memo 2945 title:', memo2[i:i + 75].split('\n')[0])
    print('memo 2946 absent:', '## Phase 2946' not in memo2)
    print('wlog 2945:', 'Phase 2945' in open(
        WLOG, encoding='utf-8').read())
    print('wmem max=2945:', 'max=2945' in open(
        WMEM, encoding='utf-8').read())
    print('OK closeout')


if __name__ == '__main__':
    main()
