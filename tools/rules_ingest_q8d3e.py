# -*- coding: utf-8 -*-
"""把 Q8-D3 / Q8-E 的三条新规则 upsert 进 data/rules/ledger.json（幂等：按 id 覆盖）。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'data' / 'rules' / 'ledger.json'
TS = datetime.now().isoformat(timespec='seconds')

E_D3Q = dict(phase='Q8-D3', script='tools/pat_q8d.py',
             data='results/2026-11-01/185433_pat_q8d3.json', model='Qwen3-4B')
E_D3G = dict(phase='Q8-D3', script='tools/pat_q8d.py',
             data='results/2026-11-01/193015_pat_q8d3.json', model='GLM-4-9B(8bit)')
E_EQ = dict(phase='Q8-E', script='tools/pat_q8e.py',
            data='results/2026-11-01/192347_pat_q8e.json', model='Qwen3-14B')
E_EV = dict(phase='Q8-E', script='tools/pat_q8e_verify.py',
            data='results/2026-11-01/192347_pat_q8e.json', model='Qwen3-14B')

RULES = [
    dict(
        id='R-MTH-30', domain='method', topic='噪声地板量的点值不可复现，判决必须用多种子均值加不等式',
        claim='无信号目标（如原始频率轴）的 held-out R^2 点值可以用单次运行报告并逐位复现。',
        status='refuted', confidence='high',
        criterion='当判据落在噪声地板上（真值≈0 或求解器受最小特征值方向支配）时：'
                  '①主脚本报 ≥3 个随机种子（折划分/初始化）的均值；'
                  '②复核器不做点复现断言，改判不等式（如 r2<0.10 且 gap≥0.30）；'
                  '③禁止把 cv_fast 与手写正规方程在同折上的差异当 bug——那是代码路径的数值细节。',
        measure=dict(symptom='同一折上 raw 轴 held-out R^2：cv_fast -0.10 vs 手写 -0.97',
                     diagnosis='因子分离确认差异来自求解器在最小特征值方向的数值细节，非代码错误',
                     fix='主脚本 3 种子均值（Q8-E raw 轴 0.023/-0.106），复核改不等式判决'),
        evidence=[dict(E_EQ, note='LOG_SCALE 判决由不等式承载，两模型 gap 0.804/0.858 远超阈'),
                  dict(E_EV, note='复核器 float32 点复现断言被判为类别错误后重写')],
        retest=dict(status='block', count=1,
                    note='任何以 R^2≈0 为预期值的对照轴/零假设轴都适用；点显著判据禁用于噪声地板量'),
        why='点值在这类量上不是数据性质而是求解器指纹；把点复现写进复核会把数值细节当回归 bug，'
            '或把 -0.10 vs -0.97 的差异误判为主脚本出错。',
        tags=['method', 'statistics', 'verify'], keywords=['噪声地板', '多种子', '不等式', 'R2', '零假设轴'],
    ),
    dict(
        id='R-MTH-31', domain='method', topic='特征/缓存键必须含数据身份（模型名、维度、折指纹）',
        claim='Gram 矩阵缓存键只含折指纹与样本数即可跨脚本/跨模型共享。',
        status='refuted', confidence='high',
        criterion='一切以 X 为输入派生的缓存（Gram、投影、统计量）键必须含 X 的完整身份：'
                  '模型名 + 维度 + 构造参数。跨模型共用 ns 的脚本，ns 必须带模型名前缀；'
                  '否则先注入一次跨模型缓存冲突（如 2560 维 G 塞给 5120 维 solve）再修。',
        measure=dict(symptom='诊断脚本两模型共用 ns，缓存中 2560 维 Gram 被喂给 5120 维 solve，维度报错',
                     fix='ns = f"{model_name}|{...}" 前缀；Q8-B 的 R-MTH-24（折指纹）是该规则在折维度的特例'),
        evidence=[dict(E_EQ, note='14B (d=5120) 与 4B (d=2560) 同池构造，缓存键补模型名后干净'),
                  dict(E_EV, note='独立复核器不复用缓存，天然免疫；主脚本修复后无冲突')],
        retest=dict(status='block', count=1,
                    note='R-MTH-24（缓存键缺折指纹）的同族扩展：身份缺失类缓存键一律视为已证伪模式'),
        why='缓存键缺身份维度时冲突是静默的或以误导性报错出现；维度恰好相容时会更隐蔽——'
            '错误的 Gram 值直接进 solve，产出看起来合理的错数。',
        tags=['method', 'caching'], keywords=['缓存键', 'Gram', '模型名', '跨模型', 'R-MTH-24'],
    ),
    dict(
        id='R-MET-40', domain='measurement', topic='复核器与主脚本必须共享唯一编码入口',
        claim='复核器可以自己实现一份输入编码（模板包装/位移拼接），只要语义等价。',
        status='refuted', confidence='high',
        criterion='输入编码入口必须唯一：复核器 import 主脚本的编码函数（或反之），'
                  '禁止在复核器里重写 wrap/apply_chat_template 路径。'
                  '任何位移/扰动 dE() 必须经过同一入口构造对比串。',
        measure=dict(symptom='Q8-D3 复核 dE() 在裸串上位移、基线在包装串上，恒定 -0.55 偏移，max|dE|=0.60',
                     fix='dE 改走 wrap() 后逐剂量偏差降到 0.004-0.012；chat 模板陷阱 '
                         '(<think> 强制 / 缺尾换行) 也只修一处'),
        evidence=[dict(E_D3Q, note='修复后 Qwen 复核 21 PASS / 0 FAIL'),
                  dict(E_D3G, note='同管线 GLM 复核通过（含 8-bit 饱和区翻转规则 p_M>=0.90）')],
        retest=dict(status='block', count=1,
                    note='双实现"语义等价"在模板层几乎不可保证——一处 enable_thinking 或尾换行差异就是系统性偏移'),
        why='编码差异不是噪声而是系统性偏移（-0.55 全局常数），会把所有判据推向 FAIL 并掩盖真实差异；'
            '唯一入口让模板陷阱只需修一次、复核自动继承。',
        tags=['measurement', 'verify', 'encoding'], keywords=['复核器', '编码入口', 'chat模板', '偏移', 'wrap'],
    ),
    dict(
        id='R-MTH-32', domain='method', topic='账本写入前必须有备份，新条目 ID 必须从账本现算最大号取',
        claim='新规则编号可以按记忆里的编号续排；覆盖式 upsert 前不需要备份。',
        status='refuted', confidence='high',
        criterion='①任何账本 JSON 写入前先落一份 .bak（cp 即可）；②新条目 ID 用代码从账本现算 '
                  'max(编号)+1，禁止凭记忆续排；③upsert 脚本遇到"按 id 覆盖"时必须先 diff 打印旧条目 '
                  'topic，topic 不同立即中止（那是撞号不是更新）。',
        measure=dict(symptom='Q8-D3 收尾时凭记忆编 R-MTH-28/R-MET-37，撞掉 Q9-A 线两条 confirmed 规则；'
                             '恢复时一次 json.dump(kwarg) 错误又把整个 ledger.json 截断为空',
                     fix='从 ledger.json.bak_q9a（116 条）恢复 + 重放 q9a ingest + 重编号 30/31/40；'
                         '零规则丢失，但多耗一整轮排查'),
        evidence=[dict(phase='Q8-D3', script='tools/rules_ingest_q8d3e.py',
                       data='data/rules/ledger.json.bak_q9a', model='-',
                       note='备份文件是唯一救命绳；HEAD 里的版本（104 条）不含当天未提交的 12 条')],
        retest=dict(status='block', count=1,
                    note='一切 JSON 账本（rules/ledger/atlas）都适用；幂等 upsert 不等于可安全覆盖'),
        why='撞号静默覆盖 confirmed 规则，且恢复路径上的任何写错误都可能把损失放大成整文件截断；'
            '备份 + 现算编号 + topic diff 三道闸成本几乎为零。',
        tags=['method', 'ledger', 'process'], keywords=['编号冲突', '备份', 'upsert', '截断', '账本'],
    ),
]


def main() -> int:
    doc = json.loads(LEDGER.read_text(encoding='utf-8'))
    by_id = {r.get('id'): i for i, r in enumerate(doc['rules'])}
    for r in RULES:
        r['updated_at'] = TS
        if r['id'] in by_id:
            old = doc['rules'][by_id[r['id']]]
            r.setdefault('created_at', old.get('created_at', TS))
            doc['rules'][by_id[r['id']]] = r
            print('updated %s' % r['id'])
        else:
            r.setdefault('created_at', TS)
            doc['rules'].append(r)
            by_id[r['id']] = len(doc['rules']) - 1
            print('appended %s' % r['id'])
    LEDGER.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                      encoding='utf-8', newline='')
    chk = json.loads(LEDGER.read_text(encoding='utf-8'))
    print('n_rules = %d' % len(chk['rules']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
