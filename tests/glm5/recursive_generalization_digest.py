"""Build an explicitly retrospective casebook from immutable existing outputs.

No inference, model training, re-thresholding of historical verdicts or formal promotion.
"""
from collections import Counter
from datetime import datetime
import html
import json
from pathlib import Path
import sys

import numpy as np

from recursive_generalization_review import ROOT, OUT, read, sha, write
from deploy.storage_policy import check_write
from shared.dist_protocol.mechanism_trace import unbits


def legacy_audit():
    base=ROOT/'research/interventions'
    result=read(base/'EXPL-v2-results.json')
    pools=read(base/'EXPL-v2/pools.json')
    causes=[]
    for p in sorted((base/'EXPL-v2').glob('causes_*.json')):
        causes.extend(read(p)['items'])
    rows=[]
    for cause in causes:
        if cause['cause']!='递归泛化':continue
        unit=next(u for u in result['units'] if u['code']==cause['code'])
        pool=pools[unit['code']]
        assert (unit['block'],unit['unit'])==(pool['block'],pool['unit'])
        rows.append(dict(**unit,pool=pool,interpretation='历史失败原因的描述性分类，不是训练中递归泛化的验证'))
    files=[base/'EXPL-v2-results.json',base/'EXPL-v2.prereg.json',base/'EXPL-v2/pools.json',
           *sorted((base/'EXPL-v2').glob('causes_*.json'))]
    write('historical-audit.json',dict(units=rows,n=len(rows),text_outcome=dict(Counter(u['text_outcome'] for u in rows)),
        retrieval=dict(Counter(u['retrieval'] for u in rows)),strict_pass=sum(u['explained_strict'] for u in rows),
        self_check=result['self_check'],self_check_target_median_relative_error=0.05,
        checks=dict(source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files}),
        limits=['每单元仅5个新正例、5个近反例、5个远反例；检索正例为普查极值21–32名。',
                '历史测量按float32重算SiLU(G)*V，不能等同于新版保存的实际BF16 down_proj输入。',
                '检索标注可见右侧10个token；后文不是当前状态的因果输入。',
                '原因分类来自隔离AI上下文，不是独立训练实验或分支干预。',
                '自检误差中位数超出0.05目标；原协议规定仅报告，历史判定带限制保留。']))
    return rows


def corpus_audit(inventory):
    rows=[]
    for folder in sorted({s['path'] for s in inventory['layers']}):
        p=Path(folder)
        docroot=p if (p/'documents.json').is_file() else ROOT/'tests/glm5/result/mlp_tail_goodfire_v1'
        d=read(docroot/'documents.json')
        tp=docroot/'tokenizer.json'
        if not tp.is_file():tp=ROOT/'models/hf/qwen3-4b/tokenizer.json'
        rows.append(dict(path=folder,document_file=str(docroot/'documents.json'),document_sha256=sha(docroot/'documents.json'),
            n_documents=len(d['docs']),positions=d['positions'],splits=dict(Counter(v['split'] for v in d['docs'])),
            tokenizer=str(tp),tokenizer_sha256=sha(tp),missing_raw_blocks=[s['block'] for s in inventory['layers'] if s['path']==folder and not s['raw_A']]))
    # All archived tokenizers and the model tokenizers used for fallback have identical bytes.
    assert len({r['tokenizer_sha256'] for r in rows})==1
    write('corpus-audit.json',dict(sources=rows,warning='不同执行版本、文档数与位置网格分别保留，不混合幅值或估计总体比例。'))


def controls(units):
    from tokenizers import Tokenizer
    records=[]
    wanted={('qwen14',26,13099),('qwen14',19,64),('qwen14',38,14670),('qwen4',28,7912)}
    for u in units:
        key=(u['model'],u['block'],u['unit'])
        if key not in wanted:continue
        p=Path(u['path']);d=read(p/'documents.json');pos=d['positions'];docs=d['docs']
        tp=p/'tokenizer.json'
        if not tp.is_file():tp=ROOT/'models/hf/qwen3-4b/tokenizer.json'
        tok=Tokenizer.from_file(str(tp))
        arrays={f:unbits(np.load(p/f"B{u['block']}-{f}.npy",mmap_mode='r')[:,:,u['unit']]).reshape(-1) for f in ['G','V','A']}
        assert all(len(a)==len(docs)*len(pos) and np.isfinite(a).all() for a in arrays.values())
        tokenids=np.array([doc['tokens'][j] for doc in docs for j in pos])
        ranks={f:np.sort(a) for f,a in arrays.items()}
        unit_examples=[]
        for e in u['examples']:
            if e['field']!='G' or e['pole']!='positive':continue
            idx=e['document']*len(pos)+pos.index(e['position'])
            assert all(float(arrays[f][idx])==e[f] for f in arrays)
            others=np.flatnonzero((tokenids==tokenids[idx]) & (np.arange(len(tokenids))//len(pos)!=e['document']))
            same_source=np.array([i for i in others if docs[i//len(pos)]['source']==e['source']],dtype=int)
            same_source_preferred=bool(len(same_source))
            if same_source_preferred:others=same_source
            selected=[]
            if len(others):
                for oi in dict.fromkeys([int(others[np.argmin(arrays['G'][others])]),int(others[np.argmax(arrays['G'][others])])]):
                    di,slot=divmod(oi,len(pos));j=pos[slot];doc=docs[di]
                    selected.append(dict(document=di,position=j,source=doc['source'],doc_id=doc['doc_id'],current=tok.decode([tokenids[oi]]),
                        **{f:float(a[oi]) for f,a in arrays.items()},prefix=tok.decode(doc['tokens'][:j+1]),
                        window=tok.decode(doc['tokens'][max(0,j-120):j+1])))
            unit_examples.append(dict(target=e,token_id=int(tokenids[idx]),
                empirical_percentile={f:float(np.searchsorted(ranks[f],e[f],side='right')/len(tokenids)) for f in arrays},
                same_token_other_document_n=len(others),source_matched=same_source_preferred,comparisons=selected))
        records.append(dict(model=u['model'],block=u['block'],unit=u['unit'],n_positions=len(tokenids),
            examples=unit_examples,selection='事后选G正极；同token、尽量同来源的其他文档中G最小/最大两个位置，仅作反例检索。'))
        print(f'controls {key} positions={len(tokenids)} examples={len(unit_examples)}',flush=True)
    write('same-token-controls.json',dict(units=records,limits='非随机、非新盲测、未配平所有位置与语义，不作显著性或因果推断。'))


CASE_NOTES=[
    ('qwen4',13,9571,'优先候选：达到或越过界限',
     '经济水平超过、处理器性能追平、地下水到达极限、氧气将耗尽，可按“量相对参照界限”的关系组织。分支之间的箭头只是假设，不是已观测到的训练先后。',
     '历史M058检索6/12，误报0/18；新正例1/5，文本tie。eyes、山等高响应仍未解释；“coming to light”也未必是阈值。'),
    ('qwen4',16,9414,'优先候选：代价与支付责任',
     '付款/收费、信用账单、广告按点击计费、担保本息、保险赔付、医疗费用等共享“获得资源或承担责任所需付出”的关系。',
     '历史N097检索9/12，误报0/18；新正例1/5，文本tie。税法/经济来源与常见搭配可能解释结果；利润创造等旁支仍需单独预测。'),
    ('qwen4',30,9517,'优先候选：地域与语言变体标记的桥接',
     'G正极有香港文化与区域关系，A/G正极还有zh-hans/zh-hant/zh等维基变体标记，涉及法律、显示器、驱动程序等不同主题。',
     '旧GF三个对照方向一致，但样本仅3且控制不均衡；S8读写一致性未过。大幅整块输入steering不能算单单元因果。现有长文极值仅保存索引，原G/V/A已缺失；不得填造幅值。'),
    ('qwen4',31,2619,'优先候选：以测量或试验作判定',
     'A/G正极涵盖听力检测、燃油评级、EMI基线测量、考试评分与软件Alpha/Beta测试。可检验“测量结果相对标准”的共同功能。',
     'test/testing及其译词是强替代解释，软件分支多个位置集中同一篇文档；原全场已缺失，暂无新语言家族或分支干预。'),
    ('qwen14',26,13099,'较弱候选：形式对象及其关系/操作',
     'G正极涵盖微分形式、环与闭包、射影变换、向量空间，以及WinRT实例、软件设计模式的类/对象、Haskell语法。',
     '数学/技术说明的语体与词汇仍未排除。软件设计模式来自同一文档；G较强不保证A同号：同篇类/对象两个位置A为-1.2265625与+1.53125。尚无冻结语义检验。'),
    ('qwen14',19,64,'较弱候选：数学量的变换与计算规则',
     'G正极涵盖坐标变换、AHP优化、滚动哈希、高阶矩阵工具、Thomas算法、Lisp数值误差与排序算法。',
     '月食过程、分位成绩区间等不都能由“算法”说明；一般技术主题是更简单解释。G/V/A写出关系不统一，未证实递归分支或推理功能。'),
    ('qwen4',15,3396,'历史边界案例：不同领域的指标量',
     '心率、盐浓度、海拔、天顶角、班级人数等可被整理为“可测量属性及其水平”。',
     'M005检索4/12未过，新正例1/5，文本tie。共同标签过宽，不能因此判成递归泛化成立。'),
    ('qwen4',28,7912,'普通主题泛化对照：课程与教育安排',
     '在线课程、教材、学校申请、考试分级与课程要求存在共同教育主题。',
     '更接近类别泛化；留出前三个高响应来自同一篇课程文档，不能当成三个独立分支。'),
    ('qwen14',38,14670,'普通主题/未解释混合对照：临床材料',
     '维生素缺乏、听力损失、血管炎、肺泡疾病与钙化防御的临床描述出现高G。',
     '这首先是跨疾病主题泛化；周公摄政与电视剧清单也有高G。不能只挑医疗样本便宣布一条递归树。'),
    ('qwen14',34,7426,'词汇与主题对照：日本相关材料',
     '日本地方产业、历史、政治、天皇等在A正极重复出现。',
     '国家词、日本实体与语料来源可直接解释大量响应；留出鸟类拉丁分类表仍未解释。不能仅凭一个国家下的多个主题证明递归。'),
]


def build():
    check_write(OUT,100*2**20)
    inv=read(OUT/'inventory.json');historical=legacy_audit();corpus_audit(inv)
    paths=list(OUT.glob('contexts-*.json'));assert len(paths)==1
    raw=read(paths[0]);units=raw['units'];controls(units)
    supplemental=[ROOT/'research/neurons/goodfire/GF-L30-U9517-local.json',
        ROOT/'research/neurons/goodfire/GF-L30-U80-local.json',
        ROOT/'tests/glm5/result/b35_recursive_v2/analysis.json',
        ROOT/'tests/glm5/result/b35_recursive_v2/prereg.json']
    write('supplemental-sources.json',dict(files={str(p.relative_to(ROOT)):sha(p) for p in supplemental},
        gf9517=read(supplemental[0])['verdict'],b35_eligible_units=read(supplemental[2])['eligible_units'],
        limits='B35的2001个单元复现短链/长链局部贡献方向；这与训练时响应域的分支形成是不同命题。'))
    cards=[]
    for model,block,unit,title,pattern,limits in CASE_NOTES:
        h=next((x for x in historical if model=='qwen4' and (x['block'],x['unit'])==(block,unit)),None)
        n=next((x for x in units if (x['model'],x['block'],x['unit'])==(model,block,unit)),None)
        cards.append(dict(model=model,block=block,unit=unit,title=title,pattern=pattern,limits=limits,
            historical_code=h['code'] if h else None,run_id=n['run_id'] if n else 'EXPL-v2',
            snapshot=n['snapshot'] if n else '6be1034f0b55a6df-bf16 (historical abbreviated identity)',
            execution=n['execution'] if n else 'historical EXPL-v2',status='exploratory_candidate_or_control',formal_promotion=False))
    write('case-cards.json',dict(cards=cards,source_context_sha256=sha(paths[0]),
        source_identity='canonical and legacy numerical versions retained separately',
        confirmed_recursive_generalization_units=0,confirmation_scope='没有当前材料能够证明训练时分支形成；0不是没有这种结构的证据。'))
    esc=html.escape
    sections=[]
    for card in cards:
        key=(card['model'],card['block'],card['unit'])
        u=next((x for x in units if (x['model'],x['block'],x['unit'])==key),None)
        hist=next((x for x in historical if x['code']==card['historical_code']),None)
        examples=[]
        if hist:
            for e in hist['pool']['rows']:
                if e['kind']=='pos':
                    # Keep original text but visually identify the future suffix; labels used it historically.
                    original=e['text'];prefix,sep,future=original.partition('】')
                    examples.append(f'<details><summary>历史正极池槽位 {e["slot"]} · {esc(e["source"])}</summary><p>{esc(prefix+sep)}</p><p class="muted">之后文本（非当前状态的因果输入）：{esc(future)}</p></details>')
        if u:
            for e in u['examples']:
                values=' / '.join(f'{f}={e[f]}' for f in ['G','V','A'])
                examples.append(f'<details><summary>{e["field"]} {e["pole"]} · {e["split"]} · doc {e["document"]} token {e["position"]} · {values}</summary><p>{esc(e["window"])}</p><details><summary>全部可见前缀</summary><pre>{esc(e["prefix"])}</pre></details><p class="muted">doc_id={esc(str(e["doc_id"]))}; source={esc(e["source"])}</p></details>')
        sections.append(f'<section><h2>{card["model"]} B{card["block"]}/U{card["unit"]} — {esc(card["title"])}</h2><p>{esc(card["pattern"])}</p><p class="limit">{esc(card["limits"])}</p><p class="muted">run={esc(card["run_id"])}; snapshot={esc(card["snapshot"])}</p><details><summary>原上下文、两极与反例 ({len(examples)} 条；不是独立样本数)</summary>{"".join(examples)}</details></section>')
    table=''.join(f'<tr><td>{h["code"]}</td><td>B{h["block"]}/U{h["unit"]}</td><td>{esc(h["explanation"])}</td><td>{h["text_outcome"]}</td><td>{h["retrieval"]}</td><td>{h["recall"]:.3f}</td><td>{h["fpr"]:.3f}</td></tr>' for h in historical)
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>递归泛化候选复核</title>
<style>body{{max-width:1080px;margin:36px auto;padding:0 20px;background:#f4f5f7;color:#182233;font:16px/1.75 system-ui}}section{{background:white;border:1px solid #d8dee6;border-radius:12px;padding:20px;margin:20px 0}}h1{{font-size:30px}}h2{{font-size:21px}}.limit{{border-left:4px solid #ba752e;padding:0 12px}}.muted{{color:#5a6574;font-size:14px;overflow-wrap:anywhere}}pre{{white-space:pre-wrap;font:inherit}}details{{margin:10px 0;padding:8px;border:1px solid #dde3ea}}summary{{cursor:pointer}}table{{border-collapse:collapse;font-size:14px}}td,th{{border:1px solid #d5dce4;padding:8px;text-align:left}}.scroll{{overflow:auto}}a{{color:#1467a3}}</style>
<h1>递归泛化：已有神经元的候选与反例</h1><p>{datetime.now().astimezone().isoformat()} · 静态归档的探索性复核 · 未运行GPU、未发布、未提升正式证据等级</p>
<p><strong>可以形成研究思路，尚不能得出已验证的训练规律。</strong> 当前观察支持“同一单元的响应可按相关分支组织”的候选；它没有显示分支实际怎样在训练中逐级长出。</p>
<p>40个已归档模型层索引覆盖4B的B18–35共175,104个单元、14B的B18–39共382,976个单元；这里没有逐个完成语义判断。按关键词检索后深入提取18个单元的G/V/A两极，另审核历史EXPL-v2的10个标签。历史13/15/16层候选在上述40层索引之外。案例选择有偏，不能据此估计全网比例。</p>
<p>旧4B尾段使用2,048篇×16个保存位置；新中段与14B使用1,024篇×8个位置。旧4B尾段只有极值索引和统计保留，标为None的幅值表示缺失；各模型、版本、语料与位置网格不混合。confirm/holdout是历史分片名称，本次解释已经见到这些材料，不能称为新盲测。</p>
<p><a href="https://already-clean.goodfire.com/">Goodfire原文</a>把递归泛化作为训练时沿相关方向长出分支的工作假设。普通主题泛化、跨深度重复计算、SiLU双负区间与这个发展性主张分别记录。</p>
<section><h2>如何理解这些候选</h2><p>已知计算式：G=gᵀz，V=uᵀz，A=SiLU(G)V，单元写入=A·d。G极相似并不意味着A同号或下游作用相同。以14B B26/U13099为例，数学与软件材料都可有较高G，但V决定了写入的方向和大小。</p>
<p>同token反例检索也表明不能只看当前词：该单元在微分形式文档的“ of”处G=2.140625，同来源另一文档的“ of”处G=-2.875。但这是事后挑选的极值对照，只排除了“此例完全由当前token身份决定”的简单解释，未排除前缀词汇、语体、位置或主题。</p>
<p>已有B35短链到长链实验中的2,001个方向复现单元，检验的是计算在更深步骤能否复用；它没有显示训练时分支如何形成。两类研究可以互补，不能交换名字。</p>
<p>候选分支图的节点应是可定义的输入家族，边是假设的词形、语义、关系或操作联系。下一步先冻结这些联系，预测未见家族，再控制词身份、格式、位置和来源，检验原单元/单元组作用。只有训练检查点或受控训练才可回答“如何长成”。</p></section>
{''.join(sections)}<section><h2>历史10个“递归泛化”标签逐项复核</h2><p>全部文本tie，6/10检索通过，严格联合通过0/10。它们是失败原因分类；检索通过不等于分支机制成立。测量自检20位置相对误差中位数0.08883，超过0.05目标，最大0.99376。旧协议允许只报告，这一限制不能省略。</p><div class="scroll"><table><tr><th>编码</th><th>单元</th><th>原解释</th><th>新文本</th><th>检索</th><th>召回</th><th>误报</th></tr>{table}</table></div></section>
<section><h2>下一阶段：分支发现—预测—作用—形成</h2><p>先对阈值、代价、测量判定、地域/语言标记、形式对象五组候选，冻结共同核心和竞争解释。新词、新句式、新领域分别留出，用新生成的家族做行为与读出可见性校准，规模按文档家族功效确定。全位置采集候选单元G/V/A、弱响应和反例，同时保存相关全层场。</p><p>在相同材料上比较词身份、n-gram/上下文基线、分支模型和适用字典方法；对通过预测的候选做单元及组干预、匹配负控制、恢复与副作用评估。最后再用训练快照或独立的小模型受控训练验证分支出现的先后。4B与14B的对应关系靠功能与响应重新对齐，不按编号对应。</p></section>
<p>复算文件：<a href="case-cards.json">候选卡</a> · <a href="historical-audit.json">历史审计</a> · <a href="same-token-controls.json">同token反例检索</a> · <a href="corpus-audit.json">语料身份</a> · <a href="inventory.json">全层索引</a></p></html>'''
    target=check_write(OUT/'report.html');target.write_text(doc,encoding='utf-8')
    # Verify meaningful boundary properties of the delivered digest.
    assert len(historical)==10 and all(h['text_outcome']=='tie' for h in historical)
    assert sum(h['retrieval']=='通过' for h in historical)==6
    assert sum(h['explained_strict'] for h in historical)==0
    assert len(cards)==10 and len(units)==18
    for link in ['case-cards.json','historical-audit.json','same-token-controls.json','corpus-audit.json','inventory.json']:
        assert (OUT/link).is_file()
    write('digest-audit.json',dict(checks='passed',historical_rows=10,case_cards=10,extracted_units=18,
        report_sha256=sha(target),script_sha256=sha(__file__),source_extraction_sha256=sha(ROOT/'tests/glm5/recursive_generalization_review.py'),
        evidence_status='exploratory_no_promotion'))
    print('digest complete; 10 case cards; 18 extracted units; 10 historical labels; audit passed',flush=True)


if __name__=='__main__':build()
