"""Build the client Goodfire catalog, then the protected historical index after pilot acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.storage_policy import check_write
from goodfire_record_cards import CARDS, GROUPS

OUT = ROOT / 'shared/dist_protocol/test_records'
MEMO = ROOT / 'research/glm5/docs/AGI_GLM5_MEMO.md'
RESULT = ROOT / 'tests/glm5/result'
AUDIT = RESULT / 'test_records_20261119'
HEAD = re.compile(r'^#{1,3}[ \t]+(?:\*\*)?Phase[ \t]*(\d+(?:\.\d+)?)[ \t]*[:：.、—–-][ \t]*(.+?)\s*$', re.I | re.M)
QWEN = 'Qwen3-14B'
MECHANISMS = {
    'update':['分析推理 / 授权更新','知识绑定 / 状态保持'],
    'route':['知识绑定 / 多记录选源','分析推理 / 条件查表'],
    'origin':['知识绑定 / 内容来源','语法 / 生成格式'],
    'position':['语言风格 / 格式起始','语法 / 上下文位置'],
    'fusion':['语言 / 中英文选择','语言风格 / 论文与代码','分析推理 / 事实更新','语法 / 括号'],
    'native':['条件响应 / 原生单元与状态读写'],
}
RELATIONS = {
    '2959': ['2957-v3'], '2961': ['2959'], '2964': ['2962'], '2967': ['2965'],
    '2968': ['2967'], '2969': ['2967','2968'], '2970': ['2964'], '2971': ['2969','2981'],
    '2972': ['2970'], '2973': ['2972'], '2974': ['2973'], '2975': ['2974'],
    '2976': ['2975','2978-v2','2979'], '2978-v1': ['2976'], '2978-v2': ['2978-v1','2979'],
    '2979': ['2976','2978-v2'], '2980': ['2978-v2','2981'], '2981': ['2980','2971'],
    '2957-v2': ['2957-v1'], '2957-v3': ['2957-v2'],
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, obj):
    data = (json.dumps(obj, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
    if path.exists() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_bytes(data)
    tmp.replace(path)


def sections(path):
    text = path.read_text(encoding='utf-8-sig')
    headings = list(HEAD.finditer(text))
    for idx, m in enumerate(headings):
        body = text[m.end(): headings[idx + 1].start() if idx + 1 < len(headings) else None].strip()
        yield m[1], re.sub(r'\s*\[[\d-]{10}[^\]]*\]\s*$', '', m[2]).strip(), body, m[2]


def source(path):
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path.read_bytes())}


def memo_id(author, phase, title):
    fingerprint = sha((author+'\n'+phase+'\n'+title).encode())
    return 'memo-'+author+'-'+phase+'-'+fingerprint[:12]


def goodfire():
    phases = {n: (title, body) for n, title, body, _ in sections(MEMO)}
    claude = {n:title for n,title,_,_ in sections(ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md')}
    release = ROOT/'release/deployment-20261118-goodfire/RELEASE_MANIFEST.json'
    public_reports = {r['run_id']:r for r in read(release)['studies']}
    records = []
    for key, group, kind, status, evidence, goal, principle, outcome, counter, gate in CARDS:
        n = key.split('-')[0]
        artifact = RESULT / evidence
        folder = artifact.parent
        run = None
        for filename in ['design.json', 'plan.json', 'manifest.json']:
            if (folder / filename).is_file():
                data = read(folder / filename)
                run = data.get('run') or data.get('run_id')
                if isinstance(run, str):
                    break
        records.append(dict(id='gf-'+key, phase=n, group=group, question=GROUPS[group], title=phases[n][0],
                            type=kind, status=status, model=QWEN, objective=goal, principle=principle,
                            result=outcome, counter=counter, gate=gate, related=['gf-'+k for k in RELATIONS.get(key, [])],
                            runs=[{'id': run if isinstance(run, str) else folder.name,
                                   'identity': '运行 ID' if isinstance(run, str) else '产物目录标识',
                                   'activity': '已有数据复算' if n in ('2968','2980') else '本地模型实测',
                                   'source': source(artifact)}],
                            memo=MEMO.relative_to(ROOT).as_posix(), phase_sha256=sha(phases[n][1].encode())))
        if run in public_reports:
            records[-1]['public_report'] = public_reports[run]['report_sha256']
        rec = records[-1]
        protocol_name = {'2965':'prereg.json','2967':'natural-plan.json','2969':'prereg.json',
                         '2971':'causal-freeze.json'}.get(n)
        protocol = next((folder/f for f in [protocol_name,'plan.json','design.json'] if f and (folder/f).is_file()),None)
        if protocol:
            rec['protocol']=source(protocol)
            frozen=read(protocol)
            snap=frozen.get('snapshot') or frozen.get('model') or frozen.get('provenance',{}).get('model_revision')
            if isinstance(snap,dict):
                snap=snap.get('snapshot')
            if isinstance(snap,str) and re.fullmatch(r'[0-9a-f]{64}',snap):
                rec['snapshot']=snap
            rec['material']=[{'split':s,'families':v.get('families'),'sha256':v['sha']}
                             for s,v in (frozen.get('material') or {}).items()
                             if isinstance(v,dict) and isinstance(v.get('sha'),str)] if isinstance(frozen.get('material'),dict) else []
    # These were reviews or release handoffs. A row of a historical review is not a new run.
    for i,line in enumerate(l for l in phases['2953'][1].splitlines() if l.startswith(('| Claude ', '| GLM5 '))):
        period, design, result = [s.strip() for s in line.strip('|').split('|')]
        older = []
        for author, part in re.findall(r'(Claude|GLM5) ([\d、–\-]+)',period):
            for span in part.split('、'):
                ends = re.split(r'[–-]', span)
                for num in range(int(ends[0]),int(ends[-1])+1):
                    n = str(num)
                    old_title = claude[n] if author=='Claude' else phases[n][0]
                    older.append({'id':memo_id(author.lower(),n,old_title),'label':f'{author} {n}', 'title':old_title})
        sentences = re.split(r'(?<=[。])',result)
        records.append(dict(id=f'gf-lineage-{i+1}', phase='2953', group='native', question=GROUPS['native'],
                            title=period, type='历史综述', status='复用已有采集', model='多模型 / 见适用范围',
                            objective='追溯原生单元到状态读写的历史证据。', principle=design,
                            result=result, counter=next((x for x in sentences if re.search('未|失败|缺|降低|限制|不',x)), ''),
                            gate='本轮只汇总历史证据。各次检验可按下列 Phase 分别查看。', earlier=older,
                            memo=MEMO.relative_to(ROOT).as_posix(), phase_sha256=sha(phases['2953'][1].encode()),
                            runs=[], related=['gf-2962']))
    notes = {
        '2954': ('native','既有实验','双留出检验提示事实、实发状态、竞争槽与答码。','kv_chain_20261117_v1'),
        '2955': ('native','既有实验','固定实发 State，改变同长插入内容，分开测试 H18 寻址和真实改码。','kv_receipt_20261117_v1'),
        '2962': ('native','路线复核','区分反向传播的训练作用和固定权重下的推理，制定可证伪测试。',None),
        '2966': ('route','路线复核','评估现有现象是否需要新的数学结构。',None),
        '2977': ('fusion','发布交接','汇总四范式研究及其既有发布记录。',None),
    }
    for n,(group,kind,goal,directory) in notes.items():
        title, body = phases[n]
        runs = []
        if directory:
            path = RESULT / directory / 'report.md'
            plan = read(path.parent/'plan.json')
            runs = [dict(id=plan.get('run') or plan['run_id'], identity='运行 ID', activity='本地模型实测', source=source(path))]
        records.append(dict(id='gf-'+n, phase=n, group=group, question=GROUPS[group], title=title,
                            type=kind, status='既有实测' if runs else '阶段综述', model=QWEN if runs else '多模型 / 见适用范围',
                            objective=goal,
                            runs=runs,related=[],memo=MEMO.relative_to(ROOT).as_posix(),phase_sha256=sha(body.encode())))
    extra = {
        '2954': ('交叉中性段、长度、竞争槽和码本，分别预测读取和真实答码。',
                 '新词既有句式中，读取预测的多数类基线达 89.94%。等幅寻址阻断与随机头分别改变 26 和 11 条首选。',
                 '原新句式面板的 117 条失败首个分歧都在 State。','原新句式面板未进入内部预测和干预。'),
        '2955': ('同提示同实发状态，交叉插入文字和长度，在新词及新句式留出观察谁被读取、是否改码。',
                 '新词 0/252 个插入视图改码，新词加新句式为 32/576。','32 次改码都发生在预设 State→Receipt 切换以外。',
                 '不能用这条寻址切换解释真实改码。'),
        '2962': ('复核自由生成、寻址和等幅干预各自能检验的命题。','形成知识绑定、推理状态更新、语法读出的可证伪设计。',
                 '两个模型同编号单元没有功能对齐证据。','研究设计；本轮没有新增前向或因果边。'),
        '2966': ('检查可计算对象、替代解释和预测增益。','提出可以用受控任务比较的因果抽象与数学路线。',
                 '多种内部实现可以产生相同输出。','理论讨论；新的数学定律有待独立检验。'),
        '2977': ('复核 2970、2972–2976 的六项研究及发布清单。','保留 1011 的条件性改善和 1111 的失败条件，完成增量交付。',
                 '1111 有 125/128 家族的二阶预测变差。','发布交接没有增加样本或提升证据等级。'),
    }
    for row in records:
        if row['phase'] in extra:
            row['principle'],row['result'],row['counter'],row['gate'] = extra[row['phase']]
        row['mechanisms']=MECHANISMS[row['group']]
        row['publication']='成果库已有同运行报告（2026-11-18 读回）' if row.get('public_report') else '本地来源索引'
        if row['phase'] in {'2967','2968','2969','2971','2981'}:
            row['objects']=['B26/H36']
        if row['phase']=='2972':
            row['objects']=['B30/H25','B30/H18','MLP G/U']
    subtests = {
        'gf-2971':[('四记录选源预测','已执行，有留出支持'),('原生头部干预','已执行，有局部注意力效应'),('最终答码首选','已测，未切换')],
        'gf-2981':[('完整自由生成','已执行，全部到 EOS'),('整组抄档案的正向行为门','已执行，方向相反'),('B26/H36 等幅阻断','未执行'),('原生恢复','未执行')],
        'gf-2976':[('请求后移联合阻断','范围内通过'),('档案后移等幅对照','未通过'),('B32 G/U 恢复优势','未通过')],
        'gf-2979':[('跨句式预测','未超过文本基线'),('原生阻断与恢复','已执行，供体覆盖不足'),('后续自然续写','未执行')],
    }
    for row in records:
        if row['id'] in subtests:
            row['stages']=subtests[row['id']]
    records.sort(key=lambda r:(int(r['phase']),r['id']),reverse=True)
    assert len({r['id'] for r in records}) == len(records)
    assert set(x for r in records for x in r['related']) <= {r['id'] for r in records}
    return {'schema':'research_test_records.v1','groups':GROUPS,'records':records}


def historical():
    rows = {}
    # Explicitly index sources. Memo entries can include proposals, technical work and reanalysis.
    for author in ('glm5','claude','gpt5','gemini','deepseek'):
        docs = ROOT / 'research' / author / 'docs'
        for path in sorted(docs.glob('AGI_'+author.upper()+'_MEMO*.md')):
            for phase,title,body,header in sections(path):
                if not title:
                    continue
                key=memo_id(author,phase,title)
                if key in rows:
                    other={'path':path.relative_to(ROOT).as_posix(),'sha256':sha(body.encode())}
                    if other['sha256']!=rows[key]['phase_sha256']:
                        rows[key].setdefault('source_versions',[]).append(other)
                    continue
                def excerpt(terms):
                    lines = [x.strip('- *') for x in body.splitlines() if 22 < len(x) < 1200
                             and not x.startswith(('#','|','```')) and not re.search(r'https?://|[A-Z]:\\|token=|Bearer |password|secret|凭据|令牌',x,re.I)]
                    return next((x[:400] for x in lines if re.search(terms,x)), '')
                kind = ('工程与交付索引' if re.search('客户端|安装包|运维|清理|部署|界面|升级包',title)
                        else '方案与复盘索引' if re.search('方案|综述|复盘|规划|路线审查',title) else '备忘录索引')
                rows[key] = dict(id=key,title=title,phase=phase,group=author,type=kind,status='待逐项复核',
                                 objective=title, principle=excerpt('方法|设计|对照|原理'),
                                 result=excerpt('实测|实际结果|实测结果|结果显示|留出结果'),
                                 counter=excerpt('反例|失败样本'), gate=excerpt('未过门|未达|失败原因'),
                                 memo=path.relative_to(ROOT).as_posix(),phase_sha256=sha(body.encode()),runs=[])
    registry = ROOT / 'ai2050_research_os'
    definitions = registry/'registry/tests.json'
    for design in read(definitions):
        key=design['id']
        rows[key] = dict(id=key,title=design['name'],type='测试设计',group='Research OS',status=design['status'],
                         objective=design['objective'],principle='对照项：'+'、'.join(design.get('controls',[])),
                         result='',counter='',gate=design['pass_rule']+' '+design['fail_rule'],runs=[],
                         evidence=source(definitions))
    contracts = read(registry/'registry/contracts.json')
    runs = read(registry/'registry/runs.json')
    for item in contracts:
        path = registry/item['path']
        design = read(path)
        related = [r for r in runs if r.get('contract_id')==item['id']]
        rows[item['id']] = dict(id=item['id'],title=design['title'],group='Research OS',
            objective=design.get('question',design['title']), phase=str(design.get('phase','')),
            type='冻结实验合同', status=item['status'], principle='；'.join(str(p.get('observable','')) for p in design.get('hypothesis_predictions',[])),
            result=' / '.join(str(r.get('verdict') or r['status']) for r in related),counter='',gate='',
            evidence=source(path), runs=[dict(id=r['id'],activity=r['status'],model=r.get('model'),result=r.get('verdict'),note=r.get('notes')) for r in related])
    for path in sorted((ROOT/'results').glob('????-??-??/*.json')):
        doc = read(path)
        if not isinstance(doc,dict) or doc.get('schema')!='result.v1':
            continue
        prov=doc.get('provenance') or {}
        payload=doc.get('payload') or {}
        if not isinstance(payload,dict):
            payload={}
        key='result-'+sha(path.read_bytes())
        if key in rows:
            rows[key].setdefault('other_sources',[]).append(path.relative_to(ROOT).as_posix())
            continue
        rows[key]=dict(id=key,title=doc['name'],type='结果产物',status='待逐项复核',group='result.v1',
                       phase=str(payload.get('phase') or ''), objective=doc['name'],principle=prov.get('tool',''),
                       model=Path(str(prov.get('model_path') or '')).name, result=str(payload.get('verdict',''))[:260],
                       counter='',gate='',evidence=source(path),runs=[])
    known = {r['source']['sha256'] for row in read(OUT/'pilot.json')['records'] for r in row['runs']}
    for directory in sorted(p for p in RESULT.iterdir() if p.is_dir()):
        report=next((directory/f for f in ('report.md','report.zh-CN.md','report.json') if (directory/f).is_file()),None)
        manifest=next((directory/f for f in ('manifest.json','plan.json','prereg.json') if (directory/f).is_file()),None)
        if not report or not manifest:
            continue
        evidence=source(report)
        if evidence['sha256'] in known:
            continue
        key='report-'+evidence['sha256']
        if key in rows:
            rows[key].setdefault('other_sources',[]).append(evidence['path'])
            continue
        meta=read(manifest)
        if not isinstance(meta,dict):
            continue
        title=directory.name
        if report.suffix=='.md':
            with report.open(encoding='utf-8-sig') as f:
                title=f.readline().strip('# \r\n') or title
        rid=meta.get('run_id') or meta.get('run')
        # A report and a frozen design share this directory. Keep only explicit identities.
        runs=[{'id':rid,'activity':'原记录所列运行'}] if isinstance(rid,str) else []
        rows[key]=dict(id=key,title=title,group='本地报告',type='原始报告索引',status='待逐项复核',
                       objective=title,principle='',result='',counter='',gate='',evidence=evidence,
                       protocol=source(manifest),runs=runs)
    return {'schema':'research_test_records.v1','records':list(rows.values())}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['pilot','history'])
    args=parser.parse_args()
    check_write(OUT,80*2**20,operation='测试记录回填')
    if args.stage=='pilot':
        doc=goodfire()
    else:
        receipt=read(AUDIT/'pilot-acceptance.json')
        if not receipt.get('passed') or receipt.get('pilot_sha256')!=source(OUT/'pilot.json')['sha256']:
            raise SystemExit('请先验收当前版本 Goodfire 记录的浏览及新建测试链路。')
        doc=historical()
    dest = OUT/'pilot.json' if args.stage=='pilot' else ROOT/'deploy/test_record_history.json'
    write(dest,doc)
    print(args.stage,len(doc['records']))

