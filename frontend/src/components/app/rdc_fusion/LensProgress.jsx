/* 脉络透镜 v4：顶层三 tab —— 行业进展 / 平台进度 / 当前机器进度
   v4（M1 接线）：平台进度 tab 接中心节点 GET /api/distributed/summary（deploy/distributed_service.py，deploy/ 不入 git），
   行业进展接 GET /api/news（arXiv 抓取+内置兜底）；失败自动回退 demo 数据（distributedData.js）
   并显示 DEMO 徽标，成功显示 LIVE 徽标。当前机器进度 = 本机 Agent 状态（demo 接入点不变）。
   三带节点点击出详情；同行差异以「对比卡」登记入账本（四级证据体系）。 */
import { useEffect, useState } from 'react';
import { LOCAL_NODE, PLATFORM_STATS, NODES, TEMPLATES, AGG_STEPS, AGG_DIMS, NEWS, MI_ERAS, RSA_ERAS, RSA_NOTE, LANG_TEMPLATES, ANALYSES, RESEARCH_KITS, TUTORIAL_STEPS, TUTORIAL_PATHS, TECH_CATEGORIES, DEMO_COVERAGE, USAGE_DOC } from './distributedData.js';

/* API 前缀：与研发透镜（LensProcess）同一惯例；禁止在组件里散落硬编码地址 */
const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');

/* 当前机器进度 · RDC 主线（demo 接入点 ← atlas_ledger.json / phase_queue.json） */
const MAINLINE=[
  {lb:'P4–P7',tt:'主轴三段\n权重绑定定 is-a',st:'done'},
  {lb:'P8–P11',tt:'写入端分布式\n栈=软门',st:'done'},
  {lb:'P17–21',tt:'w+com_V≈26\n跨精度复现',st:'done'},
  {lb:'P35',tt:'A 闸门 seal\nQ08=甲',st:'done'},
  {lb:'Q03',tt:'E_read 基线\n0.332 · 5% 门 0/3',st:'done'},
  {lb:'Q04',tt:'E_ar 装置\n4/4 PASS',st:'done'},
  {lb:'Q05',tt:'E_ar 测量\n形状 flat/sat',st:'done'},
  {lb:'Q06',tt:'C_steer 基座\n0.0000 · 无定向杠杆',st:'done'},
  {lb:'Q07',tt:'KPI 曲线 v0\nQ03–Q06 汇总',st:'now'},
  {lb:'远期',tt:'权重级证明\nN2h1-α-1',st:'todo'},
];

/* 平台进度 · 平台发布时间线（demo 接入点 ← 发布仓 / 部署记录；客户端与服务的版本时间线） */
const RELEASES=[
  {lb:'v1–v3',tt:'三部分平台改造\n3D 空间 / 研发 / 数据',st:'done'},
  {lb:':8501',tt:'InterPLM 部署\nESM-2 SAE 2548 特征',st:'done'},
  {lb:'v4–v5',tt:'融合驾驶舱 v5\n三透镜对象路由 /rdc-fusion',st:'done'},
  {lb:'3589dbb',tt:'发布仓推送\n46.97 MiB',st:'done'},
  {lb:'v5-home',tt:'v5 设为默认首页\n官网 6 页同步换肤',st:'now'},
  {lb:'next',tt:'真实数据接入\n激活←collect.npz · 队列←phase_queue',st:'todo'},
];

/* 行业进展 · 同行对比卡（demo 接入点 ← industry.json；每卡必带「对比」注脚登记视角差） */
const PEERS_ROW1=[
  {src:'Anthropic',date:'2025.03',h:'Circuit Tracing: Attribution Graphs',p:'层内→跨层电路自动化追踪；与 Q05 消融口径可桥接',
   note:'对比：Attribution Graphs 依赖人工假设遍历；本项目 Q05 用精确可加预算自动化消融 — 视角差登记入账本'},
  {src:'OpenAI',date:'2020 · archived',h:'Microscope：模型级可视化集合',p:'对象导航 + 多视图挂载的原型范式',
   note:'对比：Microscope 已归档，其「对象→多视图」范式即 v5 透镜路由来源'},
  {src:'Anthropic',date:'2022–2024',h:'Toy Models of Superposition / SAE',p:'特征叠加与稀疏字典分解基线',
   note:'对比：Superposition 对应本项目跨族近正交观察（P4–P7）；叠加上限与 G−1=5 维读位槽相关'},
];
const PEERS_ROW2=[
  {src:'开源生态',date:'已部署 :8501',h:'InterPLM — 蛋白质 SAE 对齐',p:'SAE 结论跨域同构的旁证',note:'对比：InterPLM 结论：ESM-2 SAE 叠加结论与 LLM 同构 → 支持本项目跨模型比较策略'},
  {src:'开源生态',date:'2024',h:'SAEBench — SAE 系统评估',p:'装置级评估标准的对标项',note:'SAEBench 提供 SAE 评估标准；Q04/Q05 的装置门与 metric_dict v4 口径可对标'},
];

function PeerCard({c}){
  return (
    <div className="fw-peer-card" onClick={()=>window.alert(c.note)} title={c.note}>
      <div className="src"><span>{c.src}</span><span>{c.date}</span></div>
      <h6>{c.h}</h6>
      <p>{c.p}</p>
    </div>
  );
}

/* M7-P1 技术类别查找（PROJECTIONS/era cat → 名称/颜色/短名，数据 ← TECH_CATEGORIES） */
const catOf=id=>TECH_CATEGORIES.find(c=>c.id===id)||{};
const catColor=id=>(catOf(id).color)||'#888780';
const catShort=id=>(catOf(id).short)||id;

/* 行业进展 · 领域阶段时间轴（可复用）：机制可解释性（MI_ERAS）与表征相似性分析（RSA_ERAS）各一条
   数据 ← eras（阶段数组）；live 可选——传入时最右追加「最新」实时节点（← /api/news）。
   点击节点 → 下方详情面板：阶段目标 / 成果 / 代表论文（可点击跳原文）。
   M7-P1：era 带 cat（四类技术归属）→ 顶部类别筛选 chips，非命中类别置灰（不隐藏，保时间轴完整）。 */
function EraTimeline({eras,live}){
  const [sel,setSel]=useState(eras[0].id);
  const [catSel,setCatSel]=useState('all');   // M7-P1 类别筛选
  const isLive = !!live && sel==='live';
  const era = eras.find(e=>e.id===sel) || eras[0];
  const liveNode = live ? {
    years:'NOW', title:'最新节点 · 实时动态',
    goal: live.newsLive
      ? '持续追踪领域最前沿：中心节点每 60 分钟抓取一次 arXiv「mechanistic interpretability」最新论文，前端每 30s 轮询刷新。'
      : '中心节点离线或抓取失败——当前显示内置兜底列表（最后一次成功抓取前登记的静态条目），重启中心节点后自动恢复实时。',
    outcome: live.newsLive
      ? (live.newsSrc==='arxiv' ? '来源：arXiv API · 按提交日期倒序取最新 10 篇' : '来源：中心节点缓存')
      : '来源：内置兜底列表（builtin-fallback）',
    papers: live.newsItems||[],
  } : null;
  const cur = isLive ? liveNode : era;
  return (
    <div className="fw-era">
      <div className="fw-cat-chips">
        <button type="button" className={'fw-proj-chip'+(catSel==='all'?' on':'')} onClick={()=>setCatSel('all')}><span>全部类别</span></button>
        {TECH_CATEGORIES.map(c=>(
          <button key={c.id} type="button" className={'fw-proj-chip'+(catSel===c.id?' on':'')} style={{'--cat':c.color}}
                  title={c.name} onClick={()=>setCatSel(c.id)}>
            <i style={{background:c.color}}/><span>{c.short||c.name}</span>
          </button>
        ))}
      </div>
      <div className="fw-era-tl">
        {eras.map(e=>{
          const dim=catSel!=='all'&&e.cat!==catSel;
          return (
            <button key={e.id} type="button" className={'fw-era-node'+(sel===e.id?' sel':'')+(dim?' dim':'')} onClick={()=>setSel(e.id)}>
              <span className="fw-era-dot"/>
              <span className="fw-era-years">{e.years}</span>
              <span className="fw-era-name">{e.title}</span>
              <span className="fw-era-cat" style={{color:catColor(e.cat)}}>{catShort(e.cat)}</span>
            </button>
          );
        })}
        {live && (
          <button type="button" className={'fw-era-node live'+(isLive?' sel':'')} onClick={()=>setSel('live')}
                  title={live.newsLive?'/api/news · '+live.newsSrc:'中心节点离线'}>
            <span className="fw-era-dot"/>
            <span className="fw-era-years">NOW</span>
            <span className="fw-era-name">最新{' · '}{live.newsLive?'LIVE':'兜底'}</span>
          </button>
        )}
      </div>
      <div className="fw-era-detail">
        <div className="fw-era-hd">
          <b>{cur.years} · {cur.title}</b>
          {isLive && <span className={'fw-src-chip '+(live.newsLive?'live':'demo')}>{live.newsLive?'● LIVE '+live.newsSrc:'○ FALLBACK'}</span>}
        </div>
        <div className="fw-era-body">
          <div className="fw-era-block"><h6>阶段目标</h6><p>{cur.goal}</p></div>
          <div className="fw-era-block"><h6>关键成果</h6><p>{cur.outcome}</p></div>
        </div>
        <div className="fw-era-papers">
          <h6>{isLive?'最新论文（实时流）':'代表论文'} <span>{(cur.papers||[]).length} 篇 · 点击跳原文</span></h6>
          {(cur.papers||[]).map(pw=>(
            <div key={pw.title} className="fw-era-paper" style={pw.url?{cursor:'pointer'}:undefined}
                 onClick={()=>{ if(pw.url) window.open(pw.url,'_blank','noopener'); }}>
              <span className="fw-era-pdate">{pw.date}</span>
              <span className="fw-era-ptitle">{pw.title}</span>
              <span className="fw-era-ptag">{pw.tag}</span>
              <span className="fw-era-psrc">{pw.src}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function Lane({color,label,small,children}){
  return (
    <div className="fw-lane">
      <div className="fw-lane-head"><span className="ln-dot" style={{background:color}}/>{label} <small>{small}</small></div>
      {children}
    </div>
  );
}

function Timeline({items,tag}){
  return (
    <div className="fw-tl">
      {items.map(n=>(
        <button key={n.lb} className={'fw-tl-node '+n.st} onClick={()=>window.alert(n.lb+'：'+n.tt.replace('\n',' · ')+'（详情接入 '+tag+' 后开放）')}>
          <div className="fw-tl-dot"/>
          <div className="lb">{n.lb}</div>
          <div className="tt">{n.tt.split('\n').map((t,i)=><span key={i}>{t}<br/></span>)}</div>
        </button>
      ))}
    </div>
  );
}

function fmtAgo(ts){
  if(!ts) return '—';
  const d = Math.max(0, Math.round(Date.now()/1000 - ts));
  return d<60 ? d+'s 前' : d<3600 ? Math.round(d/60)+'m 前' : Math.round(d/3600)+'h 前';
}

/* 分布式测试协议（TM 通用契约 v2，2026-10-07）：目标 / 流程 / 模板格式 / 产出格式
   与 deploy/dist_templates_seed.py + dist_runner_tm.py 严格同步；归总纪律对齐 research_os 13.x 红线 */
const PROTO_STEPS = [
  ['预注册 S2', '模板包 = 受控语料 + 固定 runner，design_sha 内容冻结；改动必须升版本'],
  ['领任务 S3', '节点 register → claim（6h 租约），下载 bundle（corpus.json + runner.py）'],
  ['采集上传 S4', '残差流目标 token（first/mid/last 层）→ 逐维 η² → summary.json + means.npz 分块上传，sha256 内容寻址，design_sha 不符 422 拒收'],
  ['分桶归总 S6', '按 模型×kind×seed 分桶；桶内 median + 符号一致性；禁止跨桶简单平均（13.1 红线）'],
  ['证据升格', '观测 → observed；过迁移干预门（换句式 / held-out 语言复现）才可登记 mechanism_evidence'],
];
const PROTO_FIELDS = [
  ['analysis', "oneway 单因素（TM-01..03）｜factorial 双因子网格（TM-04/05）"],
  ['factors', "因子名对：[topic, frame] / [entity, position] —— 句式=因子水平，变化=操纵"],
  ['layers', "采集层位 [first, mid, last]：first=第1层输出，mid=中层，last=末层"],
  ['items', "{id, cond{因子水平}, text, target}：锚点=target 末位 token，与绝对位置无关"],
  ['fingerprint', "反词嵌入指纹参照集 {within 类内词, out 类外词}——特征 ID 依据纹匹配而非激活峰值"],
  ['eta2_by_factor', "逐维 two-way 分解均值：η²(A)+η²(B)+η²(interaction)=1；交互项=推理组合特征领地"],
  ['per_layer', "每层位一组 η² 分解 —— 位置/内容效应的深度剖面"],
  ['means.npz', "keys / means / eta2_factor_dim(3×d) / cos（cell 方向矩阵）"],
];
/* 首轮结果（qwen3-4b 单节点 · 2026-10-07，登记用，非结论） */
const PROTO_FIRST = [
  ['TM-04 句式×知识', '末层 η²：句式 0.53 > 交互 0.30 > 知识 0.17；指纹类内等角 spread 0.010–0.029、类外 sep 0.046–0.057'],
  ['TM-05 实体×位置', '位置 η² 随深度 0.15→0.52 增、实体 0.77→0.29 减；「末层位置≈0」预注册预测被证伪——登记为新发现（候选解释：末层在编码下一 token 的句法规划）'],
  ['TM-06 前缀扰动（苹果用例）', 'top-32 head 集合 Jaccard：同内容跨前缀 0.73–0.94 ≫ 跨内容 0.28–0.39 → head 招募=内容主导；filler 下目标词 73% 注意力被无关前缀吸走但 head 集合保持；读出方向内容轴互聚（+0.63~0.87）/ 跨内容对立（−0.9）'],
];

function ProtocolCard(){
  return (
    <div className="fw-pl-card">
      <h6>分布式测试协议 <span>TM 通用契约 v2 · 与 deploy/ 模板包严格同步</span></h6>
      <div style={{fontSize:12, lineHeight:1.6, marginBottom:8}}>
        <b>目标</b>：把「句式万千」从噪声翻转为<b>免费的因子操纵</b>——同一知识嵌入 N 种句式、同一实体滑过 N 个位置，
        逐维方差分解把效应按因子归位；特征 ID 依据反词嵌入指纹（类内全 token 行投影等角）而非激活峰值。
        元法则：<b>变化=因子，不变=指纹；η² 定编码，干预定因果</b>。
      </div>
      <div className="fw-agg-steps">
        {PROTO_STEPS.map(([a,b],i)=>(<div key={a} className={'fw-agg-step'+(i<=2?' past':'')}><b>{i+1}. {a}</b><span>{b}</span></div>))}
      </div>
    </div>
  );
}

function ProtocolFormatCard(){
  return (
    <div className="fw-pl-card">
      <h6>模板与产出格式 <span>corpus.json（测试模板） · summary.json v2（产出模板）</span></h6>
      <div className="fw-pl-row head"><span>字段</span><span className="r">说明</span></div>
      {PROTO_FIELDS.map(([k,v])=>(
        <div key={k} className="fw-pl-row" style={{gridTemplateColumns:'110px 1fr', alignItems:'baseline'}}>
          <span className="fw-mono" style={{fontWeight:600}}>{k}</span>
          <span className="dim2" style={{fontSize:11.5, lineHeight:1.5}}>{v}</span>
        </div>
      ))}
      <div style={{marginTop:8, paddingTop:8, borderTop:'1px solid var(--fw-line, rgba(128,128,128,.2))'}}>
        <div style={{fontSize:9.5, color:'var(--fw-text-3, #94a3b8)', marginBottom:4}}>首轮登记注记（静态 · 实时分桶见下方 AGG 卡）：</div>
        {PROTO_FIRST.map(([k,v])=>(
          <div key={k} style={{fontSize:11.5, lineHeight:1.6, marginBottom:4}}>
            <span className="fw-mono" style={{fontWeight:600}}>{k}</span> · <span className="dim2">{v}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* AGG 实时桶（ui_decoupled_plan_v1 §3「首轮结果卡改为读 /api/agg 真值」）：
   对每个有结果的模板拉 GET /api/agg/{tm_id}（AGG-v0 分桶回显，模型×kind×seed），
   桶内摘要只渲染协议键（eta2_by_factor / eta2_by_cond / topk_jaccard）；
   跨桶简单平均被禁止（research_os 13.1），此处只回显不做汇总计算 */
function AggLiveCard({templates}){
  const [agg,setAgg]=useState(null);
  useEffect(()=>{
    let dead=false;
    const tms=(templates||[]).filter(t=>t.results>0).map(t=>t.tm_id);
    if(!tms.length){ setAgg({}); return undefined; }
    Promise.all(tms.map(id=>
      fetch(`${API_BASE}/api/agg/${encodeURIComponent(id)}`).then(r=>r.ok?r.json():null).catch(()=>null)
    )).then(rs=>{ if(!dead) setAgg(Object.fromEntries(tms.map((id,i)=>[id,rs[i]]))); });
    return ()=>{dead=true;};
  },[templates]);
  const entries = agg ? Object.entries(agg) : null;
  return (
    <div className="fw-pl-card">
      <h6>聚合结果 · AGG 实时分桶 <span>{agg ? 'GET /api/agg/{tm_id} · 桶=模型×kind×seed（只回显，禁跨桶平均）' : '拉取中…'}</span></h6>
      {!agg && <div className="fw-pc-empty">拉取聚合桶…</div>}
      {agg && !entries.length && <div className="fw-pc-empty">尚无结果——节点上传后此处自动出现。</div>}
      {entries && entries.map(([tm,a])=>(
        <div key={tm} className="fw-agg-tpl">
          <div className="fw-agg-tpl-h"><b className="fw-mono">{tm}</b><span className="dim2 fw-mono" style={{fontSize:9}}>{a&&a.agg_version}</span></div>
          {a && Object.entries(a.buckets||{}).map(([bk,items])=>{
            const withEta2 = items.map(it=>it.summary&&it.summary.eta2_by_factor).filter(Boolean);
            const eta2 = withEta2.length ? withEta2[0] : null;
            const cond = items.map(it=>it.summary&&it.summary.eta2_by_cond).filter(Boolean)[0]||null;
            const jacc = items.map(it=>it.summary&&it.summary.topk_jaccard).filter(Boolean)[0]||null;
            return (
              <div key={bk} className="fw-agg-bucket">
                <span className="fw-mono dim2" style={{fontSize:9.5}}>{bk}</span>
                <b className="fw-mono" style={{fontSize:10}}>{items.length} 条</b>
                {eta2 && <span className="fw-mono" style={{fontSize:9.5}}>
                  η² {Object.entries(eta2).filter(([,v])=>typeof v==='number').map(([k,v])=>k+' '+v.toFixed(2)).join('/')}
                </span>}
                {cond && <span className="fw-mono dim2" style={{fontSize:9.5}}>{Object.entries(cond).map(([k,v])=>k+' '+(v.toFixed?v.toFixed(2):v)).join(' · ')}</span>}
                {jacc && <span className="fw-mono dim2" style={{fontSize:9.5}}>{Object.entries(jacc).slice(0,2).map(([k,v])=>k+' '+(v.toFixed?v.toFixed(2):v)).join(' · ')}</span>}
              </div>
            );
          })}
          {a && !Object.keys(a.buckets||{}).length && <div className="dim2" style={{fontSize:9.5,padding:'2px 12px 6px'}}>无桶数据</div>}
        </div>
      ))}
    </div>
  );
}

/* 使用说明 · 新人教程四步卡（M6-P0，数据 ← TUTORIAL_STEPS）：接入→领任务→上传→下载研究
   每步命令可一键复制；<center>/<gpu>/<model> 为占位符，接入前替换为本机实际值
   （2026-10-08 调整：新手相关内容从「研发平台」tab 移入本「使用说明」tab） */
function TutorialCard(){
  const [copied,setCopied]=useState(-1);
  const copy=(cmd,i)=>{ try{ navigator.clipboard.writeText(cmd); }catch(e){} setCopied(i); setTimeout(()=>setCopied(-1),1400); };
  return (
    <div className="fw-pl-card fw-tut">
      <h6>新人教程 · 四步接入分布式研发 <span>从零参与到产出上传与本地复现 · 命令可复制</span></h6>
      <div className="fw-tut-steps">
        {TUTORIAL_STEPS.map((s,i)=>(
          <div key={s.n} className="fw-tut-step">
            <span className="fw-tut-n">{s.n}</span>
            <div className="fw-tut-main">
              <b>{s.t}</b>
              <div className="fw-tut-cmd">
                <code>{s.cmd}</code>
                <button type="button" className="fw-tbtn" onClick={()=>copy(s.cmd,i)}>{copied===i?'✓ 已复制':'复制'}</button>
              </div>
              <p>{s.d}</p>
            </div>
            {i<TUTORIAL_STEPS.length-1&&<span className="fw-tut-arrow">→</span>}
          </div>
        ))}
      </div>
      <p className="fw-tut-note">占位符替换：<span className="fw-mono">&lt;center&gt;</span> 中心节点地址 · <span className="fw-mono">&lt;gpu&gt;</span> GPU 型号 · <span className="fw-mono">&lt;model&gt;</span> 目标模型。全流程纪律：预注册冻结（design_sha 不符拒收）、内容寻址（sha256）、口径版本强制展示。</p>
    </div>
  );
}

/* 使用说明 · 按技术类别上手路径（M7-P1，数据 ← TUTORIAL_PATHS）：
   四类技术各一条——复用四步教程骨架，差异只在第 4 步 download 的 --kit 选择 */
function TutorialPathsCard(){
  return (
    <div className="fw-pl-card fw-tut">
      <h6>按技术类别上手 <span>四类技术 × 四条路径 · 骨架同上方四步，差异在第 4 步 --kit</span></h6>
      {TUTORIAL_PATHS.map(p=>{
        const c=catOf(p.cat);
        return (
          <div key={p.cat} className="fw-path-row" title={c.note||''}>
            <span className="fw-path-dot" style={{background:c.color}}/>
            <b>{c.name||p.cat}</b>
            <code className="fw-mono">--kit {p.kit}</code>
            <span className="dim2">{p.note}</span>
          </div>
        );
      })}
      <p className="fw-tut-note">类别状态实时看「研发平台 · 技术成熟度」卡组；缺口任务（哪个模板 × 哪类技术还是空格）在研发透镜「缺口」tab 领取。</p>
    </div>
  );
}

/* 使用说明 · 平台介绍 + 文档索引 + node_agent 子命令速查（数据 ← USAGE_DOC）
   全部内容为真实仓库文件/端点；文档行悬停提示完整路径 */
function TutorialDocCard(){
  return (
    <>
      <div className="fw-pl-card fw-tut">
        <h6>平台是什么 <span>一分钟了解这套分布式研发体系</span></h6>
        <p className="fw-tut-note" style={{marginTop:0,fontSize:10.5,lineHeight:1.85}}>{USAGE_DOC.intro}</p>
      </div>
      <div className="fw-pl-card fw-tut">
        <h6>文档索引 <span>深入阅读 · 均为仓库内真实文件与端点</span></h6>
        <div className="fw-reg-xd" style={{gap:0}}>
          {USAGE_DOC.docs.map(d=>(
            <div key={d.t} className="fw-reg-item" style={{display:'block',padding:'8px 2px'}} title={d.s}>
              <div className="fw-reg-hd"><b>{d.t}</b><span className="fw-mono dim2">{d.s}</span></div>
              <div style={{marginTop:4,fontSize:10,color:'var(--fw-text-2)',lineHeight:1.65}}>{d.d}</div>
            </div>
          ))}
        </div>
      </div>
      <div className="fw-pl-card fw-tut">
        <h6>终端命令速查 <span>node_agent.py 五个子命令 · --server 为顶层参数置于子命令前</span></h6>
        <div className="fw-reg-xd">
          {USAGE_DOC.cli.map(([c,d])=>(
            <div key={c}><i><span className="fw-mono" style={{color:'var(--fw-blue)'}}>node_agent.py {c}</span></i><span>{d}</span></div>
          ))}
        </div>
        <p className="fw-tut-note">凭据纪律：<span className="fw-mono">node_credentials.json</span> 仅存本机，浏览器不读取——本机精确状态看终端 <span className="fw-mono">status</span>，浏览器内只展示公开队列快照。</p>
      </div>
    </>
  );
}

/* 平台进度 · 技术成熟度卡组（M7-P1）：四类技术 × 平台状态机（与 Q04/Q06「装置建成→正式测量」同构）
   数据 ← TECH_CATEGORIES（类状态/算力）+ ANALYSES（各类注册数/可用数）+ DEMO_COVERAGE（覆盖格） */
function MaturityCards(){
  return (
    <div className="fw-mature-grid">
      {TECH_CATEGORIES.map(c=>{
        const techs=ANALYSES.filter(a=>a.category===c.id);
        const avail=techs.filter(t=>t.status==='available').length;
        const done=DEMO_COVERAGE.filter(x=>x.cat===c.id&&x.status==='done').length;
        return (
          <div key={c.id} className="fw-pl-card fw-mature-card" style={{borderTop:'2px solid '+c.color}} title={c.note}>
            <div className="fw-mature-hd">
              <b style={{color:c.color}}>{c.name}</b>
              <span className={'fw-tmx-agg '+(c.status==='measured'?'done':c.status==='device_built'?'collecting':'todo')}>
                {c.status==='measured'?'已测':(c.status==='device_built'?'装置已建':'装置待建')}
              </span>
            </div>
            <p className="fw-mature-note">{c.note}</p>
            <div className="fw-mature-ms">{c.methods.join(' · ')}</div>
            <div className="fw-mature-kv">
              <span>注册 {techs.length} 项</span><span>可用 {avail}</span><span>覆盖 {done}/{DEMO_COVERAGE.length} 格</span>
            </div>
            <div className="fw-mature-cost dim2">算力参考：{c.cost} · 契约 {c.input}</div>
          </div>
        );
      })}
    </div>
  );
}

/* 平台进度 · 双注册表卡（M6-P0/P1）：语言模板（LANG_TEMPLATES）× 分析技术（ANALYSES）+ 研究包
   研究包组合：离线=RESEARCH_KITS DEMO 叙事；LIVE=GET /api/kits（服务端 S7，tm×analysis），
   available 行可点击下载整包（bundle JSON：corpus/runner/contract + 结果清单 + README） */
function RegistryCards(){
  const [open,setOpen]=useState(null);   /* 展开的语言模板 id */
  const [kits,setKits]=useState(null);   /* LIVE：服务端 S7 研究包清单 */
  const [dlIng,setDlIng]=useState('');
  useEffect(()=>{
    let dead=false;
    fetch(`${API_BASE}/api/kits`).then(r=>r.ok?r.json():null).catch(()=>null)
      .then(d=>{ if(!dead&&d&&d.kits) setKits(d); });
    return ()=>{dead=true;};
  },[]);
  const downloadKit=async(kid)=>{
    setDlIng(kid);
    try{
      const r=await fetch(`${API_BASE}/api/kits/${kid}/bundle`);
      if(!r.ok) throw new Error('HTTP '+r.status);
      const b=await r.json();
      const blob=new Blob([JSON.stringify(b,null,1)],{type:'application/json'});
      const a=document.createElement('a');
      a.href=URL.createObjectURL(blob); a.download=`${kid}.bundle.json`; a.click();
      URL.revokeObjectURL(a.href);
    }catch(e){
      window.alert(`研究包 ${kid} 下载失败（${e.message}）——需中心节点已重启加载 S7 端点，或该 kit 尚无满足契约的结果。`);
    }finally{ setDlIng(''); }
  };
  const tplOf=id=>LANG_TEMPLATES.find(t=>t.id===id);
  return (
    <div className="fw-reg-row">
      <div className="fw-pl-card">
        <h6>语言模板注册表 <span>{LANG_TEMPLATES.length} 种 · 受控语料+固定口径 · 空间透镜可切换体验</span></h6>
        {LANG_TEMPLATES.map(t=>(
          <div key={t.id} className="fw-reg-item" onClick={()=>setOpen(open===t.id?null:t.id)}>
            <div className="fw-reg-hd">
              <b className="fw-mono">{t.id}</b>
              <span>{t.name}</span>
              <span className={'fw-tmx-agg '+(t.status==='measured'?'done':'todo')}>{t.status==='measured'?'measured · 已测':'demo · 占位'}</span>
            </div>
            {open===t.id&&(
              <div className="fw-reg-xd">
                <div><i>示例 token 序列</i><span className="fw-mono">{t.demo.tokens.join(' ')}</span></div>
                <div><i>焦点特征</i><span className="fw-mono">{t.demo.focus.id} · {t.demo.focus.label}</span></div>
                <div><i>状态注记</i><span>{t.status==='measured'?'E_read=0.331615（TM-07，Q09 基线）':'未测量——不挂任何实测数字，Q20 族扩展后转 measured'}</span></div>
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="fw-pl-card">
        <h6>分析技术注册表 <span>{ANALYSES.length} 种 · 按输入契约匹配可用性 · 口径各自登记</span></h6>
        {ANALYSES.map(a=>(
          <div key={a.id} className="fw-reg-item">
            <div className="fw-reg-hd">
              <b className="fw-mono">{a.id}</b>
              <span>{a.name}</span>
              <span className="fw-era-cat" style={{color:catColor(a.category)}} title={'技术类别：'+((catOf(a.category)||{}).name||a.category)}>{catShort(a.category)}</span>
              <span className="fw-pc-ev mono">{a.evidence_level}</span>
            </div>
            <div className="fw-reg-xd">
              <div><i>技术类别</i><span>{(catOf(a.category)||{}).name||a.category} · {a.status==='available'?'可运行':(a.status==='planned'?'装置待建':a.status)}</span></div>
              <div><i>输入契约</i><span className="fw-mono">{a.input.join(' + ')||'—'}</span></div>
              <div><i>口径版本</i><span className="fw-mono">{a.metric_version}</span></div>
              <div><i>说明</i><span>{a.note}</span></div>
            </div>
          </div>
        ))}
        <h6 style={{marginTop:10}}>
          研究包组合
          <span>{kits ? 'LIVE · 服务端 S7 · available 行点击下载整包' : 'DEMO · 语言模板 × 分析技术（中心节点离线）'}</span>
          {!kits && <span className="fw-src-chip demo" style={{marginLeft:'auto'}}>○ DEMO</span>}
          {kits && <span className="fw-src-chip live" style={{marginLeft:'auto'}}>● LIVE S7</span>}
        </h6>
        {kits ? kits.kits.map(k=>(
          <div key={k.kit_id} className="fw-reg-item"
               onClick={()=>{ if(k.status==='available') downloadKit(k.kit_id); }}
               title={k.status==='available'?'点击下载研究包（bundle JSON：corpus/runner/contract + 结果清单 + README）':'结果库暂无满足该 input 契约的结果'}>
            <div className="fw-reg-hd">
              <b className="fw-mono">{k.kit_id}</b>
              <span className="dim2">{k.analysis_name} · {k.metric_version}</span>
              {k.status==='available'
                ? <button type="button" className="fw-tbtn" style={{marginLeft:'auto',flexShrink:0}}
                          onClick={e=>{e.stopPropagation();downloadKit(k.kit_id);}}
                          disabled={dlIng===k.kit_id}>{dlIng===k.kit_id?'下载中…':'↓ 下载'}</button>
                : <span className={'fw-tmx-agg todo'} style={{marginLeft:'auto',flexShrink:0}}>pending · 等结果</span>}
            </div>
          </div>
        )) : RESEARCH_KITS.map(k=>(
          <div key={k.kit_id} className="fw-reg-item">
            <div className="fw-reg-hd">
              <b className="fw-mono">{tplOf(k.lang_tpl)?tplOf(k.lang_tpl).name:k.lang_tpl}</b>
              <span className="dim2">× {k.analysis}</span>
              <span className={'fw-tmx-agg '+(k.status==='available'?'done':'todo')}>{k.status==='available'?'available · 可下载':'pending · 等结果'}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function LensProgress({on,onGo}){
  const [view,setView]=useState('machine');
  /* 中心节点实时态（M1 接线）：summary=平台进度、newsLive=行业进展；拉不到则回退 demo */
  const [dist,setDist]=useState(null);
  const [distNews,setDistNews]=useState(null);
  const [newsSrc,setNewsSrc]=useState('');
  const [schedQ,setSchedQ]=useState(null);      // M6-P2 /api/tasks（S8 调度队列只读快照，本机状态条数据源）
  useEffect(()=>{
    let dead=false;
    const load=()=>{
      const ctl=new AbortController();
      const t=setTimeout(()=>ctl.abort(),4000);
      Promise.all([
        fetch(`${API_BASE}/api/distributed/summary`,{signal:ctl.signal}).then(r=>r.ok?r.json():null).catch(()=>null),
        fetch(`${API_BASE}/api/news`,{signal:ctl.signal}).then(r=>r.ok?r.json():null).catch(()=>null),
        fetch(`${API_BASE}/api/tasks`,{signal:ctl.signal}).then(r=>r.ok?r.json():null).catch(()=>null),
      ]).then(([s,n,q])=>{
        if(dead) return;
        if(s&&s.version) setDist(s);
        if(n&&n.items&&n.items.length){ setDistNews(n.items); setNewsSrc(n.source||''); }
        setSchedQ(q&&q.stats?q:null);
      }).catch(()=>{}).finally(()=>clearTimeout(t));
    };
    load();
    const iv=setInterval(load,30000);
    return ()=>{dead=true;clearInterval(iv);};
  },[]);

  /* 平台进度数据源：LIVE（中心节点）或 DEMO（distributedData.js）；研究包统计 ← 注册表（协议第七件） */
  const liveStats = [
    ...(dist ? [
      {n:dist.nodes_online, l:'在线节点', s:'共 '+dist.nodes_total+' 注册'},
      {n:dist.templates.filter(t=>t.status==='open').length, l:'开放模板', s:'模板注册表 S2'},
      {n:dist.results_total, l:'已上传结果', s:'sha256 内容寻址'},
      {n:dist.downloads_total, l:'结果下载', s:'他人引用计数'},
      {n:dist.agg_version, l:'聚合版本', s:dist.version},
    ] : PLATFORM_STATS),
    {n:RESEARCH_KITS.filter(k=>k.status==='available').length, l:'可用研究包', s:'语言模板 × 分析技术'},
  ];
  const liveNodes = dist ? dist.nodes.map(n=>({
    id:n.node_id, gpu:n.gpu||'—', model:n.model||n.name||'—', tm:'—', prog:0,
    cls: (Date.now()/1000-n.last_seen<120)?'run':'idle', up:fmtAgo(n.last_seen),
  })) : NODES;
  const liveTpls = dist ? dist.templates.map(t=>({
    id:t.tm_id, dim:t.dim, name:t.name, me:false,
    agg: t.results>0 ? 'collecting' : 'todo',
    nodes:t.claims_active, cells:t.results+' 结果',
  })) : TEMPLATES;
  const tplTip = t => t.name+' · '+(t.nodes||0)+' 节点执行中 · '+t.cells;
  const tplAggTxt = t => t.agg==='done'?'已聚合':(t.agg==='collecting'?'收集中':'待分发');
  const newsItems = distNews || NEWS;
  return (
    <section className={'fw-view fw-progress'+(on?' on':'')}>
      <div className="fw-rg-inner">
        {/* ===== 顶层三 tab：行业进展 / 平台进度 / 当前机器进度 ===== */}
        <div className="fw-pt-tabs" role="tablist">
          <button type="button" role="tab" className={'fw-pt-tab'+(view==='news'?' on':'')} onClick={()=>setView('news')}>行业进展<small>机制可解释性动态</small></button>
          <button type="button" role="tab" className={'fw-pt-tab'+(view==='platform'?' on':'')} onClick={()=>setView('platform')}>研发平台<small>{dist ? dist.nodes_online+' 节点在线 · '+dist.templates.length+' 模板' : '47 节点 · 12 模板'}</small></button>
          <button type="button" role="tab" className={'fw-pt-tab'+(view==='machine'?' on':'')} onClick={()=>setView('machine')}>本地进度<small>NODE-A3F7 · 本机</small></button>
          <button type="button" role="tab" className={'fw-pt-tab'+(view==='tutorial'?' on':'')} onClick={()=>setView('tutorial')}>使用说明<small>新手教程 · 文档</small></button>
        </div>

        {/* ===== 当前机器进度：本机节点 + RDC 主线 ===== */}
        {view==='machine'&&(<>
          {/* 本机节点横幅（demo：接 GET /api/distributed/summary 后取真实本机状态） */}
          <div className="fw-node-banner">
            <span className="fw-nid">{LOCAL_NODE.id}</span>
            <span className="fw-nchip">{LOCAL_NODE.gpu}</span>
            <span className="fw-nchip">{LOCAL_NODE.model}</span>
            <span className={'fw-nstat '+LOCAL_NODE.status}>● 执行中</span>
            <span className="fw-ntask">{LOCAL_NODE.task}</span>
            <span className="fw-nprog"><i style={{width:(LOCAL_NODE.progress*100).toFixed(0)+'%'}}/></span>
            <span className="fw-nmono">{LOCAL_NODE.cells}/{LOCAL_NODE.cells_total} cells</span>
            <span className="fw-nmono">上传 {LOCAL_NODE.uploaded}</span>
            <span className="fw-nmono dim">{LOCAL_NODE.server}</span>
          </div>

          {/* 本机 agent 状态条（M6-P2）：调度队列只读快照（GET /api/tasks）+ 终端详细状态指路。
              浏览器无法读取本机 node_credentials.json（凭据不出终端），故中心侧只给队列快照；
              本机 agent 精确状态（含凭据鉴权的 /api/nodes/me）用终端 `node_agent.py status` 查看 */}
          <div className="fw-sched-bar">
            {schedQ ? (<>
              <span className="fw-sched-dot on">●</span>
              <b>调度队列快照</b>
              <span className="fw-mono">活跃 {schedQ.stats.claimed||0}</span>
              <span className="fw-mono">完成 {schedQ.stats.done||0}</span>
              <span className="fw-mono">失败 {schedQ.stats.failed||0}</span>
              <span className="fw-mono">过期 {schedQ.stats.expired||0}</span>
              <span className="fw-mono dim">单租约 {schedQ.lease_hours}h · GET /api/tasks</span>
            </>) : (<>
              <span className="fw-sched-dot">○</span>
              <b>调度队列快照</b>
              <span className="dim">中心节点不可达——无调度数据（本横幅为 demo 叙事，不代表本机实时状态）</span>
            </>)}
            <span className="fw-sched-tip" title="凭据不出终端：node_credentials.json 仅存本机，浏览器不读取">本机 agent 详细状态 → 终端 <code>node_agent.py status</code> · 全队列 → <code>node_agent.py queue</code></span>
          </div>

          <Lane color="#0284c7" label="自己的 · RDC 主线" small="F#3734 所在证据链 · Phase 4 → Q07 · 点击节点看详情">
            <Timeline items={MAINLINE} tag="atlas_ledger.json"/>
          </Lane>

          <div className="fw-rg-note">
            <b>路线图逻辑</b>：三个 tab 共享同一套证据口径——本机进度挂真实实验（节点必带 sealed 产物），平台进度对应可运行的版本与聚合产物（AGG-v0 分桶 → v1 跨节点 η²），业界以新闻流与对比卡登记视角差（has_data / observed / generalization_checked / mechanism_evidence 四级体系）。节点接入：见 deploy/README_CENTOS.md（node_agent.py register → run）。
          </div>
          <div style={{textAlign:'center',padding:'4px 0 10px'}}>
            <button className="fw-tbtn" onClick={()=>onGo('process')}>回到当前任务 →</button>
          </div>
        </>)}

        {/* ===== 平台进度：分布式统计 + 节点 + 模板矩阵 + 聚合分析 + 发布时间线 ===== */}
        {view==='platform'&&(<>
          <div className="fw-pl-stats">
            <span className={'fw-src-chip '+(dist?'live':'demo')} title={dist?dist.version:'接 GET /api/distributed/summary 后显示实时数据'}>
              {dist?'● LIVE 中心节点':'○ DEMO（中心节点离线）'}
            </span>
            {liveStats.map(s=>(
              <div key={s.l} className="fw-pl-stat"><div className="n">{s.n}</div><div className="l">{s.l}</div><div className="s">{s.s}</div></div>
            ))}
          </div>
          {/* ===== 双注册表卡（C1：语言模板 × 分析技术 × 研究包）——新人教程已移至「使用说明」tab ===== */}
          <RegistryCards/>

          {/* ===== 技术成熟度卡组（M7-P1）：四类技术 × 平台状态机 ===== */}
          <MaturityCards/>

          <div className="fw-pl-grid">
            {/* 节点列表（LIVE=summary.nodes 最近心跳；DEMO=distributedData.js） */}
            <div className="fw-pl-card">
              <h6>节点目录 <span>{dist?'按最近心跳排序 · '+liveNodes.length+' 台展示':'8/47 展示 · 接 /api/distributed/summary'}</span></h6>
              <div className="fw-pl-row head"><span>节点</span><span>硬件</span><span>当前模板</span><span>进度</span><span className="r">上传</span></div>
              {liveNodes.map(n=>(
                <div key={n.id} className="fw-pl-row">
                  <span className={'fw-mono'+(n.me?' me':'')}>{n.id}{n.me?'（本机）':''}</span>
                  <span className="dim2">{n.gpu} · {n.model}</span>
                  <span>{n.tm}</span>
                  <span><span className="fw-pl-prog"><i style={{width:(n.prog*100).toFixed(0)+'%'}}/></span></span>
                  <span className={'r fw-mono fw-st-'+n.cls}>{n.cls==='agg'?'聚合中':(n.cls==='run'?'在线':'空闲')} · {n.up}</span>
                </div>
              ))}
            </div>
            {/* 模板矩阵 + 聚合管线 */}
            <div style={{display:'flex',flexDirection:'column',gap:10,minWidth:0}}>
              <div className="fw-pl-card">
                <h6>模板测试矩阵 <span>每模板=受控语料+固定口径，可分发任意机器</span></h6>
                <div className="fw-tmx">
                  {liveTpls.map(t=>(
                    <div key={t.id} className={'fw-tmx-cell'+(t.me?' me':'')} title={tplTip(t)}>
                      <b>{t.id}</b>
                      <span className="fw-tmx-dim">{t.dim}</span>
                      <span className={'fw-tmx-agg '+t.agg}>{tplAggTxt(t)}</span>
                      <span className="fw-tmx-n">{t.nodes||0}节点 · {t.cells}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="fw-pl-card">
                <h6>聚合分析管线 <span>综合大量模板结果 → 语言编码机制</span></h6>
                <div className="fw-agg-steps">
                  {AGG_STEPS.map(([a,b],i)=>(
                    <div key={a} className={'fw-agg-step'+(i<=3?' past':'')}>
                      <b>{i+1}. {a}</b><span>{b}</span>
                    </div>
                  ))}
                </div>
                <div className="fw-agg-dims">
                  {AGG_DIMS.map(d=>(
                    <div key={d.dim} className="fw-agg-dim">
                      <b>{d.dim}</b>
                      <span className={'fw-tmx-agg '+d.state}>{d.state==='done'?'已聚合':(d.state==='partial'?'部分':'待开始')}</span>
                      <p>{d.note} <em>{d.src}</em></p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* ===== 分布式测试协议（v2 通用契约）：目标/流程/格式 —— 任意句型可扩展 ===== */}
          <div style={{display:'grid', gridTemplateColumns:'minmax(280px,5fr) minmax(320px,7fr)', gap:10, alignItems:'stretch'}}>
            <ProtocolCard/>
            <ProtocolFormatCard/>
          </div>

          {/* ===== AGG 实时分桶（/api/agg 真值，协议键渲染）——首轮静态登记见上卡注记 ===== */}
          <AggLiveCard templates={dist ? dist.templates : []}/>

          <Lane color="#059669" label="平台发布" small="可视化客户端与服务端的版本时间线">
            <Timeline items={RELEASES} tag="release registry"/>
          </Lane>
        </>)}

        {/* ===== 使用说明：平台介绍 + 新人教程四步卡 + 文档索引 + 命令速查（新手内容集中于此） ===== */}
        {view==='tutorial'&&(<>
          <TutorialCard/>
          <TutorialPathsCard/>
          <TutorialDocCard/>
        </>)}

        {/* ===== 行业进展：领域阶段时间轴（点击节点看阶段/论文） + 实时动态流 + 同行对比卡 ===== */}
        {view==='news'&&(<>
          <div className="fw-news-head">
            行业动态 · 机制可解释性
            <span className={'fw-src-chip '+(distNews?'live':'demo')} title={distNews?'GET /api/news · '+newsSrc:'中心节点离线，显示内置兜底列表'}>
              {distNews?'● LIVE '+(newsSrc==='arxiv'?'arXiv 抓取':'中心节点'):'○ DEMO（内置兜底）'}
            </span>
            <span>领域五阶段时间轴 · 最新论文见时间轴 NOW 节点{distNews?' · 每 30s 自动刷新':' · 接入 RSS / arXiv / 官方博客后自动更新'}</span>
          </div>
          <EraTimeline eras={MI_ERAS} live={{newsItems,newsLive:!!distNews,newsSrc}}/>

          {/* ===== 表征相似性分析（RSA）：第二条阶段时间轴（格式与机制可解释性时间轴一致） ===== */}
          <div className="fw-news-head" style={{marginTop:16}}>
            表征相似性分析（RSA）
            <span>方法演进时间轴 · 比较的是「关系结构」而非坐标 · 跨模型/跨物种可比 · 点击节点看阶段与论文</span>
          </div>
          <EraTimeline eras={RSA_ERAS}/>
          <div className="fw-rg-note" style={{marginTop:10}}>
            <b>与本项目证据链的对比</b>：{RSA_NOTE}
          </div>

          <Lane color="#6366f1" label="行业主流 · 对比卡" small="点击对比卡 → 与本项目证据链的视角差">
            <div className="fw-peer-row">{PEERS_ROW1.map((c,i)=><PeerCard key={i} c={c}/>)}</div>
            <div className="fw-peer-row" style={{marginBottom:0}}>
              {PEERS_ROW2.map((c,i)=><PeerCard key={i} c={c}/>)}
            </div>
          </Lane>
        </>)}
      </div>
    </section>
  );
}
