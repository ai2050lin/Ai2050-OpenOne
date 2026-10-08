/* ============================================================
   RdcFusionWorkspace demo 数据层 —— 全部为真实研究数值
   （M3 红线：demo 内容只允许存在于本文件，组件内禁止内联常量数据）
   接入点：
   - EVENTS / TICKER ← research OS 事件总线 / .workbuddy ledger
   - ACT_LINES       ← collect.npz（F#3734 top 激活 token，值→emerald 透明度）
   - CMDK_GROUPS     ← atlas_ledger.json / phase_queue / industry.json
   - DEMO_QUEUE      ← /api/ai-rnd/queue（phase_queue_v1.json）离线回退
   - DEMO_WORKSPACE  ← /api/ai-rnd/workspace?path=tests/deepseek 离线回退
   - DEMO_ARTIFACTS  ← Q05 运行工件节选（collect_ar.py / result.json / review）
   - DEMO_OBJECT     ← /api/object/F#3734（object_registry.json）离线回退
   ============================================================ */

export const EVENTS=[
  {tm:'08:02',txt:'Q05 sealed · metric_dict v3→v4'},
  {tm:'07:31',txt:'发布仓 3589dbb · 46.97 MiB'},
];

export const TICKER=[
  '[Q05] sealed 1acb1e78 · 独立复核 47/0',
  '[metric_dict] v3→v4 · E_ar.status=measured',
  '[InterPLM] :8501 ESM-2 SAE 2548 特征就绪',
  '[ledger] n=304 bbda63df · N 线 P3–P7 待补',
  '[queue] 7/30 sealed · next Q07 KPI v0',
];

export const ACT_LINES=[
  {toks:[['苹果',0],['是',0],['一种',0],['水果',0.95],['，',0],['富含',0.45],['维生素',0]]},
  {toks:[['香蕉',0],['、',0],['菠萝',0],['、',0],['芒果',0],['都',0],['属于',0],['水果',0.95]]},
  {toks:[['The',0],['crinoletta',0],['is',0],['a',0],['hybrid',0.55],['apple',0.75],['variety',0.35]]},
  {dfa:true,toks:[['我',0],['买了',0],['一台',0],['苹果',0.12],['笔记本',0]]},
];

export const CMDK_GROUPS=[
  {gl:'对象 · FEATURES',items:[
    {k:'F#3734',t:'is-a 上位关系：水果族',d:'L6 · mech_evidence',go:'spatial'},
    {k:'F#0821',t:'is-a 下位：个体实例方向',d:'L6 · observed',go:'spatial'},
  ]},
  {gl:'任务 · TASKS',items:[
    {k:'Q07',t:'KPI 曲线 v0 汇总 · NEXT',d:'Q03–Q06 → 单调曲线',go:'process'},
    {k:'Q06',t:'C_steer 基座测量 · SEALED',d:'5f88ed7e · 复核 16/0',go:'process'},
    {k:'Q05',t:'E_ar(k) 正式测量 · SEALED',d:'1acb1e78',go:'process'},
  ]},
  {gl:'论文与节点 · PROGRESS',items:[
    {k:'paper',t:'Circuit Tracing: Attribution Graphs',d:'Anthropic 2025',go:'progress'},
    {k:'P35',t:'A 闸门 seal（R8）',d:'Q08=甲 · C 全接受',go:'progress'},
  ]},
];

/* ── 研发队列 demo（/api/ai-rnd/queue 回退；条目 schema 与 phase_queue_v1 对齐） ── */
export const DEMO_QUEUE={
  source:'demo', count:5, sealed:4,
  queue:[
    {id:'Q07',q:7,title:'KPI 曲线 v0 汇总',status:'pending',block:'B KPI',kpi:'all',gpu:'zero',
     deliverable:'把 Q03–Q06 + 397 条历史判决映射为 advance/catalog，产出第一张单调曲线'},
    {id:'Q06',q:6,title:'C_steer 基座测量',status:'sealed',block:'B KPI',kpi:'C_steer',gpu:'mid',
     seal_record:'tests/deepseek/result/q06_result.json',res_sha8:'5f88ed7e',
     note:'C_steer_main=0.0000 · Wilson95 上界 1.0% · 复核 16/0'},
    {id:'Q05',q:5,title:'E_ar(k) 正式测量',status:'sealed',block:'B KPI',kpi:'E_ar',gpu:'mid',
     seal_record:'tests/deepseek/result/q05_result.json',res_sha8:'1acb1e78',
     note:'形状 flat/saturating/flat · D4 桥 PASS · 复核 47/0'},
    {id:'Q04',q:4,title:'E_ar(k) 装置建造',status:'device_built',block:'B KPI',kpi:'E_ar',gpu:'mid',
     seal_record:'tests/deepseek/result/q04_smoke_result.json',prereg_design_sha:'33ddf69d',
     note:'SMOKE 4/4 装置门 · 复核 14/0'},
    {id:'Q03',q:3,title:'E_read 统一基线复算',status:'sealed',block:'B KPI',kpi:'E_read',gpu:'low',
     seal_record:'tests/deepseek/result/q03_result.json',
     note:'池化 0.373350 · 5% 门 0/3'},
  ],
};

/* ── 工作区文件树 demo（/api/ai-rnd/workspace?path=tests/deepseek 回退） ── */
export const DEMO_WORKSPACE={
  path:'tests/deepseek', name:'deepseek', truncated:false,
  dirs:[
    {name:'q05_ar_sweep',path:'tests/deepseek/q05_ar_sweep',type:'dir'},
    {name:'q06_c_steer',path:'tests/deepseek/q06_c_steer',type:'dir'},
    {name:'shared',path:'tests/deepseek/shared',type:'dir'},
  ],
  files:[
    {name:'collect_ar.py',path:'tests/deepseek/collect_ar.py',type:'file',size:8421,demo:'code'},
    {name:'metric_dict_v4.json',path:'tests/deepseek/metric_dict_v4.json',type:'file',size:5210,demo:'result'},
    {name:'review_report.txt',path:'tests/deepseek/review_report.txt',type:'file',size:1204,demo:'review'},
  ],
};

/* ── 运行工件 demo 正文（Q05 节选，真实数值） ── */
export const DEMO_ARTIFACTS={
  code:{label:'collect_ar.py（节选）',text:[
    '# Q05 · E_ar(k) 正式测量 — 四臂 738×K16（节选）',
    'def sweep_ar(model, K=16, arms=("is_a", "attr", "syntax", "rand")):',
    '    share = exact_additive_budget(model)          # 铁律 (a) 精确可加向量预算',
    '    for arm in arms:',
    '        E = [share.write(arm, k=k) for k in range(1, K+1)]',
    '        rel = (E[0] - E) / E[0]                   # S_rel 归一化',
    '        yield arm, rel',
    '',
    '# D4 桥（4b bf16 ↔ 4-bit NF4 口径桥）',
    'assert max(abs(rel_bf16 - rel_nf4)) < 0.05   # PASS: 0.0489',
  ].join('\n')},
  result:{label:'result.json',text:[
    '{',
    '  "phase": "Q05", "status": "measured",',
    '  "d4_bridge_max_delta_rel": 0.0489,',
    '  "shape": {"4b": "flat", "14b": "saturating", "9b": "flat"},',
    '  "s_rel_min": [0.6858, 0.6939, 0.7392],',
    '  "sealed_sha": "1acb1e78",',
    '  "review": {"pass": 47, "fail": 0}',
    '}',
  ].join('\n')},
  review:{label:'review_report.txt',text:[
    '# 独立复核结论（节选）',
    '装置门 4/4 通过；份额全程使用精确可加向量预算；',
    '冻结锚逐位复现（drift 0.00e+00）；D4 桥在预注册门内。',
    '未支持结论：k→∞ 外推、跨层迁移、权重级因果证明。',
  ].join('\n')},
};

/* ── 终端 demo 行（SSE 事件流离线回退） ── */
export const DEMO_TERMINAL=[
  '$ python q05_ar_sweep/collect_ar.py --all-arms --K 16',
  '[Q05] D4 bridge max|Δrel| = 0.0489 < 0.05 PASS',
  '[Q05] shape 4b=flat · 14B=saturating · 9B=flat',
  '[Q06] C_steer_main = 0.0000 · 95%CI ≤ 1.0% · collateral clean 93.3%',
  '[review] independent TOTAL PASS=47 FAIL=0 → sealed 1acb1e78',
  '[Q06] sealed 5f88ed7e · 复核 16/0 → Ledger n=306',
];

/* ── 对象卡 demo（/api/object/{fid} 回退；schema=object_card.v1 与 object_registry.json 对齐） ── */
export const DEMO_OBJECT={
  id:'F#3734',
  label:'is-a 上位关系：水果族',
  layer:'L6 · write 端',
  evidence:'mechanism_evidence',
  collection:'6-l6-teal',
  metrics:[
    {k:'E_read',v:0.331615,note:'4b 基线 · Q03 sealed（metric_dict v4）'},
    {k:'share_max',v:0.03,note:'单头份额 3.0% 内（P4–P7）'},
    {k:'读位槽 G−1',v:5,note:'维（P4–P7）'},
  ],
  activations:ACT_LINES,
  links:[
    {lens:'spatial',text:'水果族簇 · top-5 邻居在 0.31–0.44'},
    {lens:'process',text:'Q05 四臂消融已覆盖 · Q06 steering 计划中'},
    {lens:'progress',text:'支撑 Q05/Q06 节点 · 对标 Attribution Graphs'},
  ],
  queue_refs:['Q03','Q05','Q06'],
  tm_ids:['TM-04'],
};
