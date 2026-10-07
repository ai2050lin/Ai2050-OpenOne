/* ============================================================
   RdcFusionWorkspace demo 数据层 —— 全部为真实研究数值
   接入点：
   - EVENTS / TICKER ← research OS 事件总线 / .workbuddy ledger
   - ACT_LINES       ← collect.npz（F#3734 top 激活 token，值→emerald 透明度）
   - CMDK_GROUPS     ← atlas_ledger.json / phase_queue / industry.json
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
  '[queue] 6/30 sealed · next Q06 C_steer',
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
    {k:'Q06',t:'C_steer 基座测量 · RUNNING',d:'step 2/5',go:'process'},
    {k:'Q05',t:'E_ar(k) 正式测量 · SEALED',d:'1acb1e78',go:'process'},
  ]},
  {gl:'论文与节点 · PROGRESS',items:[
    {k:'paper',t:'Circuit Tracing: Attribution Graphs',d:'Anthropic 2025',go:'progress'},
    {k:'P35',t:'A 闸门 seal（R8）',d:'Q08=甲 · C 全接受',go:'progress'},
  ]},
];
