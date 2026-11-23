// Offline adapter for archived observations. All indices are zero based.
(() => {
  'use strict';
  const {modelRecords, modelExplorerData, explorerSystemField: raw, AI2050HeadArchive: headZ} = window;
  const family = modelExplorerData;
  const sameRun = (a,b) => a.run === b.run && a.snapshot === b.snapshot;
  function recordFor(run, id) {
    const record = modelRecords.find(r => sameRun(r,run) && r.caseId === id);
    if (!record) throw Error('找不到运行、模型快照和题号都匹配的归档。');
    return record;
  }

  if (!sameRun(family,headZ) || headZ.block !== 26 || headZ.head !== 36 || headZ.boundary !== 'head_z')
    throw Error('注意力来源与头通道归档的身份不匹配。');

  const translations = {
    'confirm-000-00':'在当前档案中，找出名字为 Tebaka 的记录的代码。',
    'confirm-000-01':'在当前档案中，找出印刷行标签为 ROW D 的记录的代码。',
    'confirm-000-02':'在当前档案中，找出 BEGIN 正下方第一条数据记录的代码。'
  };
  const headCases = family.cases.map(c => {
    const record = recordFor(family,c.id);
    const z = headZ.cases[c.id];
    if (c.rows.length !== 5 || !c.rows[c.gold] || !z || z.values.length !== headZ.headWidth)
      throw Error('该题的记录索引或头通道向量不完整。');
    const winner = c.rows.reduce((best,row,i) => row.mass > c.rows[best].mass ? i : best,0);
    return {
      ...c, recordId:record.id, record, question:record.question, questionZh:translations[c.id],
      winner, position:z.position, tokenId:z.tokenId, z:z.values
    };
  });

  const record = recordFor(raw,raw.caseId);
  const [hiddenLayers,sites,hiddenWidth] = raw.hiddenShape;
  const [ffnSites,width] = raw.mlpShape;
  const observed = family.feature;
  if (observed.run !== raw.run || observed.caseId !== raw.caseId || raw.block !== observed.block ||
      ffnSites !== sites || observed.positions.length !== sites ||
      observed.positions.some((p,i) => p.index !== raw.positions[i].index))
    throw Error('FFN 归档与位置标签或组干预记录不一致。');

  const nativeRoot = 'tests/glm5/result/b36_transition14_20261114_v1/';
  const headRoot = 'tests/glm5/result/clt_b26_three_cues_v3/';
  const ffn = {
    record, positions:raw.positions.map((p,i) => ({...p,label:observed.positions[i].name})),
    block:raw.block, width, hiddenLayers, hiddenWidth,
    // Exact frozen text in plan.json rows[id=t01eb4d20ccb2964:counted0].prompt.
    prompt:"<|im_start|>user\nExecute these Python assignments in order:\n```python\nx = 22\nx -= 2\nx += 7\n```\nOptions:\nRow 1: value 27, code A\nRow 2: value 28, code B\nRow 3: value 20, code C\nRow 4: value 22, code D\nWrite exactly four lines: \"Steps: \" then values after each update separated by commas (exclude the initial assignment); \"Value: \" then the final numeric value; \"Slot: \" then the row number whose value equals that final value; \"Code: \" then the code of the row whose value equals that final value. There are exactly 2 update statements. Steps must contain exactly 2 values, one per update statement. Stop immediately after the requested lines. No explanation or code fences.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    questionZh:'按顺序计算 x = 22、x 减 2、x 加 7，按表查出最终数值所在行的代码，并分四行回答。',
    prefix:record.question, output:'A', unit:observed.unit,
    groupSize:observed.groupSize, baseCorrect:observed.baseCorrect, blockedCorrect:observed.blockedCorrect, total:observed.total,
    boundary:'B36 的 down_proj 输入，逐单元记录 SiLU(gate) × up。',
    hiddenBoundary:'L0 记录输入嵌入，L1–L40 记录各块输出的残差状态；L40 在最终归一化之前。',
    limit:'四个位置来自同一道题。固定 32 单元组的阻断和基线在 92 个配对家族上都答对 90 个；U2429 的单独因果作用尚未检验。',
    sources:[
      {label:'冻结题目与原始输入',path:nativeRoot+'plan.json',sha256:'cb573fa357ddd56751c00d079d0c44b3ff3d54d8d4119c80293685403f31f4ab'},
      {label:'模型生成原文',path:nativeRoot+'counted0.json'},
      {label:'原场与干预记录',path:nativeRoot+'native/confirm-000/record.json',sha256:raw.source.recordSha256},
      {label:'原生向量',path:nativeRoot+raw.source.file,sha256:raw.source.sha256},
      {label:'固定单元组干预汇总',path:nativeRoot+'native-feature-card.json'}
    ]
  };

  // Expand the retained BF16 bits once, only when the corresponding view is opened.
  const fields = {};
  function decode(key, shape) {
    if (fields[key]) return fields[key];
    const size = shape.reduce((n,x) => n*x,1), binary = atob(raw[key]);
    if (binary.length !== size*2) throw Error('归档原场长度与声明形状不一致。');
    const values = new Float32Array(size), words = new Uint32Array(values.buffer);
    for (let i=0;i<size;i++) {
      const hi = binary.charCodeAt(2*i+1), lo = binary.charCodeAt(2*i);
      words[i] = ((hi << 8) | lo) << 16;
    }
    fields[key] = values;
    return values;
  }
  const indexFits = (i,n) => Number.isInteger(i) && i>=0 && i<n;
  function ffnValues(site) {
    if (!indexFits(site,sites)) return null;
    return decode('mlp',raw.mlpShape).subarray(site*width,(site+1)*width);
  }
  function hiddenValues(layer,site) {
    if (!indexFits(layer,hiddenLayers) || !indexFits(site,sites)) return null;
    const start = (layer*sites+site)*hiddenWidth;
    return decode('hidden',raw.hiddenShape).subarray(start,start+hiddenWidth);
  }

  const head = {
    model:headCases[0].record.model, run:headZ.run,snapshot:headZ.snapshot,
    block:headZ.block, index:headZ.head, width:headZ.headWidth,
    boundary:'最后输入 token 对每一行内所有 token 的注意力总和。Z 取该位置进入 o_proj 前的 128 个通道。',
    limit:'物理首行条件下模型答出 silver，注意力峰值在第 4 行。单头的注意力峰值不足以解释此题的答案。',
    cohort:{familyCount:128,caseCount:1536,scope:'四种交换格分别问三种规则。表中按规则统计五行中目标记录的注意力最大比例。',
      fractions:[0.990234375,0.998046875,0.333984375],metric:'natural[].all_five_gold'},
    sources:[
      {label:'冻结材料',path:headCases[0].record.materialSource},
      {label:'确认集采集清单与分片',path:headRoot+'confirm-collection.json',sha256:headCases[0].record.artifactSha},
      {label:'头内 Z 通道',path:headZ.source,sha256:headZ.sha256},
      {label:'原题注意力、输出与确认家族汇总',path:'frontend/src/components/v8/threeCueEvidence.json'}
    ]
  };
  // Chinese paired archive entries belong to a separate snapshot and measure.
  const chinesePairs = modelRecords.filter(r => r.series==='Claude' && r.promptType==='原始最小句对');
  window.EncodingEvidence = Object.freeze({headCases,head,ffn,ffnValues,hiddenValues,chinesePairs});
})();
