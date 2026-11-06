const SHA = /^[0-9a-f]{64}$/i;
const PROTOCOL = /^([^@\s]+)@([0-9a-f]{64})$/i;
const normalize = text => String(text || '').normalize('NFKC').trim().toLowerCase();

export const SCOPE_FIELDS = [
  { key: 'model', label: '模型快照', placeholder: '完整 SHA256' },
  { key: 'data', label: '数据与划分', placeholder: '冻结语料 SHA256' },
  { key: 'protocol', label: '实验口径', placeholder: '子实验 ID@预注册 SHA256' },
];

export function ruleScope(rule) {
  const evidence = rule.evidence || [];
  const scope = rule.scope || {};
  const models = [...new Set(evidence.map(item => item.model).filter(Boolean))];
  return {
    model: scope.model || models.join(' / '),
    data: scope.data || [...new Set(evidence.map(item => item.data).filter(Boolean))].join(' · '),
    protocol: scope.protocol || rule.criterion,
    modelId: SHA.test(scope.model_snapshot) ? scope.model_snapshot : '',
    dataId: SHA.test(scope.data_sha256) ? scope.data_sha256 : '',
    protocolId: scope.protocol_id && SHA.test(scope.design_sha256) ? `${scope.protocol_id}@${scope.design_sha256}` : '',
  };
}

function tokens(value) {
  const text = normalize(value);
  const terms = new Set(text.match(/[a-z0-9][a-z0-9_-]*/g) || []);
  for (const group of text.match(/[\u4e00-\u9fff]+/g) || []) {
    for (let i = 0; i < group.length - 1; i++) terms.add(group.slice(i, i + 2));
  }
  return terms;
}

function similarity(query, value) {
  const terms = tokens(value);
  if (!query.size || !terms.size) return 0;
  const hits = [...query].filter(term => terms.has(term)).length;
  return hits / Math.sqrt(query.size * terms.size);
}

function compareScope(rule, input) {
  const scope = ruleScope(rule);
  return SCOPE_FIELDS.map(({ key, label }) => {
    const supplied = normalize(input[key]);
    const registered = scope[`${key}Id`];
    const valid = key === 'protocol' ? PROTOCOL.test(supplied) : SHA.test(supplied);
    const outcome = !supplied ? '待填写'
      : !valid ? '需完整身份' : !registered ? '账本缺少可核对身份'
        : supplied === normalize(registered) ? '一致' : '不同';
    return { key, label, outcome, supplied: input[key], registered, description: scope[key] };
  });
}

export function checkRules(rules, query, input) {
  const words = tokens(query);
  const requested = normalize(query);
  const byId = rules.find(rule => normalize(rule.id) === requested);
  const hits = (byId ? [byId] : rules).flatMap(rule => {
    const explicitId = normalize(rule.id) === requested;
    const score = explicitId ? 2 : Math.max(
      similarity(words, [rule.title, rule.claim, rule.topic].join(' ')),
      1.15 * similarity(words, (rule.keywords || []).join(' ')),
    );
    if (score <= 0.06) return [];
    const comparison = compareScope(rule, input);
    const sameScope = comparison.every(item => item.outcome === '一致');
    const mismatched = comparison.some(item => item.outcome === '不同');
    const settled = ['refuted', 'confirmed'].includes(rule.status) && rule.retest?.status === 'block';
    const decision = sameScope && settled ? '同范围已测'
      : mismatched ? '部分重合'
        : '证据不足';
    return [{ rule, score, decision, comparison }];
  });
  return hits.sort((a, b) => Number(b.decision === '同范围已测') - Number(a.decision === '同范围已测') || b.score - a.score).slice(0, 5);
}
