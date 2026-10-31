export const AUDIT_3D_FOCUS_EVENT = 'openone:audit-3d-focus';
export const AUDIT_3D_FOCUS_STORAGE_KEY = 'openone.audit3dFocus';

const STAGE_FOCUS_MAP = {
  stage71: {
    stageId: 'stage71',
    stageLabel: '阶段71 统一主核',
    theoryObject: 'protocol_bridge',
    analysisMode: 'cross_layer_transport',
    animationMode: 'protocol_bridge',
    summary: '把高层统一主核映射到协议桥与跨层运输，观察统一结论如何依赖多个下游块。',
  },
  stage73: {
    stageId: 'stage73',
    stageLabel: '阶段73 判伪边界',
    theoryObject: 'admissible_update',
    analysisMode: 'counterfactual',
    animationMode: 'counterfactual_split',
    summary: '把判伪边界映射成反事实分叉，重点看局部更新与边界触发是否来自内部构造。',
  },
  stage80: {
    stageId: 'stage80',
    stageLabel: '阶段80 失效图谱',
    theoryObject: 'restricted_readout',
    analysisMode: 'feature_decomposition',
    animationMode: 'margin_breathing',
    summary: '把失效图谱映射到读出热点与边界呼吸，观察最坏裂缝集中在哪些结构位置。',
  },
  stage81: {
    stageId: 'stage81',
    stageLabel: '阶段81 前后向统一',
    theoryObject: 'stage_conditioned_transport',
    analysisMode: 'cross_layer_transport',
    animationMode: 'stage_transition',
    summary: '把前后向统一映射到阶段条件运输，检查统一是否真的贯穿多层路径。',
  },
  stage82: {
    stageId: 'stage82',
    stageLabel: '阶段82 新颖泛化修复',
    theoryObject: 'successor_aligned_transport',
    analysisMode: 'robustness',
    animationMode: 'successor_transport',
    summary: '把新颖泛化修复映射到后继对齐运输，重点看 sqrt 优势是否足够稳定。',
  },
  stage83: {
    stageId: 'stage83',
    stageLabel: '阶段83 定理主核',
    theoryObject: 'protocol_bridge',
    analysisMode: 'minimal_circuit',
    animationMode: 'minimal_circuit_peeloff',
    summary: '把定理主核映射到最小回路剥离，观察当前闭合是否依赖少量关键节点。',
  },
  stage84: {
    stageId: 'stage84',
    stageLabel: '阶段84 可判伪计算核',
    theoryObject: 'admissible_update',
    analysisMode: 'causal_intervention',
    animationMode: 'ablation_shockwave',
    summary: '把可判伪计算核映射到因果干预与消融冲击波，观察计算核的真实脆弱点。',
  },
  stage87: {
    stageId: 'stage87',
    stageLabel: '阶段87 证据独立性审计',
    theoryObject: 'restricted_readout',
    analysisMode: 'minimal_circuit',
    animationMode: 'prototype_instance_tug',
    summary: '把证据独立性问题映射到关键读出与原型-实例拉扯，帮助观察回灌链是否过重。',
  },
  stage88: {
    stageId: 'stage88',
    stageLabel: '阶段88 外部反例扩展',
    theoryObject: 'relation_context_fiber',
    analysisMode: 'counterfactual',
    animationMode: 'counterfactual_split',
    summary: '把外部反例扩展映射到关系/语境纤维，观察多家族反例如何沿路径分叉。',
  },
};

export function getAudit3DFocus(stageId) {
  return STAGE_FOCUS_MAP[stageId] || {
    stageId: stageId || 'unknown',
    stageLabel: stageId || '未知阶段',
    theoryObject: 'protocol_bridge',
    analysisMode: 'cross_layer_transport',
    animationMode: 'stage_transition',
    summary: '当前阶段没有专门映射，回退到协议桥与跨层运输。',
  };
}

export function persistAudit3DFocus(focus) {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(AUDIT_3D_FOCUS_STORAGE_KEY, JSON.stringify({
      ...focus,
      emittedAt: Date.now(),
    }));
  } catch {
    // ignore localStorage failures
  }
}

export function readPersistedAudit3DFocus() {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(AUDIT_3D_FOCUS_STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function dispatchAudit3DFocus(stageId, extra = {}) {
  if (typeof window === 'undefined') return null;
  const focus = {
    ...getAudit3DFocus(stageId),
    ...extra,
  };
  persistAudit3DFocus(focus);
  window.dispatchEvent(new CustomEvent(AUDIT_3D_FOCUS_EVENT, { detail: focus }));
  return focus;
}
