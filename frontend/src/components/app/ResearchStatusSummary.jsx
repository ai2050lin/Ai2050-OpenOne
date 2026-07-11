import { Activity, Database, Network } from 'lucide-react';

import {
  dedupePatternAtlasUnits,
  patternAtlasEvidenceLabel,
  patternAtlasNodeColor,
  patternAtlasUnitAddressLabel,
} from '../../researchKernel/patternAtlasEvidence';

import './ResearchStatusSummary.css';

const EVIDENCE_LABELS = {
  key: '关键候选',
  natural: '自然交叉',
  group: '组级支持',
  confirmed: '扩大确认',
};

function display(value, fallback = '-') {
  return value === undefined || value === null || value === '' ? fallback : value;
}

export function ResearchStatusSummary({
  sourceMode,
  modelLabel,
  effectiveReplayRun,
  liveJob,
  apiAvailable,
  activeRunId,
  traceReady,
  traceLoading,
  traceError,
  playing,
  currentLayer,
  currentSubphase,
  currentSubphaseId,
  layerCount,
  layerData,
  overlayEnabled,
  atlas,
  evidenceFocus,
  maxUnits,
  atlasUnits = [],
}) {
  const metrics = atlas.partition?.metrics;
  const traceStatus = traceLoading
    ? '读取中'
    : traceError
      ? '读取失败'
      : traceReady
        ? '真实组件 Trace 已就绪'
        : 'Trace 未就绪';
  const playbackStatus = currentLayer == null ? '未开始' : playing ? '播放中' : '已暂停';
  const mappingStatus = !overlayEnabled
    ? '叠层已隐藏'
    : atlas.loading
      ? '正在加载证据'
      : atlas.mapped
        ? '真实物理候选已叠加'
        : '真实单元尚未映射';
  const run = sourceMode === 'replay' ? effectiveReplayRun : liveJob;
  const currentComponent = ['qkv', 'attn_score', 'softmax', 'attn_out'].includes(currentSubphaseId)
    ? 'attention'
    : ['ffn_up', 'ffn_act', 'ffn_down'].includes(currentSubphaseId)
      ? 'mlp'
      : null;
  const currentAtlasUnits = overlayEnabled && currentLayer != null
    ? dedupePatternAtlasUnits(atlasUnits.filter((node) => (
        Number(node.layer) === Number(currentLayer)
        && (!currentComponent || node.component === currentComponent)
      )))
    : [];

  return (
    <div className="research-status-summary">
      <section className="research-status-summary__section">
        <h3><Database size={14} />运行与 Trace</h3>
        <dl>
          <dt>运行来源</dt><dd>{sourceMode === 'live' ? '实时分析' : '证据回放'}</dd>
          <dt>模型</dt><dd>{display(modelLabel)}</dd>
          <dt>Run</dt><dd className="is-mono">{display(activeRunId)}</dd>
          <dt>Trace</dt><dd className={traceError ? 'is-error' : traceReady ? 'is-ready' : ''}>{traceStatus}</dd>
          <dt>API</dt><dd>{apiAvailable ? '已连接' : '静态回放降级'}</dd>
          <dt>证据</dt><dd>{display(run?.evidence_level, sourceMode === 'replay' ? 'L2' : '未发布')}</dd>
          <dt>事件数</dt><dd>{display(run?.event_count, 0)}</dd>
          <dt>状态</dt><dd>{sourceMode === 'replay' ? (run?.validated === false ? '未校验' : '冻结证据') : display(run?.status, '待创建')}</dd>
        </dl>
        {traceError && <p className="research-status-summary__error">{traceError}</p>}
      </section>

      <section className="research-status-summary__section">
        <h3><Activity size={14} />统一播放</h3>
        <dl>
          <dt>播放状态</dt><dd>{playbackStatus}</dd>
          <dt>当前位置</dt><dd>{currentLayer == null ? '-' : `L${currentLayer} · ${currentSubphase}`}</dd>
          <dt>层进度</dt><dd>{currentLayer == null ? `0/${layerCount}` : `${currentLayer + 1}/${layerCount}`}</dd>
          <dt>Attention</dt><dd>{layerData?.attention?.norm?.toFixed?.(2) || '-'}</dd>
          <dt>MLP</dt><dd>{layerData?.ffn?.norm?.toFixed?.(2) || '-'}</dd>
          <dt>Residual</dt><dd>{layerData?.residual_norm?.toFixed?.(2) || '-'}</dd>
          <dt>图谱物理单元</dt><dd>{currentAtlasUnits.length}</dd>
        </dl>
        {currentAtlasUnits.length > 0 && (
          <div className="research-status-summary__unit-list" aria-label="当前组件图谱物理单元">
            {currentAtlasUnits.slice(0, 12).map((node) => (
              <span
                key={`${node.model}:${node.layer}:${node.component}:${node.unit_kind}:${node.unit_index}`}
                className="research-status-summary__unit"
                style={{ '--unit-color': patternAtlasNodeColor(node) }}
                title={`L${node.layer} ${node.component} ${patternAtlasEvidenceLabel(node)}`}
              >
                {patternAtlasUnitAddressLabel(node)}
              </span>
            ))}
            {currentAtlasUnits.length > 12 && (
              <span className="research-status-summary__unit-more">+{currentAtlasUnits.length - 12}</span>
            )}
          </div>
        )}
      </section>

      <section className="research-status-summary__section">
        <h3><Network size={14} />语言模式族物理叠层</h3>
        <dl>
          <dt>叠层状态</dt><dd>{mappingStatus}</dd>
          <dt>模式族</dt><dd>{display(atlas.family?.family_name)}</dd>
          <dt>模型分区</dt><dd>{display(atlas.model)}</dd>
          <dt>证据范围</dt><dd>{display(EVIDENCE_LABELS[evidenceFocus])}</dd>
          <dt>显示候选</dt><dd>{maxUnits}</dd>
          <dt>关键层</dt><dd>{metrics?.candidate_layer_count || 0}</dd>
          <dt>唯一候选</dt><dd>{metrics?.unique_unit_count || 0}</dd>
          <dt>自然交叉</dt><dd>{metrics?.natural_overlap_count || 0}</dd>
          <dt>组级支持</dt><dd>{metrics?.group_supported_candidate_count || 0}</dd>
          <dt>扩大确认</dt><dd>{metrics?.expanded_confirmed_candidate_count || 0}</dd>
        </dl>
        <p className="research-status-summary__boundary">证据边界：单神经元因果 0；同屏高亮不等于因果闭合。</p>
      </section>
    </div>
  );
}
