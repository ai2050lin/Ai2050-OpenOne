import {
  Activity,
  AlertTriangle,
  ArrowDown,
  BrainCircuit,
  CheckCircle2,
  CircleDot,
  Cpu,
  Network,
  Radio,
  ShieldAlert,
} from 'lucide-react';

import { CURRENT_SNN_RESEARCH } from '../../researchKernel/snnResearchState';

import './SNNResearchDashboard.css';

const EFFECT_ICONS = {
  positive: CheckCircle2,
  mixed: CircleDot,
  negative: AlertTriangle,
};

function countActiveSpikes(spikes) {
  if (!spikes || typeof spikes !== 'object') return 0;
  return Object.values(spikes).reduce(
    (total, indices) => total + (Array.isArray(indices) ? indices.length : 0),
    0,
  );
}

export function SNNResearchDashboard({ runtimeState }) {
  const research = CURRENT_SNN_RESEARCH;
  const neuronCount = runtimeState?.structure?.neurons?.length || 0;
  const connectionCount = runtimeState?.structure?.connections?.length || 0;
  const spikeCount = countActiveSpikes(runtimeState?.spikes);

  return (
    <section className="snn-research" aria-label="当前 SNN 研究方案与模型效果">
      <header className="snn-research__header">
        <div className="snn-research__eyebrow">
          <Radio size={12} />
          {research.evidenceLabel} · {research.evidenceDate}
        </div>
        <div className="snn-research__title-row">
          <div>
            <h3>{research.title}</h3>
            <p>{research.id}</p>
          </div>
          <span>{research.status}</span>
        </div>
        <div className="snn-research__tags">
          {research.tags.map((tag) => <span key={tag}>{tag}</span>)}
        </div>
        <div className="snn-research__boundary">
          <ShieldAlert size={13} />
          <span>{research.boundary}</span>
        </div>
      </header>

      <div className="snn-research__section-heading">
        <BrainCircuit size={14} />
        <div>
          <strong>当前方案</strong>
          <span>从事件选择到群体读出的非 Attention 候选主干</span>
        </div>
      </div>

      <div className="snn-research__flow">
        {research.architecture.map((step, index) => (
          <div className="snn-research__flow-item" key={step.id}>
            <div className="snn-research__flow-index">{String(index + 1).padStart(2, '0')}</div>
            <div>
              <strong>{step.label}</strong>
              <code>{step.code}</code>
              <p>{step.detail}</p>
            </div>
            {index < research.architecture.length - 1 && <ArrowDown className="snn-research__flow-arrow" size={12} />}
          </div>
        ))}
      </div>

      <div className="snn-research__regions">
        {research.regions.map((region) => (
          <div key={region.id}>
            <strong>{region.label}</strong>
            <span>{region.role}</span>
          </div>
        ))}
      </div>

      <div className="snn-research__module-list">
        <div className="snn-research__module-title"><Network size={12} /> 系统闭环</div>
        {research.systemModules.map((module) => <span key={module}>{module}</span>)}
      </div>

      <div className="snn-research__rule">
        <strong>{research.learningRule.label}</strong>
        <code>{research.learningRule.formula}</code>
        <p>{research.learningRule.boundary}</p>
      </div>

      <div className="snn-research__section-heading is-effects">
        <Activity size={14} />
        <div>
          <strong>模型效果</strong>
          <span>冻结测试结果；数值不是完整语言能力百分比</span>
        </div>
      </div>

      <div className="snn-research__profile">
        <div><span>实测原型</span><strong>{research.modelProfile.measuredParameters}</strong></div>
        <div><span>功能区域</span><strong>{research.modelProfile.regions}</strong></div>
        <div><span>Patch slots</span><strong>{research.modelProfile.patchSlots}</strong></div>
        <div><span>训练尺度</span><strong>{research.modelProfile.trainedScale}</strong></div>
      </div>

      <div className="snn-research__effects">
        {research.effects.map((effect) => {
          const Icon = EFFECT_ICONS[effect.tone] || CircleDot;
          return (
            <article className={`is-${effect.tone}`} key={effect.id}>
              <div><Icon size={12} /><span>{effect.label}</span></div>
              <strong>{effect.value}</strong>
              <p>{effect.detail}</p>
            </article>
          );
        })}
      </div>

      <div className="snn-research__generation">
        <div><Cpu size={12} /> 真实生成预览</div>
        <code><span>{research.generation.prompt}</span> → {research.generation.output}</code>
        <p>{research.generation.verdict}</p>
      </div>

      <div className="snn-research__scale">
        <strong>规模化路线（尚未实训）</strong>
        <div>
          {research.scalingPlan.map((item) => (
            <span key={item.phase}><b>Phase-{item.phase}</b>{item.parameters}<small>{item.status}</small></span>
          ))}
        </div>
      </div>

      <div className="snn-research__verdicts">
        <div className="is-verified">
          <strong>已经看到</strong>
          {research.verified.map((item) => <span key={item}><CheckCircle2 size={11} />{item}</span>)}
        </div>
        <div className="is-unverified">
          <strong>还没有证明</strong>
          {research.unverified.map((item) => <span key={item}><AlertTriangle size={11} />{item}</span>)}
        </div>
      </div>

      <div className="snn-research__runtime">
        <div>
          <strong>实时 3D LIF 演示</strong>
          <span>旧 NeuroFiber 工具，仅用于观察放电传播，不等于上方 SpikeICSPB 模型。</span>
        </div>
        <div className={runtimeState?.initialized ? 'is-online' : 'is-offline'}>
          <span>{runtimeState?.initialized ? '已初始化' : '未初始化'}</span>
          <b>{neuronCount} neurons</b>
          <b>{connectionCount} edges</b>
          <b>{spikeCount} active</b>
        </div>
      </div>
    </section>
  );
}

