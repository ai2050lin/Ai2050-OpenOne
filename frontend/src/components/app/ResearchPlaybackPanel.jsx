import {
  Activity,
  Brain,
  Database,
  History,
  Pause,
  Play,
  Square,
  StepForward,
} from 'lucide-react';

import { PatternFamilyAtlasControls } from './PatternFamilyAtlasControls';
import './ResearchPlaybackPanel.css';

const MODEL_OPTIONS = [
  { value: 'qwen3-4b', label: 'Qwen3-4B', detail: '36L · d2560' },
  { value: 'glm4-9b', label: 'GLM4-9B-Chat', detail: '40L · d4096' },
  { value: 'ds7b', label: 'DeepSeek-R1-7B', detail: '28L · d3584' },
];

const COLOR_OPTIONS = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'brown', 'black', 'white', 'gray', 'silver'];

export function ResearchPlaybackPanel({
  sourceMode,
  onSourceModeChange,
  modelKey,
  onModelChange,
  prompt,
  onPromptChange,
  targetLabel,
  onTargetLabelChange,
  replayRuns,
  replayRunId,
  effectiveReplayRun,
  onReplayRunChange,
  liveJob,
  liveTraceReusable,
  traceReady,
  traceLoading,
  apiAvailable,
  currentLayer,
  layerCount,
  playing,
  speed,
  onSpeedChange,
  onRun,
  onStop,
  onStep,
  overlayProps,
}) {
  const busy = sourceMode === 'live' && ['queued', 'running'].includes(liveJob?.status);
  const canRun = sourceMode === 'replay'
    ? Boolean(effectiveReplayRun && traceReady && !traceLoading)
    : Boolean(prompt.trim() && targetLabel && apiAvailable && !busy);
  const progress = currentLayer == null ? 0 : ((currentLayer + 1) / Math.max(1, layerCount)) * 100;

  return (
    <div className="research-playback">
      <section className="research-playback__section">
        <div className="research-playback__label"><Activity size={14} />运行来源</div>
        <div className="research-playback__source-tabs" role="group" aria-label="运行来源">
          <button type="button" className={sourceMode === 'live' ? 'is-active' : ''} onClick={() => onSourceModeChange('live')}>
            <Brain size={14} />实时分析
          </button>
          <button type="button" className={sourceMode === 'replay' ? 'is-active' : ''} onClick={() => onSourceModeChange('replay')}>
            <History size={14} />证据回放
          </button>
        </div>
      </section>

      <section className="research-playback__section research-playback__config">
        {sourceMode === 'live' ? (
          <>
            <label>
              <span>模型</span>
              <select value={modelKey} onChange={(event) => onModelChange(event.target.value)}>
                {MODEL_OPTIONS.map((option) => <option key={option.value} value={option.value}>{option.label} · {option.detail}</option>)}
              </select>
            </label>
            <label>
              <span>输入语句</span>
              <input value={prompt} onChange={(event) => onPromptChange(event.target.value)} placeholder="输入要捕获的提示词" />
            </label>
            <label>
              <span>目标颜色</span>
              <select value={targetLabel} onChange={(event) => onTargetLabelChange(event.target.value)}>
                {COLOR_OPTIONS.map((color) => <option key={color} value={color}>{color}</option>)}
              </select>
            </label>
            <div className="research-playback__source-status">
              <span className={apiAvailable ? 'is-ready' : 'is-error'}>{apiAvailable ? 'Trace API 已连接' : 'Trace API 未连接'}</span>
              <span>{liveJob?.status || '待创建 run'}</span>
            </div>
          </>
        ) : (
          <>
            <label>
              <span>冻结实验</span>
              <select value={effectiveReplayRun?.run_id || replayRunId || ''} onChange={(event) => onReplayRunChange(event.target.value)}>
                {replayRuns.map((run) => (
                  <option key={run.run_id} value={run.run_id}>{run.model} · {run.run_id}</option>
                ))}
              </select>
            </label>
            <div className="research-playback__run-meta">
              <span>{effectiveReplayRun?.evidence_level || 'L2'}</span>
              <span>{effectiveReplayRun?.event_count || 0} events</span>
              <span>{effectiveReplayRun?.validated === false ? '未校验' : '冻结证据'}</span>
            </div>
          </>
        )}
      </section>

      <section className="research-playback__section">
        <PatternFamilyAtlasControls {...overlayProps} variant="panel" showModel={false} showDetails={false} modelKey={modelKey} onModelChange={onModelChange} />
      </section>

      <section className="research-playback__section research-playback__transport">
        <div className="research-playback__label"><Database size={14} />统一播放</div>
        <div className="research-playback__buttons">
          <button type="button" onClick={onRun} disabled={playing || !canRun} title={playing ? '正在运行' : '运行'}>
            {playing || busy ? <Pause size={15} /> : <Play size={15} />}
            {busy ? '捕获中' : playing ? '播放中' : sourceMode === 'live' && !liveTraceReusable ? '创建 Trace' : '运行'}
          </button>
          <button type="button" onClick={onStop} disabled={currentLayer == null && !busy} title="停止">
            <Square size={14} />停止
          </button>
          <button type="button" onClick={onStep} disabled={currentLayer == null || busy} title="下一步">
            <StepForward size={15} />下一步
          </button>
        </div>

        <label className="research-playback__speed">
          <span>速度</span>
          <input type="range" min="200" max="2000" step="100" value={speed} onChange={(event) => onSpeedChange(Number(event.target.value))} />
          <output>{speed}ms</output>
        </label>

        <div className="research-playback__progress"><span style={{ width: `${progress}%` }} /></div>
      </section>
    </div>
  );
}
