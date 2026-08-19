import {
  Activity,
  AlertCircle,
  BookOpen,
  Brain,
  CheckCircle2,
  Clock,
  TrendingUp,
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import DNNAnalysisPanel from './components/evaluation/DNNAnalysisPanel';
import MilestoneProgressPanel from './components/evaluation/MilestoneProgressPanel';
import ProgressRiskDualAxis from './components/evaluation/ProgressRiskDualAxis';
import RouteABComparePanel from './components/evaluation/RouteABComparePanel';
import RouteScoreTrendPanel from './components/evaluation/RouteScoreTrendPanel';
import RouteTimelineBoard from './components/evaluation/RouteTimelineBoard';
import StageSwimlaneBoard from './components/evaluation/StageSwimlaneBoard';
import TheoryAuditPanel from './components/evaluation/TheoryAuditPanel';
import WeeklyReportPanel from './components/evaluation/WeeklyReportPanel';
import { API_ENDPOINTS } from './config/api';

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function asNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const payload = await res.json();
  return payload;
}

export const AGIProgressDashboard = () => {
  const [progressData, setProgressData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [timeline, setTimeline] = useState(null);
  const [timelineLoading, setTimelineLoading] = useState(true);
  const [timelineError, setTimelineError] = useState(null);
  const [auditData, setAuditData] = useState(null);
  const [auditLoading, setAuditLoading] = useState(true);
  const [auditError, setAuditError] = useState(null);

  useEffect(() => {
    const progressUrl = API_ENDPOINTS.agi.test.replace('/agi/test', '/agi/progress');

    fetchJson(progressUrl)
      .then((payload) => {
        if (payload?.status !== 'success') {
          throw new Error(payload?.message || 'invalid progress payload');
        }
        setProgressData(payload);
      })
      .catch((err) => {
        console.error('progress fetch error:', err);
        setError('无法加载 AGI 研究进度');
      })
      .finally(() => setLoading(false));

    fetchJson(API_ENDPOINTS.runtime.experimentTimeline(120))
      .then((payload) => {
        if (payload?.status !== 'success') {
          throw new Error(payload?.message || 'invalid timeline payload');
        }
        setTimeline(payload.timeline);
      })
      .catch((err) => {
        console.error('timeline fetch error:', err);
        setTimelineError('无法加载路线测试时间线');
      })
      .finally(() => setTimelineLoading(false));

    fetchJson(API_ENDPOINTS.research.auditLatest)
      .then((payload) => {
        if (payload?.status !== 'success') {
          throw new Error(payload?.message || 'invalid audit payload');
        }
        setAuditData(payload.audit || null);
      })
      .catch((err) => {
        console.error('audit fetch error:', err);
        setAuditError('无法加载严格审查结果');
      })
      .finally(() => setAuditLoading(false));
  }, []);

  const phases = useMemo(() => asArray(progressData?.phases), [progressData]);
  const logs = useMemo(() => asArray(progressData?.research_logs), [progressData]);
  const latest = progressData?.latest_test || {};
  const latestAcc = asNumber(latest.val_accuracy, 0);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-zinc-400 gap-2">
        <Activity size={18} className="animate-spin" />
        正在加载 AGI 研究进度...
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-red-400 p-6 text-center gap-2">
        <AlertCircle size={40} />
        <div className="font-bold">加载失败</div>
        <div className="text-sm opacity-80">{error}</div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-[#0a0a0c] overflow-hidden text-zinc-200">
      <div className="p-6 border-b border-white/5 bg-gradient-to-r from-blue-900/10 to-transparent">
        <div className="flex items-center justify-between mb-2">
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <TrendingUp className="text-blue-400" />
            语言机制研究进度中心
          </h1>
          <div className="text-xs text-zinc-500 flex items-center gap-1">
            <Clock size={12} />
            更新时间{' '}
            {progressData?.last_updated
              ? new Date(progressData.last_updated * 1000).toLocaleString()
              : '-'}
          </div>
        </div>
        <p className="text-sm text-zinc-400">
          当前基线：Phase 1236 / C001-WP00。结构发现、行为资格、因果闭合与工程状态分别记账。
        </p>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-8">
        <section>
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <TrendingUp className="text-blue-400" size={18} />
            研究驾驶舱总览
          </h2>
          <div className="grid grid-cols-1 xl:grid-cols-[1.35fr,1fr] gap-4">
            <ProgressRiskDualAxis
              progressData={progressData}
              timeline={timeline}
              auditData={auditData}
            />
            <TheoryAuditPanel
              auditData={auditData}
              loading={auditLoading}
              error={auditError}
            />
          </div>
        </section>

        <section>
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Clock className="text-cyan-400" size={18} />
            阶段泳道总览
          </h2>
          <StageSwimlaneBoard
            phases={phases}
            timeline={timeline}
            auditData={auditData}
            logs={logs}
          />
        </section>

        <section>
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <CheckCircle2 className="text-green-400" size={18} />
            阶段路线图
          </h2>
          {phases.length === 0 ? (
            <div className="p-4 rounded-xl border border-white/10 bg-zinc-900/40 text-zinc-500 text-sm">
              暂无阶段数据
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {phases.map((phase, idx) => {
                const statusText = String(phase?.status || '').toLowerCase();
                const isDone =
                  statusText.includes('done') ||
                  statusText.includes('complete') ||
                  statusText.includes('完成');
                const isInProgress =
                  statusText.includes('progress') ||
                  statusText.includes('进行') ||
                  statusText.includes('active');
                return (
                  <div
                    key={`${phase?.title || 'phase'}-${idx}`}
                    className={`p-4 rounded-xl border transition-all ${
                      isInProgress
                        ? 'bg-blue-900/10 border-blue-500/30'
                        : 'bg-zinc-900/50 border-white/5'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-zinc-400">Phase {idx + 1}</span>
                      <span
                        className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                          isDone
                            ? 'bg-green-500/10 text-green-400 border border-green-500/20'
                            : isInProgress
                              ? 'bg-blue-500/20 text-blue-300 border border-blue-500/30'
                              : 'bg-zinc-800 text-zinc-500 border border-white/5'
                        }`}
                      >
                        {phase?.status || 'unknown'}
                      </span>
                    </div>
                    <div className="font-semibold text-white mb-1">
                      {phase?.title || 'Untitled Phase'}
                    </div>
                    <div className="text-xs text-zinc-400 line-clamp-3">
                      {phase?.summary || '暂无阶段总结'}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>

        <section>
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <CheckCircle2 className="text-emerald-400" size={18} />
            历史曲率演示（非当前机制证据）
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* GPT-2 验证结果 */}
            <div className="p-4 rounded-xl border border-green-500/30 bg-green-900/10">
              <div className="flex items-center justify-between mb-3">
                <span className="font-semibold text-white">GPT-2 Small</span>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-green-500/20 text-green-300 border border-green-500/30">
                  历史记录
                </span>
              </div>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <div className="text-zinc-500 text-xs">参数量</div>
                  <div className="text-zinc-200 font-mono">124M</div>
                </div>
                <div>
                  <div className="text-zinc-500 text-xs">层数 / 维度</div>
                  <div className="text-zinc-200 font-mono">12 / 768</div>
                </div>
                <div>
                  <div className="text-zinc-500 text-xs">平均曲率</div>
                  <div className="text-green-400 font-mono font-bold">0.014</div>
                </div>
                <div>
                  <div className="text-zinc-500 text-xs">曲率范围</div>
                  <div className="text-zinc-200 font-mono">0.000 - 0.050</div>
                </div>
              </div>
              <div className="mt-3 text-xs text-zinc-400">
                生成测试: "The capital of France is" → " the"
              </div>
            </div>

            {/* Qwen2.5 验证结果 */}
            <div className="p-4 rounded-xl border border-blue-500/30 bg-blue-900/10">
              <div className="flex items-center justify-between mb-3">
                <span className="font-semibold text-white">Qwen2.5-0.5B</span>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-blue-500/20 text-blue-300 border border-blue-500/30">
                  历史记录
                </span>
              </div>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <div className="text-zinc-500 text-xs">参数量</div>
                  <div className="text-zinc-200 font-mono">494M</div>
                </div>
                <div>
                  <div className="text-zinc-500 text-xs">层数 / 维度</div>
                  <div className="text-zinc-200 font-mono">24 / 896</div>
                </div>
                <div>
                  <div className="text-zinc-500 text-xs">平均曲率</div>
                  <div className="text-blue-400 font-mono font-bold">0.012</div>
                </div>
                <div>
                  <div className="text-zinc-500 text-xs">曲率范围</div>
                  <div className="text-zinc-200 font-mono">0.008 - 0.015</div>
                </div>
              </div>
              <div className="mt-3 text-xs text-zinc-400">
                生成测试: "The capital of France is" → " Paris"
              </div>
            </div>
          </div>

          {/* 分析结论 */}
          <div className="mt-4 p-4 rounded-xl border border-white/10 bg-zinc-900/40">
            <div className="text-sm font-semibold text-zinc-200 mb-2">证据边界</div>
            <ul className="text-xs text-zinc-400 space-y-1.5">
              <li className="flex items-start gap-2">
                <span className="text-green-400 mt-0.5">•</span>
                <span>这些数值来自早期演示口径，未进入 C001 当前不可变证据注册表。</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-400 mt-0.5">•</span>
                <span>不能由两个模型的局部估计推出“模型越大表示空间越平滑”的尺度规律。</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-violet-400 mt-0.5">•</span>
                <span>曲率读数不能证明流形理论、NFB 或完整语言数学结构；当前理论扩张保持冻结。</span>
              </li>
            </ul>
          </div>
        </section>

        {/* 深度神经网络数学结构分析 */}
        <section>
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Brain className="text-violet-400" size={18} />
            深度神经网络数学结构分析
          </h2>
          <DNNAnalysisPanel />
        </section>

        <section>
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <CheckCircle2 className="text-violet-400" size={18} />
            自动里程碑更新
          </h2>
          <MilestoneProgressPanel timeline={timeline} loading={timelineLoading} />
        </section>

        <section>
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Clock className="text-cyan-400" size={18} />
            路线测试时间线
          </h2>
          <RouteTimelineBoard
            timeline={timeline}
            loading={timelineLoading}
            error={timelineError}
          />
        </section>

        <section>
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <TrendingUp className="text-amber-400" size={18} />
            可行性评分趋势
          </h2>
          <RouteScoreTrendPanel timeline={timeline} />
        </section>

        <section>
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <TrendingUp className="text-cyan-400" size={18} />
            路线 A/B 对照
          </h2>
          <RouteABComparePanel timeline={timeline} />
        </section>

        <section>
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <BookOpen className="text-emerald-400" size={18} />
            周报导出
          </h2>
          <WeeklyReportPanel />
        </section>

        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="bg-zinc-900/50 border border-white/5 rounded-xl p-5">
            <div className="text-sm text-zinc-400 mb-2">最新测试指标</div>
            <div className="text-xs text-zinc-500 mb-1">任务类型</div>
            <div className="text-zinc-200 font-mono mb-3">{latest.task_type || 'N/A'}</div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm text-zinc-500">验证准确率</span>
              <span className="text-2xl font-bold text-green-400">
                {(latestAcc * 100).toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-zinc-800 rounded-full h-1.5 overflow-hidden">
              <div
                className="bg-green-500 h-full"
                style={{ width: `${Math.max(0, Math.min(100, latestAcc * 100))}%` }}
              />
            </div>
          </div>

          <div className="lg:col-span-2">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <BookOpen className="text-orange-400" size={18} />
              研究日志摘要
            </h2>
            <div className="space-y-3">
              {logs.length === 0 ? (
                <div className="p-4 bg-zinc-900/50 border border-white/5 rounded-xl text-sm text-zinc-500">
                  暂无研究日志
                </div>
              ) : (
                logs.map((log, i) => (
                  <div
                    key={`log-${i}`}
                    className="p-4 bg-zinc-900/50 border border-white/5 rounded-xl"
                  >
                    <div className="text-xs text-zinc-500 mb-2">Milestone Update</div>
                    <p className="text-sm text-zinc-300 leading-relaxed">{log}</p>
                  </div>
                ))
              )}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};
