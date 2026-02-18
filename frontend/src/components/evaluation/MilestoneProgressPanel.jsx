import { CheckCircle2, Circle, Loader2 } from 'lucide-react';
import { useMemo } from 'react';

function clamp01(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(1, n));
}

function toPct(value) {
  return Math.round(clamp01(value) * 100);
}

function getRoutes(timeline) {
  return Array.isArray(timeline?.routes) ? timeline.routes : [];
}

function collectStats(routes) {
  const tests = [];
  routes.forEach((routeItem) => {
    const routeTests = Array.isArray(routeItem?.tests) ? routeItem.tests : [];
    routeTests.forEach((test) => {
      tests.push(test);
    });
  });

  const totalRuns = tests.length;
  const completedRuns = tests.filter((t) => t?.status === 'completed').length;
  const failedRuns = tests.filter((t) => t?.status === 'failed').length;
  const completionRate = totalRuns > 0 ? completedRuns / totalRuns : 0;

  const scored = tests
    .map((t) => Number(t?.evaluation?.score))
    .filter((v) => Number.isFinite(v));
  const avgScore = scored.length > 0 ? scored.reduce((a, b) => a + b, 0) / scored.length : 0;

  const analysisRouteMap = new Map();
  tests.forEach((test) => {
    const analysis = test?.analysis_type;
    const route = test?.route;
    if (!analysis || !route) return;
    if (!analysisRouteMap.has(analysis)) {
      analysisRouteMap.set(analysis, new Set());
    }
    analysisRouteMap.get(analysis).add(route);
  });

  const compareCoverage = Array.from(analysisRouteMap.values()).filter(
    (routeSet) => routeSet.size >= 2
  ).length;

  const hasFailureReason = tests.some((t) => Boolean(t?.failure_reason));

  return {
    totalRuns,
    completedRuns,
    failedRuns,
    completionRate,
    avgScore,
    compareCoverage,
    hasFailureReason,
  };
}

function buildMilestones(summary) {
  const dataScale = clamp01(summary.totalRuns / 20);
  const stableScale = clamp01(summary.completionRate / 0.8);
  const feasibilityScale = clamp01(Math.min(summary.avgScore / 0.65, summary.completedRuns / 10));
  const compareScale = clamp01(summary.compareCoverage / 3);
  const governanceScale = summary.failedRuns <= 0 ? 0.6 : summary.hasFailureReason ? 1 : 0.3;

  return [
    {
      key: 'data_volume',
      title: '数据积累',
      desc: `目标 >=20 次测试，当前 ${summary.totalRuns} 次`,
      progress: dataScale,
    },
    {
      key: 'stability',
      title: '执行稳定性',
      desc: `目标完成率 >=80%，当前 ${Math.round(summary.completionRate * 100)}%`,
      progress: stableScale,
    },
    {
      key: 'feasibility',
      title: '可行性证据',
      desc: `目标 avg_score >=0.65 且 completed >=10，当前 ${summary.avgScore.toFixed(3)} / ${summary.completedRuns}`,
      progress: feasibilityScale,
    },
    {
      key: 'ab_coverage',
      title: '路线对照覆盖',
      desc: `目标 >=3 个分析类型可做跨路线对照，当前 ${summary.compareCoverage}`,
      progress: compareScale,
    },
    {
      key: 'failure_governance',
      title: '失败治理',
      desc: summary.failedRuns > 0 ? '目标失败记录带原因，当前已追踪 failure_reason' : '当前无失败记录',
      progress: governanceScale,
    },
  ];
}

export default function MilestoneProgressPanel({ timeline, loading }) {
  const routes = getRoutes(timeline);
  const summary = useMemo(() => collectStats(routes), [routes]);
  const milestones = useMemo(() => buildMilestones(summary), [summary]);

  const weakest = useMemo(() => {
    if (milestones.length === 0) return null;
    return [...milestones].sort((a, b) => a.progress - b.progress)[0];
  }, [milestones]);

  if (loading) {
    return (
      <div className="p-4 rounded-xl border border-white/10 bg-zinc-900/40 text-zinc-400 text-sm flex items-center gap-2">
        <Loader2 size={14} className="animate-spin" />
        正在计算里程碑...
      </div>
    );
  }

  if (routes.length === 0) {
    return (
      <div className="p-4 rounded-xl border border-white/10 bg-zinc-900/40 text-zinc-500 text-sm">
        暂无时间线数据，无法自动更新里程碑。
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-white/10 bg-zinc-900/40 p-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {milestones.map((item) => {
          const done = item.progress >= 1;
          return (
            <div key={item.key} className="rounded-lg border border-white/10 bg-black/20 p-3">
              <div className="flex items-center justify-between mb-1 text-sm">
                <span className="inline-flex items-center gap-1 text-zinc-200">
                  {done ? <CheckCircle2 size={13} className="text-green-400" /> : <Circle size={13} className="text-zinc-500" />}
                  {item.title}
                </span>
                <span className={done ? 'text-green-300 text-xs' : 'text-zinc-400 text-xs'}>
                  {toPct(item.progress)}%
                </span>
              </div>
              <div className="h-1.5 bg-black/30 rounded overflow-hidden mb-2">
                <div
                  className={done ? 'h-full bg-green-500/80' : 'h-full bg-blue-500/80'}
                  style={{ width: `${toPct(item.progress)}%` }}
                />
              </div>
              <div className="text-xs text-zinc-500">{item.desc}</div>
            </div>
          );
        })}
      </div>

      {weakest ? (
        <div className="mt-3 text-xs text-zinc-300">
          当前优先项: <span className="text-amber-300">{weakest.title}</span>
        </div>
      ) : null}
    </div>
  );
}
