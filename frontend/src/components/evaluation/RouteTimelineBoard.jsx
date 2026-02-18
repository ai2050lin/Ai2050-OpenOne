import {
  AlertCircle,
  CheckCircle2,
  Clock3,
  Download,
  Route,
  XCircle,
} from 'lucide-react';
import { useMemo, useState } from 'react';

function formatTime(iso) {
  if (!iso) return '-';
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function getStatusColor(status) {
  if (status === 'completed') return '#22c55e';
  if (status === 'failed') return '#ef4444';
  if (status === 'running') return '#3b82f6';
  return '#a1a1aa';
}

function toTimestamp(iso) {
  const ts = Date.parse(iso || '');
  return Number.isFinite(ts) ? ts : 0;
}

function exportJsonFile(filename, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: 'application/json;charset=utf-8',
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function filterByTimeWindow(test, timeWindow) {
  if (timeWindow === 'all') return true;
  const ts = toTimestamp(test.timestamp);
  if (ts <= 0) return false;
  const now = Date.now();
  const days = timeWindow === '7d' ? 7 : 30;
  return now - ts <= days * 24 * 3600 * 1000;
}

export default function RouteTimelineBoard({ timeline, loading, error }) {
  const [selectedRoute, setSelectedRoute] = useState('all');
  const [statusFilter, setStatusFilter] = useState('all');
  const [analysisFilter, setAnalysisFilter] = useState('all');
  const [timeWindow, setTimeWindow] = useState('all');

  const routes = useMemo(
    () => (Array.isArray(timeline?.routes) ? timeline.routes : []),
    [timeline]
  );

  const analysisOptions = useMemo(() => {
    const set = new Set();
    routes.forEach((routeItem) => {
      const tests = Array.isArray(routeItem?.tests) ? routeItem.tests : [];
      tests.forEach((test) => {
        if (test?.analysis_type) set.add(test.analysis_type);
      });
    });
    return Array.from(set).sort();
  }, [routes]);

  const filteredRoutes = useMemo(() => {
    const routeSubset =
      selectedRoute === 'all'
        ? routes
        : routes.filter((r) => r.route === selectedRoute);

    return routeSubset
      .map((routeItem) => {
        const tests = Array.isArray(routeItem?.tests) ? routeItem.tests : [];
        const filteredTests = tests.filter((test) => {
          if (statusFilter !== 'all' && test?.status !== statusFilter) return false;
          if (analysisFilter !== 'all' && test?.analysis_type !== analysisFilter) return false;
          if (!filterByTimeWindow(test, timeWindow)) return false;
          return true;
        });
        return { ...routeItem, tests: filteredTests };
      })
      .filter((routeItem) => routeItem.tests.length > 0);
  }, [routes, selectedRoute, statusFilter, analysisFilter, timeWindow]);

  const filteredSummary = useMemo(() => {
    const routeCount = filteredRoutes.length;
    const testCount = filteredRoutes.reduce(
      (sum, routeItem) => sum + routeItem.tests.length,
      0
    );
    return { routeCount, testCount };
  }, [filteredRoutes]);

  const failureBuckets = useMemo(() => {
    const counts = new Map();
    filteredRoutes.forEach((routeItem) => {
      const tests = Array.isArray(routeItem.tests) ? routeItem.tests : [];
      tests.forEach((test) => {
        const isFailure = test?.status === 'failed' || Boolean(test?.failure_reason) || Boolean(test?.error);
        if (!isFailure) return;
        const rawReason =
          test?.failure_reason ||
          test?.error ||
          (test?.evaluation && test.evaluation.summary) ||
          'unknown_failure';
        const reason = String(rawReason).trim() || 'unknown_failure';
        counts.set(reason, (counts.get(reason) || 0) + 1);
      });
    });
    return Array.from(counts.entries())
      .map(([reason, count]) => ({ reason, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 8);
  }, [filteredRoutes]);

  if (loading) {
    return (
      <div className="p-4 rounded-xl border border-white/10 bg-zinc-900/40 text-zinc-400 text-sm flex items-center gap-2">
        <Clock3 size={14} className="animate-spin" />
        正在加载路线测试时间线...
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 rounded-xl border border-red-500/30 bg-red-900/10 text-red-300 text-sm flex items-center gap-2">
        <AlertCircle size={14} />
        {error}
      </div>
    );
  }

  if (routes.length === 0) {
    return (
      <div className="p-4 rounded-xl border border-white/10 bg-zinc-900/40 text-zinc-500 text-sm">
        暂无测试记录。运行一次结构分析或路线测试后会自动写入 JSON 时间线。
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-white/10 bg-zinc-900/40 p-3">
        <div className="flex flex-wrap gap-2 items-center text-xs">
          <select
            className="bg-black/40 border border-white/10 rounded px-2 py-1 text-zinc-200"
            value={selectedRoute}
            onChange={(e) => setSelectedRoute(e.target.value)}
          >
            <option value="all">全部路线</option>
            {routes.map((routeItem) => (
              <option key={routeItem.route} value={routeItem.route}>
                {routeItem.route}
              </option>
            ))}
          </select>

          <select
            className="bg-black/40 border border-white/10 rounded px-2 py-1 text-zinc-200"
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
          >
            <option value="all">全部状态</option>
            <option value="completed">completed</option>
            <option value="failed">failed</option>
            <option value="running">running</option>
            <option value="pending">pending</option>
          </select>

          <select
            className="bg-black/40 border border-white/10 rounded px-2 py-1 text-zinc-200"
            value={analysisFilter}
            onChange={(e) => setAnalysisFilter(e.target.value)}
          >
            <option value="all">全部分析</option>
            {analysisOptions.map((analysisType) => (
              <option key={analysisType} value={analysisType}>
                {analysisType}
              </option>
            ))}
          </select>

          <select
            className="bg-black/40 border border-white/10 rounded px-2 py-1 text-zinc-200"
            value={timeWindow}
            onChange={(e) => setTimeWindow(e.target.value)}
          >
            <option value="all">全部时间</option>
            <option value="7d">最近 7 天</option>
            <option value="30d">最近 30 天</option>
          </select>

          <button
            className="ml-auto inline-flex items-center gap-1 bg-blue-500/20 hover:bg-blue-500/30 border border-blue-400/30 rounded px-2 py-1 text-blue-200"
            onClick={() =>
              exportJsonFile('agi_filtered_timeline.json', {
                exported_at: new Date().toISOString(),
                filters: {
                  route: selectedRoute,
                  status: statusFilter,
                  analysis_type: analysisFilter,
                  time_window: timeWindow,
                },
                routes: filteredRoutes,
              })
            }
          >
            <Download size={12} />
            导出筛选结果
          </button>
        </div>

        <div className="mt-2 text-xs text-zinc-400">
          当前筛选：{filteredSummary.routeCount} 条路线，{filteredSummary.testCount} 次测试
        </div>
      </div>

      <div className="rounded-xl border border-white/10 bg-zinc-900/40 p-3">
        <div className="text-xs text-zinc-400 mb-2">失败原因聚合（按当前筛选）</div>
        {failureBuckets.length === 0 ? (
          <div className="text-xs text-zinc-500">未发现失败记录</div>
        ) : (
          <div className="space-y-2">
            {failureBuckets.map((item) => (
              <div key={item.reason} className="text-xs">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-zinc-300 truncate pr-2">{item.reason}</span>
                  <span className="text-red-300">{item.count}</span>
                </div>
                <div className="h-1.5 bg-black/30 rounded overflow-hidden">
                  <div
                    className="h-full bg-red-500/70"
                    style={{
                      width: `${Math.min(100, (item.count / Math.max(1, failureBuckets[0].count)) * 100)}%`,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {filteredRoutes.length === 0 ? (
        <div className="p-4 rounded-xl border border-white/10 bg-zinc-900/40 text-zinc-500 text-sm">
          当前筛选条件下没有结果。
        </div>
      ) : (
        filteredRoutes.map((routeItem) => {
          const tests = Array.isArray(routeItem.tests) ? routeItem.tests : [];
          const stats = routeItem.stats || {};
          return (
            <div key={routeItem.route} className="p-4 rounded-xl border border-white/10 bg-zinc-900/40">
              <div className="flex items-center justify-between mb-3 gap-2">
                <div className="flex items-center gap-2 text-white">
                  <Route size={14} className="text-blue-400" />
                  <span className="font-semibold">{routeItem.route}</span>
                </div>
                <div className="text-xs text-zinc-400">
                  总计 {stats.total_runs || 0} | 成功 {stats.completed_runs || 0} | 失败 {stats.failed_runs || 0} | 平均评分 {stats.avg_score ?? 0}
                </div>
              </div>

              <div className="mb-3">
                <button
                  className="inline-flex items-center gap-1 bg-black/30 hover:bg-black/40 border border-white/10 rounded px-2 py-1 text-xs text-zinc-300"
                  onClick={() =>
                    exportJsonFile(`agi_route_${routeItem.route}_timeline.json`, {
                      exported_at: new Date().toISOString(),
                      route: routeItem.route,
                      stats: routeItem.stats || {},
                      tests,
                    })
                  }
                >
                  <Download size={12} />
                  导出该路线
                </button>
              </div>

              <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
                {tests.map((test) => {
                  const evaluation = test.evaluation || {};
                  const scorePct = Math.round(Number(evaluation.score || 0) * 100);
                  return (
                    <div key={test.test_id} className="rounded-lg border border-white/10 bg-black/20 p-3">
                      <div className="flex items-center justify-between text-xs mb-2">
                        <span style={{ color: getStatusColor(test.status) }} className="font-semibold">
                          {test.status === 'completed' ? (
                            <CheckCircle2 size={12} className="inline mr-1" />
                          ) : null}
                          {test.status === 'failed' ? (
                            <XCircle size={12} className="inline mr-1" />
                          ) : null}
                          {test.status}
                        </span>
                        <span className="text-zinc-400">{formatTime(test.timestamp)}</span>
                      </div>

                      <div className="text-sm text-zinc-200">
                        <span className="text-zinc-400">分析:</span> {test.analysis_type}
                      </div>
                      <div className="text-xs text-zinc-400 mt-1">
                        run_id: {test.run_id} | events: {test.event_count || 0}
                      </div>
                      <div className="mt-2 text-xs text-zinc-300">
                        评估: 等级 {evaluation.grade || '-'} | 可行性 {evaluation.feasibility || '-'} | 评分 {scorePct}%
                      </div>
                      <div className="mt-1 text-xs text-zinc-500 line-clamp-2">
                        {evaluation.summary || '暂无评估摘要'}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })
      )}
    </div>
  );
}
