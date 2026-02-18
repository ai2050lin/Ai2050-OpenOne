import { BarChart3, Trophy } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

function toTimestamp(value) {
  const ts = Date.parse(value || '');
  return Number.isFinite(ts) ? ts : 0;
}

function scoreToPct(score) {
  const value = Number(score || 0);
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, Math.round(value * 100)));
}

export default function RouteABComparePanel({ timeline }) {
  const routes = Array.isArray(timeline?.routes) ? timeline.routes : [];

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

  const [selectedAnalysis, setSelectedAnalysis] = useState('all');

  useEffect(() => {
    if (analysisOptions.length === 0) {
      setSelectedAnalysis('all');
      return;
    }
    if (selectedAnalysis === 'all') {
      setSelectedAnalysis(analysisOptions[0]);
      return;
    }
    if (!analysisOptions.includes(selectedAnalysis)) {
      setSelectedAnalysis(analysisOptions[0]);
    }
  }, [analysisOptions, selectedAnalysis]);

  const compared = useMemo(() => {
    return routes
      .map((routeItem) => {
        const tests = Array.isArray(routeItem?.tests) ? routeItem.tests : [];
        const scoped = tests.filter((test) =>
          selectedAnalysis === 'all' ? true : test?.analysis_type === selectedAnalysis
        );
        const sorted = [...scoped].sort(
          (a, b) => toTimestamp(b?.timestamp) - toTimestamp(a?.timestamp)
        );
        const latest = sorted[0];
        if (!latest) return null;
        return {
          route: routeItem.route,
          status: latest.status,
          analysis_type: latest.analysis_type,
          timestamp: latest.timestamp,
          score: Number(latest?.evaluation?.score || 0),
          grade: latest?.evaluation?.grade || '-',
          feasibility: latest?.evaluation?.feasibility || '-',
          summary: latest?.evaluation?.summary || '',
        };
      })
      .filter(Boolean)
      .sort((a, b) => b.score - a.score);
  }, [routes, selectedAnalysis]);

  const winner = compared[0] || null;

  if (routes.length === 0) {
    return (
      <div className="p-4 rounded-xl border border-white/10 bg-zinc-900/40 text-zinc-500 text-sm">
        暂无路线数据，无法执行 A/B 对照。
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-white/10 bg-zinc-900/40 p-4">
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <div className="text-sm text-zinc-300 inline-flex items-center gap-1">
          <BarChart3 size={14} className="text-cyan-400" />
          A/B 路线对照
        </div>
        <select
          className="ml-auto bg-black/40 border border-white/10 rounded px-2 py-1 text-xs text-zinc-200"
          value={selectedAnalysis}
          onChange={(e) => setSelectedAnalysis(e.target.value)}
        >
          {analysisOptions.length === 0 ? (
            <option value="all">无分析类型</option>
          ) : (
            analysisOptions.map((analysisType) => (
              <option key={analysisType} value={analysisType}>
                {analysisType}
              </option>
            ))
          )}
        </select>
      </div>

      {winner ? (
        <div className="mb-3 text-xs text-zinc-300 inline-flex items-center gap-2">
          <Trophy size={12} className="text-amber-300" />
          当前领先路线：<span className="font-semibold text-white">{winner.route}</span>
          <span className="text-zinc-400">({scoreToPct(winner.score)}%)</span>
        </div>
      ) : null}

      {compared.length === 0 ? (
        <div className="text-xs text-zinc-500">当前分析类型下暂无可比较结果。</div>
      ) : (
        <div className="space-y-2">
          {compared.map((item) => (
            <div key={`${item.route}-${item.analysis_type}`} className="rounded-lg border border-white/10 bg-black/20 p-3">
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-zinc-200 font-semibold">{item.route}</span>
                <span className="text-zinc-400">{new Date(item.timestamp).toLocaleString()}</span>
              </div>
              <div className="text-xs text-zinc-400 mb-2">
                状态 {item.status} | 等级 {item.grade} | 可行性 {item.feasibility}
              </div>
              <div className="h-2 bg-black/30 rounded overflow-hidden mb-1">
                <div className="h-full bg-cyan-500/80" style={{ width: `${scoreToPct(item.score)}%` }} />
              </div>
              <div className="text-xs text-zinc-300">评分 {scoreToPct(item.score)}%</div>
              {item.summary ? (
                <div className="text-xs text-zinc-500 mt-1 line-clamp-2">{item.summary}</div>
              ) : null}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
