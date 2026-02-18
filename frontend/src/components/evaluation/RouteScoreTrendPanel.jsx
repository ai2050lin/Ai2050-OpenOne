import { Activity, TrendingUp } from 'lucide-react';
import { useMemo, useState } from 'react';

function toTimestamp(value) {
  const ts = Date.parse(value || '');
  return Number.isFinite(ts) ? ts : 0;
}

function toPct(score) {
  const n = Number(score);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(100, Math.round(n * 100)));
}

function buildPolyline(scores, width = 220, height = 56) {
  if (!scores.length) return '';
  if (scores.length === 1) {
    const y = height - (scores[0] / 100) * height;
    return `0,${y} ${width},${y}`;
  }
  return scores
    .map((value, idx) => {
      const x = (idx / (scores.length - 1)) * width;
      const y = height - (value / 100) * height;
      return `${x},${y}`;
    })
    .join(' ');
}

function pickTests(routeItem, analysisType, maxPoints) {
  const tests = Array.isArray(routeItem?.tests) ? routeItem.tests : [];
  let scoped = tests.filter((test) =>
    analysisType === 'all' ? true : test?.analysis_type === analysisType
  );
  scoped = scoped
    .filter((test) => Number.isFinite(Number(test?.evaluation?.score)))
    .sort((a, b) => toTimestamp(a?.timestamp) - toTimestamp(b?.timestamp));
  if (maxPoints > 0 && scoped.length > maxPoints) {
    scoped = scoped.slice(scoped.length - maxPoints);
  }
  return scoped;
}

export default function RouteScoreTrendPanel({ timeline }) {
  const routes = Array.isArray(timeline?.routes) ? timeline.routes : [];
  const [analysisType, setAnalysisType] = useState('all');
  const [windowSize, setWindowSize] = useState(20);

  const analysisOptions = useMemo(() => {
    const set = new Set();
    routes.forEach((routeItem) => {
      const tests = Array.isArray(routeItem?.tests) ? routeItem.tests : [];
      tests.forEach((test) => {
        if (test?.analysis_type) set.add(test.analysis_type);
      });
    });
    return ['all', ...Array.from(set).sort()];
  }, [routes]);

  const trendData = useMemo(() => {
    return routes
      .map((routeItem) => {
        const scoped = pickTests(routeItem, analysisType, windowSize);
        const points = scoped.map((test) => toPct(test?.evaluation?.score));
        if (points.length === 0) return null;
        const latest = points[points.length - 1];
        const prev = points.length > 1 ? points[points.length - 2] : latest;
        return {
          route: routeItem.route,
          points,
          latest,
          delta: latest - prev,
          lastTime: scoped[scoped.length - 1]?.timestamp,
        };
      })
      .filter(Boolean)
      .sort((a, b) => b.latest - a.latest);
  }, [routes, analysisType, windowSize]);

  if (routes.length === 0) {
    return (
      <div className="p-4 rounded-xl border border-white/10 bg-zinc-900/40 text-zinc-500 text-sm">
        暂无路线数据，无法展示趋势。
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-white/10 bg-zinc-900/40 p-4">
      <div className="flex flex-wrap items-center gap-2 mb-3 text-xs">
        <select
          className="bg-black/40 border border-white/10 rounded px-2 py-1 text-zinc-200"
          value={analysisType}
          onChange={(e) => setAnalysisType(e.target.value)}
        >
          {analysisOptions.map((opt) => (
            <option key={opt} value={opt}>
              {opt === 'all' ? '全部分析' : opt}
            </option>
          ))}
        </select>

        <select
          className="bg-black/40 border border-white/10 rounded px-2 py-1 text-zinc-200"
          value={windowSize}
          onChange={(e) => setWindowSize(Number(e.target.value))}
        >
          <option value={10}>最近 10 次</option>
          <option value={20}>最近 20 次</option>
          <option value={50}>最近 50 次</option>
        </select>
      </div>

      {trendData.length === 0 ? (
        <div className="text-xs text-zinc-500">当前筛选下没有可计算的评分趋势。</div>
      ) : (
        <div className="space-y-3">
          {trendData.map((item) => {
            const up = item.delta >= 0;
            return (
              <div key={item.route} className="rounded-lg border border-white/10 bg-black/20 p-3">
                <div className="flex items-center justify-between mb-2">
                  <div className="inline-flex items-center gap-1 text-zinc-200 text-sm">
                    <Activity size={13} className="text-cyan-400" />
                    {item.route}
                  </div>
                  <div className="text-xs text-zinc-400">
                    {item.lastTime ? new Date(item.lastTime).toLocaleString() : '-'}
                  </div>
                </div>

                <div className="flex items-center gap-4">
                  <svg viewBox="0 0 220 56" className="w-[220px] h-14 shrink-0">
                    <polyline
                      fill="none"
                      stroke={up ? '#22c55e' : '#f43f5e'}
                      strokeWidth="2.5"
                      points={buildPolyline(item.points)}
                    />
                  </svg>
                  <div className="text-xs">
                    <div className="text-zinc-300 inline-flex items-center gap-1">
                      <TrendingUp size={12} className={up ? 'text-green-400' : 'text-rose-400'} />
                      最新评分 {item.latest}%
                    </div>
                    <div className={up ? 'text-green-300 mt-1' : 'text-rose-300 mt-1'}>
                      {up ? '+' : ''}
                      {item.delta}% (对比前一次)
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
