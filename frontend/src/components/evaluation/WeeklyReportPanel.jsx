import { Download, FileText, Loader2, RefreshCw } from 'lucide-react';
import { useState } from 'react';
import { API_ENDPOINTS } from '../../config/api';

function downloadText(filename, content, mime = 'text/plain;charset=utf-8') {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

async function fetchWeeklyReport(days, persist) {
  const url = API_ENDPOINTS.runtime.weeklyReport(days, persist);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const payload = await res.json();
  if (payload?.status !== 'success') {
    throw new Error(payload?.message || 'weekly report failed');
  }
  return payload;
}

export default function WeeklyReportPanel() {
  const [days, setDays] = useState(7);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [report, setReport] = useState(null);
  const [markdown, setMarkdown] = useState('');
  const [savedFiles, setSavedFiles] = useState(null);

  const handleGenerate = async (persist = false) => {
    setLoading(true);
    setError(null);
    try {
      const payload = await fetchWeeklyReport(days, persist);
      setReport(payload.report || null);
      setMarkdown(payload.markdown || '');
      setSavedFiles(payload.saved_files || null);
    } catch (err) {
      console.error('weekly report error:', err);
      setError(err.message || 'failed to generate weekly report');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="rounded-xl border border-white/10 bg-zinc-900/40 p-4">
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <div className="text-sm text-zinc-200 inline-flex items-center gap-1">
          <FileText size={14} className="text-emerald-400" />
          自动周报
        </div>
        <select
          className="ml-auto bg-black/40 border border-white/10 rounded px-2 py-1 text-xs text-zinc-200"
          value={days}
          onChange={(e) => setDays(Number(e.target.value))}
        >
          <option value={7}>最近 7 天</option>
          <option value={14}>最近 14 天</option>
          <option value={30}>最近 30 天</option>
        </select>
        <button
          className="inline-flex items-center gap-1 bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-400/30 rounded px-2 py-1 text-xs text-emerald-100"
          onClick={() => handleGenerate(false)}
          disabled={loading}
        >
          {loading ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
          生成
        </button>
        <button
          className="inline-flex items-center gap-1 bg-blue-500/20 hover:bg-blue-500/30 border border-blue-400/30 rounded px-2 py-1 text-xs text-blue-100"
          onClick={() => handleGenerate(true)}
          disabled={loading}
        >
          {loading ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
          生成并落盘
        </button>
      </div>

      {error ? (
        <div className="text-xs text-red-300 mb-2">{error}</div>
      ) : null}

      {!report ? (
        <div className="text-xs text-zinc-500">
          点击“生成”创建周报（JSON + Markdown）。
        </div>
      ) : (
        <div className="space-y-3">
          <div className="text-xs text-zinc-300">
            统计：tests {report?.totals?.total_tests || 0} | completed {report?.totals?.completed_tests || 0} | failed {report?.totals?.failed_tests || 0} | completion {Math.round((Number(report?.totals?.completion_rate || 0)) * 100)}%
          </div>
          <div className="text-xs text-zinc-400">
            亮点：best route {report?.highlights?.best_route || '-'} | most active {report?.highlights?.most_active_route || '-'} | best run {report?.highlights?.best_run_id || '-'}
          </div>

          <div className="space-y-1">
            {(report?.route_summaries || []).slice(0, 5).map((item) => (
              <div key={item.route} className="text-xs text-zinc-300">
                {item.route}: tests {item.total_tests}, avg_score {item.avg_score}
              </div>
            ))}
          </div>

          <div className="flex flex-wrap gap-2">
            <button
              className="inline-flex items-center gap-1 bg-black/30 hover:bg-black/40 border border-white/10 rounded px-2 py-1 text-xs text-zinc-200"
              onClick={() =>
                downloadText(
                  `agi_weekly_report_${report.generated_at?.replace(/[:.]/g, '-') || 'latest'}.json`,
                  JSON.stringify(report, null, 2),
                  'application/json;charset=utf-8'
                )
              }
            >
              <Download size={12} />
              导出 JSON
            </button>
            <button
              className="inline-flex items-center gap-1 bg-black/30 hover:bg-black/40 border border-white/10 rounded px-2 py-1 text-xs text-zinc-200"
              onClick={() =>
                downloadText(
                  `agi_weekly_report_${report.generated_at?.replace(/[:.]/g, '-') || 'latest'}.md`,
                  markdown || '',
                  'text/markdown;charset=utf-8'
                )
              }
            >
              <Download size={12} />
              导出 Markdown
            </button>
          </div>

          {savedFiles?.json_path || savedFiles?.markdown_path ? (
            <div className="text-[11px] text-zinc-500">
              已落盘：{savedFiles?.json_path || '-'} | {savedFiles?.markdown_path || '-'}
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
}
