
import {
    Activity,
    AlertCircle,
    BarChart3,
    BookOpen,
    CheckCircle2,
    Clock,
    FileText,
    Search,
    TrendingUp
} from 'lucide-react';
import { useEffect, useState } from 'react';

export const AGIProgressDashboard = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('http://localhost:5002/agi/progress')
      .then(res => res.json())
      .then(d => {
        if (d.status === 'success') {
          setData(d);
        } else {
          setError(d.message);
        }
        setLoading(false);
      })
      .catch(err => {
        console.error("Fetch Error:", err);
        setError("无法连接到服务器后端");
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-zinc-400">
        <Activity className="animate-spin mr-2" size={20} />
        正在加载 AGI 研发进度...
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-red-400 p-6 text-center">
        <AlertCircle size={40} className="mb-4" />
        <div className="text-lg font-bold mb-2">加载失败</div>
        <div className="text-sm opacity-80">{error}</div>
      </div>
    );
  }

  const { phases = [], latest_test = {} } = data || {};

  return (
    <div className="flex flex-col h-full bg-[#0a0a0c] overflow-hidden text-zinc-200">
      {/* Header Section */}
      <div className="p-6 border-b border-white/5 bg-gradient-to-r from-blue-900/10 to-transparent">
        <div className="flex items-center justify-between mb-2">
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <TrendingUp className="text-blue-400" />
            AGI 研发进度管理中心
          </h1>
          <div className="text-xs text-zinc-500 flex items-center gap-1">
            <Clock size={12} />
            最后更新: {new Date(data.last_updated * 1000).toLocaleString()}
          </div>
        </div>
        <p className="text-sm text-zinc-400">
          全息追踪 "Project Genesis" 路线图，监控几何动力学验证状态。
        </p>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-8">
        {/* Phase Roadmap Section */}
        <section>
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <CheckCircle2 className="text-green-400" size={18} />
            研发阶段路线图 (Engineering Phases)
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {phases.map((phase, idx) => (
              <div 
                key={idx} 
                className={`p-4 rounded-xl border transition-all hover:bg-white/5 ${
                  phase.status === '进行中' 
                    ? 'bg-blue-900/10 border-blue-500/30' 
                    : 'bg-zinc-900/50 border-white/5'
                }`}
              >
                <div className="flex justify-between items-start mb-2">
                  <span className="text-xs font-bold text-blue-400 uppercase tracking-wider">
                    Phase {idx + 4}
                  </span>
                  <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                    phase.status === '已完成' 
                      ? 'bg-green-500/10 text-green-400 border border-green-500/20' 
                      : phase.status === '进行中'
                        ? 'bg-blue-500/20 text-blue-300 border border-blue-500/30 animate-pulse'
                        : 'bg-zinc-800 text-zinc-500 border border-white/5'
                  }`}>
                    {phase.status}
                  </span>
                </div>
                <h3 className="font-bold text-white mb-2">{phase.title}</h3>
                <div className="text-xs text-zinc-400 line-clamp-3 mb-3">
                  {phase.summary || "暂无阶段总结..."}
                </div>
                {phase.target && (
                  <div className="mt-auto pt-3 border-t border-white/5 flex items-start gap-2">
                    <Activity size={12} className="text-zinc-500 mt-0.5" />
                    <span className="text-[11px] text-zinc-500 italic">{phase.target}</span>
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>

        {/* Real-time Metrics Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <section className="lg:col-span-1">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <BarChart3 className="text-purple-400" size={18} />
              最新实验指标
            </h2>
            <div className="bg-zinc-900/50 border border-white/5 rounded-xl p-5 space-y-4">
              <div className="flex justify-between items-center text-sm">
                <span className="text-zinc-500">任务类型</span>
                <span className="text-zinc-200 font-mono">{latest_test.task_type || "N/A"}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-zinc-500">验证准确率 (Accuracy)</span>
                <span className="text-2xl font-bold text-green-400">
                  {latest_test.val_accuracy ? (latest_test.val_accuracy * 100).toFixed(1) : "0"}%
                </span>
              </div>
              <div className="w-full bg-zinc-800 rounded-full h-1.5 overflow-hidden">
                <div 
                  className="bg-green-500 h-full transition-all duration-1000 shadow-[0_0_10px_rgba(34,197,94,0.5)]" 
                  style={{ width: `${(latest_test.val_accuracy || 0) * 100}%` }}
                />
              </div>
            </div>
            {/* Model Topology Health (Added) */}
            <div className="bg-zinc-900/50 border border-blue-500/20 rounded-xl p-5 mt-6 space-y-4">
              <h2 className="text-lg font-semibold flex items-center gap-2">
                <Search className="text-blue-400" size={18} />
                GPT-2 全谱扫描状态 (Glass Matrix HUD)
              </h2>
              <div className="flex justify-between items-center text-sm">
                <span className="text-zinc-500">模型节点</span>
                <span className="text-blue-400 font-bold">GPT-2 (12 Layers)</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-zinc-500">扫描完成度</span>
                <span className="text-xl font-mono text-blue-400">100%</span>
              </div>
              <div className="w-full bg-zinc-800 rounded-full h-1.5 overflow-hidden">
                <div className="bg-blue-500 h-full w-full shadow-[0_0_10px_rgba(59,130,246,0.5)]" />
              </div>
              <div className="pt-2">
                <div className="text-[10px] text-zinc-500 font-bold mb-2 uppercase tracking-tight">扫描产出 (Artifacts)</div>
                <div className="flex flex-wrap gap-2">
                  <span className="px-2 py-1 bg-white/5 border border-white/10 rounded text-[10px] font-mono">topology.json (1.5MB)</span>
                  <span className="px-2 py-1 bg-white/5 border border-white/10 rounded text-[10px] font-mono">betti_curves.json</span>
                </div>
              </div>
            </div>
          </section>

          {/* Research Insights / Logs */}
          <section className="lg:col-span-2">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <BookOpen className="text-orange-400" size={18} />
              研究分析简报 (Phase V - Analysis)
            </h2>
            <div className="space-y-3">
              {data.research_logs && data.research_logs.length > 0 ? (
                data.research_logs.map((log, i) => (
                  <div key={i} className="p-4 bg-zinc-900/50 border border-white/5 rounded-xl hover:bg-white/5 transition-colors group">
                    <div className="flex items-center gap-2 mb-2">
                      <FileText size={14} className="text-zinc-500 group-hover:text-orange-400" />
                      <span className="text-xs font-bold text-zinc-500">2026-02-15 | Milestone Update</span>
                    </div>
                    <p className="text-sm text-zinc-300 leading-relaxed">{log}</p>
                  </div>
                ))
              ) : (
                <>
                  <div className="p-4 bg-zinc-900/50 border border-white/5 rounded-xl hover:bg-white/5 transition-colors group">
                    <div className="flex items-center gap-2 mb-2">
                      <FileText size={14} className="text-zinc-500 group-hover:text-orange-400" />
                      <span className="text-xs font-bold text-zinc-500">2026-02-14 | Ricci Flow Optimization</span>
                    </div>
                    <p className="text-sm text-zinc-300 leading-relaxed">
                      验证了基于微分几何流的正则化机制。Ricci Flow 成功通过惩罚非正交性降低了 $S_3$ 群任务的曲率熵，使 FiberNet 突破了 27.1% 的泛化准确率，初步验证了几何自律性。
                    </p>
                  </div>
                  <div className="p-4 bg-zinc-900/50 border border-white/5 rounded-xl hover:bg-white/5 transition-colors group">
                    <div className="flex items-center gap-2 mb-2">
                      <FileText size={14} className="text-zinc-500 group-hover:text-orange-400" />
                      <span className="text-xs font-bold text-zinc-500">2026-02-12 | Fiber Bundle Connection Layer</span>
                    </div>
                    <p className="text-sm text-zinc-300 leading-relaxed">
                      定义了全纯裂隙与曲率热力图的光学映射规则。实现了联络层 $\nabla$ 对语义漂移的纠偏作用，这是从单纯的特征叠加向结构化纤维丛演进的关键一步。
                    </p>
                  </div>
                </>
              )}
              <button className="w-full py-3 border border-dashed border-white/10 rounded-xl text-xs text-zinc-500 hover:text-zinc-300 hover:border-white/20 transition-all">
                查看 AGI_RESEARCH_MEMO.md 完整日志...
              </button>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};

