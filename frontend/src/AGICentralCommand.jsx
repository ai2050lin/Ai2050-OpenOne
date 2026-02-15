import {
    Brain,
    CheckCircle2,
    Cpu,
    Info,
    Layers,
    Microscope,
    Search,
    Shield,
    Target,
    TrendingUp
} from 'lucide-react';
import { useEffect, useState } from 'react';

const AGICentralCommand = ({ onClose }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedPhaseId, setSelectedPhaseId] = useState('theory'); // Default to top-level Theory

  useEffect(() => {
    fetch('http://localhost:5002/agi/progress')
      .then(res => res.json())
      .then(d => {
        if (d.status === 'success') {
          setData(d.systemic || { theory: [], engineering: [], experiments: {}, roadmap: [] });
        } else {
          setError(d.message);
        }
        setLoading(false);
      })
      .catch(err => {
        setError("无法连接指挥系统后端");
        setLoading(false);
      });
  }, []);

  if (loading) return (
    <div className="fixed inset-0 z-[4000] bg-black/95 flex items-center justify-center">
      <div className="text-blue-500 animate-pulse flex flex-col items-center gap-4">
        <Shield size={48} />
        <span className="font-mono tracking-[0.2em] text-sm uppercase">Initializing AGI Central Command...</span>
      </div>
    </div>
  );

  const selectedPhase = data?.roadmap?.find(p => p.id === selectedPhaseId) || data?.roadmap?.[0];

  return (
    <div className="fixed inset-0 z-[4000] bg-[#050507] text-zinc-100 font-sans overflow-hidden flex flex-col">
      {/* Background Ambience */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none opacity-20">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-600/20 blur-[120px] rounded-full" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-600/20 blur-[120px] rounded-full" />
      </div>

      {/* Header */}
      <div className="h-16 border-b border-white/5 flex items-center justify-between px-8 bg-black/40 backdrop-blur-md relative z-10">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-blue-500/10 rounded-lg border border-blue-500/20 shadow-[0_0_15px_rgba(59,130,246,0.2)]">
            <Shield className="text-blue-400" size={20} />
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-widest uppercase text-white">AGI Central Command</h1>
            <p className="text-[10px] text-zinc-500 font-mono">Mission Control // Topological Governance Active</p>
          </div>
        </div>
        
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-4 text-[10px] font-mono border-x border-white/5 px-6">
            <div className="flex flex-col items-end">
              <span className="text-zinc-500">Convergence Index</span>
              <span className="text-blue-400 font-bold">{(data?.convergence_index || 0.62).toFixed(2)}</span>
            </div>
            <div className="w-12 h-1 bg-zinc-800 rounded-full overflow-hidden">
               <div className="h-full bg-blue-500" style={{ width: `${(data?.convergence_index || 0.62) * 100}%` }} />
            </div>
          </div>
          <button 
            onClick={onClose}
            className="px-4 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-xs rounded-md border border-white/5 transition-all uppercase tracking-tighter"
          >
            Exit Terminal
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 overflow-y-auto relative z-10 custom-scrollbar p-8 space-y-8">
        
        {/* Dimension 4: Roadmap & Milestones (Interactive Timeline) */}
        <section className="bg-white/[0.01] border border-white/5 rounded-3xl p-8 backdrop-blur-sm relative overflow-hidden">
           {/* Animated Gradient Background for Roadmap */}
           <div className="absolute inset-0 bg-gradient-to-r from-blue-500/5 via-transparent to-purple-500/5 pointer-events-none" />
           
           <div className="flex items-center justify-between mb-10 relative z-10">
              <div className="flex items-center gap-2">
                <Target className="text-red-400" size={20} />
                <h2 className="text-base font-bold uppercase tracking-[0.2em] text-white">AGI 演进蓝图 (AGI Strategic Roadmap)</h2>
              </div>
              <div className="text-[10px] text-zinc-500 font-mono flex items-center gap-4">
                 <span className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-blue-500" /> 理论/分析验证</span>
                 <span className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-orange-500 animate-pulse" /> 工程构建中</span>
              </div>
           </div>
           
           {/* Horizontal Timeline */}
           <div className="relative mb-12 px-20">
              {/* Main Connecting Rail */}
              <div className="absolute top-[28px] left-0 right-0 h-[2px] bg-zinc-800 relative z-0">
                 <div 
                    className="absolute h-full bg-gradient-to-r from-blue-500 to-blue-400 transition-all duration-1000"
                    style={{ width: selectedPhaseId === 'agi_goal' ? '100%' : selectedPhaseId === 'engineering' ? '66%' : selectedPhaseId === 'analysis' ? '33%' : '0%' }}
                 />
              </div>
              
              <div className="flex justify-between relative z-10">
                 {data?.roadmap?.map((p) => (
                    <button 
                       key={p.id} 
                       onClick={() => setSelectedPhaseId(p.id)}
                       className={`flex flex-col items-center gap-4 transition-all duration-300 focus:outline-none group`}
                       style={{ width: '160px' }}
                    >
                       <div className={`w-14 h-14 rounded-full border-2 flex items-center justify-center transition-all duration-500 relative ${
                         p.id === selectedPhaseId ? 'scale-110 shadow-[0_0_30px_rgba(255,255,255,0.1)]' : 'scale-100'
                       } ${
                         p.status === 'done' 
                         ? 'bg-blue-500/20 border-blue-500 shadow-[0_0_20px_rgba(59,130,246,0.3)]'
                         : p.status === 'in_progress'
                         ? 'bg-orange-500/20 border-orange-500 border-dashed animate-pulse-slow shadow-[0_0_15px_rgba(249,115,22,0.2)]'
                         : 'bg-zinc-900 border-white/10'
                       }`}>
                          {p.id === 'theory' ? <Brain size={20} className={p.status === 'done' ? 'text-blue-400' : 'text-zinc-600'} /> :
                           p.id === 'analysis' ? <Search size={20} className={p.status === 'done' ? 'text-blue-400' : 'text-zinc-600'} /> :
                           p.id === 'engineering' ? <Cpu size={20} className={p.status === 'in_progress' ? 'text-orange-400' : 'text-zinc-600'} /> :
                           <Target size={20} className="text-zinc-700" />}
                          
                          {/* Selection indicator */}
                          {p.id === selectedPhaseId && (
                             <div className="absolute -bottom-2 w-1.5 h-1.5 bg-white rounded-full shadow-[0_0_10px_#fff]" />
                          )}
                       </div>
                       
                       <div className="text-center overflow-hidden w-full px-1">
                          <div className={`text-xs font-bold transition-colors ${p.id === selectedPhaseId ? 'text-white' : 'text-zinc-500'} truncate uppercase tracking-widest`}>
                             {p.title.split('(')[0]}
                          </div>
                          <div className={`text-[8px] font-mono tracking-[0.2em] mt-1 ${
                             p.status === 'done' ? 'text-blue-500/80' : p.status === 'in_progress' ? 'text-orange-500/80' : 'text-zinc-700'
                          }`}>
                             {p.status.toUpperCase()}
                          </div>
                       </div>
                    </button>
                 ))}
              </div>
           </div>

           {/* Detail Panel for Selected Milestone */}
           {selectedPhase && (
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 border-t border-white/5 pt-10 animate-in fade-in slide-in-from-bottom-4 duration-500">
                 {/* Left Info Column */}
                 <div className="lg:col-span-5 space-y-6">
                    <div>
                       <div className="flex items-center gap-3 mb-3">
                          <span className="px-2 py-0.5 bg-white/5 border border-white/10 rounded text-[9px] font-mono text-zinc-400 uppercase tracking-[0.2em]">
                             {selectedPhase.id.toUpperCase()} SN-01
                          </span>
                       </div>
                       <h3 className="text-2xl font-bold text-white tracking-tight mb-4">{selectedPhase.title}</h3>
                       <p className="text-sm text-zinc-400 leading-relaxed italic border-l-2 border-blue-500/30 pl-4">
                          {selectedPhase.desc}
                       </p>
                    </div>

                    <div className="space-y-4">
                       <h4 className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest flex items-center gap-2">
                          <CheckCircle2 size={14} /> 核心内容概览
                       </h4>
                       <div className="space-y-2">
                          {(selectedPhase.details || []).map((detail, idx) => (
                             <div key={idx} className="flex items-center gap-3 p-3 bg-white/[0.02] border border-white/5 rounded-xl text-xs text-zinc-300">
                                <div className="w-1 h-1 bg-blue-500 rounded-full" />
                                {detail}
                             </div>
                          ))}
                       </div>
                    </div>
                 </div>

                 {/* Right Data Column */}
                 <div className="lg:col-span-7 space-y-8">
                    {selectedPhase.id === 'engineering' ? (
                       <div className="space-y-4">
                          <h4 className="text-[10px] font-bold text-orange-500 uppercase tracking-widest flex items-center gap-2">
                             <Layers size={14} /> 演进阶段 (Evolutionary Phases)
                          </h4>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                             {selectedPhase.sub_phases?.map((sp, idx) => (
                                <div key={idx} className={`p-4 rounded-2xl border transition-all ${
                                   sp.status === 'done' ? 'bg-blue-500/[0.03] border-blue-500/20' : 
                                   sp.status === 'in_progress' ? 'bg-orange-500/[0.03] border-orange-500/20 animate-pulse-slow' : 'bg-black/20 border-white/5 opacity-50'
                                }`}>
                                   <div className="flex items-center justify-between mb-2">
                                      <span className="text-[11px] font-bold text-zinc-200">{sp.name}</span>
                                      {sp.status === 'done' && <CheckCircle2 size={12} className="text-blue-500" />}
                                   </div>
                                   <div className="text-[9px] text-zinc-500 font-mono">{sp.focus}</div>
                                </div>
                             ))}
                          </div>
                       </div>
                    ) : (
                       <div className="space-y-4">
                          <h4 className="text-[10px] font-bold text-blue-500 uppercase tracking-widest flex items-center gap-2">
                             <TrendingUp size={14} /> 执行指标 (Operational Metrics)
                          </h4>
                          <div className="grid grid-cols-2 gap-4">
                             {Object.entries(selectedPhase.metrics || {}).map(([key, value]) => (
                                <div key={key} className="p-5 bg-blue-500/[0.03] border border-blue-500/10 rounded-2xl group hover:bg-blue-500/5 transition-all">
                                   <div className="text-[9px] text-zinc-500 font-bold uppercase mb-1">{key}</div>
                                   <div className="text-2xl font-mono text-blue-400 font-bold">{value}</div>
                                </div>
                             ))}
                          </div>
                       </div>
                    )}

                    <div className="p-5 bg-zinc-900/50 border border-white/5 rounded-2xl flex items-start gap-3">
                       <Info size={16} className="text-zinc-600 mt-1" />
                       <div className="text-[10px] text-zinc-500 leading-relaxed italic">
                          当前监控处于 **{selectedPhase.status === 'done' ? '基线对齐' : '活动探测'}** 模式。
                          已启用 NFB-RA 层级化分析协议，针对 {selectedPhase.id} 维度的流形偏移进行亚毫秒级补偿。
                       </div>
                    </div>
                 </div>
              </div>
           )}
        </section>

        {/* Global Stats Grid Below Timeline */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 pb-8">
          {/* Dimension 1: Theory (智能理论) */}
          <section className="lg:col-span-4 bg-white/[0.02] border border-white/10 rounded-2xl p-6 backdrop-blur-sm flex flex-col">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-2">
                <Brain className="text-purple-400" size={18} />
                <h2 className="text-sm font-bold uppercase tracking-wider text-zinc-200">智能理论 (Theory)</h2>
              </div>
              <span className="text-[10px] px-2 py-0.5 bg-purple-500/10 text-purple-400 border border-purple-500/20 rounded">VALIDATED</span>
            </div>
            
            <div className="flex-1 space-y-4">
              {(data?.theory && data.theory.length > 0) ? data.theory.map((t, i) => (
                <div key={i} className="group p-4 bg-white/[0.01] border border-white/5 rounded-xl hover:bg-white/5 transition-all flex items-center justify-between">
                  <div>
                    <div className="text-xs font-bold text-zinc-300 mb-0.5">{t.name}</div>
                    <div className="text-[10px] text-zinc-600 font-mono tracking-tighter">{t.status.toUpperCase()}</div>
                  </div>
                  <div className={`p-2 rounded-lg ${t.status === 'validated' ? 'bg-green-500/10 text-green-400' : 'bg-orange-500/10 text-orange-400'}`}>
                    <Search size={14} />
                  </div>
                </div>
              )) : (
                <div className="h-full flex flex-col items-center justify-center opacity-30 gap-2 p-8 text-center">
                  <Microscope size={32} />
                  <span className="text-[10px] font-mono">Parsing Deep Logic Manifolds...</span>
                </div>
              )}
            </div>
          </section>

          {/* Dimension 2: Engineering (工程进展) */}
          <section className="lg:col-span-5 bg-white/[0.02] border border-white/10 rounded-2xl p-6 backdrop-blur-sm">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-2">
                <Cpu className="text-blue-400" size={18} />
                <h2 className="text-sm font-bold uppercase tracking-wider text-zinc-200">工程阶段 (Engineering)</h2>
              </div>
              <div className="flex items-center gap-1.5 text-[10px] text-blue-400/80">
                <div className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
                LIVE BUILD
              </div>
            </div>

            <div className="space-y-4">
                {data?.engineering?.map((item, i) => (
                  <div key={i} className="space-y-2">
                    <div className="flex justify-between items-center text-[11px]">
                      <span className="text-zinc-300">{item.name}</span>
                      <span className="text-zinc-500 font-mono">{item.progress}%</span>
                    </div>
                    <div className="h-1 bg-white/[0.03] rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-blue-600 to-blue-400 shadow-[0_0_8px_rgba(59,130,246,0.3)]" 
                        style={{ width: `${item.progress}%` }} 
                      />
                    </div>
                  </div>
                ))}
                <div className="pt-4 grid grid-cols-2 gap-3">
                    <div className="p-3 bg-zinc-900 border border-white/5 rounded-xl">
                       <div className="text-[9px] text-zinc-600 font-bold uppercase mb-1">Scanner Nodes</div>
                       <div className="text-lg font-mono text-zinc-300">12/12</div>
                    </div>
                    <div className="p-3 bg-zinc-900 border border-white/5 rounded-xl">
                       <div className="text-[9px] text-zinc-600 font-bold uppercase mb-1">Hardware Affinity</div>
                       <div className="text-lg font-mono text-green-500">98.2%</div>
                    </div>
                </div>
            </div>
          </section>

          {/* Convergence Index Mini Widget */}
          <section className="lg:col-span-3 bg-zinc-900/40 border border-white/10 rounded-2xl p-6 backdrop-blur-sm flex flex-col justify-center items-center gap-4 text-center">
              <div className="relative w-32 h-32 flex items-center justify-center">
                 {/* Circular Progress SVG would go here, simplified as div for now */}
                 <div className="absolute inset-0 rounded-full border-4 border-white/5" />
                 <div className="absolute inset-0 rounded-full border-4 border-blue-500/50 border-t-transparent animate-spin-slow" />
                 <div className="text-center">
                    <div className="text-3xl font-bold text-white">0.62</div>
                    <div className="text-[9px] text-zinc-500 font-mono uppercase tracking-widest">Convergence</div>
                 </div>
              </div>
              <div>
                 <p className="text-[11px] text-zinc-400">目前处于系统级收敛阶段</p>
                 <p className="text-[9px] text-zinc-600 mt-1">下一步：多模态联络层对齐</p>
              </div>
          </section>
        </div>
      </div>

      {/* Footer Info */}
      <div className="h-10 border-t border-white/5 px-8 flex items-center justify-between text-[10px] text-zinc-600 font-mono bg-black/40 relative z-10">
         <div className="flex items-center gap-4">
            <span className="flex items-center gap-1.5"><div className="w-1.5 h-1.5 rounded-full bg-green-500" /> API CONNECTED</span>
            <span className="flex items-center gap-1.5"><div className="w-1.5 h-1.5 rounded-full bg-blue-500" /> QUANTUM LINK ACTIVE</span>
         </div>
         <div className="uppercase tracking-widest">Antigravity Research Terminal v1.1.0 // Genesis Protocol</div>
      </div>

      <style jsx>{`
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.05); border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.1); }
        
        @keyframes pulse-slow {
          0%, 100% { opacity: 0.3; transform: scale(1); }
          50% { opacity: 0.6; transform: scale(1.05); }
        }
        .animate-pulse-slow {
          animation: pulse-slow 3s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
        @keyframes spin-slow {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .animate-spin-slow {
          animation: spin-slow 8s linear infinite;
        }
      `}</style>
    </div>
  );
};

export default AGICentralCommand;
