import { Brain, X, Activity, Target, Search, Play, Zap, CheckCircle } from 'lucide-react';
import { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { pollRuntimeWithFallback } from './utils/runtimeClient';
import { beginBackendRequest, clearBackendUnavailable, isFetchNetworkError, markBackendUnavailable } from './utils/backendAvailability';
import { ProjectRoadmapTab } from './blueprint/ProjectRoadmapTab';
import { LanguageAnalysisTab } from './blueprint/LanguageAnalysisTab';
import { DeepAnalysisTab } from './blueprint/DeepAnalysisTab';
import { ResearchProgressTab } from './blueprint/ResearchProgressTab';
import { SystemStatusTab } from './blueprint/SystemStatusTab';
import { AtlasControlDashboard } from './blueprint/AtlasControlDashboard';
import { AIRnDConsoleTab } from './AIRnD/AIRnDConsoleTab';
import { AIRnDConfigTab } from './AIRnD/AIRnDConfigTab';
import { RESEARCH_PHASES } from './AIRnD/aiRnDConfig';
import {
  PHASES,
  IMPROVEMENTS,
  DNN_ANALYSIS_PLAN,
  EVIDENCE_DRIVEN_PLAN,
  EXECUTION_PLAYBOOK,
  MATH_ROUTE_SYSTEM_PLAN,
} from './blueprint/blueprintConfig';
import { API_BASE, mapLegacyConsciousField, mapRuntimeConsciousField } from './blueprint/blueprintRuntimeUtils';

const BLUEPRINT_TABS = new Set(['roadmap', 'atlas_control', 'language', 'analysis', 'progress', 'system', 'rnd_console', 'rnd_config']);
const THEORY_TABS = new Set(['roadmap', 'atlas_control', 'language', 'analysis', 'progress', 'system']);
const RND_TABS = new Set(['rnd_console', 'rnd_config']);

const PHASE_ICONS = {
  analyze: Search,
  plan: Target,
  generate: Zap,
  execute: Play,
  summarize: CheckCircle,
};

export const HLAIBlueprint = ({ onClose, initialTab = 'roadmap', mode = 'overlay', scope = 'theory' }) => {
  const normalizedInitialTab = scope === 'rnd'
    ? (RND_TABS.has(initialTab) ? initialTab : 'rnd_console')
    : (THEORY_TABS.has(initialTab) ? initialTab : 'roadmap');
  const [activeTab, setActiveTab] = useState(normalizedInitialTab); // roadmap, progress, system
  const [lastTheoryTab, setLastTheoryTab] = useState('roadmap');
  const [selectedRouteId, setSelectedRouteId] = useState('fiber_bundle');

  const handleTabChange = (tabId) => {
    if (scope === 'rnd' && !RND_TABS.has(tabId)) return;
    if (scope === 'theory' && !THEORY_TABS.has(tabId)) return;
    setActiveTab(tabId);
    if (THEORY_TABS.has(tabId)) {
      setLastTheoryTab(tabId);
    }
  };

  // AI Auto R&D states and lifecycle
  const [sessionStatus, setSessionStatus] = useState('idle'); // idle | running | paused | stopped | waiting_step
  const [sessionMode, setSessionMode] = useState('auto'); // auto | manual
  const [currentPhase, setCurrentPhase] = useState(null);
  const [round, setRound] = useState(0);
  const [logs, setLogs] = useState([]);
  const [findings, setFindings] = useState([]);
  const [generatedCode, setGeneratedCode] = useState('');
  const [executionResult, setExecutionResult] = useState(null);
  const [researchState, setResearchState] = useState(null);
  const [rndError, setRndError] = useState(null);
  const eventSourceRef = useRef(null);

  const fetchSessionStatus = useCallback(async () => {
    if (!beginBackendRequest()) return;
    try {
      const res = await fetch(`${API_BASE}/api/ai-rnd/session/status`);
      if (res.ok) {
        clearBackendUnavailable();
        const data = await res.json();
        setSessionStatus(data.status || 'idle');
        setSessionMode(data.mode || 'auto');
        setCurrentPhase(data.current_phase || null);
        setRound(data.round || 0);
        setResearchState(data.research_state || null);
      }
    } catch (e) {
      if (isFetchNetworkError(e)) markBackendUnavailable();
    }
  }, []);

  const handleEvent = useCallback((data) => {
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = { ...data, timestamp, id: Date.now() + Math.random() };
    setLogs(prev => [...prev.slice(-500), logEntry]);

    if (data.type === 'phase_change') {
      setCurrentPhase(data.phase);
    } else if (data.type === 'round_change') {
      setRound(data.round);
    } else if (data.type === 'status_change') {
      setSessionStatus(data.status);
    } else if (data.type === 'mode_change') {
      setSessionMode(data.mode);
    } else if (data.type === 'finding') {
      setFindings(prev => [...prev.slice(-200), data.finding]);
    } else if (data.type === 'code_generated') {
      setGeneratedCode(data.code || '');
    } else if (data.type === 'execution_result') {
      setExecutionResult(data.result);
    } else if (data.type === 'error') {
      setRndError(data.message);
    }
  }, []);

  const connectEventStream = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    const es = new EventSource(`${API_BASE}/api/ai-rnd/session/events`);
    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleEvent(data);
      } catch (e) {
        console.warn('Failed to parse SSE event:', e);
      }
    };
    es.onerror = () => {
      es.close();
      eventSourceRef.current = null;
    };
    eventSourceRef.current = es;
  }, [handleEvent]);

  // Fetch initial session status and auto-connect SSE if running
  useEffect(() => {
    fetchSessionStatus().then(() => {
      if (!beginBackendRequest()) return;
      fetch(`${API_BASE}/api/ai-rnd/session/status`)
        .then(res => res.ok ? res.json() : null)
        .then(data => {
          if (data && data.status === 'running') {
            connectEventStream();
          }
        })
        .catch((e) => {
          if (isFetchNetworkError(e)) markBackendUnavailable();
        });
    });
    
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, [fetchSessionStatus, connectEventStream]);

  const startSession = useCallback(async (objective = '') => {
    try {
      setRndError(null);
      const trimmedObjective = String(objective || '').trim();
      const res = await fetch(`${API_BASE}/api/ai-rnd/session/start`, {
        method: 'POST',
        headers: trimmedObjective ? { 'Content-Type': 'application/json' } : undefined,
        body: trimmedObjective ? JSON.stringify({ objective: trimmedObjective }) : undefined,
      });
      if (res.ok) {
        setSessionStatus('running');
        connectEventStream();
      } else {
        const err = await res.json();
        setRndError(err.detail || '启动失败');
      }
    } catch (e) {
      setRndError(`连接后端失败: ${e.message}`);
    }
  }, [connectEventStream]);

  const pauseSession = useCallback(async () => {
    try {
      await fetch(`${API_BASE}/api/ai-rnd/session/pause`, { method: 'POST' });
      setSessionStatus('paused');
    } catch (e) { setRndError(e.message); }
  }, []);

  const resumeSession = useCallback(async () => {
    try {
      await fetch(`${API_BASE}/api/ai-rnd/session/start`, { method: 'POST' });
      setSessionStatus('running');
    } catch (e) { setRndError(e.message); }
  }, []);

  const stopSession = useCallback(async () => {
    try {
      await fetch(`${API_BASE}/api/ai-rnd/session/stop`, { method: 'POST' });
      setSessionStatus('stopped');
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    } catch (e) { setRndError(e.message); }
  }, []);

  const toggleMode = useCallback(async () => {
    const newMode = sessionMode === 'auto' ? 'manual' : 'auto';
    try {
      const res = await fetch(`${API_BASE}/api/ai-rnd/session/mode`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: newMode }),
      });
      if (res.ok) {
        setSessionMode(newMode);
      }
    } catch (e) { setRndError(e.message); }
  }, [sessionMode]);

  const stepNext = useCallback(async () => {
    try {
      setRndError(null);
      const res = await fetch(`${API_BASE}/api/ai-rnd/session/step`, { method: 'POST' });
      if (!res.ok) {
        const err = await res.json();
        setRndError(err.detail || 'Step failed');
      }
    } catch (e) { setRndError(e.message); }
  }, []);
  const [timelineRoutes, setTimelineRoutes] = useState([]);
  const [expandedFormulaIdx, setExpandedFormulaIdx] = useState(null);
  const [expandedParam, setExpandedParam] = useState(null);
  const [expandedEngPhase, setExpandedEngPhase] = useState(null);
  const [expandedImprovementPhase, setExpandedImprovementPhase] = useState(IMPROVEMENTS[0]?.id || null);
  const [expandedImprovementTest, setExpandedImprovementTest] = useState(null);
  const [consciousField, setConsciousField] = useState(null);
  const [multimodalSummary, setMultimodalSummary] = useState(null);
  const [multimodalView, setMultimodalView] = useState('multimodal_connector');
  const [multimodalError, setMultimodalError] = useState(null);
  const [runtimeStatusSummary, setRuntimeStatusSummary] = useState(null);
  const runtimeStepRef = useRef(0);

  useEffect(() => {
    const targetTab = BLUEPRINT_TABS.has(initialTab) ? initialTab : 'roadmap';
    setActiveTab(targetTab);
    if (['roadmap', 'atlas_control', 'language', 'analysis', 'progress', 'system'].includes(targetTab)) {
      setLastTheoryTab(targetTab);
    }
  }, [initialTab]);

  // Real-time Consciousness Polling
  useEffect(() => {
    let mounted = true;
    let retryAfter = 0;

    const fetchLegacyConsciousField = async () => {
      if (!beginBackendRequest()) throw new Error('backend unavailable cooldown');
      const res = await fetch(`${API_BASE}/nfb_ra/unified_conscious_field`);
      if (!res.ok) throw new Error(`legacy conscious field failed: ${res.status}`);
      clearBackendUnavailable();
      const data = await res.json();
      if (data?.status !== 'success') throw new Error('legacy conscious field unavailable');
      return mapLegacyConsciousField(data);
    };

    const pollConsciousField = async () => {
      if (Date.now() < retryAfter) return;
      const stepId = runtimeStepRef.current++;
      try {
        const result = await pollRuntimeWithFallback({
          apiBase: API_BASE,
          runRequest: {
            route: 'fiber_bundle',
            analysis_type: 'unified_conscious_field',
            params: { step_id: stepId, noise_scale: 0.4 },
            input_payload: {},
          },
          mapRuntimeEvents: mapRuntimeConsciousField,
          fetchLegacy: fetchLegacyConsciousField,
          eventLimit: 20,
        });
        if (!mounted) return;
        retryAfter = 0;
        setConsciousField({ ...result.data, source: result.source });
      } catch (error) {
        if (!mounted) return;
        if (isFetchNetworkError(error)) markBackendUnavailable();
        retryAfter = Date.now() + 10000;
        setConsciousField(null);
      }
    };

    pollConsciousField();
    const interval = setInterval(pollConsciousField, 2000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    let mounted = true;

    const fetchRuntimeStatusSummary = async () => {
      if (!beginBackendRequest()) return;
      try {
        const res = await fetch(`${API_BASE}/api/system_status/runtime_summary`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        clearBackendUnavailable();
        const payload = await res.json();
        if (!mounted || payload?.status !== 'success') return;
        setRuntimeStatusSummary(payload);
      } catch (error) {
        if (!mounted) return;
        if (isFetchNetworkError(error)) markBackendUnavailable();
        setRuntimeStatusSummary(null);
      }
    };

    fetchRuntimeStatusSummary();
    const interval = setInterval(fetchRuntimeStatusSummary, 10000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    let mounted = true;
    const fetchTimelineRoutes = async () => {
      if (!beginBackendRequest()) return;
      try {
        const res = await fetch(`${API_BASE}/api/v1/experiments/timeline?limit=120`);
        if (!res.ok) return;
        clearBackendUnavailable();
        const payload = await res.json();
        if (!mounted || payload?.status !== 'success') return;
        const routes = Array.isArray(payload?.timeline?.routes) ? payload.timeline.routes : [];
        setTimelineRoutes(routes);
      } catch (error) {
        if (isFetchNetworkError(error)) markBackendUnavailable();
        // Keep local defaults when runtime API is unavailable.
      }
    };
    fetchTimelineRoutes();
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    let mounted = true;
    const fetchMultimodalSummary = async () => {
      if (!beginBackendRequest()) return;
      try {
        const res = await fetch(`${API_BASE}/nfb/multimodal/summary`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        clearBackendUnavailable();
        const payload = await res.json();
        if (!mounted) return;
        if (payload?.status !== 'success') throw new Error('invalid payload');
        setMultimodalSummary(payload);
        setMultimodalError(null);
      } catch (err) {
        if (!mounted) return;
        if (isFetchNetworkError(err)) markBackendUnavailable();
        setMultimodalError(err?.message || 'multimodal summary unavailable');
      }
    };
    fetchMultimodalSummary();
    const interval = setInterval(fetchMultimodalSummary, 15000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);


  useEffect(() => {
    const available = Array.isArray(multimodalSummary?.available_views)
      ? multimodalSummary.available_views
      : [];
    if (available.length > 0 && !available.includes(multimodalView)) {
      setMultimodalView(available[0]);
    }
  }, [multimodalSummary, multimodalView]);

  const baseStatusData = PHASES.find(p => p.id === 'agi_status');
  const statusData = useMemo(() => {
    if (!baseStatusData) return baseStatusData;
    const runtimeModelSummary = runtimeStatusSummary?.model_summary || {};
    const runtimeLanguage = runtimeStatusSummary?.runtime_language || {};
    const phaseaRuntime = runtimeStatusSummary?.phasea_runtime || {};
    const researchOverview = runtimeStatusSummary?.research_overview || {};
    return {
      ...baseStatusData,
      model_summary: {
        ...(baseStatusData.model_summary || {}),
        ...runtimeModelSummary,
      },
      runtime_language: runtimeLanguage,
      phasea_runtime: {
        ...(baseStatusData.phasea_runtime || {}),
        ...phaseaRuntime,
      },
      research_overview: {
        ...(baseStatusData.research_overview || {}),
        ...researchOverview,
      },
    };
  }, [baseStatusData, runtimeStatusSummary]);
  const roadmapData = PHASES.find(p => p.id === 'roadmap');
  const theoryPhase = PHASES.find(p => p.id === 'theory');
  const analysisPhase = PHASES.find(p => p.id === 'analysis');
  const engineeringPhase = PHASES.find(p => p.id === 'engineering');
  const milestonePhase = PHASES.find(p => p.id === 'agi_goal');

  const routeBlueprints = useMemo(
    () => ({
      fiber_bundle: {
        id: 'fiber_bundle',
        title: 'Fiber Bundle',
        subtitle: '几何原生智能路线',
        routeDescription: '以神经纤维丛与几何推理为核心，验证结构化智能的可行性。',
        engineeringProcessDescription:
          '计算流程：输入先映射到底流形进行逻辑定位，再进入纤维记忆检索候选语义；通过联络传输层完成跨束对齐，最后由全局工作空间执行 Top-K 裁决并输出结果。',
        theoryTitle: theoryPhase?.definition?.headline || 'Intelligence = Geometry + Physics',
        theorySummary: theoryPhase?.definition?.summary || '',
        theoryBullets: (theoryPhase?.theory_content || []).slice(0, 4).map((item) => item.title),
        theoryFormulas: [
          {
            title: '神经纤维丛原理 (NFB Principle)',
            formula: 'φ(x) = M ⊗ F',
            detail:
              '把智能状态拆成"逻辑骨架 (底流形)"和"知识内容 (纤维)"的张量积，逻辑稳定、内容可扩展。',
          },
          {
            title: '全局工作空间 (Global Workspace)',
            formula: 'W_G = ∫ (w_i · P_i) dμ',
            detail:
              '将多模块竞争后的有效信息做全局聚合，形成当前时刻的统一意识场与决策上下文。',
          },
          {
            title: '高维全息编码 (SHDC Encoding)',
            formula: '⟨v_i, v_j⟩ ≈ δ_ij',
            detail:
              '利用高维近似正交，让特征编码尽量互不干扰，从而支持高容量、低串扰的知识表示。',
          },
          {
            title: '联络与推理 (Connection Equation)',
            formula: '∇_X s = 0',
            detail:
              '将推理视为语义流形上的平行移动，约束语义在传输中保持一致，减少无关漂移。',
          },
        ],
        engineeringItems: [
          {
            name: 'Base Manifold Controller',
            status: 'done',
            focus: '底流形调度与全局约束',
            detail: '维护逻辑骨架状态，统一管理各子模块输入输出与全局稳定性边界。',
          },
          {
            name: 'Fiber Memory Bank',
            status: 'done',
            focus: '知识纤维写入与检。',
            detail: '负责高维语义纤维存储、索引与按联络条件的快速检索。',
          },
          {
            name: 'Connection Transport Layer',
            status: 'in_progress',
            focus: '跨束信息传输',
            detail: '执行底流形与纤维空间间的并行传输与语义一致性对齐。',
          },
          {
            name: 'Ricci Flow Optimizer',
            status: 'in_progress',
            focus: '流形平滑与冲突修。',
            detail: '在离。在线周期中优化曲率分布，减少推理路径扭曲与幻觉风险。',
          },
          {
            name: 'Global Workspace Arbiter',
            status: 'in_progress',
            focus: '全局工作空间竞争裁决',
            detail: '对多模块候选表征执。Top-K 选择，形成当前时刻统一决策上下文。',
          },
          {
            name: 'Alignment & Surgery Interface',
            status: 'done',
            focus: '可交互价值对。',
            detail: '通过流形手术接口对语义方向进行可控干预，支持偏差修复与对齐验证。',
          },
        ],
        nfbtProcessSteps: [
          { step: '1. 邻域图', input: 'X[N,D]', output: '邻居索引', complexity: 'O(N^2D)', op: '距离计算 / 近邻搜索' },
          { step: '2. 局部坐标', input: '邻居点', output: 'basis[d,D]', complexity: 'O(kDd)', op: '局部SVD / 随机SVD' },
          { step: '3. 度量张量', input: 'coords', output: 'g[d,d]', complexity: 'O(kd^2)', op: '局部协方差与正则化' },
          { step: '4. 联络', input: 'g, dg', output: 'Γ[d,d,d]', complexity: 'O(d^3)', op: '偏导组合与指标变换' },
          { step: '5. 曲率', input: 'Γ', output: 'R[d,d,d,d]', complexity: 'O(d^4)', op: '张量收缩与对称化' },
          { step: '6. 平行移动', input: 'Γ, v, dx', output: 'v_new', complexity: 'O(d^2)', op: '联络驱动向量更新' },
          { step: '7. Ricci Flow', input: 'R, X', output: 'X_new', complexity: 'O(T*n*d^4)', op: '离散演化迭代' },
        ],
        nfbtOptimization:
          '关键优化：d << D（如 d=4, D=128），将核心几何计算从 O(D^4) 降到 O(d^4)，并结合近似kNN、截断SVD与曲率张量对称约简降低总成本。',
        milestoneTitle: '里程碑（原 AGI 终点）',
        milestoneGoals: milestonePhase?.goals || [],
        milestoneMetrics: milestonePhase?.metrics || {},
        milestoneStages: [
          {
            id: 'prototype',
            name: '原型阶段',
            status: 'done',
            featurePoints: [
              '完成 FiberNet 核心逻辑层与底流形构建',
              '打通 NFB 几何编码与基础推理链路',
              '建立最小可用结构分析工具链（Logit Lens/TDA等）',
            ],
            tests: [
              {
                name: 'Z113 逻辑闭包验证',
                params: 'layers=12, d=4, D=128, optimizer=adamw, lr=1e-3',
                dataset: 'Z113 模运算合成数据集',
                result: '准确率 99.4%，可恢复稳定环面结构',
                summary: '证明原型具备几何逻辑骨架，不是纯统计拟合。',
              },
              {
                name: '基础拓扑可解释性测试',
                params: 'topk_heads=8, tda_threshold=0.1',
                dataset: '内部语义提示词基准集 v1',
                result: '关键层拓扑特征可稳定复现',
                summary: '原型阶段已具备可观测、可解释的结构分析能力。',
              },
            ],
          },
          {
            id: 'scale',
            name: '规模化阶段',
            status: 'in_progress',
            featurePoints: [
              '完成参数规模 × 数据规模的系统化训练矩阵验证（full preset）',
              '完成 8.5M 大模型专项调参（warmup + grad accumulation）并恢复收敛',
              '完成 5-seed 大规模稳定性复现实验，沉淀统计报告与基准文件',
              '完成 d_100k 低资源长程训练对照（36 epochs），确认当前瓶颈主要来自数据规模而非训练轮次',
            ],
            tests: [
              {
                name: 'Full Matrix 基线测试（16 runs）',
                params: 'preset=full, epochs=12, batch=256, eval_batch=2048, device=cuda',
                dataset: 'Modular Addition 合成集：d_100k/d_300k/d_700k/d_1200k',
                result: '总耗时 18.15 分钟；m_0.4m/m_1.4m/m_3.2m 在中大数据规模可收敛；m_8.5m 在默认超参下失稳（~0.009）。',
                summary: '验证了"参数放大后训练策略敏感性显著增加"，大模型不可直接复用小模型超参。',
              },
              {
                name: 'm_8.5m 专项调参测试（4 runs）',
                params: 'epochs=24, lr=2e-4, weight_decay=0.01, warmup=0.1, min_lr_scale=0.1, grad_accum=2, grad_clip=0.5, dropout=0.0',
                dataset: '同 full 数据规模四档：d_100k/d_300k/d_700k/d_1200k',
                result: 'best_val_acc：0.7984 / 0.9905 / 0.9999 / 1.0000（由默认配置的 ~0.009 全面恢复）。',
                summary: '调参后 8.5M 已具备稳定收敛能力，且随数据规模增加表现持续提升。',
              },
              {
                name: 'm_8.5m 多随机种子稳定性（5 seeds, 20 runs）',
                params: '固定 tuned 配置，seed 组：42 / 314 / 2026 / 4096 / 8192',
                dataset: 'd_100k/d_300k/d_700k/d_1200k',
                result: '均值(best)：0.793640 / 0.990581 / 0.999949 / 1.000000；std：0.004160 / 0.000616 / 0.000042 / 0.000000',
                summary: '300k+ 数据规模下结果稳定且高分，100k 档位仍存在数据瓶颈（约 0.79 上限）。',
              },
              {
                name: 'm_8.5m 低资源长程训练（3 seeds, d_100k, epochs=36）',
                params: '同 tuned 配置，epochs 从 24 提升到 36；seed：10001 / 20002 / 30003',
                dataset: 'd_100k',
                result: 'best_val_acc：0.794689 / 0.788311 / 0.791244；mean=0.791415，std=0.002607',
                summary: '与 epochs=24 的 d_100k 结果相比无显著提升，说明低资源场景应优先补充数据或引入更强正则与数据增强策略。',
              },
              {
                name: 'WikiText 几何涌现 (Phase 3)',
                params: '20M Params, Split Stream',
                dataset: 'WikiText-2 (10M Tokens)',
                result: 'Loss: 0.529, ID Peak: 31.5',
                summary: '时间: 2026-02-19 | 数据: Ep 1-47 | 分析: 观测到完整的 ID 压缩(10.5)->膨胀(31.5)->微调(29.3) 呼吸周期。 | 结论: 验证了 SHMC 理论中流形动态重组的物理机制。',
              },
            ],
          },
          {
            id: 'agi',
            name: 'AGI阶段',
            status: 'planned',
            featurePoints: [
              '构建统一意识裁决中心（多路线仲裁。',
              '实现具身控制闭环与安全对齐机。',
              '完成跨模型迁移与长期自治学习框架',
            ],
            tests: [
              {
                name: '全局工作空间端到端压。',
                params: 'modules>=7, arbitration=Top-K, latency<200ms',
                dataset: 'Multi-Agent Conflict Suite',
                result: '待执行',
                summary: '用于验证复杂冲突场景下的稳定裁决能力。',
              },
              {
                name: '具身控制闭环测试',
                params: 'control_horizon=128, safety_guard=on',
                dataset: 'Embodied Interaction Set',
                result: '待执行',
                summary: '用于验证感知-决策-行动闭环的一致性与安全边界。',
              },
            ],
          },
        ],
        milestonePlanEvaluation: {
          assessment:
            '里程碑已从"功能演示"升级为"规模化证据链"：完成 full 矩阵、专项调参与多 seed 复现，证明大模型可训练性与稳定性。',
          suggestions: [

            '将每阶段验收门槛量化（准确率、稳定性、时延、成本）。',
            '规模化阶段增加故障注入与恢复时间指标（MTTR）。',
            'Phase 4: TinyStories 规模化结晶实验 (100M Params) 正在运行 (Batch 500+, ID~12.0)。',

            '将规模化阶段验收门槛固定为：mean/std、训练耗时、吞吐、失败率四项硬指标。',
            '补充 OOD 与噪声扰动测试，验证高分是否可迁移而非数据内记忆。',
            '针对 d_100k 低资源场景继续优化（更长训练、正则与学习率策略），形成小数据稳态方案。',
          ],
        },
      },

    }),
    [engineeringPhase?.sub_phases, milestonePhase?.goals, milestonePhase?.metrics, theoryPhase?.definition?.headline, theoryPhase?.definition?.summary, theoryPhase?.theory_content]
  );

  const routeList = useMemo(() => {
    const runtimeIds = timelineRoutes
      .map((item) => item?.route)
      .filter((id) => typeof id === 'string' && id.length > 0);
    const baseIds = Object.keys(routeBlueprints);
    const allIds = Array.from(new Set([...baseIds, ...runtimeIds]));

    return allIds.map((id) => {
      const base = routeBlueprints[id] || {
        id,
        title: id,
        subtitle: '实验路线',
        routeDescription: '该路线正在构建中，描述信息待补充。',
        engineeringProcessDescription: '计算流程说明待补充。',
        theoryTitle: '待补充理论',
        theorySummary: '该路线尚未配置详细理论描述。',
        theoryBullets: [],
        theoryFormulas: [],
        engineeringItems: [],
        nfbtProcessSteps: [],
        nfbtOptimization: '',
        milestoneTitle: '里程碑目标（AGI 终点）',
        milestoneGoals: [],
        milestoneMetrics: {},
        milestoneStages: [],
        milestonePlanEvaluation: null,
      };
      const runtime = timelineRoutes.find((item) => item?.route === id);
      const stats = runtime?.stats || {};
      const totalRuns = Number(stats.total_runs || 0);
      const completedRuns = Number(stats.completed_runs || 0);
      const avgScore = Number(stats.avg_score || 0);
      const routeProgress =
        totalRuns > 0
          ? Math.max(
            0,
            Math.min(100, Math.round((completedRuns / Math.max(1, totalRuns)) * 60 + avgScore * 40))
          )
          : 0;
      return {
        ...base,
        stats: {
          totalRuns,
          completedRuns,
          failedRuns: Number(stats.failed_runs || 0),
          avgScore,
          routeProgress,
        },
      };
    });
  }, [routeBlueprints, timelineRoutes]);

  useEffect(() => {
    if (routeList.length === 0) return;
    if (!routeList.some((item) => item.id === selectedRouteId)) {
      setSelectedRouteId(routeList[0].id);
    }
  }, [routeList, selectedRouteId]);

  useEffect(() => {
    setExpandedFormulaIdx(null);
    setExpandedEngPhase(null);
    setExpandedParam(null);
  }, [selectedRouteId]);

  const selectedRoute = routeList.find((item) => item.id === selectedRouteId) || routeList[0];
  const systemRouteOptions = routeList.filter((item) =>
    ['fiber_bundle'].includes(item.id)
  );
  const selectedMultimodalData = multimodalSummary?.views?.[multimodalView] || null;
  const selectedMultimodalReport = selectedMultimodalData?.report || null;
  const selectedMultimodalBest = selectedMultimodalReport?.summary?.best || null;
  const selectedMultimodalLatest = selectedMultimodalData?.latest_test || null;

  const multimodalMetricRows = useMemo(() => {
    if (!selectedMultimodalBest) return [];
    if (multimodalView === 'vision_alignment') {
      return [
        { label: '最佳轮次', value: selectedMultimodalBest.epoch },
        { label: 'Val Accuracy', value: Number(selectedMultimodalBest.val_acc || 0).toFixed(4) },
        { label: 'Anchor Cos', value: Number(selectedMultimodalBest.val_anchor_cos || 0).toFixed(4) },
        { label: 'Val Loss', value: Number(selectedMultimodalBest.val_loss || 0).toFixed(4) },
      ];
    }
    return [
      { label: '最佳轮次', value: selectedMultimodalBest.epoch },
      { label: 'Val Fused Acc', value: Number(selectedMultimodalBest.val_fused_acc || 0).toFixed(4) },
      { label: 'Retrieval@1', value: Number(selectedMultimodalBest.val_retrieval_top1 || 0).toFixed(4) },
      { label: 'Align Cos', value: Number(selectedMultimodalBest.val_alignment_cos || 0).toFixed(4) },
    ];
  }, [selectedMultimodalBest, multimodalView]);

  const getRouteImpl = (capability) => {
    const map = capability?.implementation_by_route || {};
    return (
      map[selectedRouteId] ||
      map[selectedRoute?.id] ||
      capability?.desc ||
      '该路线实现描述待补充。'
    );
  };

  const systemProfiles = useMemo(
    () => ({
      fiber_bundle: {
        metricCards: [
          {
            label: '内稳态调。',
            brain_ability: '稳态维持与资源分配',
            value: consciousField ? `${((consciousField.stability || 0) * 100).toFixed(1)}%` : '92.0%',
            color: '#10b981',
          },
          {
            label: '工作记忆负载',
            brain_ability: '短时记忆与上下文保持',
            value: consciousField ? `${consciousField.memory_load || 0}%` : '68%',
            color: '#00d2ff',
          },
          {
            label: '跨域共振',
            brain_ability: '跨模态联想整。',
            value: consciousField ? (consciousField.resonance || 0).toFixed(3) : '0.742',
            color: '#ffaa00',
          },
          {
            label: '意识竞争强度',
            brain_ability: '注意焦点竞争与广。',
            value: consciousField ? (consciousField.gws_intensity || 0).toFixed(2) : '0.81',
            color: '#a855f7',
          },
        ],
        parameterCards: [
          {
            name: '几何潜空间配。',
            brain_ability: '抽象结构建模',
            route_param: 'd=4, D=128, manifold=riemannian',
            detail: '低维几何。+ 高维语义外壳',
            desc: '通过 d<<D 降低核心几何计算复杂度，同时保持语义表达容量。',
            value_meaning: '兼顾可解释性、稳定性与计算成本。',
            why_important: '决定几何推理是否可持续扩展。',
          },
          {
            name: '联络与平行移。',
            brain_ability: '推理路径保持',
            route_param: 'transport=connection_based, step=adaptive',
            detail: 'Γ 驱动语义平移',
            desc: '沿流形执行平行移动，减少语义漂移。',
            value_meaning: '推理链更稳定，抗扰动能力更强。',
            why_important: '是从"拟合"走向"结构推理"的关键。'
          },
          {
            name: 'Ricci Flow 婕斿寲',
            brain_ability: '睡眠重整与冲突修。',
            route_param: 'iterations=100, reg=1e-3',
            detail: '离线曲率平滑',
            desc: '通过流形平滑降低逻辑尖峰与幻觉风险。',
            value_meaning: '提升长期稳定性与一致性。',
            why_important: '支持系统持续自我修复。'
          },
          {
            name: '全局工作空间',
            brain_ability: '意识竞争裁决',
            route_param: 'top_k=8, arbitration=winner_take_all',
            detail: '多模块竞争广。',
            desc: '在冲突候选中选取最优表示并广播。',
            value_meaning: '保证实时决策聚焦有效信息。',
            why_important: '直接影响系统响应质量与时延。'
          },
        ],
        validationRecords: (statusData?.passed_tests || []).map((t) => ({
          ...t,
          brain_ability: t.brain_ability || '结构推理稳定与记忆重。',
          route_param_focus: t.route_param_focus || 'manifold_dim=4, top_k=8, ricci_iterations=100',
        })),
      },
    }),
    [consciousField, selectedRouteId, statusData?.passed_tests]
  );

  const activeSystemProfile = systemProfiles[selectedRouteId] || systemProfiles.fiber_bundle;
  const mergedMilestoneStages = useMemo(() => {
    const baseStages = selectedRoute?.milestoneStages || [];
    const routeTests = activeSystemProfile?.validationRecords || [];
    if (!routeTests.length) return baseStages;

    const routeValidationStage = {
      id: 'route_validation',
      name: '路线测试记录',
      status: routeTests.every((t) => String(t?.result || '').toUpperCase().includes('PASS')) ? 'done' : 'in_progress',
      featurePoints: [
        `来源：系统状态 / ${selectedRoute?.title || selectedRouteId}`,
        `测试数量：${routeTests.length}`,
        '作为里程碑验收证据沉淀到研发进展',
      ],
      tests: routeTests.map((t) => ({
        name: t.name || '未命名测试',
        params: t.route_param_focus || t.params || '-',
        dataset: t.dataset || (t.date ? `验证日期: ${t.date}` : '-'),
        result: t.result || '-',
        summary: t.significance || t.summary || '-',
      })),
    };

    return [...baseStages, routeValidationStage];
  }, [selectedRoute, selectedRouteId, activeSystemProfile]);
  return (
    <div style={mode === 'sidebar' ? {
      position: 'absolute', top: '20px', right: '20px', bottom: '20px', width: '500px',
      backgroundColor: 'rgba(10, 10, 18, 0.98)', zIndex: 102,
      display: 'flex', flexDirection: 'column', color: '#fff',
      transform: 'translateZ(0)',
      willChange: 'auto',
      fontFamily: '"SF Mono", "Roboto Mono", monospace', overflow: 'hidden',
      borderRadius: '12px',
      border: '1px solid rgba(0, 210, 255, 0.2)',
      boxShadow: '0 16px 32px rgba(0,0,0,0.42)'
    } : {
      position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh',
      backgroundColor: 'rgba(5, 5, 10, 0.98)', backdropFilter: 'blur(30px)', zIndex: 2000,
      display: 'flex', flexDirection: 'column', color: '#fff',
      fontFamily: '"SF Mono", "Roboto Mono", monospace', overflow: 'hidden'
    }}>
      {/* Custom Keyframes */}
      <style>{`
        @keyframes roadmapFade { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes brainPulse { from { scale: 0.95; opacity: 0.8; } to { scale: 1.05; opacity: 1; } }
        @keyframes brainRotate { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes brainRotateReverse { from { transform: rotate(0deg); } to { transform: rotate(-360deg); } }
        @keyframes synapsePulse { 0%, 100% { opacity: 0.3; scale: 0.8; } 50% { opacity: 1; scale: 1.2; } }
      `}</style>

      {/* Top Header / Navigation */}
      <div style={mode === 'sidebar' ? {
        padding: '0 16px', height: '60px', display: 'flex', justifyContent: 'space-between',
        alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.1)', background: 'rgba(0,0,0,0.3)',
        flexShrink: 0
      } : {
        padding: '0 40px', height: '80px', display: 'flex', justifyContent: 'space-between',
        alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.1)', background: 'rgba(0,0,0,0.3)',
        flexShrink: 0
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: mode === 'sidebar' ? '12px' : '50px', height: '100%', overflow: 'hidden' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexShrink: 0 }}>
            <Brain size={mode === 'sidebar' ? 20 : 28} color="#00d2ff" />
            <span style={{ fontSize: mode === 'sidebar' ? '14px' : '18px', fontWeight: 'bold', letterSpacing: '1px' }}>
              {scope === 'rnd' ? '自动研发' : '理论分析'}
            </span>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>
          <button onClick={onClose} style={{
            background: 'rgba(255,255,255,0.05)', border: 'none', color: '#fff', cursor: 'pointer',
            width: mode === 'sidebar' ? '30px' : '40px', height: mode === 'sidebar' ? '30px' : '40px',
            borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center'
          }} onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,100,100,0.2)'} onMouseLeave={e => e.currentTarget.style.background = 'rgba(255,255,255,0.05)'}>
            <X size={mode === 'sidebar' ? 16 : 22} />
          </button>
        </div>
      </div>

      {/* Secondary Sub-Navbar for Theory */}
      {scope === 'theory' && THEORY_TABS.has(activeTab) && (
        <div style={{
          display: 'flex', gap: '6px', padding: '10px 16px 0',
          background: 'rgba(0,0,0,0.15)',
          flexShrink: 0, overflowX: 'auto', scrollbarWidth: 'none', msOverflowStyle: 'none'
        }}>
          <button
            onClick={() => handleTabChange('atlas_control')}
            style={{
              background: activeTab === 'atlas_control' ? 'rgba(0, 210, 255, 0.12)' : 'rgba(255,255,255,0.02)',
              border: `1px solid ${activeTab === 'atlas_control' ? 'rgba(0, 210, 255, 0.25)' : 'rgba(255,255,255,0.05)'}`,
              borderRadius: '6px',
              color: activeTab === 'atlas_control' ? '#00d2ff' : '#999',
              fontSize: '11px', fontWeight: 'bold', cursor: 'pointer',
              padding: '6px 12px', flexShrink: 0,
              transition: 'all 0.2s'
            }}
          >
            图谱总控
          </button>
        </div>
      )}
      {scope === 'theory' && THEORY_TABS.has(activeTab) && (
        <div style={{
          display: 'flex', gap: '6px', padding: '10px 16px',
          background: 'rgba(0,0,0,0.15)', borderBottom: '1px solid rgba(255,255,255,0.06)',
          flexShrink: 0, overflowX: 'auto', scrollbarWidth: 'none', msOverflowStyle: 'none'
        }}>
          {[
            { id: 'roadmap', label: '项目大纲' },
            { id: 'language', label: '语言分析' },
            { id: 'analysis', label: '智能理论' },
            { id: 'progress', label: '模型研发' },
            { id: 'system', label: '系统状态' },
          ].map(t => (
            <button
              key={t.id}
              onClick={() => handleTabChange(t.id)}
              style={{
                background: activeTab === t.id ? 'rgba(0, 210, 255, 0.12)' : 'rgba(255,255,255,0.02)',
                border: `1px solid ${activeTab === t.id ? 'rgba(0, 210, 255, 0.25)' : 'rgba(255,255,255,0.05)'}`,
                borderRadius: '6px',
                color: activeTab === t.id ? '#00d2ff' : '#999',
                fontSize: '11px', fontWeight: 'bold', cursor: 'pointer',
                padding: '6px 12px', flexShrink: 0,
                transition: 'all 0.2s'
              }}
              onMouseEnter={e => {
                if (activeTab !== t.id) {
                  e.currentTarget.style.color = '#fff';
                  e.currentTarget.style.background = 'rgba(255,255,255,0.05)';
                }
              }}
              onMouseLeave={e => {
                if (activeTab !== t.id) {
                  e.currentTarget.style.color = '#999';
                  e.currentTarget.style.background = 'rgba(255,255,255,0.02)';
                }
              }}
            >
              {t.label}
            </button>
          ))}
        </div>
      )}

      {/* Secondary Sub-Navbar for AI Auto R&D */}
      {scope === 'rnd' && RND_TABS.has(activeTab) && (
        <div style={{
          display: 'flex', gap: '6px', padding: '10px 16px',
          background: 'rgba(0,0,0,0.15)', borderBottom: '1px solid rgba(167,139,250,0.12)',
          flexShrink: 0, overflowX: 'auto', scrollbarWidth: 'none', msOverflowStyle: 'none'
        }}>
          {[
            { id: 'rnd_console', label: '研发控制台' },
            { id: 'rnd_config', label: '研发配置' },
          ].map(t => (
            <button
              key={t.id}
              onClick={() => handleTabChange(t.id)}
              style={{
                background: activeTab === t.id ? 'rgba(167, 139, 250, 0.14)' : 'rgba(255,255,255,0.02)',
                border: `1px solid ${activeTab === t.id ? 'rgba(167, 139, 250, 0.35)' : 'rgba(255,255,255,0.05)'}`,
                borderRadius: '6px',
                color: activeTab === t.id ? '#c4b5fd' : '#999',
                fontSize: '11px', fontWeight: 'bold', cursor: 'pointer',
                padding: '6px 12px', flexShrink: 0,
                transition: 'all 0.2s'
              }}
              onMouseEnter={e => {
                if (activeTab !== t.id) {
                  e.currentTarget.style.color = '#fff';
                  e.currentTarget.style.background = 'rgba(255,255,255,0.05)';
                }
              }}
              onMouseLeave={e => {
                if (activeTab !== t.id) {
                  e.currentTarget.style.color = '#999';
                  e.currentTarget.style.background = 'rgba(255,255,255,0.02)';
                }
              }}
            >
              {t.label}
            </button>
          ))}
        </div>
      )}

      {/* Main Content Area */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

        {/* Sub-Sidebar for Research Progress */}
        {scope === 'theory' && activeTab === 'progress' && mode !== 'sidebar' && (
          <div style={{
            width: '280px', borderRight: '1px solid rgba(255,255,255,0.1)',
            padding: '30px 20px', background: 'rgba(0,0,0,0.2)', overflowY: 'auto',
            position: 'relative'
          }}>
            <div style={{ fontSize: '10px', color: '#444', textTransform: 'uppercase', marginBottom: '30px', letterSpacing: '2px', fontWeight: 'bold' }}>Research Routes</div>

            {/* Vertical Timeline Line */}
            <div style={{
              position: 'absolute', left: '38px', top: '80px', bottom: '40px',
              width: '1px', background: 'linear-gradient(to bottom, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%)',
              zIndex: 0
            }} />

            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', position: 'relative', zIndex: 1 }}>
              {routeList.map((routeItem) => (
                <div key={routeItem.id} style={{ position: 'relative' }}>
                  {/* Timeline Dot */}
                  <div style={{
                    position: 'absolute', left: '15px', top: '22px',
                    width: '6px', height: '6px', borderRadius: '50%',
                    background: selectedRoute?.id === routeItem.id ? '#00d2ff' : '#222',
                    border: `2px solid ${selectedRoute?.id === routeItem.id ? '#000' : 'rgba(255,255,255,0.1)'}`,
                    boxShadow: selectedRoute?.id === routeItem.id ? '0 0 10px #00d2ff' : 'none',
                    transition: 'all 0.3s'
                  }} />

                  <button
                    onClick={() => setSelectedRouteId(routeItem.id)}
                    style={{
                      width: '100%', padding: '12px 12px 12px 45px', borderRadius: '14px',
                      textAlign: 'left', cursor: 'pointer',
                      background: selectedRoute?.id === routeItem.id ? 'rgba(255,255,255,0.03)' : 'transparent',
                      border: 'none',
                      color: selectedRoute?.id === routeItem.id ? '#fff' : '#666', transition: 'all 0.3s',
                      display: 'flex', flexDirection: 'column', gap: '4px'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                      <span style={{ fontSize: '14px', fontWeight: 'bold', color: selectedRoute?.id === routeItem.id ? '#fff' : '#888' }}>
                        {routeItem.title}
                      </span>
                      <span style={{
                        fontSize: '11px', fontFamily: 'monospace', fontWeight: 'bold',
                        color: selectedRoute?.id === routeItem.id ? '#00d2ff' : '#444'
                      }}>
                        {routeItem.stats.routeProgress}%
                      </span>
                    </div>
                    <div style={{ fontSize: '10px', color: '#444' }}>
                      run {routeItem.stats.totalRuns} | success {routeItem.stats.completedRuns} | avg {(routeItem.stats.avgScore * 100).toFixed(1)}%
                    </div>
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Content Details */}
        <div style={{
          flex: 1, padding: mode === 'sidebar' ? '20px 24px' : '50px 80px', overflowY: 'auto',
          background: 'radial-gradient(circle at 50% 10%, rgba(0, 100, 200, 0.05) 0%, transparent 70%)'
        }}>

          {/* TAB: Project Roadmap */}
          {scope === 'theory' && activeTab === 'roadmap' && (
            <ProjectRoadmapTab
              roadmapData={roadmapData}
              analysisPhase={analysisPhase}
              evidenceDrivenPlan={EVIDENCE_DRIVEN_PLAN}
              executionPlaybook={EXECUTION_PLAYBOOK}
              mathRouteSystemPlan={MATH_ROUTE_SYSTEM_PLAN}
              improvements={IMPROVEMENTS}
              expandedImprovementPhase={expandedImprovementPhase}
              setExpandedImprovementPhase={setExpandedImprovementPhase}
              expandedImprovementTest={expandedImprovementTest}
              setExpandedImprovementTest={setExpandedImprovementTest}
            />
          )}

          {/* TAB: Language Analysis */}
          {scope === 'theory' && activeTab === 'language' && (
            <LanguageAnalysisTab />
          )}

          {/* TAB: Pattern Family Atlas Control */}
          {scope === 'theory' && activeTab === 'atlas_control' && (
            <AtlasControlDashboard />
          )}

          {/* TAB: Deep Analysis / Model Comparison */}
          {scope === 'theory' && activeTab === 'analysis' && (
            <DeepAnalysisTab
              evidenceDrivenPlan={EVIDENCE_DRIVEN_PLAN}
              improvements={IMPROVEMENTS}
              expandedImprovementPhase={expandedImprovementPhase}
              setExpandedImprovementPhase={setExpandedImprovementPhase}
              expandedImprovementTest={expandedImprovementTest}
              setExpandedImprovementTest={setExpandedImprovementTest}
            />
          )}

          {/* TAB: Research Progress (Route-Centric Command) */}
          {scope === 'theory' && activeTab === 'progress' && selectedRoute && (
            <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: '16px' }}>
              {mode === 'sidebar' && (
                <div style={{ marginBottom: '8px', flexShrink: 0 }}>
                  <label style={{ fontSize: '11px', color: '#888', display: 'block', marginBottom: '6px', fontWeight: 'bold', letterSpacing: '1px' }}>选择研究路线 (Research Route)</label>
                  <select
                    value={selectedRouteId}
                    onChange={(e) => setSelectedRouteId(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '8px 12px',
                      background: 'rgba(0,0,0,0.6)',
                      border: '1px solid rgba(0, 210, 255, 0.3)',
                      borderRadius: '8px',
                      color: '#fff',
                      fontSize: '13px',
                      outline: 'none',
                      cursor: 'pointer',
                      fontFamily: 'inherit'
                    }}
                  >
                    {routeList.map((r) => (
                      <option key={r.id} value={r.id} style={{ background: '#111' }}>
                        {r.title} ({r.stats.routeProgress}%)
                      </option>
                    ))}
                  </select>
                </div>
              )}
              <ResearchProgressTab
                selectedRoute={selectedRoute}
                expandedFormulaIdx={expandedFormulaIdx}
                setExpandedFormulaIdx={setExpandedFormulaIdx}
                dnnAnalysisPlan={DNN_ANALYSIS_PLAN}
                expandedEngPhase={expandedEngPhase}
                setExpandedEngPhase={setExpandedEngPhase}
                mergedMilestoneStages={mergedMilestoneStages}
                multimodalView={multimodalView}
                setMultimodalView={setMultimodalView}
                multimodalError={multimodalError}
                selectedMultimodalReport={selectedMultimodalReport}
                selectedMultimodalData={selectedMultimodalData}
                selectedMultimodalLatest={selectedMultimodalLatest}
                multimodalMetricRows={multimodalMetricRows}
              />
            </div>
          )}

          {/* TAB: AGI System Status */}
          {scope === 'theory' && activeTab === 'system' && (
            <SystemStatusTab
              consciousField={consciousField}
              systemRouteOptions={systemRouteOptions}
              routeList={routeList}
              setSelectedRouteId={setSelectedRouteId}
              selectedRouteId={selectedRouteId}
              activeSystemProfile={activeSystemProfile}
              statusData={statusData}
              selectedRoute={selectedRoute}
              getRouteImpl={getRouteImpl}
              expandedParam={expandedParam}
              setExpandedParam={setExpandedParam}
            />
          )}

          {/* TAB: AI Auto R&D Console */}
          {scope === 'rnd' && activeTab === 'rnd_console' && (
            <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: '16px' }}>
              {/* Phase Progress Bar */}
              {sessionStatus === 'running' && (
                <div style={{
                  display: 'flex', alignItems: 'center', gap: 4, padding: '12px 16px',
                  background: 'rgba(10, 10, 16, 0.4)', borderBottom: '1px solid rgba(167, 139, 250, 0.15)',
                  borderRadius: '8px', overflowX: 'auto', scrollbarWidth: 'none', msOverflowStyle: 'none',
                  width: '100%', boxSizing: 'border-box'
                }}>
                  {RESEARCH_PHASES.map((phase, i) => {
                    const isCurrent = currentPhase === phase.id;
                    const Icon = PHASE_ICONS[phase.id] || Search;
                    return (
                      <div key={phase.id} style={{ display: 'flex', alignItems: 'center', gap: 3, flexShrink: 0 }}>
                        <div style={{
                          padding: '4px 8px', borderRadius: '16px', fontSize: '11px', fontWeight: 'bold',
                          background: isCurrent ? `${phase.color}20` : 'rgba(255,255,255,0.02)',
                          border: `1px solid ${isCurrent ? phase.color : 'rgba(255,255,255,0.08)'}`,
                          color: isCurrent ? phase.color : '#888',
                          transition: 'all 0.3s',
                          boxShadow: isCurrent ? `0 0 12px ${phase.color}40` : 'none',
                          display: 'flex', alignItems: 'center', gap: '4px',
                        }}>
                          <Icon size={12} style={{ filter: isCurrent ? `drop-shadow(0 0 3px ${phase.color})` : 'none' }} />
                          <span>{phase.label}</span>
                        </div>
                        {i < RESEARCH_PHASES.length - 1 && (
                          <div style={{ 
                            width: 12, height: 2, 
                            background: isCurrent ? `linear-gradient(90deg, ${phase.color}, rgba(255,255,255,0.1))` : 'rgba(255,255,255,0.08)',
                            transition: 'all 0.3s',
                            flexShrink: 0
                          }} />
                        )}
                      </div>
                    );
                  })}
                  <div style={{ marginLeft: 'auto', fontSize: '11px', color: '#9ca3af', fontFamily: 'monospace', fontWeight: 'bold', flexShrink: 0, paddingLeft: 8 }}>
                    R{round}
                  </div>
                </div>
              )}

              {/* R&D Error banner */}
              {rndError && (
                <div style={{
                  padding: '8px 16px', background: 'rgba(255,50,50,0.1)', borderBottom: '1px solid rgba(255,50,50,0.3)',
                  borderRadius: '6px', color: '#ff6666', fontSize: '13px',
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                }}>
                  <span>{rndError}</span>
                  <button onClick={() => setRndError(null)} style={{ background: 'none', border: 'none', color: '#ff6666', cursor: 'pointer' }}>
                    <X size={14} />
                  </button>
                </div>
              )}

              <AIRnDConsoleTab
                sessionStatus={sessionStatus}
                sessionMode={sessionMode}
                currentPhase={currentPhase}
                round={round}
                logs={logs}
                generatedCode={generatedCode}
                executionResult={executionResult}
                findings={findings}
                onStart={startSession}
                onPause={pauseSession}
                onResume={resumeSession}
                onStop={stopSession}
                onToggleMode={toggleMode}
                onStep={stepNext}
                onClear={() => setLogs([])}
              />
            </div>
          )}

          {/* TAB: AI Auto R&D Config */}
          {scope === 'rnd' && activeTab === 'rnd_config' && (
            <AIRnDConfigTab />
          )}

        </div>
      </div>
    </div>
  );
};






