
import { Activity, Network, Play, Zap, Send, Bot, User, RefreshCw, Loader2, Cpu, ChevronRight, BarChart2, MessageSquare, Waves, Info } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import axios from 'axios';

// UI Components
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');

const FiberNetPanel = () => {
  const [language, setLanguage] = useState('en');
  const [inputText, setInputText] = useState('I love her');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('observer'); // observer | energy | chat

  // Energy (Mother Engine) State
  const [energyPrompt, setEnergyPrompt] = useState("The artificial");
  const [energySteps, setEnergySteps] = useState(15);
  const [isEnergyGenerating, setIsEnergyGenerating] = useState(false);
  const [energyResult, setEnergyResult] = useState(null);
  const [renderedTraces, setRenderedTraces] = useState([]);
  const [renderedEnergyText, setRenderedEnergyText] = useState('');

  // Chat State
  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [chatStatus, setChatStatus] = useState({ is_ready: false, status_msg: 'Checking...' });

  const canvasRef = useRef(null);
  const messagesEndRef = useRef(null);
  const animRef = useRef(null);

  const presets = {
    en: "I love her",
    fr: "Je aime la"
  };

  useEffect(() => {
    setInputText(presets[language] || "");
    setResult(null);
  }, [language]);

  const checkChatStatus = async () => {
    try {
      const res = await axios.get(`${API_BASE}/api/agi_chat/status`);
      setChatStatus(res.data);
    } catch (e) {
      setChatStatus({ is_ready: false, status_msg: 'Engine Offline' });
    }
  };

  useEffect(() => {
    checkChatStatus();
    const interval = setInterval(checkChatStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const handleInference = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/fibernet/inference`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText, lang: language })
      });
      const data = await response.json();
      setResult(data);

      setMessages(prev => [...prev, { role: 'sys', content: `Topology synchronized: ${data.tokens.length} nodes collapsed.` }]);
    } catch (error) {
      console.error("Inference failed:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleEnergyGenerate = async () => {
    if (!energyPrompt.trim() || isEnergyGenerating) return;

    setIsEnergyGenerating(true);
    setEnergyResult(null);
    setRenderedTraces([]);
    setRenderedEnergyText('');

    try {
      const response = await axios.post(`${API_BASE}/api/mother-engine/generate`, {
        prompt: energyPrompt,
        steps: energySteps
      });

      if (response.data && response.data.status === 'success') {
        setEnergyResult(response.data);
        animateEnergy(response.data.traces);
        setMessages(prev => [...prev, { role: 'sys', content: `Energy Pulse: ${response.data.traces.length} steps of O(1) collapse triggered.` }]);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setIsEnergyGenerating(false);
    }
  };

  const animateEnergy = (traces) => {
    if (!traces || traces.length === 0) return;
    let p = 0;
    const tick = () => {
      if (p < traces.length) {
        const currentTrace = traces[p];
        setRenderedTraces(prev => [...prev, currentTrace]);
        setRenderedEnergyText(prev => prev + currentTrace.token_str);
        p++;
        animRef.current = setTimeout(tick, 150 + Math.random() * 150);
      }
    };
    tick();
  };

  const handleChatSend = async () => {
    if (!chatInput.trim() || !chatStatus.is_ready) return;

    const userMsg = chatInput.trim();
    setChatInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setIsTyping(true);

    try {
      const res = await axios.post(`${API_BASE}/api/agi_chat/generate`, {
        prompt: userMsg,
        max_tokens: 50
      });

      if (res.data && res.data.generated_text) {
        setMessages(prev => [...prev, { role: 'agi', content: res.data.generated_text }]);
      }
    } catch (e) {
      setMessages(prev => [...prev, { role: 'sys', content: `Error: ${e.response?.data?.detail || e.message}` }]);
    } finally {
      setIsTyping(false);
    }
  };

  useEffect(() => {
    return () => { if (animRef.current) clearTimeout(animRef.current); };
  }, []);

  const syncToInference = (text) => {
    setInputText(text);
    setActiveTab('observer');
  };

  const resetChat = async () => {
    try {
      await axios.post(`${API_BASE}/api/agi_chat/reset`);
      setMessages([]);
    } catch (e) { console.error(e); }
  };

  // Draw Heatmap
  useEffect(() => {
    if (!result || !result.attention || !canvasRef.current) return;

    const attn = result.attention[0];
    const tokens = result.tokens;
    const size = attn.length;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const cellSize = 50;
    const padding = 40;

    canvas.width = size * cellSize + padding * 2;
    canvas.height = size * cellSize + padding * 2;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const weight = attn[i][j];
        const intensity = Math.floor((1 - weight) * 255);
        ctx.fillStyle = `rgb(${intensity}, ${intensity}, 255)`;
        ctx.fillRect(padding + j * cellSize, padding + i * cellSize, cellSize, cellSize);

        ctx.fillStyle = weight > 0.5 ? 'white' : 'black';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(weight.toFixed(2), padding + j * cellSize + cellSize / 2, padding + i * cellSize + cellSize / 2);
      }
    }

    ctx.fillStyle = 'black';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    tokens.forEach((token, idx) => {
      ctx.fillText(token, padding + idx * cellSize + cellSize / 2, padding - 10);
    });
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    tokens.forEach((token, idx) => {
      ctx.fillText(token, padding - 10, padding + idx * cellSize + cellSize / 2);
    });
  }, [result]);

  return (
    <div className="h-full flex flex-col p-4 bg-slate-950 font-sans text-slate-200 overflow-y-auto">
      <Card className="mb-6 bg-slate-900 border-slate-800 shadow-2xl">
        <CardHeader className="pb-4">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <CardTitle className="flex items-center gap-3 text-slate-100">
              <div className="p-2 bg-indigo-500/10 rounded-lg border border-indigo-500/20">
                <Cpu className="h-6 w-6 text-indigo-400" />
              </div>
              <div className="flex flex-col">
                <span className="text-xl tracking-tight font-black uppercase">FiberNet Integrated Lab</span>
                <span className="text-[10px] text-slate-500 font-mono tracking-widest uppercase opacity-80">Phase XXXII: Physical Conscious Field</span>
              </div>
            </CardTitle>

            <div className="flex bg-slate-950/50 p-1 rounded-xl border border-slate-800 self-start">
              {[
                { id: 'observer', label: 'Theory Flow', icon: Waves },
                { id: 'energy', label: 'Energy Spikes', icon: Zap },
                { id: 'chat', label: 'Neural Terminal', icon: MessageSquare }
              ].map(tab => (
                <Button
                  key={tab.id}
                  variant={activeTab === tab.id ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setActiveTab(tab.id)}
                  className={`rounded-lg transition-all px-4 ${activeTab === tab.id ? 'bg-indigo-600 shadow-lg shadow-indigo-600/20' : 'text-slate-500 hover:text-indigo-400'}`}
                >
                  <tab.icon className={`h-3.5 w-3.5 mr-2 ${activeTab === tab.id ? 'text-white' : ''}`} />
                  <span className="text-[11px] font-bold uppercase tracking-wider">{tab.label}</span>
                </Button>
              ))}
            </div>
          </div>
        </CardHeader>

        <CardContent>
          {activeTab === 'observer' && (
            <div className="animate-in fade-in slide-in-from-bottom-2 duration-500">
              <div className="flex flex-col md:flex-row gap-4 mb-6">
                <div className="flex bg-slate-950 rounded-lg p-1 border border-slate-800 h-10">
                  <Button variant={language === 'en' ? 'secondary' : 'ghost'} onClick={() => setLanguage('en')} size="sm">EN</Button>
                  <Button variant={language === 'fr' ? 'secondary' : 'ghost'} onClick={() => setLanguage('fr')} size="sm">FR</Button>
                </div>
                <Input
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  className="flex-1 bg-slate-950 border-slate-800 h-10 text-slate-100 font-mono"
                  placeholder="Input concept for topology visualization..."
                />
                <Button onClick={handleInference} disabled={loading} className="bg-indigo-600 hover:bg-indigo-700 h-10 px-6 font-bold">
                  {loading ? <Activity className="animate-spin mr-2 h-4 w-4" /> : <Play className="mr-2 h-4 w-4" />}
                  ACTIVATE FLOW
                </Button>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card className="bg-slate-950 border-slate-800">
                  <CardHeader className="py-3 border-b border-slate-800">
                    <CardTitle className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">Substantial Heatmap</CardTitle>
                  </CardHeader>
                  <CardContent className="flex justify-center p-6 min-h-[300px]">
                    {result ? <canvas ref={canvasRef} className="rounded" /> : <div className="text-slate-700 flex flex-col items-center justify-center gap-4 opacity-40"><BarChart2 className="h-12 w-12" /><span className="text-xs font-mono uppercase tracking-widest">Awaiting manifold collapse...</span></div>}
                  </CardContent>
                </Card>

                <Card className="bg-slate-950 border-slate-800">
                  <CardHeader className="py-3 border-b border-slate-800">
                    <CardTitle className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">Entropy Analysis</CardTitle>
                  </CardHeader>
                  <CardContent className="p-6">
                    {result ? (
                      <div className="space-y-6">
                        <div className="p-4 bg-indigo-500/5 rounded-xl border border-indigo-500/10">
                          <div className="text-2xl font-mono text-slate-100 font-black tracking-tighter">{result.tokens.length} <span className="text-xs text-slate-500 uppercase font-bold">Nodes</span></div>
                        </div>
                        <div className="text-lg font-bold flex flex-wrap items-center gap-2 leading-relaxed bg-slate-900 p-4 rounded-xl border border-slate-800">
                          {result.tokens.map((t, idx) => <span key={idx} className="bg-slate-800 text-slate-300 px-2 py-0.5 rounded">{t}</span>)}
                          <Zap className="h-4 w-4 text-indigo-400 animate-pulse" />
                          <span className="text-indigo-400 font-black px-2 py-0.5 bg-indigo-500/10 rounded border border-indigo-500/30">{result.next_token}</span>
                        </div>
                      </div>
                    ) : <div className="text-slate-800 flex items-center justify-center min-h-[300px] text-[10px] font-black uppercase tracking-[0.2em]">Ready for observation</div>}
                  </CardContent>
                </Card>
              </div>
            </div>
          )}

          {activeTab === 'energy' && (
            <div className="animate-in fade-in slide-in-from-bottom-2 duration-500 space-y-6">
              <div className="bg-slate-950 p-6 rounded-2xl border border-slate-800 border-l-4 border-l-emerald-500/50">
                <div className="flex flex-col md:flex-row gap-6 mb-6">
                  <div className="flex-1 space-y-4">
                    <Input value={energyPrompt} onChange={e => setEnergyPrompt(e.target.value)} className="bg-slate-900 border-slate-800 text-slate-100 h-12" />
                    <div className="flex items-center gap-6 bg-slate-900/50 p-4 rounded-xl border border-slate-800">
                      <input type="range" min="1" max="50" value={energySteps} onChange={e => setEnergySteps(parseInt(e.target.value))} className="w-full accent-emerald-500 cursor-pointer" />
                      <Button onClick={handleEnergyGenerate} disabled={isEnergyGenerating} className="bg-emerald-600 hover:bg-emerald-700 h-10 px-8 font-black">POUR ENERGY</Button>
                    </div>
                  </div>
                  <div className="w-full md:w-1/3 bg-slate-900 p-5 rounded-2xl border border-slate-800">
                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-3 bg-slate-950 rounded-lg border border-slate-800"><div className="text-[8px] text-slate-500 font-black mb-1">VOCAB RECEPTOR</div><div className="text-lg font-black font-mono text-slate-200">{energyResult?.physics_details?.vocab || '50257'}</div></div>
                      <div className="p-3 bg-slate-950 rounded-lg border border-slate-800"><div className="text-[8px] text-slate-500 font-black mb-1">MANIFOLD DIM</div><div className="text-lg font-black font-mono text-slate-200">{energyResult?.physics_details?.represent_dim || '3000'}</div></div>
                    </div>
                  </div>
                </div>
                <div className="bg-slate-950 p-6 rounded-2xl border border-slate-800 min-h-[300px]">
                  <div className="flex justify-between items-center mb-6 border-b border-slate-800 pb-3"><span className="text-[10px] font-black text-slate-500 uppercase tracking-widest font-mono">Calculated Manifold Collapse Trail</span><span className="text-[10px] font-mono text-emerald-500 bg-emerald-500/10 px-2 py-0.5 rounded ring-1 ring-emerald-500/20">{renderedTraces.length}/{energySteps} STEPS</span></div>
                  <div className="text-2xl font-mono text-slate-100 transition-all"><span className="text-emerald-400 font-black">{energyPrompt}</span><span className="ml-2">{renderedEnergyText}</span>{isEnergyGenerating && <span className="animate-pulse text-emerald-500 ml-1">|</span>}</div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-2 mt-8">
                    {renderedTraces.map((t, idx) => (
                      <div key={idx} className="p-2 bg-slate-900/50 rounded-lg border border-slate-800"><div className="text-xs font-black text-slate-100 font-mono">'{t.token_str}'</div><div className="text-[8px] font-mono text-slate-600 uppercase mt-1">{t.resonance_energy?.toFixed(2)} eV</div></div>
                    ))}
                  </div>
                  {(energyResult || renderedTraces.length > 0) && <div className="mt-8 pt-4 border-t border-slate-800 flex justify-end"><Button variant="ghost" size="sm" onClick={() => syncToInference(energyPrompt + renderedEnergyText)} className="text-[10px] font-black uppercase text-slate-500 hover:text-indigo-400"><RefreshCw className="h-3 w-3 mr-2" /> Sync to Inference</Button></div>}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'chat' && (
            <div className="animate-in fade-in slide-in-from-bottom-2 duration-500 flex flex-col h-[550px] bg-slate-950 rounded-2xl border border-slate-800 overflow-hidden">
              <div className="bg-slate-900/50 border-b border-slate-800 px-4 py-2 flex justify-between items-center">
                <div className="text-[10px] font-black text-slate-500 uppercase tracking-widest flex items-center gap-2"><Bot className="h-3.5 w-3.5" /> Interaction Console</div>
                <div className={`text-[9px] font-black tracking-widest ${chatStatus.is_ready ? 'text-emerald-500' : 'text-orange-500'}`}>{chatStatus.status_msg.toUpperCase()}</div>
              </div>
              <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin">
                {messages.map((m, idx) => (
                  <div key={idx} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[85%] p-3 rounded-xl ${m.role === 'user' ? 'bg-indigo-600 text-white' : (m.role === 'agi' ? 'bg-slate-900 border border-slate-800 text-slate-200' : 'bg-slate-950 text-slate-600 text-[9px] w-full text-center border border-slate-900 font-mono')}`}>
                      {m.content}
                    </div>
                  </div>
                ))}
                {isTyping && <div className="text-slate-500 text-[10px] font-mono animate-pulse">Collapsing Manifold Matrix...</div>}
                <div ref={messagesEndRef} />
              </div>
              <div className="p-4 bg-slate-900/80 border-t border-slate-800 flex gap-2">
                <Input value={chatInput} onChange={e => setChatInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleChatSend()} placeholder="Inject stimulation..." className="bg-slate-950 border-slate-800 text-sm h-10" />
                <Button onClick={handleChatSend} disabled={!chatStatus.is_ready || isTyping} className="bg-indigo-600 h-10 w-10 shrink-0"><Send className="h-4 w-4" /></Button>
                <Button variant="ghost" onClick={resetChat} className="h-10 w-10 shrink-0 text-slate-500"><RefreshCw className="h-4 w-4" /></Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default FiberNetPanel;

