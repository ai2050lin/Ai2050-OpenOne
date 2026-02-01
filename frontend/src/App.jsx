import { ContactShadows, OrbitControls, PerspectiveCamera, Text } from '@react-three/drei';
import { Canvas, useFrame } from '@react-three/fiber';
import axios from 'axios';
import { Brain, HelpCircle, Loader2, RotateCcw, Search, Settings, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import ErrorBoundary from './ErrorBoundary';
import { SimplePanel } from './SimplePanel';
import { CompositionalVisualization3D, FeatureVisualization3D, FiberBundleVisualization3D, LayerDetail3D, ManifoldVisualization3D, NetworkGraph3D, SNNVisualization3D, StructureAnalysisControls, ValidityVisualization3D } from './StructureAnalysisPanel';

import { locales } from './locales';

const API_BASE = 'http://localhost:8888';




// 3D Bar Component
function Bar({ position, height, color, label, actual, probability, layer, posIndex, onHover, isActiveLayer }) {
  const mesh = useRef();
  const [hovered, setHover] = useState(false);

  // Smooth animation for height
  useFrame((state) => {
    if (mesh.current) {
      mesh.current.scale.y = THREE.MathUtils.lerp(mesh.current.scale.y, height, 0.1);
      mesh.current.position.y = mesh.current.scale.y / 2;
    }
  });

  return (
    <group position={position}>
      <mesh
        ref={mesh}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHover(true);
          onHover({ label, actual, probability, layer, posIndex });
        }}
        onPointerOut={() => {
          setHover(false);
          onHover(null);
        }}
      >
        <boxGeometry args={[0.8, 1, 0.8]} />
        <meshStandardMaterial 
          color={color} 
          emissive={color} 
          emissiveIntensity={isActiveLayer ? 1.5 : (hovered ? 0.5 : 0.1)}
          transparent
          opacity={0.8}
        />
      </mesh>
    </group>
  );
}

// Probability to Color mapping (Viridis-like)
const getColor = (prob) => {
  const colors = [
    '#440154', // dark purple
    '#3b528b', // blue
    '#21918c', // teal
    '#5ec962', // green
    '#fde725'  // yellow
  ];
  const idx = Math.min(Math.floor(prob * colors.length), colors.length - 1);
  return colors[idx];
};

function Visualization({ data, hoveredInfo, setHoveredInfo, activeLayer }) {
  if (!data) return null;

  const { logit_lens, tokens } = data;
  const nLayers = logit_lens.length;
  const seqLen = tokens.length;

  return (
    <>
      <group position={[-seqLen / 2, 0, -nLayers / 2]}>
        {logit_lens.map((layerData, layerIdx) => (
          layerData.map((posData, posIdx) => (
            <Bar
              key={`${layerIdx}-${posIdx}`}
              position={[posIdx * 1.2, 0, layerIdx * 1.2]}
              height={posData.prob * 5 + 0.1}
              color={getColor(posData.prob)}
              label={posData.token}
              actual={posData.actual_token}
              probability={posData.prob}
              layer={layerIdx}
              posIndex={posIdx}
              onHover={setHoveredInfo}
              isActiveLayer={layerIdx === activeLayer}
            />
          ))
        ))}

        {/* Axis Labels */}
        {tokens.map((token, i) => (
          <Text
            key={`x-label-${i}`}
            position={[i * 1.2, -0.5, -1]}
            rotation={[-Math.PI / 2, 0, 0]}
            fontSize={0.3}
            color="white"
          >
            {token}
          </Text>
        ))}

        {Array.from({ length: nLayers }).map((_, i) => (
          <Text
            key={`z-label-${i}`}
            position={[-1.5, -0.5, i * 1.2]}
            rotation={[-Math.PI / 2, 0, 0]}
            fontSize={0.3}
            color="gray"
          >
            L{i}
          </Text>
        ))}
      </group>

      {/* Info panel moved to DOM overlay - see bottom-left panel */}
    </>
  );
}

// Flow Particles Component - shows information flow between layers
function FlowParticles({ nLayers, seqLen, isPlaying }) {
  const particlesRef = useRef();
  const [particles, setParticles] = useState([]);
  
  // Generate particles
  useFrame((state) => {
    if (!isPlaying || !particlesRef.current) return;
    
    // Generate new particles more frequently (20% chance instead of 5%)
    if (Math.random() < 0.2) {
      const newParticle = {
        id: Math.random(),
        x: (Math.random() - 0.5) * seqLen * 1.2,
        z: 0,
        targetZ: (nLayers - 1) * 1.2,
        progress: 0,
        speed: 0.3 + Math.random() * 0.4
      };
      setParticles(prev => [...prev.slice(-50), newParticle]);
    }
    
    // Update particle positions
    setParticles(prev => prev.map(p => ({
      ...p,
      progress: Math.min(1, p.progress + 0.008 * p.speed)
    })).filter(p => p.progress < 1));
  });
  
  if (!isPlaying) return null;
  
  return (
    <group ref={particlesRef} position={[-seqLen / 2, 4, -nLayers / 2]}>
      {particles.map(p => {
        const currentZ = p.z + (p.targetZ - p.z) * p.progress;
        const opacity = Math.sin(p.progress * Math.PI);
        
        return (
          <mesh key={p.id} position={[p.x, 0, currentZ]}>
            <sphereGeometry args={[0.15, 16, 16]} />
            <meshStandardMaterial 
              color="#00d2ff" 
              emissive="#00d2ff"
              emissiveIntensity={3}
              transparent
              opacity={opacity * 0.9}
            />
          </mesh>
        );
      })}
    </group>
  );
}

// Attention Heatmap Component using Canvas
function AttentionHeatmap({ pattern, tokens, headIdx }) {
  const canvasRef = useRef(null);
  
  useEffect(() => {
    if (!canvasRef.current || !pattern) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const size = pattern.length;
    const cellSize = Math.min(200 / size, 40);
    
    canvas.width = size * cellSize;
    canvas.height = size * cellSize;
    
    // Draw heatmap
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const value = pattern[i][j];
        const intensity = Math.floor(value * 255);
        ctx.fillStyle = `rgb(${intensity}, ${Math.floor(intensity * 0.5)}, ${255 - intensity})`;
        ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
      }
    }
    
    // Draw grid
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= size; i++) {
      ctx.beginPath();
      ctx.moveTo(i * cellSize, 0);
      ctx.lineTo(i * cellSize, size * cellSize);
      ctx.stroke();
      
      ctx.beginPath();
      ctx.moveTo(0, i * cellSize);
      ctx.lineTo(size * cellSize, i * cellSize);
      ctx.stroke();
    }
  }, [pattern]);
  
  return (
    <div style={{ marginBottom: '12px' }}>
      <div style={{ fontSize: '11px', fontWeight: 'bold', marginBottom: '4px', color: '#00d2ff' }}>
        头 {headIdx}
      </div>
      <canvas 
        ref={canvasRef} 
        style={{ 
          border: '1px solid #444', 
          borderRadius: '4px',
          maxWidth: '100%',
          imageRendering: 'pixelated'
        }} 
      />
    </div>
  );
}

// MLP Activation Bar Chart using Canvas
function MLPActivationChart({ distribution }) {
  const canvasRef = useRef(null);
  
  useEffect(() => {
    if (!canvasRef.current || !distribution) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = 300;
    const height = 100;
    const barCount = Math.min(distribution.length, 100);
    const barWidth = width / barCount;
    
    canvas.width = width;
    canvas.height = height;
    
    // Find max for scaling
    const maxVal = Math.max(...distribution.slice(0, barCount));
    
    // Draw bars
    for (let i = 0; i < barCount; i++) {
      const value = distribution[i];
      const barHeight = (value / maxVal) * height;
      const hue = (value / maxVal) * 120; // 0 (red) to 120 (green)
      ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
      ctx.fillRect(i * barWidth, height - barHeight, barWidth, barHeight);
    }
  }, [distribution]);
  
  return (
    <div>
      <div style={{ fontSize: '11px', fontWeight: 'bold', marginBottom: '4px', color: '#00d2ff' }}>
        MLP激活分布
      </div>
      <canvas 
        ref={canvasRef} 
        style={{ 
          border: '1px solid #444', 
          borderRadius: '4px',
          width: '100%'
        }} 
      />
    </div>
  );
}

// Global Config Panel Component
// Global Config Panel Component
function GlobalConfigPanel({ visibility, onToggle, onClose, onReset, lang, onSetLang, t }) {
  const getLabelFor = (key) => {
    return t(`panels.${key}`) || key;
  };

  return (
    <SimplePanel
      title={t('panels.globalConfig')}
      onClose={onClose}
      icon={<Settings />}
      style={{
        position: 'absolute', top: 60, left: 20, zIndex: 100,
        minWidth: '220px'
      }}
    >
      {/* Language Switcher */}
      <div style={{ marginBottom: '16px', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '12px' }}>
        <div style={{fontSize: '12px', color: '#aaa', marginBottom: '8px'}}>{t('common.language')}</div>
        <div style={{display: 'flex', gap: '8px'}}>
          <button 
            onClick={() => onSetLang('zh')}
            style={{
              flex: 1, padding: '4px', borderRadius: '4px',
              border: lang === 'zh' ? '1px solid #4488ff' : '1px solid #444',
              background: lang === 'zh' ? 'rgba(68, 136, 255, 0.2)' : 'transparent',
              color: lang === 'zh' ? '#fff' : '#888',
              cursor: 'pointer', fontSize: '12px'
            }}
          >
            中文
          </button>
          <button 
             onClick={() => onSetLang('en')}
             style={{
              flex: 1, padding: '4px', borderRadius: '4px',
              border: lang === 'en' ? '1px solid #4488ff' : '1px solid #444',
              background: lang === 'en' ? 'rgba(68, 136, 255, 0.2)' : 'transparent',
              color: lang === 'en' ? '#fff' : '#888',
              cursor: 'pointer', fontSize: '12px'
            }}
          >
            English
          </button>
        </div>
      </div>
      
      <div style={{ marginBottom: '16px' }}>
      {Object.entries(visibility).map(([key, isVisible]) => (
        <div key={key} style={{display:'flex', justifyContent:'space-between', marginBottom:'12px', fontSize:'13px', alignItems:'center'}}>
          <span style={{color: '#ccc'}}>{getLabelFor(key)}</span>
          <button 
            onClick={() => onToggle(key)}
            style={{
              background: isVisible ? '#4488ff' : '#333',
              border: 'none', borderRadius: '12px', width: '36px', height: '20px',
              position: 'relative', cursor: 'pointer', transition: 'background 0.2s'
            }}
          >
            <div style={{
              position: 'absolute', top: '2px', left: isVisible ? '18px' : '2px',
              width: '16px', height: '16px', background: '#fff', borderRadius: '50%',
              transition: 'left 0.2s'
            }}/>
          </button>
        </div>
      ))}
      </div>

      <button onClick={onReset} style={{
        width: '100%', padding: '8px', backgroundColor: '#333', color: '#fff', border: 'none',
        borderRadius: '4px', cursor: 'pointer', fontSize: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
        transition: 'background 0.2s', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '12px'
      }}>
        <RotateCcw size={12} /> {t('panels.resetLayout')}
      </button>
    </SimplePanel>
  );
}

export default function App() {
  const [lang, setLang] = useState('zh');
  const t = (key, params = {}) => {
    const keys = key.split('.');
    let val = locales[lang];
    for (const k of keys) {
      val = val?.[k];
    }
    if (!val) return key;
    if (params) {
      for (const [pKey, pVal] of Object.entries(params)) {
        val = val.replace(`{{${pKey}}}`, pVal);
      }
    }
    return val;
  };

  const [prompt, setPrompt] = useState('The quick brown fox');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [hoveredInfo, setHoveredInfo] = useState(null);
  const [modelConfig, setModelConfig] = useState(null);
  const [selectedLayer, setSelectedLayer] = useState(null);
  const [layerData, setLayerData] = useState(null);
  const [loadingLayerData, setLoadingLayerData] = useState(false);
  const [isAnimationPlaying, setIsAnimationPlaying] = useState(true);
  const [showStructurePanel, setShowStructurePanel] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [layerNeuronState, setLayerNeuronState] = useState(null);
  const [loadingNeurons, setLoadingNeurons] = useState(false);
  const [layerInfo, setLayerInfo] = useState(null); // For 3D visualization
  
  // Animation states for layer computation visualization
  const [isAnimating, setIsAnimating] = useState(false);
  const [activeLayer, setActiveLayer] = useState(null);
  const [computationPhase, setComputationPhase] = useState(null); // 'attention' | 'mlp' | 'output'
  const [activeLayerInfo, setActiveLayerInfo] = useState(null);
  
  // Auto-analysis state
  const [autoAnalysisResult, setAutoAnalysisResult] = useState(null);
  const [stepAnalysisMode, setStepAnalysisMode] = useState('features'); // 'features', 'circuit', 'causal', 'none'
  const [analysisResult, setAnalysisResult] = useState(null);
  const [structureTab, setStructureTab] = useState('circuit');

  // Analysis Forms State (Lifted from StructureAnalysisPanel)
  const [circuitForm, setCircuitForm] = useState({
    clean_prompt: 'The capital of France is Paris',
    corrupted_prompt: 'The capital of France is Berlin',
    threshold: 0.1,
    target_token_pos: -1
  });
  
  const [featureForm, setFeatureForm] = useState({
    prompt: 'The quick brown fox jumps',
    layer_idx: 6,
    hidden_dim: 1024,
    sparsity_coef: 0.001,
    n_epochs: 30
  });
  
  const [causalForm, setCausalForm] = useState({
    prompt: 'The quick brown fox',
    target_token_pos: -1,
    importance_threshold: 0.01
  });
  
  const [manifoldForm, setManifoldForm] = useState({
    prompt: 'The quick brown fox',
    layer_idx: 0
  });

  // System Type State for Structure Analysis
  const [systemType, setSystemType] = useState('dnn');

  // SNN State
  const [snnState, setSnnState] = useState({
    initialized: false,
    layers: [],
    structure: null, // [NEW] Store 3D structure
    time: 0,
    spikes: {},
    isPlaying: false
  });

  const initializeSNN = async () => {
    try {
      const res = await axios.post(`${API_BASE}/snn/initialize`, {
        layers: {
          "Retina_Shape": 20,
          "Retina_Color": 20,
          "Object_Fiber": 20
        },
        connections: [
          { src: "Retina_Shape", tgt: "Object_Fiber", type: "one_to_one", weight: 0.8 },
          { src: "Retina_Color", tgt: "Object_Fiber", type: "one_to_one", weight: 0.8 }
        ]
      });
      setSnnState(prev => ({ 
          ...prev, 
          initialized: true, 
          layers: res.data.layers,
          structure: res.data.structure
      }));

    } catch (err) {
      console.error(err);
      if (err.message === 'Network Error') {
         alert("连接服务器失败。请检查后端服务器 (server.py) 是否正在运行。如果已崩溃，请重启它。");
      } else {
         alert("SNN 初始化失败: " + err.message);
      }
    }
  };

  const injectSNNStimulus = async (layer, patternIdx) => {
    try {
      await axios.post(`${API_BASE}/snn/stimulate`, {
        layer_name: layer,
        pattern_idx: patternIdx,
        intensity: 2.0
      });
      // Immediately step to see effect
      await stepSNN();
    } catch (err) {
      console.error(err);
      if (err.message === 'Network Error') {
         alert("连接服务器失败。请检查后端服务器 (server.py) 是否正在运行。如果已崩溃，请重启它。");
      }
    }
  };

  const stepSNN = async () => {
    try {
      const res = await axios.post(`${API_BASE}/snn/step`, { steps: 5 });
      setSnnState(prev => ({
        ...prev,
        time: res.data.time,
        spikes: res.data.spikes
      }));
    } catch (err) {
      console.error(err);
      if (err.message === 'Network Error') {
         alert("连接服务器失败。请检查后端服务器 (server.py) 是否正在运行。如果已崩溃，请重启它。");
      }
    }
  };
   
  // SNN Auto-play effect
  useEffect(() => {
    let interval;
    if (snnState.isPlaying && snnState.initialized) {
      interval = setInterval(stepSNN, 200); // 5 steps every 200ms = 25 steps/sec
    }
    return () => clearInterval(interval);
  }, [snnState.isPlaying, snnState.initialized]);

  const [infoPanelTab, setInfoPanelTab] = useState('model'); // 'model' | 'detail'
  const [displayInfo, setDisplayInfo] = useState(null); // Persisted hover info

  // Auto-switch Info Panel tab on hover and persist info
  useEffect(() => {
    if (hoveredInfo) {
      setInfoPanelTab('detail');
      setDisplayInfo(hoveredInfo);
    }
  }, [hoveredInfo]);
  
  // UI Tabs State
  const [inputPanelTab, setInputPanelTab] = useState('dnn'); // 'dnn' | 'snn'

  // Sync Auto Analysis Result (Single Step) to Main Result State
  // This ensures results show up even if StructureAnalysisControls is not mounted (Basic Tab)
  useEffect(() => {
    if (autoAnalysisResult) {
      setAnalysisResult(autoAnalysisResult.data);
      if (autoAnalysisResult.type !== 'none') {
        setStructureTab(autoAnalysisResult.type);
      }
    }
  }, [autoAnalysisResult]);
  
  // Head Analysis Panel State
  const [headPanel, setHeadPanel] = useState({
    isOpen: false,
    layerIdx: null,
    headIdx: null
  });

  // Global Visibility State
  const [showConfigPanel, setShowConfigPanel] = useState(false);
  const [compForm, setCompForm] = useState({
    layer_idx: 0,
    raw_phrases: "black, cat, black cat\nParis, France, Paris France\nking, man, king",
    phrases: [["black", "cat", "black cat"], ["Paris", "France", "Paris France"], ["king", "man", "king"]]
  });

  const [panelVisibility, setPanelVisibility] = useState({
    inputPanel: true,
    infoPanel: true,
    layersPanel: true,
    structurePanel: true,
    neuronPanel: true,
    headPanel: true,

  });

  const togglePanelVisibility = (key) => {
    setPanelVisibility(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };
  
  const handleHeadClick = (layerIdx, headIdx) => {
    setHeadPanel({
      isOpen: true,
      layerIdx,
      headIdx
    });
  };
  
  // Reusable draggable hook
  const useDraggable = (storageKey, defaultPosition) => {
    const [position, setPosition] = useState(() => {
      const saved = localStorage.getItem(storageKey);
      return saved ? JSON.parse(saved) : defaultPosition;
    });
    const [isDragging, setIsDragging] = useState(false);
    const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
    
    useEffect(() => {
      localStorage.setItem(storageKey, JSON.stringify(position));
    }, [position, storageKey]);
    
    const handleMouseDown = (e) => {
      setIsDragging(true);
      setDragStart({
        x: e.clientX - position.x,
        y: e.clientY - position.y
      });
    };
    
    const handleMouseMove = (e) => {
      if (isDragging) {
        setPosition({
          x: e.clientX - dragStart.x,
          y: e.clientY - dragStart.y
        });
      }
    };
    
    const handleMouseUp = () => {
      setIsDragging(false);
    };
    
    useEffect(() => {
      if (isDragging) {
        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);
        return () => {
          window.removeEventListener('mousemove', handleMouseMove);
          window.removeEventListener('mouseup', handleMouseUp);
        };
      }
    }, [isDragging, dragStart]);
    
    return { position, setPosition, isDragging, handleMouseDown };
  };
  
  // Draggable panels
  const structurePanel = useDraggable('structureAnalysisPanel', { x: window.innerWidth - 400, y: 20 });
  const headPanelDrag = useDraggable('headAnalysisPanel', { x: 400, y: 100 });
  const neuronPanel = useDraggable('neuronStatePanel', { x: 20, y: window.innerHeight - 600 });
  const layerInfoPanel = useDraggable('layerInfoPanel', { x: 400, y: window.innerHeight - 450 });
  const layerDetailPanel = useDraggable('layerDetailPanel', { x: window.innerWidth - 850, y: 20 });

  const resetConfiguration = () => {
    // Clear all localStorage
    localStorage.removeItem('structureAnalysisPanel');
    localStorage.removeItem('headAnalysisPanel');
    localStorage.removeItem('neuronStatePanel');
    localStorage.removeItem('layerInfoPanel');
    localStorage.removeItem('layerDetailPanel');
    
    // Reset panel positions
    structurePanel.setPosition({ x: window.innerWidth - 400, y: 20 });
    headPanelDrag.setPosition({ x: 400, y: 100 });
    neuronPanel.setPosition({ x: 20, y: window.innerHeight - 600 });
    layerInfoPanel.setPosition({ x: 400, y: window.innerHeight - 450 });
    layerDetailPanel.setPosition({ x: window.innerWidth - 850, y: 20 });
    
    // Clear states
    setPrompt('');
    setData(null);
    setSelectedLayer(null);
    setLayerNeuronState(null);
    setActiveLayer(null);
    setActiveLayerInfo(null);
    setAutoAnalysisResult(null);
    
    alert('✅ 配置已重置到初始状态');
  };


  const analyze = async () => {
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/analyze`, { prompt });
      setData(res.data);
    } catch (err) {
      console.error(err);
      alert('Error analyzing text. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };


  const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

  const animateLayerComputation = async () => {
    if (!data?.model_config || !prompt) return;
    
    const nLayers = data.model_config.n_layers;
    
    for (let layer = 0; layer < nLayers; layer++) {
      setActiveLayer(layer);
      console.log(`[Animation] Processing layer ${layer}/${nLayers}`);
      
      // Fetch both layer config and neuron state for the active layer
      try {
        console.log(`[Animation] Fetching data for layer ${layer}...`);
        const [configRes, stateRes] = await Promise.all([
          axios.get(`${API_BASE}/layer_detail/${layer}`),
          axios.post(`${API_BASE}/layer_details`, {
            prompt,
            layer_idx: layer
          })
        ]);
        
        console.log(`[Animation] Layer ${layer} data received:`, {
          config: configRes.data,
          state: stateRes.data
        });
        
        setActiveLayerInfo(configRes.data);
        setLayerNeuronState(stateRes.data); // Display attention patterns and MLP stats
      } catch (err) {
        console.error(`[Animation] Error fetching layer ${layer} info:`, err);
        alert(`获取第${layer}层数据时出错: ${err.message}`);
      }
      
      // Auto-run feature extraction for this layer (every layer)
      if (true) {
        try {
          console.log(`[Animation] Running feature extraction for layer ${layer}...`);
          const featureRes = await axios.post(`${API_BASE}/extract_features`, {
            prompt,
            layer_idx: layer,
            hidden_dim: 512,  // Reduced for speed
            sparsity_coef: 0.001,
            n_epochs: 10  // Reduced for speed
          });
          
          setAutoAnalysisResult({
            layer: layer,
            type: 'features',
            data: featureRes.data
          });
          
          console.log(`[Animation] Feature extraction complete for layer ${layer}`);
        } catch (err) {
          console.error(`[Animation] Feature extraction failed for layer ${layer}:`, err);
        }
      }
      
      // Attention phase
      setComputationPhase('attention');
      await sleep(150);
      
      // MLP phase
      setComputationPhase('mlp');
      await sleep(120);
      
      // Output phase
      setComputationPhase('output');
      await sleep(80);
    }
    
    // Clear animation state
    setActiveLayer(null);
    setComputationPhase(null);
    setActiveLayerInfo(null);
    setLayerNeuronState(null);
    setAutoAnalysisResult(null);
  };


  const generateNext = async () => {
    setGenerating(true);
    setIsAnimating(true);
    
    try {
      // First, run the layer computation animation
      await animateLayerComputation();
      
      // Then perform actual generation
      const res = await axios.post(`${API_BASE}/generate_next`, { 
        prompt, 
        num_tokens: 1,
        temperature: 0.7
      });
      setPrompt(res.data.generated_text);
      
      // Auto-analyze after generation
      setTimeout(() => {
        analyze();
      }, 100);
    } catch (err) {
      console.error(err);
      alert('Error generating text. Is the backend running?');
    } finally {
      setGenerating(false);
      setIsAnimating(false);
    }
  };

  const stepToNextLayer = async () => {
    if (!data?.model_config || !prompt) {
      alert('请先运行分析！');
      return;
    }
    
    const nLayers = data.model_config.n_layers;
    const nextLayer = activeLayer === null ? 0 : activeLayer + 1;
    
    if (nextLayer >= nLayers) {
      alert('已到达最后一层！');
      return;
    }
    
    setIsAnimating(true);
    setActiveLayer(nextLayer);
    
    try {
      console.log(`[Step] Processing layer ${nextLayer}/${nLayers}`);
      
      // Fetch layer config and neuron state
      const [configRes, stateRes] = await Promise.all([
        axios.get(`${API_BASE}/layer_detail/${nextLayer}`),
        axios.post(`${API_BASE}/layer_details`, {
          prompt,
          layer_idx: nextLayer
        })
      ]);
      
      setActiveLayerInfo(configRes.data);
      setLayerNeuronState(stateRes.data);
      
      // Auto-run analysis based on selected mode
      if (stepAnalysisMode !== 'none') {
        try {
          console.log(`[Step] Running ${stepAnalysisMode} analysis for layer ${nextLayer}...`);
          
          let resultData = null;
          let resultType = stepAnalysisMode;

          if (stepAnalysisMode === 'features') {
            const featureRes = await axios.post(`${API_BASE}/extract_features`, {
              prompt,
              layer_idx: nextLayer,
              hidden_dim: 512,
              sparsity_coef: 0.001,
              n_epochs: 10
            });
            resultData = featureRes.data;
          }
          else if (stepAnalysisMode === 'circuit') {
            const circuitRes = await axios.post(`${API_BASE}/discover_circuit`, {
              ...circuitForm, // Use detailed form settings
              target_layer: nextLayer
            });
            resultData = circuitRes.data;
          }
          else if (stepAnalysisMode === 'causal') {
            const causalRes = await axios.post(`${API_BASE}/causal_analysis`, {
              ...causalForm, // Use detailed form settings
              target_layer: nextLayer
            });
            resultData = causalRes.data;
          }
          else if (stepAnalysisMode === 'manifold') {
            const manifoldRes = await axios.post(`${API_BASE}/manifold_analysis`, {
              ...manifoldForm, // Use detailed form settings
              layer_idx: nextLayer
            });
            resultData = manifoldRes.data;
          }
          
          if (resultData) {
            setAutoAnalysisResult({
              layer: nextLayer,
              type: resultType,
              data: resultData
            });
          }
        } catch (err) {
          console.error(`[Step] Analysis (${stepAnalysisMode}) failed:`, err);
        }
      }
      
      // Animate computation phases
      setComputationPhase('attention');
      await sleep(150);
      
      setComputationPhase('mlp');
      await sleep(120);
      
      setComputationPhase('output');
      await sleep(80);
      
      // Keep layer visible, just set phase to idle
      setComputationPhase('idle');
      
    } catch (err) {
      console.error(`[Step] Error:`, err);
      alert('单步执行失败');
    } finally {
      setIsAnimating(false);
    }
  };


  const loadLayerDetails = async (layerIdx) => {
    if (!prompt) return;
    setLoadingNeurons(true);
    try {
      // Fetch both layer config and neuron state in parallel
      const [configRes, stateRes] = await Promise.all([
        axios.get(`${API_BASE}/layer_detail/${layerIdx}`),
        axios.post(`${API_BASE}/layer_details`, {
          prompt,
          layer_idx: layerIdx
        })
      ]);
      
      setLayerInfo(configRes.data);
      setLayerNeuronState(stateRes.data);
    } catch (err) {
      console.error(err);
      alert('Error loading layer details.');
    } finally {
      setLoadingNeurons(false);
    }
  };

  return (
    <div style={{ width: '100vw', height: '100vh', background: '#050505', color: 'white' }}>
      
      {/* Global Settings Button */}
      <button
        onClick={() => setShowConfigPanel(!showConfigPanel)}
        style={{
          position: 'absolute', top: 20, left: 20, zIndex: 101, // Higher than panels
          background: showConfigPanel ? '#4488ff' : 'rgba(20, 20, 25, 0.8)',
          border: '1px solid rgba(255,255,255,0.1)',
          borderRadius: '8px',
          padding: '8px',
          cursor: 'pointer',
          color: 'white',
          backdropFilter: 'blur(10px)',
          display: 'flex', alignItems: 'center', justifyContent: 'center'
        }}
        title="界面配置"
      >
        <Settings size={20} />
      </button>

      {/* Global Config Panel */}
      {showConfigPanel && (
        <GlobalConfigPanel 
          visibility={panelVisibility} 
          onToggle={togglePanelVisibility} 
          onClose={() => setShowConfigPanel(false)} 
          onReset={() => {
            resetConfiguration();
            setShowConfigPanel(false);
          }}
          lang={lang}
          onSetLang={setLang}
          t={t}
        />
      )}

      {/* Top-left Input Panel */}
      {panelVisibility.inputPanel && (
      <div style={{
        position: 'absolute', top: 60, left: 20, zIndex: 10, // Moved down to avoid overlap with settings button
        background: 'rgba(20, 20, 25, 0.9)', padding: '20px', borderRadius: '12px',
        backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.1)',
        width: '380px', maxHeight: '85vh', display: 'flex', flexDirection: 'column'
      }}>
        <h1 style={{ margin: '0 0 16px 0', fontSize: '20px', fontWeight: 'bold', background: 'linear-gradient(45deg, #00d2ff, #3a7bd5)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Brain size={24} color="#00d2ff"/> AGI智能理论分析
        </h1>
        
        {/* Tabs for Input Panel */}
        <div style={{ display: 'flex', borderBottom: '1px solid rgba(255,255,255,0.1)', marginBottom: '16px', background: 'rgba(0,0,0,0.2)', borderRadius: '6px', padding: '2px' }}>
          <button
             onClick={() => {
                 setInputPanelTab('dnn');
                 setSystemType('dnn');
             }}
             style={{
               flex: 1, padding: '8px', background: inputPanelTab === 'dnn' ? '#3a7bd5' : 'transparent', border: 'none', borderRadius: '4px',
               color: inputPanelTab === 'dnn' ? '#fff' : '#888',
               cursor: 'pointer', fontWeight: '600', fontSize: '12px', transition: 'all 0.2s'
             }}
          >
            深度神经网络 (DNN)
          </button>
          <button
             onClick={() => {
                 setInputPanelTab('snn'); 
                 setSystemType('snn');
             }}
             style={{
               flex: 1, padding: '8px', background: inputPanelTab === 'snn' ? '#4ecdc4' : 'transparent', border: 'none', borderRadius: '4px',
               color: inputPanelTab === 'snn' ? '#000' : '#888',
               cursor: 'pointer', fontWeight: '600', fontSize: '12px', transition: 'all 0.2s'
             }}
          >
            脉冲神经网络 (SNN)
          </button>
        </div>

        {/* Content Container with Scroll */}
        <div style={{ flex: 1, overflowY: 'auto', paddingRight: '4px' }}>
        
            {/* DNN Content: Generation + Structure Analysis */}
            {inputPanelTab === 'dnn' && (
              <div className="animate-fade-in">
                {/* Generation Section */}
                <div style={{ background: 'rgba(255,255,255,0.03)', padding: '12px', borderRadius: '8px', marginBottom: '16px', border: '1px solid rgba(255,255,255,0.05)' }}>
                    <div style={{ fontSize: '12px', color: '#aaa', marginBottom: '8px', fontWeight: 'bold', display: 'flex', justifyContent: 'space-between' }}>
                        <span>文本生成与提示词</span>
                        {generating && <span style={{color: '#5ec962'}}>Generating...</span>}
                    </div>
                    
                    <textarea
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      placeholder="输入提示词..."
                      rows={3}
                      style={{
                        width: '100%', background: '#1a1a1f', border: '1px solid #333',
                        color: 'white', padding: '10px', borderRadius: '6px', outline: 'none',
                        resize: 'vertical', fontSize: '13px', fontFamily: 'sans-serif'
                      }}
                    />
                    
                    <div style={{ display: 'flex', gap: '8px', marginTop: '10px' }}>
                         <button
                            onClick={analyze}
                            disabled={loading || !prompt}
                            style={{
                              flex: 1, background: '#333', border: '1px solid #444', color: 'white',
                              padding: '8px', borderRadius: '6px', cursor: 'pointer',
                              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
                              fontSize: '12px'
                            }}
                            title="仅分析当前提示词"
                          >
                            {loading ? <Loader2 className="animate-spin" size={14} /> : <Search size={14} />} 分析
                          </button>
                          
                          <button
                            onClick={generateNext}
                            disabled={generating || !prompt}
                            style={{
                              flex: 2,
                              background: generating ? '#888' : 'linear-gradient(45deg, #5ec962, #96c93d)',
                              border: 'none',
                              color: 'white',
                              padding: '8px',
                              borderRadius: '6px',
                              cursor: generating || !prompt ? 'not-allowed' : 'pointer',
                              fontSize: '12px',
                              fontWeight: '600',
                              opacity: generating || !prompt ? 0.7 : 1,
                              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px'
                            }}
                          >
                            {generating ? '生成中...' : 'Generate Next Token'}
                          </button>
                    </div>
                </div>

                {/* Structure Analysis Section */}
                <div style={{ marginBottom: '10px' }}>
                     {/* Pass systemType='dnn' expressly */}
                     <StructureAnalysisControls
                       autoResult={autoAnalysisResult}
                       systemType={systemType} 
                       setSystemType={setSystemType}
                       circuitForm={circuitForm} setCircuitForm={setCircuitForm}
                       featureForm={featureForm} setFeatureForm={setFeatureForm}
                       causalForm={causalForm} setCausalForm={setCausalForm}
                       manifoldForm={manifoldForm} setManifoldForm={setManifoldForm}
                       compForm={compForm} setCompForm={setCompForm}
                       onResultUpdate={setAnalysisResult}
                       activeTab={structureTab}
                       setActiveTab={setStructureTab}
                       t={t}
                       // SNN Props
                       snnState={snnState}
                       onInitializeSNN={initializeSNN}
                       onToggleSNNPlay={() => setSnnState(s => ({...s, isPlaying: !s.isPlaying}))}
                       onStepSNN={stepSNN}
                       onInjectStimulus={injectSNNStimulus}
                       containerStyle={{ 
                          background: 'transparent', 
                          borderLeft: 'none', 
                          backdropFilter: 'none',
                          padding: 0
                       }}
                     />
                </div>
                
                {/* Step Execution Controls */}
                <div style={{ marginTop: '12px', padding: '12px', background: 'rgba(0,0,0,0.2)', borderRadius: '8px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                        <span style={{ fontSize: '11px', color: '#aaa', fontWeight: 'bold' }}>单步调试 (Step-by-Step)</span>
                         <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '10px', color: '#888', cursor: 'pointer' }}>
                            <input 
                              type="checkbox" 
                              checked={stepAnalysisMode !== 'none'}
                              onChange={(e) => setStepAnalysisMode(e.target.checked ? structureTab : 'none')}
                              style={{ accentColor: '#4ecdc4' }}
                            />
                            启用分析
                          </label>
                      </div>
                      
                      <button
                        onClick={stepToNextLayer}
                        disabled={isAnimating || !data}
                        style={{
                          width: '100%',
                          background: isAnimating || !data ? '#444' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                          border: 'none',
                          color: 'white',
                          padding: '8px',
                          borderRadius: '6px',
                          cursor: isAnimating || !data ? 'not-allowed' : 'pointer',
                          fontSize: '12px',
                          display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
                          opacity: isAnimating || !data ? 0.6 : 1
                        }}
                      >
                        {isAnimating ? <Loader2 className="animate-spin" size={14} /> : '▶️'} 
                        执行单层步进 {activeLayer !== null ? `(当前: L${activeLayer})` : '(从 L0 开始)'}
                      </button>
                </div>
              </div>
            )}

            {/* SNN Content */}
            {inputPanelTab === 'snn' && (
               <div className="animate-fade-in">
                   <div style={{ padding: '12px', background: 'rgba(78, 205, 196, 0.1)', borderRadius: '8px', border: '1px solid rgba(78, 205, 196, 0.2)', marginBottom: '16px' }}>
                        <div style={{ display: 'flex', gap: '8px', alignItems: 'start' }}>
                            <Brain size={16} color="#4ecdc4" />
                            <div>
                                <h4 style={{margin: '0 0 4px 0', fontSize: '13px', color: '#4ecdc4'}}>NeuroFiber SNN 仿真</h4>
                                <p style={{fontSize: '11px', color: '#bfd', margin: 0, lineHeight: '1.4'}}>
                                    探索基于神经纤维丛理论的脉冲神经网络动力学。
                                </p>
                            </div>
                        </div>
                    </div>
                    
                    {/* Pass systemType='snn' expressly */}
                    <StructureAnalysisControls
                       autoResult={autoAnalysisResult}
                       systemType="snn"
                       setSystemType={setSystemType}
                       circuitForm={circuitForm} setCircuitForm={setCircuitForm}
                       featureForm={featureForm} setFeatureForm={setFeatureForm}
                       causalForm={causalForm} setCausalForm={setCausalForm}
                       manifoldForm={manifoldForm} setManifoldForm={setManifoldForm}
                       compForm={compForm} setCompForm={setCompForm}
                       onResultUpdate={setAnalysisResult}
                       activeTab={structureTab}
                       setActiveTab={setStructureTab}
                       t={t}
                       // SNN Props
                       snnState={snnState}
                       onInitializeSNN={initializeSNN}
                       onToggleSNNPlay={() => setSnnState(s => ({...s, isPlaying: !s.isPlaying}))}
                       onStepSNN={stepSNN}
                       onInjectStimulus={injectSNNStimulus}
                       containerStyle={{ 
                          background: 'transparent', 
                          borderLeft: 'none', 
                          backdropFilter: 'none',
                          padding: 0
                       }}
                     />
               </div>
            )}
        
        </div>
      </div>
      )}

      {/* Bottom-left Info Panel */}
      {/* Model Info Panel (Top-Right) */}
      {panelVisibility.infoPanel && (
      <SimplePanel
        title={t('panels.modelInfo')}
        style={{
          position: 'absolute', top: 20, right: 20, zIndex: 100,
          minWidth: '320px', maxWidth: '400px',
          maxHeight: '80vh',
          display: 'flex', flexDirection: 'column',
          userSelect: 'text', // Explicitly allow text selection
          cursor: 'auto'
        }}
        headerStyle={{ marginBottom: '0', cursor: 'grab' }}
        actions={
           <button
             onClick={() => setShowHelp(true)}
             style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: '#888', padding: '4px', display: 'flex', transition: 'color 0.2s' }}
             onMouseOver={(e) => e.currentTarget.style.color = '#fff'}
             onMouseOut={(e) => e.currentTarget.style.color = '#888'}
             title="算法原理说明"
           >
             <HelpCircle size={16} />
           </button>
        }
      >
        {/* Content - Two Sections: Model Info & Structure Analysis Info */}
        <div style={{ padding: '0', height: '100%', display: 'flex', flexDirection: 'column' }}>

          {/* SECTION 1: Model / System Information */}
          <div style={{ flex: '0 0 auto', marginBottom: '12px' }}>
              <div style={{ fontSize: '11px', fontWeight: 'bold', color: '#888', marginBottom: '8px', textTransform: 'uppercase' }}>
                  {systemType === 'snn' ? 'SNN 网络状态' : '模型配置'}
              </div>

              {systemType === 'snn' ? (
                 /* SNN System Info */
                 <div style={{ fontSize: '12px', lineHeight: '1.6', background: 'rgba(255,255,255,0.03)', padding: '8px', borderRadius: '6px' }}>
                    <div style={{ display: 'grid', gridTemplateColumns: '100px 1fr', gap: '4px', color: '#aaa' }}>
                        <span>状态:</span>
                        <span style={{ color: snnState.initialized ? '#4ecdc4' : '#666', fontWeight: 'bold' }}>
                            {snnState.initialized ? (snnState.isPlaying ? '运行中' : '就绪') : '未初始化'}
                        </span>

                        <span>仿真时间:</span>
                        <span style={{ color: '#fff' }}>{snnState.time.toFixed(1)} ms</span>

                        <span>神经元数:</span>
                        <span style={{ color: '#fff' }}>{snnState.structure?.neurons?.length || 0}</span>
                    </div>
                 </div>
              ) : (
                 /* DNN Model Info */
                 data?.model_config ? (
                    <div style={{ fontSize: '12px', lineHeight: '1.6', background: 'rgba(255,255,255,0.03)', padding: '8px', borderRadius: '6px' }}>
                      <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr', gap: '4px', color: '#aaa' }}>
                        <span>架构:</span>
                        <span style={{ color: '#fff', fontWeight: 'bold' }}>{data.model_config.name}</span>

                        <span>层数:</span>
                        <span style={{ color: '#fff' }}>{data.model_config.n_layers}</span>

                        <span>模型维度:</span>
                        <span style={{ color: '#fff' }}>{data.model_config.d_model} (H: {data.model_config.n_heads})</span>

                        <span>参数量:</span>
                        <span style={{ color: '#fff' }}>{(data.model_config.total_params / 1e9).toFixed(2)}B</span>
                      </div>
                    </div>
                 ) : (
                     <div style={{ color: '#666', fontStyle: 'italic', fontSize: '12px', padding: '8px' }}>未加载模型</div>
                 )
              )}
          </div>

          {/* Divider */}
          <div style={{ height: '1px', background: 'rgba(255,255,255,0.1)', marginBottom: '12px' }} />

          {/* SECTION 2: Analysis / Detail Information */}
          <div style={{ flex: 1, overflowY: 'auto' }}>
              <div style={{ fontSize: '11px', fontWeight: 'bold', color: '#888', marginBottom: '8px', textTransform: 'uppercase' }}>
                  {systemType === 'snn' ? '实时动态' : '结构分析详情'}
              </div>

              {systemType === 'snn' ? (
                 /* SNN Live Details */
                 <div style={{ fontSize: '12px' }}>
                    <div style={{ marginBottom: '8px', color: '#aaa', fontSize: '11px' }}>
                        实时脉冲活动 (STDP 已启用)
                    </div>
                    {/* Compact Spike Visualization */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                          {snnState.layers.map(layer => {
                              const isActive = snnState.spikes[layer] && snnState.spikes[layer].length > 0;
                              return (
                                 <div key={layer} style={{
                                    padding: '6px',
                                    borderRadius: '4px',
                                    background: isActive ? 'rgba(255,159,67,0.15)' : 'transparent',
                                    border: isActive ? '1px solid rgba(255,159,67,0.3)' : '1px solid rgba(255,255,255,0.05)',
                                    display: 'flex', justifyContent: 'space-between', alignItems: 'center'
                                 }}>
                                    <span style={{ color: isActive ? '#fff' : '#888', fontSize: '11px' }}>{layer}</span>
                                    {isActive && <span style={{ fontSize: '9px', color: '#ff9f43', fontWeight: 'bold' }}>活跃</span>}
                                 </div>
                              );
                          })}
                    </div>
                    <div style={{ marginTop: '12px', fontSize: '11px', color: '#666' }}>
                        使用左侧面板控制注入刺激信号。
                    </div>
                 </div>
              ) : (
                 /* DNN Analysis Details - Handles both Hover and Active Analysis */
                 (displayInfo || hoveredInfo || analysisResult) ? (
                    <div>
                        {/* 2A. Hover/Selected Info (Highest Priority for immediate feedback) */}
                        {(displayInfo || hoveredInfo) && (
                           <div style={{ marginBottom: '16px', background: 'rgba(0,0,0,0.2)', padding: '10px', borderRadius: '6px', borderLeft: '3px solid #00d2ff' }}>
                              <div style={{ fontSize: '11px', fontWeight: 'bold', color: '#00d2ff', marginBottom: '6px' }}>
                                  选中信息
                              </div>
                              <div style={{ fontSize: '12px', lineHeight: '1.5', color: '#ddd' }}>
                                  {(hoveredInfo || displayInfo).type === 'feature' ? (
                                    <div>
                                      <div>特证 <strong>#{(hoveredInfo || displayInfo).featureId}</strong></div>
                                      <div>激活值: <span style={{ color: '#4ecdc4' }}>{(hoveredInfo || displayInfo).activation?.toFixed(4)}</span></div>
                                      <div style={{ fontSize: '10px', color: '#aaa', marginTop: '4px' }}>
                                          潜在表示单元。
                                      </div>
                                    </div>
                                  ) : (hoveredInfo || displayInfo).type === 'manifold' ? (
                                    <div>
                                      <div>数据点: {(hoveredInfo || displayInfo).index}</div>
                                      <div>PC1/2/3: {(hoveredInfo || displayInfo).pc1?.toFixed(2)}, {(hoveredInfo || displayInfo).pc2?.toFixed(2)}, {(hoveredInfo || displayInfo).pc3?.toFixed(2)}</div>
                                    </div>
                                  ) : (
                                    <div>
                                       <div>词元: <strong>"{(hoveredInfo || displayInfo).label}"</strong></div>
                                       <div>概率: <span style={{ color: getColor((hoveredInfo || displayInfo).probability) }}>{((hoveredInfo || displayInfo).probability * 100).toFixed(1)}%</span></div>
                                       {(hoveredInfo || displayInfo).actual && <div>实际: "{(hoveredInfo || displayInfo).actual}"</div>}
                                    </div>
                                  )}
                              </div>
                           </div>
                        )}

                        {/* 2B. Analysis Method Summary (Context) */}
                        {analysisResult && !hoveredInfo && (
                             <div style={{ fontSize: '12px', color: '#aaa' }}>
                                 <div style={{ color: '#fff', marginBottom: '4px' }}>
                                     当前分析方法: {structureTab.toUpperCase()}
                                 </div>

                                 {structureTab === 'circuit' && (
                                     <div>
                                         在因果图中发现 {analysisResult.nodes?.length} 个节点和 {analysisResult.graph?.edges?.length} 条边。
                                     </div>
                                 )}
                                 {structureTab === 'features' && (
                                     <div>
                                         从第 {featureForm.layer_idx} 层提取了 {analysisResult.top_features?.length} 个稀疏特征。
                                         <br/>重构误差: {analysisResult.reconstruction_error?.toFixed(5)}
                                     </div>
                                 )}
                             </div>
                        )}

                        {!analysisResult && !hoveredInfo && !displayInfo && (
                            <div style={{ color: '#666', fontStyle: 'italic', fontSize: '12px' }}>
                                悬停在可视化元素上查看详情。
                            </div>
                        )}
                    </div>
                 ) : (
                    <div style={{ fontSize: '12px', color: '#666', fontStyle: 'italic', padding: '20px 0', textAlign: 'center' }}>
                        与模型交互以查看分析详情。
                    </div>
                 )
              )}
          </div>
        </div>
      </SimplePanel>
      )}

      {/* Algo Explanation Modal */}
      {showHelp && (
          <div style={{
              position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
              background: 'rgba(0,0,0,0.8)', backdropFilter: 'blur(5px)',
              zIndex: 1000, display: 'flex', justifyContent: 'center', alignItems: 'center'
          }} onClick={() => setShowHelp(false)}>
              <div
                 onClick={e => e.stopPropagation()}
                 style={{
                    background: '#1a1a1f', border: '1px solid #333', borderRadius: '12px',
                    width: '600px', maxHeight: '80vh', display: 'flex', flexDirection: 'column',
                    boxShadow: '0 10px 40px rgba(0,0,0,0.8)'
                 }}
              >
                  <div style={{ padding: '16px', borderBottom: '1px solid #333', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <h2 style={{ fontSize: '16px', fontWeight: 'bold', color: '#fff', margin: 0 }}>
                          {systemType === 'snn' ? '脉冲神经网络 (SNN) 算法原理' : 'Transformer 模型架构说明'}
                      </h2>
                      <button onClick={() => setShowHelp(false)} style={{ background: 'transparent', border: 'none', color: '#888', cursor: 'pointer' }}>
                          <X size={20} />
                      </button>
                  </div>
                  <div style={{ padding: '20px', overflowY: 'auto', lineHeight: '1.6', fontSize: '13px', color: '#ddd' }}>
                       {systemType === 'snn' ? (
                          <>
                             <div style={{ color: '#4ecdc4', fontWeight: 'bold', borderBottom: '1px solid rgba(78, 205, 196, 0.2)', paddingBottom: '4px', marginBottom: '10px' }}>[A] 专业原理解析 (Professional)</div>
                             <h3 style={{fontSize: '14px', marginTop: 0, color: '#fff'}}>1. 核心模型：Leaky Integrate-and-Fire (LIF)</h3>
                             <p>
                                 SNN 模拟了生物神经元的电生理特性。每个神经元维护一个<strong>膜电位 (Membrane Potential, v)</strong>，遵循以下差分方程：
                             </p>
                             <div style={{ background: '#000', padding: '10px', borderRadius: '6px', fontFamily: 'monospace', marginBottom: '10px', fontSize: '11px' }}>
                                 v[t] = v[t-1] × (1 - dt/τ) + Σ(I_ext + Σ w_ij ⋅ x_j)
                             </div>
                             <ul style={{ paddingLeft: '20px', color: '#aaa', fontSize: '12px' }}>
                                 <li><strong>积分与泄漏：</strong> 电位随输入增加，随时间常数 τ 衰减。</li>
                                 <li><strong>发放与重置：</strong> 超过阈值时发放脉冲 (Spike) 并重置电位。</li>
                                 <li><strong>STDP 学习：</strong> 时间相关的突触可塑性，根据脉冲因果调整权重。</li>
                             </ul>

                             <div style={{ color: '#ff9f43', fontWeight: 'bold', borderBottom: '1px solid rgba(255, 159, 67, 0.2)', paddingBottom: '4px', marginTop: '20px', marginBottom: '10px' }}>[B] 直观理解 (Simplified)</div>
                             <h3 style={{fontSize: '14px', marginTop: 0, color: '#fff'}}>模拟大脑的运作</h3>
                             <p>
                                 脉冲神经网络就像一群在跳舞的精灵：
                             </p>
                             <ul style={{ paddingLeft: '20px', color: '#aaa', fontSize: '12px' }}>
                                 <li><strong>蓄能：</strong> 精灵们像拿小杯子接水，接到一定程度就会大喊一声（发信号）。</li>
                                 <li><strong>遗忘：</strong> 如果水接得太慢，杯子底部的洞会让水流掉（漏水机制）。</li>
                                 <li><strong>同步：</strong> 当两个精灵经常一起喊叫，它们就会变得更有默契（学习过程）。</li>
                             </ul>
                          </>
                      ) : (
                          <>
                             <div style={{ color: '#4ecdc4', fontWeight: 'bold', borderBottom: '1px solid rgba(78, 205, 196, 0.2)', paddingBottom: '4px', marginBottom: '10px' }}>[A] 专业架构解析 (Professional)</div>
                             <h3 style={{fontSize: '14px', marginTop: 0, color: '#fff'}}>1. Transformer 架构</h3>
                             <p>
                                 基于标准的 Decoder-only 架构。核心计算由<strong>多头自注意力 (MHA)</strong> 和<strong>前馈网络 (MLP)</strong> 构成，信息通过<strong>残差流 (Residual Stream)</strong> 传递。
                             </p>
                             <div style={{ background: '#000', padding: '10px', borderRadius: '6px', fontFamily: 'monospace', marginBottom: '10px', fontSize: '11px' }}>
                                 x_{'{l+1}'} = x_{'{l}'} + MHA(LN(x_{'{l}'})) + MLP(LN(x_{'{l}'} + MHA(LN(x_{'{l}'}))))
                             </div>
                             <ul style={{ paddingLeft: '20px', color: '#aaa', fontSize: '12px' }}>
                                 <li><strong>Logit Lens：</strong> 将每一层残差流通过 unembedding 矩阵映射回词汇空间。</li>
                                 <li><strong>激活分析：</strong> 观察神经元在处理特定的语义或语法任务时的响应方向。</li>
                             </ul>

                             <div style={{ color: '#ff9f43', fontWeight: 'bold', borderBottom: '1px solid rgba(255, 159, 67, 0.2)', paddingBottom: '4px', marginTop: '20px', marginBottom: '10px' }}>[B] 直观理解 (Simplified)</div>
                             <h3 style={{fontSize: '14px', marginTop: 0, color: '#fff'}}>AI 的“联想传送带”</h3>
                             <p>
                                 Transformer 就像一个超级高效的组装线：
                             </p>
                             <ul style={{ paddingLeft: '20px', color: '#aaa', fontSize: '12px' }}>
                                 <li><strong>传送带（残差流）：</strong> 信息像零件一样在传送带上走，每一层都会给它贴上新的标签。</li>
                                 <li><strong>聚光灯（注意力）：</strong> 模型在读到某个词时，会把光投向前面相关的词，寻找线索。</li>
                                 <li><strong>逻辑开关（MLP）：</strong> 内部有无数个小开关，负责识别“这是个地名”或者“这在表达赞美”。</li>
                             </ul>
                          </>
                      )}
                  </div>
              </div>
          </div>
      )}

      {/* Right-side Layer Detail Panel */}
      {selectedLayer !== null && data?.layer_details && (
        <SimplePanel
          title={`第 ${selectedLayer} 层详情`}
          onClose={() => {
            setSelectedLayer(null);
            setLayerInfo(null);
          }}
          style={{
            position: 'absolute', right: 340, bottom: 20, zIndex: 10,
            minWidth: '450px', maxWidth: '550px', maxHeight: '80vh'
          }}
        >
          
          {(() => {
            const layerDetail = data.layer_details[selectedLayer];
            if (!layerDetail) return <div style={{padding:'20px', color:'#aaa'}}>加载层详情中...</div>;

            return (
              <div style={{ fontSize: '13px', lineHeight: '1.8' }}>
                {/* 3D Visualization */}
                {layerInfo && (
                  <div style={{ 
                    height: '350px', 
                    background: '#0a0a0a',
                    borderRadius: '8px',
                    marginBottom: '16px',
                    border: '1px solid #333'
                  }}>
                    <ErrorBoundary>
                      <Canvas>
                        <PerspectiveCamera makeDefault position={[0, 0, 12]} fov={50} />
                        <OrbitControls enableDamping dampingFactor={0.05} />
                        <ambientLight intensity={0.4} />
                        <pointLight position={[10, 10, 10]} intensity={0.8} />
                        <pointLight position={[-10, -10, 10]} intensity={0.3} color="#00d2ff" />
                        <LayerDetail3D 
                          layerIdx={selectedLayer} 
                          layerInfo={layerInfo} 
                          onHeadClick={handleHeadClick}
                        />
                      </Canvas>
                    </ErrorBoundary>
                    <div style={{ 
                      padding: '8px', 
                      fontSize: '10px', 
                      color: '#666',
                      textAlign: 'center'
                    }}>
                      💡 拖动旋转 • 滚轮缩放 • 右键平移
                    </div>
                  </div>
                )}
                
                <div style={{ marginBottom: '14px' }}>
                  <h3 style={{ margin: '0 0 8px 0', fontSize: '14px', color: '#fff', fontWeight: '600' }}>
                    架构
                  </h3>
                  <div style={{ display: 'grid', gridTemplateColumns: '140px 1fr', gap: '6px', color: '#aaa' }}>
                    <span>注意力头数:</span>
                    <span style={{ color: '#fff' }}>{layerDetail.n_heads}</span>
                    
                    <span>头维度:</span>
                    <span style={{ color: '#fff' }}>{layerDetail.d_head}</span>
                    
                    <span>MLP隐藏维度:</span>
                    <span style={{ color: '#fff' }}>{layerDetail.d_mlp}</span>
                  </div>
                </div>
                
                <div style={{ borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '14px' }}>
                  <h3 style={{ margin: '0 0 8px 0', fontSize: '14px', color: '#fff', fontWeight: '600' }}>
                    参数
                  </h3>
                  <div style={{ display: 'grid', gridTemplateColumns: '140px 1fr', gap: '6px', color: '#aaa' }}>
                    <span>注意力:</span>
                    <span style={{ color: '#5ec962' }}>
                      {(layerDetail.attn_params / 1e6).toFixed(2)}M
                    </span>
                    
                    <span>MLP (前馈):</span>
                    <span style={{ color: '#5ec962' }}>
                      {(layerDetail.mlp_params / 1e6).toFixed(2)}M
                    </span>
                    
                    <span style={{ fontWeight: '600' }}>总计:</span>
                    <span style={{ color: '#00d2ff', fontWeight: '600' }}>
                      {(layerDetail.total_params / 1e6).toFixed(2)}M
                    </span>
                  </div>
                </div>
                
                <div style={{ 
                  marginTop: '14px', 
                  padding: '10px', 
                  background: 'rgba(0, 210, 255, 0.1)', 
                  borderRadius: '6px',
                  fontSize: '11px',
                  color: '#aaa'
                }}>
                  💡 点击其他层查看详情，或点击 × 关闭
                </div>
              </div>
            );
          })()}
        </SimplePanel>
      )}

      {/* Neuron State Visualization Panel */}
      {layerNeuronState && panelVisibility.neuronPanel && (
        <SimplePanel
          title={t('panels.neuronStateTitle', { layer: layerNeuronState.layer_idx })}
          onClose={() => setLayerNeuronState(null)}
          dragHandleProps={{ onMouseDown: neuronPanel.handleMouseDown }}
          headerStyle={{ cursor: 'grab' }}
          style={{
            position: 'absolute',
            left: `${neuronPanel.position.x}px`,
            top: `${neuronPanel.position.y}px`,
            zIndex: 15,
            width: '350px',
            maxHeight: '60vh'
          }}
        >

          {loadingNeurons ? (
            <div style={{ textAlign: 'center', padding: '40px', color: '#888' }}>
              加载神经元状态中...
            </div>
          ) : (
            <div>
              <div style={{ marginBottom: '20px' }}>
                <h3 style={{ margin: '0 0 12px 0', fontSize: '16px', color: '#fff', fontWeight: '600' }}>
                  注意力模式 ({layerNeuronState.n_heads} 个头)
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px' }}>
                  {layerNeuronState.attention_heads.map(head => (
                    <AttentionHeatmap 
                      key={head.head_idx}
                      pattern={head.pattern}
                      tokens={layerNeuronState.tokens}
                      headIdx={head.head_idx}
                    />
                  ))}
                </div>
              </div>

              <div style={{ borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '16px' }}>
                <h3 style={{ margin: '0 0 12px 0', fontSize: '16px', color: '#fff', fontWeight: '600' }}>
                  MLP激活
                </h3>
                <div style={{ marginBottom: '12px' }}>
                  <MLPActivationChart distribution={layerNeuronState.mlp_stats.activation_distribution} />
                </div>
                <div style={{ fontSize: '11px', color: '#aaa', lineHeight: '1.6' }}>
                  <div>均值: <span style={{ color: '#fff' }}>{layerNeuronState.mlp_stats.mean.toFixed(3)}</span></div>
                  <div>标准差: <span style={{ color: '#fff' }}>{layerNeuronState.mlp_stats.std.toFixed(3)}</span></div>
                  <div>范围: <span style={{ color: '#fff' }}>[{layerNeuronState.mlp_stats.min.toFixed(3)}, {layerNeuronState.mlp_stats.max.toFixed(3)}]</span></div>
                </div>
              </div>

              <div style={{
                marginTop: '16px',
                padding: '10px',
                background: 'rgba(0, 210, 255, 0.1)',
                borderRadius: '6px',
                fontSize: '10px',
                color: '#aaa'
              }}>
                <div><strong>热图:</strong> 从行(查询)到列(键)的注意力</div>
                <div><strong>颜色:</strong> 蓝色(低) → 紫色(中) → 红色(高)</div>
              </div>
            </div>
          )}
        </SimplePanel>
      )}

      {/* Bottom-right Layers Panel */}
      {panelVisibility.layersPanel && (
      <SimplePanel 
        title="层信息"
        style={{
          position: 'absolute', bottom: 20, right: 20, zIndex: 10,
          maxWidth: '300px', maxHeight: '400px'
        }}
      >
        
        {data?.logit_lens && (
          <div style={{ fontSize: '12px' }}>
            {data.logit_lens.map((layerData, layerIdx) => {
              // Calculate average confidence for this layer
              const avgConfidence = layerData.reduce((sum, pos) => sum + pos.prob, 0) / layerData.length;
              const isHovered = hoveredInfo?.layer === layerIdx;
              const isSelected = selectedLayer === layerIdx;
              
              return (
                <div 
                  key={layerIdx}
                  onClick={() => {
                    setSelectedLayer(layerIdx);
                    loadLayerDetails(layerIdx);
                  }}
                  style={{
                    padding: '8px',
                    marginBottom: '6px',
                    background: isSelected ? 'rgba(0, 210, 255, 0.2)' : isHovered ? 'rgba(0, 210, 255, 0.1)' : 'rgba(255,255,255,0.05)',
                    border: isSelected ? '2px solid rgba(0, 210, 255, 0.8)' : isHovered ? '1px solid rgba(0, 210, 255, 0.5)' : '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '6px',
                    transition: 'all 0.2s',
                    cursor: 'pointer'
                  }}
                >
                  <div style={{ fontWeight: 'bold', color: '#fff', marginBottom: '4px' }}>
                    {t('validity.layer', { layer: layerIdx })}
                  </div>
                  <div style={{ color: '#aaa', fontSize: '11px' }}>
                    平均置信度: <span style={{ color: avgConfidence > 0.5 ? '#5ec962' : '#fde725' }}>
                      {(avgConfidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
        
        {!data && (
          <div style={{ fontSize: '13px', color: '#888', fontStyle: 'italic' }}>
            暂无数据。运行分析以查看层信息。
          </div>
        )}
      </SimplePanel>
      )}

      {/* 3D Canvas */}
      <Canvas shadows>
        <PerspectiveCamera makeDefault position={[15, 15, 15]} fov={50} />
        <OrbitControls makeDefault />
        
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} castShadow />
        <spotLight position={[-10, 20, 10]} angle={0.15} penumbra={1} intensity={1} />
        
        {/* Standard LogitLens Visualization - Always visible if data exists */}
        {data && (
          <Text position={[0, 12, -5]} fontSize={1} color="#ffffff" anchorX="center" anchorY="bottom">
            Logit Lens (Token Probabilities)
          </Text>
        )}
        <Visualization data={data} hoveredInfo={hoveredInfo} setHoveredInfo={setHoveredInfo} activeLayer={activeLayer} />

        {/* Analysis Results - Rendered side-by-side if available */}
        {analysisResult && (
          <group position={[-(data?.tokens?.length || 10) - 20, 0, 0]}>
             {/* Add a label or visual separator */}
             <Text position={[0, 10, 0]} fontSize={1} color="#4ecdc4" anchorX="center">
                {structureTab === 'circuit' && '回路分析结果'}
                {structureTab === 'features' && '特征提取结果'}
                {structureTab === 'causal' && '因果分析结果'}
                {structureTab === 'manifold' && '流形分析结果'}
                {structureTab === 'compositional' && t('structure.compositional.title')}
             </Text>
             
             {structureTab === 'circuit' && <NetworkGraph3D graph={analysisResult.graph || analysisResult} />}
             {structureTab === 'features' && <FeatureVisualization3D features={analysisResult.top_features} layerIdx={analysisResult.layer_idx} onLayerClick={setSelectedLayer} selectedLayer={selectedLayer} onHover={setHoveredInfo} />}
             {structureTab === 'causal' && <NetworkGraph3D graph={analysisResult.causal_graph} />}
             {structureTab === 'manifold' && <ManifoldVisualization3D pcaData={analysisResult.pca || analysisResult} onHover={setHoveredInfo} />}
             {structureTab === 'compositional' && <CompositionalVisualization3D result={analysisResult} t={t} />}
             {structureTab === 'agi' && <FiberBundleVisualization3D result={analysisResult} t={t} />}
             {structureTab === 'fiber' && <FiberBundleVisualization3D result={analysisResult} t={t} />}
             {structureTab === 'validity' && <ValidityVisualization3D result={analysisResult} t={t} />}
          </group>
        )}

        {/* Debug Log for SNN Rendering Conditions */}
        {(() => {
             if (infoPanelTab === 'snn' || snnState.initialized) {
                 console.log('[App] SNN Render Check:', { infoPanelTab, initialized: snnState.initialized, hasStructure: !!snnState.structure });
             }
             return null;
        })()}

        {/* SNN Visualization - Independent of structure analysis result */}
        {(infoPanelTab === 'snn' || systemType === 'snn') && snnState.initialized && (
           <group position={(!data || systemType === 'snn') ? [0, 0, 0] : [-(data?.tokens?.length || 10) - 20, 0, 0]}>
              <SNNVisualization3D 
                  t={t} 
                  structure={snnState.structure}
                  activeSpikes={snnState.spikes}
              />
           </group>
        )}
        
        {/* Magnified Layer Visualization during generation */}
        {activeLayer !== null && activeLayerInfo && (
          <group position={[30, 0, 0]}>
            {/* Phase indicator */}
            <Text
              position={[0, 8, 0]}
              fontSize={0.5}
              color="#00d2ff"
              anchorX="center"
            >
              {computationPhase === 'attention' && t('app.computingAttention')}
              {computationPhase === 'mlp' && t('app.processingMlp')}
              {computationPhase === 'output' && t('app.generatingOutput')}
            </Text>
            
            <LayerDetail3D 
              layerIdx={activeLayer} 
              layerInfo={activeLayerInfo}
              animationPhase={computationPhase}
              isActive={true}
              onHeadClick={handleHeadClick}
            />
          </group>
        )}
        
        <ContactShadows resolution={1024} scale={20} blur={2} opacity={0.35} far={10} color="#000000" />
        <gridHelper args={[100, 50, '#222', '#111']} position={[0, -0.6, 0]} />
      </Canvas>

      {/* Head Analysis Panel - Draggable */}
      {panelVisibility.headPanel && headPanel.isOpen && (
        <SimplePanel
          title={t ? t('head.title', { layer: headPanel.layerIdx, head: headPanel.headIdx }) : `Layer ${headPanel.layerIdx} Head ${headPanel.headIdx}`}
          onClose={() => setHeadPanel({ ...headPanel, isOpen: false })}
          dragHandleProps={{ onMouseDown: headPanelDrag.handleMouseDown }}
          headerStyle={{ cursor: 'grab' }}
          style={{
            position: 'absolute',
            left: `${headPanelDrag.position.x}px`,
            top: `${headPanelDrag.position.y}px`,
            zIndex: 25,
            width: '500px',
            height: '400px'
          }}
        >
          <HeadAnalysisPanel 
            layerIdx={headPanel.layerIdx} 
            headIdx={headPanel.headIdx} 
            prompt={prompt}
            t={t}
          />
        </SimplePanel>
      )}

    </div>
  );
}
