import { Html, Line, OrbitControls, PerspectiveCamera, Sphere, Stars, TransformControls } from '@react-three/drei';
import { Canvas } from '@react-three/fiber';
import { useEffect, useMemo, useState } from 'react';
import * as THREE from 'three';
import { Vector3 } from 'three';

const API_BASE = "http://localhost:5002";

// Locales mock for standalone functionality
const locales = {
  en: { glassMatrix: "Glass Matrix", multimodal: "Multi-modal", model: "Model" },
  zh: { glassMatrix: "玻璃矩阵", multimodal: "多模态", model: "模型" }
};
const lang = 'zh';

const ManifoldGeometry = ({ data, currentLayer, onPointSelect, selectedId }) => {
  const points = useMemo(() => {
    if (!data || !data[currentLayer]) return [];
    return data[currentLayer].pca.map(p => new Vector3(p[0] * 5, p[1] * 5, p[2] * 5));
  }, [data, currentLayer]);

  if (points.length === 0) return null;

  return (
    <group>
      {points.map((p, i) => (
        <Sphere 
          key={i} 
          args={[0.07, 8, 8]} 
          position={p}
          onClick={(e) => {
            e.stopPropagation();
            onPointSelect(i, p.clone());
          }}
        >
          <meshStandardMaterial 
            color={selectedId === i ? "#ffffff" : new THREE.Color().setHSL(i / points.length, 0.8, 0.5)} 
            emissive={selectedId === i ? "#ffffff" : new THREE.Color().setHSL(i / points.length, 0.8, 0.2)}
            emissiveIntensity={selectedId === i ? 2 : 1}
          />
        </Sphere>
      ))}
      <Line 
        points={points.slice(0, 50)} 
        color="#00ffff"
        lineWidth={0.5}
        transparent
        opacity={0.1}
      />
    </group>
  );
};

const VisionAlignmentOverlay = ({ anchors }) => {
  return (
    <group>
      {anchors.map((anchor, i) => {
        const pos = new Vector3(anchor.projection[0] * 5, anchor.projection[1] * 5, anchor.projection[2] * 5);
        return (
          <group key={i} position={pos}>
            <Sphere args={[0.15, 16, 16]}>
              <meshBasicMaterial color="#ffcc00" transparent opacity={0.8} />
            </Sphere>
            <Html distanceFactor={10}>
              <div className="bg-black/80 text-[#ffcc00] p-1 border border-[#ffcc00] text-[10px] rounded whitespace-nowrap">
                {anchor.label}
              </div>
            </Html>
          </group>
        );
      })}
    </group>
  );
};

const LocusOfAttention = ({ data }) => {
  if (!data) return null;
  const pos = new Vector3(data.position[0] * 5, data.position[1] * 5, data.position[2] * 5);
  
  return (
    <group position={pos}>
      <Sphere args={[0.3, 32, 32]}>
        <meshStandardMaterial 
          color="#ff00ff" 
          emissive="#ff00ff" 
          emissiveIntensity={5} 
          transparent 
          opacity={0.6} 
        />
      </Sphere>
      <pointLight color="#ff00ff" intensity={4} distance={5} />
      <Html distanceFactor={10}>
        <div className="bg-pink-600/90 text-white p-1 font-bold text-[8px] rounded uppercase animate-pulse">
          Locus of Attention
        </div>
      </Html>
    </group>
  );
};

const AlignmentFibers = ({ visionAnchors, topologyData, currentLayer }) => {
  const fibers = useMemo(() => {
    if (!visionAnchors || !topologyData || !topologyData[currentLayer]) return [];
    
    // 寻找逻辑流形点与视觉锚点的对应关系
    const links = [];
    visionAnchors.forEach(anchor => {
      // 简化逻辑：匹配数字标签
      const digitStr = anchor.label.split('_')[1]; // e.g. "MNIST_3" -> "3"
      const logicalIdx = parseInt(digitStr) + 1; // 假设逻辑点 1-10 对应数字 0-9
      
      if (topologyData[currentLayer].pca[logicalIdx]) {
        const vPos = new Vector3(anchor.projection[0] * 5, anchor.projection[1] * 5, anchor.projection[2] * 5);
        const p = topologyData[currentLayer].pca[logicalIdx];
        const lPos = new Vector3(p[0] * 5, p[1] * 5, p[2] * 5);
        links.push({ start: vPos, end: lPos });
      }
    });
    return links;
  }, [visionAnchors, topologyData, currentLayer]);

  return (
    <group>
      {fibers.map((f, i) => (
        <Line 
          key={i}
          points={[f.start, f.end]}
          color="#ff00ff"
          lineWidth={1}
          transparent
          opacity={0.3}
          dashed
          dashScale={5}
          dashSize={0.2}
          gapSize={0.1}
        />
      ))}
    </group>
  );
};

export default function GlassMatrix3D() {
  const [topologyData, setTopologyData] = useState(null);
  const [currentLayer, setCurrentLayer] = useState("0");
  const [selectedModel, setSelectedModel] = useState("gpt2");
  const [multimodalMode, setMultimodalMode] = useState(false);
  const [visionAnchors, setVisionAnchors] = useState([]);
  const [showFlux, setShowFlux] = useState(true);
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [surgeryStatus, setSurgeryStatus] = useState("IDLE");
  const [gwtStatus, setGwtStatus] = useState(null);

  // Load Topology Data
  useEffect(() => {
    fetch(`${API_BASE}/nfb/topology?model=${selectedModel}`)
      .then(res => res.json())
      .then(data => {
        if (data.layers) {
          setTopologyData(data.layers);
          const layers = Object.keys(data.layers).sort((a,b) => parseInt(a)-parseInt(b));
          if (layers.length > 0) setCurrentLayer(layers[0]);
        }
      })
      .catch(err => console.error("Topology fetch error:", err));
  }, [selectedModel]);

  // Load Vision Anchors
  useEffect(() => {
    if (multimodalMode && visionAnchors.length === 0) {
      fetch(`${API_BASE}/nfb/multimodal/align?model=${selectedModel}`)
        .then(res => res.json())
        .then(data => {
          if (data.anchors) setVisionAnchors(data.anchors);
        })
        .catch(err => console.error("Vision fetch error:", err));
    }
  }, [multimodalMode, selectedModel, visionAnchors.length]);

  // Sync Global Workspace Status
  useEffect(() => {
    const timer = setInterval(() => {
      fetch(`${API_BASE}/nfb/gwt/status`)
        .then(res => res.json())
        .then(data => setGwtStatus(data))
        .catch(err => console.error("GWT fetch error:", err));
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="w-full h-full relative" style={{ background: '#050505' }}>
      {/* HUD Layer */}
      <div className="absolute top-4 left-4 z-10 flex flex-col gap-2 pointer-events-none">
        <div className="bg-black/60 backdrop-blur-md p-4 rounded-lg border border-white/10 pointer-events-auto">
          <h2 className="text-pink-400 font-bold tracking-wider text-sm mb-2 select-none uppercase">
            {locales[lang].glassMatrix}
          </h2>
          
          <div className="flex flex-col gap-3">
             <div className="flex items-center justify-between gap-4">
              <span className="text-white/50 text-xs font-medium uppercase tracking-tighter">Model</span>
              <select 
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="bg-zinc-800 text-white text-[10px] rounded px-2 py-1 border border-white/20 focus:outline-none focus:border-pink-500 transition-colors pointer-events-auto"
              >
                <option value="gpt2">GPT-2 (124M)</option>
                <option value="qwen3">Qwen3 (4B)</option>
              </select>
            </div>

            <div className="flex items-center justify-between gap-4">
              <span className="text-white/50 text-xs font-medium uppercase tracking-tighter">Multi-modal</span>
              <button 
                onClick={() => setMultimodalMode(!multimodalMode)}
                className={`text-[10px] rounded px-3 py-1 border transition-all duration-300 pointer-events-auto ${multimodalMode ? 'bg-pink-500/20 border-pink-500 text-pink-400' : 'bg-zinc-800 border-white/20 text-white/40'}`}
              >
                {multimodalMode ? 'ACTIVE' : 'OFF'}
              </button>
            </div>

            <div className="flex items-center justify-between gap-4 mt-2 pt-2 border-t border-white/5">
              <span className="text-white/50 text-xs font-medium uppercase tracking-tighter">Layer</span>
              <select 
                value={currentLayer} 
                onChange={(e) => setCurrentLayer(e.target.value)}
                className="bg-zinc-800 text-white text-[10px] rounded px-2 py-1 border border-white/20 focus:outline-none focus:border-pink-500 transition-colors pointer-events-auto"
              >
                {topologyData && Object.keys(topologyData).sort((a,b) => parseInt(a)-parseInt(b)).map(l => (
                  <option key={l} value={l}>Block {l}</option>
                ))}
              </select>
            </div>

            <div className="text-[9px] text-zinc-500 mt-2 select-none">
                SURGERY: {surgeryStatus}
            </div>
            {selectedPoint && (
              <div className="text-[9px] text-zinc-400">
                POINT SELECTED: #{selectedPoint.id}
              </div>
            )}
          </div>
        </div>
      </div>

      <Canvas shadows dpr={[1, 2]}>
        <PerspectiveCamera makeDefault position={[0, 0, 15]} fov={50} />
        <OrbitControls enableDamping dampingFactor={0.05} />
        
        <ambientLight intensity={0.2} />
        <pointLight position={[10, 10, 10]} intensity={1.5} color="#00ffff" />
        <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
        
        {topologyData && (
          <ManifoldGeometry 
            data={topologyData} 
            currentLayer={currentLayer} 
            onPointSelect={(idx, pos) => setSelectedPoint({ id: idx, position: pos })}
            selectedId={selectedPoint?.id}
          />
        )}
        
        {multimodalMode && <VisionAlignmentOverlay anchors={visionAnchors} />}
        {multimodalMode && topologyData && (
          <AlignmentFibers 
            visionAnchors={visionAnchors} 
            topologyData={topologyData} 
            currentLayer={currentLayer} 
          />
        )}
        
        <LocusOfAttention data={gwtStatus} />
        
        {selectedPoint && (
          <TransformControls 
            position={selectedPoint.position} 
            mode="translate"
            onChange={() => {
              // 实时同步干预状态
              if (selectedPoint) {
                const pos = selectedPoint.position;
                fetch(`${API_BASE}/nfb/sync/interfere`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    modality: multimodalMode ? "vision" : "text",
                    layer_idx: parseInt(currentLayer),
                    x: pos.x / 5,
                    y: pos.y / 5,
                    z: pos.z / 5
                  })
                }).catch(err => console.error("Sync error:", err));
              }
            }}
            onMouseUp={(e) => {
              setSurgeryStatus("OPERATING...");
              setTimeout(() => setSurgeryStatus("SYNCED"), 1000);
            }}
          />
        )}
        
        <fog attach="fog" args={['#050505', 10, 25]} />
      </Canvas>
    </div>
  );
}
