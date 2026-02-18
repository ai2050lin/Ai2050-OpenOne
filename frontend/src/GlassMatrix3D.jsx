import { Html, Line, OrbitControls, PerspectiveCamera, Sphere, Stars, TransformControls } from '@react-three/drei';
import { Canvas, useThree } from '@react-three/fiber';
import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { Vector3 } from 'three';
import { pollRuntimeWithFallback } from './utils/runtimeClient';

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');
const locales = {
  en: { glassMatrix: 'Glass Matrix' },
  zh: { glassMatrix: '玻璃矩阵' },
};
const lang = 'zh';
const DEFAULT_SCALE = 5;
const TARGET_RADIUS = 7;

const getScaleForLayer = (layerData) => {
  if (!layerData || !Array.isArray(layerData.pca) || layerData.pca.length === 0) {
    return DEFAULT_SCALE;
  }
  const absValues = [];
  layerData.pca.forEach((p) => {
    if (Array.isArray(p) && p.length >= 3) {
      absValues.push(Math.abs(p[0]), Math.abs(p[1]), Math.abs(p[2]));
    }
  });
  if (absValues.length === 0) return DEFAULT_SCALE;
  absValues.sort((a, b) => a - b);
  const p95 = absValues[Math.floor(absValues.length * 0.95)] || 0;
  if (p95 < 1e-6) return DEFAULT_SCALE;
  return TARGET_RADIUS / p95;
};

const buildLayerPoints = (layerData, coordScale) => {
  if (!layerData || !Array.isArray(layerData.pca)) return [];
  return layerData.pca
    .filter((p) => Array.isArray(p) && p.length >= 3)
    .map((p) => new Vector3(p[0] * coordScale, p[1] * coordScale, p[2] * coordScale));
};

const normalizePoint3D = (point) => {
  if (!Array.isArray(point)) return null;
  const normalized = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    const value = Number(point[i] ?? 0);
    if (!Number.isFinite(value)) return null;
    normalized[i] = value;
  }
  return normalized;
};

const normalizeTopologyLayers = (layersData) => {
  if (!layersData || typeof layersData !== 'object') return null;
  const normalized = {};
  Object.entries(layersData).forEach(([layerId, layerData]) => {
    const rawPoints = Array.isArray(layerData?.pca)
      ? layerData.pca
      : Array.isArray(layerData?.projections)
        ? layerData.projections
        : [];
    const validPoints = rawPoints.map(normalizePoint3D).filter(Boolean);
    normalized[layerId] = {
      ...layerData,
      pca: validPoints,
      rawPointCount: rawPoints.length,
      validPointCount: validPoints.length,
    };
  });
  return normalized;
};

const normalizeGwsPosition = (raw) => {
  if (!Array.isArray(raw) || raw.length < 3) return null;
  const position = [];
  for (let i = 0; i < 3; i++) {
    const value = Number(raw[i]);
    if (!Number.isFinite(value)) return null;
    position.push(value);
  }
  return position;
};

const AutoFitCamera = ({ points, controlsRef }) => {
  const { camera } = useThree();

  useEffect(() => {
    if (!points || points.length === 0) return;

    const box = new THREE.Box3().setFromPoints(points);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z, 1);
    const fov = (camera.fov || 50) * (Math.PI / 180);
    const distance = (maxDim / (2 * Math.tan(fov / 2))) * 1.35;

    camera.position.set(center.x, center.y, center.z + distance);
    camera.near = Math.max(0.01, distance / 200);
    camera.far = Math.max(100, distance * 25);
    camera.lookAt(center);
    camera.updateProjectionMatrix();

    if (controlsRef?.current) {
      controlsRef.current.target.copy(center);
      controlsRef.current.update();
    }
  }, [points, camera, controlsRef]);

  return null;
};

const ManifoldGeometry = ({ points, onPointSelect, selectedId }) => {
  if (points.length === 0) return null;

  return (
    <group>
      {points.map((p, i) => (
        <Sphere
          key={i}
          args={[0.09, 10, 10]}
          position={p}
          onClick={(e) => {
            e.stopPropagation();
            onPointSelect(i, p.clone());
          }}
        >
          <meshStandardMaterial
            color={selectedId === i ? '#ffffff' : new THREE.Color().setHSL(i / points.length, 0.8, 0.5)}
            emissive={selectedId === i ? '#ffffff' : new THREE.Color().setHSL(i / points.length, 0.8, 0.2)}
            emissiveIntensity={selectedId === i ? 2 : 1}
          />
        </Sphere>
      ))}
      <Line points={points} color="#00ffff" lineWidth={0.5} transparent opacity={0.1} />
    </group>
  );
};

const VisionAlignmentOverlay = ({ anchors, coordScale }) => (
  <group>
    {anchors.map((anchor, i) => {
      const pos = new Vector3(
        anchor.projection[0] * coordScale,
        anchor.projection[1] * coordScale,
        anchor.projection[2] * coordScale
      );
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

const LocusOfAttention = ({ data, coordScale }) => {
  if (!data?.position) return null;
  const pos = new Vector3(data.position[0] * coordScale, data.position[1] * coordScale, data.position[2] * coordScale);

  return (
    <group position={pos}>
      <Sphere args={[0.3, 32, 32]}>
        <meshStandardMaterial color="#ff00ff" emissive="#ff00ff" emissiveIntensity={5} transparent opacity={0.6} />
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

const AlignmentFibers = ({ visionAnchors, topologyData, currentLayer, coordScale }) => {
  const fibers = useMemo(() => {
    if (!visionAnchors || !topologyData || !topologyData[currentLayer]) return [];

    const links = [];
    visionAnchors.forEach((anchor) => {
      const digitStr = anchor.label?.split('_')?.[1];
      const logicalIdx = Number.parseInt(digitStr, 10) + 1;
      if (!Number.isFinite(logicalIdx)) return;

      const point = topologyData[currentLayer].pca?.[logicalIdx];
      if (!point) return;

      const vPos = new Vector3(
        anchor.projection[0] * coordScale,
        anchor.projection[1] * coordScale,
        anchor.projection[2] * coordScale
      );
      const lPos = new Vector3(point[0] * coordScale, point[1] * coordScale, point[2] * coordScale);
      links.push({ start: vPos, end: lPos });
    });
    return links;
  }, [visionAnchors, topologyData, currentLayer, coordScale]);

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
  const [currentLayer, setCurrentLayer] = useState('0');
  const [selectedModel, setSelectedModel] = useState('gpt2');
  const [multimodalMode, setMultimodalMode] = useState(false);
  const [visionAnchors, setVisionAnchors] = useState([]);
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [surgeryStatus, setSurgeryStatus] = useState('IDLE');
  const [gwtStatus, setGwtStatus] = useState(null);
  const [runtimeProtocol, setRuntimeProtocol] = useState('legacy');
  const [topologyProtocol, setTopologyProtocol] = useState('legacy');
  const [loadError, setLoadError] = useState(null);
  const controlsRef = useRef(null);
  const runtimeStepRef = useRef(0);

  const currentLayerData = useMemo(() => {
    if (!topologyData || !topologyData[currentLayer]) return null;
    return topologyData[currentLayer];
  }, [topologyData, currentLayer]);

  const coordScale = useMemo(() => getScaleForLayer(currentLayerData), [currentLayerData]);
  const currentPoints = useMemo(() => buildLayerPoints(currentLayerData, coordScale), [currentLayerData, coordScale]);

  useEffect(() => {
    let isMounted = true;
    setLoadError(null);
    setSelectedPoint(null);
    setTopologyData(null);

    const fetchLegacyTopology = async () => {
      const res = await fetch(`${API_BASE}/nfb/topology?model=${selectedModel}`);
      if (!res.ok) throw new Error(`legacy topology failed: ${res.status}`);
      const payload = await res.json();
      if (payload?.status === 'error') {
        throw new Error(payload?.message || 'Topology data unavailable');
      }
      const layersData = payload?.layers || payload?.data?.layers;
      const normalizedLayers = normalizeTopologyLayers(layersData);
      if (!normalizedLayers || Object.keys(normalizedLayers).length === 0) {
        throw new Error('Topology payload has no layers');
      }
      return normalizedLayers;
    };

    const mapRuntimeTopologyEvents = (events) => {
      const topologyEvent = events.find((e) => e?.event_type === 'TopologySignal');
      const layersData = topologyEvent?.payload?.layers;
      const normalizedLayers = normalizeTopologyLayers(layersData);
      if (!normalizedLayers || Object.keys(normalizedLayers).length === 0) {
        return null;
      }
      return normalizedLayers;
    };

    const loadTopology = async () => {
      try {
        const result = await pollRuntimeWithFallback({
          apiBase: API_BASE,
          runRequest: {
            route: 'fiber_bundle',
            analysis_type: 'topology_snapshot',
            model: selectedModel,
            params: {},
            input_payload: {},
          },
          mapRuntimeEvents: mapRuntimeTopologyEvents,
          fetchLegacy: fetchLegacyTopology,
          eventLimit: 5,
        });

        if (!isMounted) return;
        setTopologyProtocol(result.source);
        setTopologyData(result.data);
        const layers = Object.keys(result.data).sort((a, b) => Number.parseInt(a, 10) - Number.parseInt(b, 10));
        if (layers.length > 0) setCurrentLayer(layers[0]);
      } catch (err) {
        if (!isMounted) return;
        console.error('Topology fetch error:', err);
        setLoadError(String(err));
      }
    };

    loadTopology();
    return () => {
      isMounted = false;
    };
  }, [selectedModel]);

  useEffect(() => {
    if (multimodalMode && visionAnchors.length === 0) {
      fetch(`${API_BASE}/nfb/multimodal/align?model=${selectedModel}`)
        .then((res) => res.json())
        .then((data) => {
          if (data.anchors) setVisionAnchors(data.anchors);
        })
        .catch((err) => console.error('Vision fetch error:', err));
    }
  }, [multimodalMode, selectedModel, visionAnchors.length]);

  useEffect(() => {
    let isMounted = true;

    const fetchLegacyGwtStatus = async () => {
      const res = await fetch(`${API_BASE}/nfb/gwt/status`);
      if (!res.ok) throw new Error(`legacy gwt status failed: ${res.status}`);
      const data = await res.json();
      return data;
    };

    const mapRuntimeEvents = (events) => {
      const activation = events.find((e) => e?.event_type === 'ActivationSnapshot');
      const alignment = events.find((e) => e?.event_type === 'AlignmentSignal');
      const statePosition = normalizeGwsPosition(activation?.payload?.gws_state);
      if (!statePosition) return null;
      return {
        position: statePosition,
        winner_module: alignment?.payload?.winner_module || null,
        signal_norm: Number(activation?.payload?.signal_norm || 0),
      };
    };

    const tick = async () => {
      const stepId = runtimeStepRef.current++;
      try {
        const result = await pollRuntimeWithFallback({
          apiBase: API_BASE,
          runRequest: {
          route: 'fiber_bundle',
          analysis_type: 'unified_conscious_field',
          model: selectedModel,
          params: { step_id: stepId, noise_scale: 0.35 },
          input_payload: {},
          },
          mapRuntimeEvents,
          fetchLegacy: fetchLegacyGwtStatus,
          eventLimit: 20,
        });
        if (!isMounted) return;
        setRuntimeProtocol(result.source);
        setGwtStatus({ ...result.data, source: result.source });
      } catch (err) {
        if (!isMounted) return;
        setGwtStatus(null);
        console.error('GWT fetch error:', err);
      }
    };

    tick();
    const timer = setInterval(tick, 1500);
    return () => {
      isMounted = false;
      clearInterval(timer);
    };
  }, [selectedModel]);

  return (
    <div className="w-full h-full relative" style={{ background: '#050505' }}>
      <div className="absolute top-4 left-4 z-10 flex flex-col gap-2 pointer-events-none">
        <div className="bg-black/60 backdrop-blur-md p-4 rounded-lg border border-white/10 pointer-events-auto">
          <h2 className="text-pink-400 font-bold tracking-wider text-sm mb-2 select-none uppercase">{locales[lang].glassMatrix}</h2>

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
                {topologyData &&
                  Object.keys(topologyData)
                    .sort((a, b) => Number.parseInt(a, 10) - Number.parseInt(b, 10))
                    .map((l) => (
                      <option key={l} value={l}>
                        Block {l}
                      </option>
                    ))}
              </select>
            </div>

            <div className="text-[9px] text-zinc-500 mt-2 select-none">SURGERY: {surgeryStatus}</div>
            <div className="text-[9px] text-zinc-500 select-none">MONITOR: {runtimeProtocol.toUpperCase()}</div>
            <div className="text-[9px] text-zinc-500 select-none">TOPOLOGY: {topologyProtocol.toUpperCase()}</div>
            {currentLayerData && (
              <div className="text-[9px] text-zinc-500 select-none">
                POINTS: {currentLayerData.validPointCount ?? currentPoints.length}/{currentLayerData.rawPointCount ?? currentPoints.length}
              </div>
            )}
            {selectedPoint && <div className="text-[9px] text-zinc-400">POINT SELECTED: #{selectedPoint.id}</div>}
            {loadError && <div className="text-[9px] text-red-400 max-w-[260px] break-words">ERROR: {loadError}</div>}
          </div>
        </div>
      </div>

      <Canvas shadows dpr={[1, 2]}>
        <PerspectiveCamera makeDefault position={[0, 0, 15]} fov={50} />
        <OrbitControls ref={controlsRef} enableDamping dampingFactor={0.05} />
        <AutoFitCamera points={currentPoints} controlsRef={controlsRef} />

        <ambientLight intensity={0.2} />
        <pointLight position={[10, 10, 10]} intensity={1.5} color="#00ffff" />
        <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />

        {topologyData && (
          <ManifoldGeometry
            points={currentPoints}
            onPointSelect={(idx, pos) => setSelectedPoint({ id: idx, position: pos })}
            selectedId={selectedPoint?.id}
          />
        )}

        {multimodalMode && <VisionAlignmentOverlay anchors={visionAnchors} coordScale={coordScale} />}
        {multimodalMode && topologyData && (
          <AlignmentFibers
            visionAnchors={visionAnchors}
            topologyData={topologyData}
            currentLayer={currentLayer}
            coordScale={coordScale}
          />
        )}

        <LocusOfAttention data={gwtStatus} coordScale={coordScale} />

        {selectedPoint && (
          <TransformControls
            position={selectedPoint.position}
            mode="translate"
            onChange={() => {
              if (selectedPoint) {
                const pos = selectedPoint.position;
                fetch(`${API_BASE}/nfb/sync/interfere`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    modality: multimodalMode ? 'vision' : 'text',
                    layer_idx: Number.parseInt(currentLayer, 10),
                    x: pos.x / coordScale,
                    y: pos.y / coordScale,
                    z: pos.z / coordScale,
                  }),
                }).catch((err) => console.error('Sync error:', err));
              }
            }}
            onMouseUp={() => {
              setSurgeryStatus('OPERATING...');
              setTimeout(() => setSurgeryStatus('SYNCED'), 1000);
            }}
          />
        )}

        <fog attach="fog" args={['#050505', 15, 80]} />
      </Canvas>
    </div>
  );
}
