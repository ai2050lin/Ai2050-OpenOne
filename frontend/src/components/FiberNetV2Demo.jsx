
import { Line, OrbitControls, Text } from '@react-three/drei';
import { Canvas } from '@react-three/fiber';
import { Network, Scissors } from 'lucide-react';
import { useEffect, useState } from 'react';
import * as THREE from 'three';
import { SimplePanel } from '../SimplePanel';

const FiberNetV2Demo = ({ t }) => {

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  /* Animation State */
  const [animState, setAnimState] = useState({
      injecting: false,
      transporting: false,
      constraining: false
  });

  const [dataMode, setDataMode] = useState('mock'); // 'mock' | 'real'

  // Fetch simulation data from backend
  const fetchData = (mode = dataMode) => {
      setLoading(true);
      const endpoint = mode === 'real' ? '/nfb_ra/data' : '/fibernet_v2/demo';
      fetch(`http://localhost:5002${endpoint}`)
          .then(res => res.json())
          .then(d => {
              if (d.error || d.detail) {
                  console.warn("Backend Error:", d.error || d.detail);
                  setDataMode('mock'); // Auto-revert
                  return;
              }
              setData(d);
              setLoading(false);
          })
          .catch(err => {
              console.error(err);
              setLoading(false);
              setDataMode('mock'); // Fallback on network error
          });
  };

  useEffect(() => {
      fetchData(dataMode);
  }, [dataMode]);

  /* Surgery State */
  const [surgeryMode, setSurgeryMode] = useState(false);
  const [surgeryTool, setSurgeryTool] = useState('graft'); // 'graft' | 'ablate'
  const [selection, setSelection] = useState([]); // [source_id, target_id]

  /* Surgery Handlers */
  const handleSurgeryClick = (nodeId) => {
      if (!surgeryMode) return;

      if (surgeryTool === 'ablate') {
          // Immediate Ablation Confirmation
          if (confirm(`Confirm Ablation of Concept Node ${nodeId}?`)) {
              performSurgery('ablate', nodeId);
          }
      } else if (surgeryTool === 'graft') {
          // Source-Target Selection
          if (selection.length === 0) {
              setSelection([nodeId]);
          } else if (selection.length === 1) {
              if (selection[0] === nodeId) return; // Ignore self-select
              if (confirm(`Graft connection from ${selection[0]} to ${nodeId}?`)) {
                  performSurgery('graft', selection[0], nodeId);
              }
              setSelection([]); // Reset
          }
      }
  };

  const performSurgery = (action, src, tgt = null) => {
      // Validation: Mock Mode
      if (dataMode === 'mock') {
          alert("Simulation Mode: Surgery is visual-only. Please switch to Real Data (NFB-RA) to apply actual hooks.");
          return;
      }

      // Parse IDs safely
      const s_id = parseInt(src);
      const t_id = tgt ? parseInt(tgt) : null;

      // Validation: Invalid IDs
      if (isNaN(s_id) || (action === 'graft' && isNaN(t_id))) {
          console.error("Invalid Node IDs for Surgery:", { src, tgt, s_id, t_id });
          alert("Error: Invalid Node IDs. Cannot perform surgery on this selection.");
          return;
      }
      
      const payload = {
          action: action,
          source_id: action === 'graft' ? s_id : null,
          target_id: action === 'graft' ? t_id : s_id, // For ablate, target is src
          layer: 6,
          strength: 1.5
      };

      console.log("Sending Surgery Payload:", payload);

      fetch('http://localhost:5000/nfb_ra/surgery', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
      })
      .then(async res => {
          const d = await res.json();
          if (!res.ok) throw new Error(d.detail || res.statusText);
          return d;
      })
      .then(d => {
          alert(`Surgery Success: ${d.message}`);
          fetchData(); 
      })
      .catch(err => {
          console.error("Surgery Error:", err);
          alert("Surgery Failed: " + err.message);
      });
  };

  // Controls
  const handleInject = () => {
      setAnimState(prev => ({ ...prev, injecting: true }));
      setTimeout(() => setAnimState(prev => ({ ...prev, injecting: false })), 2000);
      if (dataMode === 'mock') fetchData(); 
  };

  // ... (keep other handlers) ...
  const handleTransport = () => {
      setAnimState(prev => ({ ...prev, transporting: true }));
      setTimeout(() => setAnimState(prev => ({ ...prev, transporting: false })), 3000);
  };

  const handleConstraint = () => {
      setAnimState(prev => ({ ...prev, constraining: true }));
      setTimeout(() => setAnimState(prev => ({ ...prev, constraining: false })), 2000);
  };

  if (loading) return <div style={{color: 'white', textAlign: 'center', padding: '20px'}}>Loading Neural Fiber Simulation...</div>;

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative', display: 'flex', flexDirection: 'column' }}>
        {/* Control Window - Moved to SimplePanel for better Z-Index/Layout */}
        <SimplePanel
            title="Neural Fiber Controls"
            style={{
                position: 'absolute', bottom: 80, right: 20, zIndex: 1000,
                width: '300px', pointerEvents: 'auto', maxHeight: '600px',
                background: 'rgba(20, 20, 25, 0.9)'
            }}
            headerStyle={{ cursor: 'move' }}
        >
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                
                {/* Data Source Switch */}
                <button
                    onClick={() => setDataMode(prev => prev === 'mock' ? 'real' : 'mock')}
                    style={{ 
                        background: '#333', color: 'white', border: '1px solid #555', 
                        padding: '8px', cursor: 'pointer', borderRadius: '4px', 
                        fontSize: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px'
                    }}
                >
                    <Network size={14} />
                    {dataMode === 'mock' ? 'Switch to Real Data (NFB-RA)' : 'Switch to Mock Demo'}
                </button>

                {/* Validation Modes */}
                <div style={{ fontSize: '12px', color: '#aaa', fontWeight: 'bold' }}>Animation Modes</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '5px' }}>
                    <button onClick={handleInject} style={{ background: animState.injecting ? '#44ff44' : '#333', color: animState.injecting ? '#000' : 'white', border: '1px solid #555', padding: '6px', borderRadius: '4px', fontSize: '11px', cursor: 'pointer' }}>
                        Inject
                    </button>
                    <button onClick={handleTransport} style={{ background: animState.transporting ? '#4488ff' : '#333', color: 'white', border: '1px solid #555', padding: '6px', borderRadius: '4px', fontSize: '11px', cursor: 'pointer' }}>
                        Transport
                    </button>
                    <button onClick={handleConstraint} style={{ background: animState.constraining ? '#ff4444' : '#333', color: 'white', border: '1px solid #555', padding: '6px', borderRadius: '4px', fontSize: '11px', cursor: 'pointer' }}>
                        Manifold
                    </button>
                </div>

                <div style={{ borderTop: '1px solid rgba(255,255,255,0.1)', margin: '4px 0' }} />

                {/* Surgery Toolkit */}
                <div style={{ fontSize: '12px', color: '#ff6b6b', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Scissors size={14} /> Manifold Surgery
                </div>
                
                <button 
                    onClick={() => setSurgeryMode(!surgeryMode)}
                    style={{ 
                        background: surgeryMode ? '#ff4444' : '#333', 
                        color: 'white', 
                        border: '1px solid #ff4444', 
                        padding: '8px', 
                        borderRadius: '4px', 
                        fontWeight: 'bold',
                        cursor: 'pointer',
                        fontSize: '12px'
                    }}
                >
                    {surgeryMode ? 'DISABLE SURGERY MODE' : 'ENABLE SURGERY MODE'}
                </button>
                
                {surgeryMode && (
                    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '8px', background: 'rgba(255,0,0,0.1)', padding: '8px', borderRadius: '6px' }}>
                        <div style={{ display: 'flex', gap: '5px' }}>
                            <button 
                                onClick={() => setSurgeryTool('graft')}
                                style={{ 
                                    flex: 1,
                                    background: surgeryTool === 'graft' ? '#4488ff' : '#222', 
                                    color: 'white', border: '1px solid #4488ff', 
                                    padding: '6px', borderRadius: '4px', fontSize: '11px', cursor: 'pointer' 
                                }}
                            >
                                Graft (Connect)
                            </button>
                            <button 
                                onClick={() => setSurgeryTool('ablate')}
                                style={{ 
                                    flex: 1,
                                    background: surgeryTool === 'ablate' ? '#ff4444' : '#222', 
                                    color: 'white', border: '1px solid #ff4444', 
                                    padding: '6px', borderRadius: '4px', fontSize: '11px', cursor: 'pointer' 
                                }}
                            >
                                Ablate (Cut)
                            </button>
                        </div>
                        
                        <button 
                            onClick={() => performSurgery('clear', 0)}
                            style={{ background: '#222', color: '#aaa', border: '1px solid #444', padding: '4px', borderRadius: '4px', fontSize: '10px', cursor: 'pointer' }}
                        >
                            Reset All Hooks
                        </button>

                        <div style={{ fontSize: '11px', color: '#aaa', fontStyle: 'italic', textAlign: 'center' }}>
                            {surgeryTool === 'graft' && (selection.length === 0 ? "Select Source Node..." : "Select Target Node...")}
                            {surgeryTool === 'ablate' && "Click a node to remove it."}
                        </div>
                    </div>
                )}
            </div>
        </SimplePanel>

        {/* 3D Scene */}
        <div style={{ flex: 1 }}>
            <Canvas camera={{ position: [5, 5, 8], fov: 45 }}>
                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} intensity={1} />
                <OrbitControls />

                {/* Grid Helper for Manifold */}
                <gridHelper args={[10, 10, 0x444444, 0x222222]} position={[0, -0.1, 0]} />

                {/* Manifold Constraint Plane (Visualizes Low-Dim Surface) */}
                <mesh rotation={[-Math.PI/2, 0, 0]} position={[0, -0.15, 0]}>
                    <planeGeometry args={[10, 10]} />
                    <meshBasicMaterial color="#00ffff" transparent opacity={0.1} side={THREE.DoubleSide} />
                </mesh>

                {/* Data Points */}
                {data && (
                    <group>
                        {/* Manifold Point Cloud */}
                        {data.manifold_points?.map((pt, i) => (
                            <mesh key={`pt-${i}`} position={pt.pos}>
                                <sphereGeometry args={[pt.type === 'concept' ? 0.08 : 0.03, 8, 8]} />
                                <meshBasicMaterial 
                                    color={pt.type === 'concept' ? '#ffaa00' : '#888888'} 
                                    transparent 
                                    opacity={pt.type === 'concept' ? 1.0 : 0.3} 
                                />
                                {pt.type === 'concept' && (
                                     <Text position={[0, 0.2, 0]} fontSize={0.15} color="#ffaa00" anchorX="center" anchorY="bottom">
                                         {pt.text.length > 20 ? pt.text.substring(0, 20) + '...' : pt.text}
                                     </Text>
                                )}
                            </mesh>
                        ))}

                        {/* Manifold Nodes (The Grid Points) */}
                        {data.manifold_nodes?.map((node, i) => {
                            const isSelected = selection.includes(node.id);
                            const isSource = selection[0] === node.id;
                            
                            return (
                                <mesh 
                                    key={node.id} 
                                    position={node.pos}
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        handleSurgeryClick(node.id);
                                    }}
                                    onPointerOver={(e) => {
                                        e.stopPropagation();
                                        if (surgeryMode) document.body.style.cursor = 'crosshair';
                                    }}
                                    onPointerOut={(e) => {
                                        document.body.style.cursor = 'default';
                                    }}
                                >
                                    <sphereGeometry args={[isSelected ? 0.15 : 0.1, 16, 16]} />
                                    <meshStandardMaterial 
                                        color={isSelected ? (isSource ? "#44ff44" : "#ffaa00") : (animState.constraining ? "#ff0000" : "#00ffff")} 
                                        emissive={isSelected ? (isSource ? "#44ff44" : "#ffaa00") : (animState.constraining ? "#ff0000" : "#0044aa")}
                                        emissiveIntensity={isSelected ? 1.0 : 0.5}
                                    />
                                    <Text position={[0, -0.3, 0]} fontSize={0.2} color="#aaa">{node.id}</Text>
                                </mesh>
                            );
                        })}

                        {/* Fibers (Vertical Vectors) */}
                        {data.fibers?.map((fiber, i) => {
                            const parent = data.manifold_nodes?.find(n => n.id === fiber.parent_id);
                            if (!parent) return null;
                            const isInjectTarget = i === data.fibers.length - 1 && animState.injecting;
                            
                            return (
                                <group key={i} position={parent.pos}>
                                    {/* Fiber Stick */}
                                    <mesh position={[0, fiber.height/2, 0]}>
                                        <cylinderGeometry args={[0.02, 0.02, fiber.height, 8]} />
                                        <meshStandardMaterial 
                                            color={isInjectTarget ? "#44ff44" : "#ffffff"} 
                                            emissive={isInjectTarget ? "#44ff44" : "#ffffff"} 
                                            emissiveIntensity={0.5} 
                                        />
                                    </mesh>
                                    {/* Fiber Head */}
                                    <mesh position={[0, fiber.height, 0]}>
                                        <sphereGeometry args={[0.15, 16, 16]} />
                                        <meshStandardMaterial 
                                            color={`hsl(${fiber.color_intensity * 360}, 80%, 50%)`} 
                                            emissive={`hsl(${fiber.color_intensity * 360}, 80%, 50%)`}
                                            emissiveIntensity={0.8}
                                        />
                                    </mesh>
                                    {/* Animation: Injection Pulse */}
                                    {isInjectTarget && (
                                        <mesh position={[0, fiber.height, 0]}>
                                            <sphereGeometry args={[0.4, 16, 16]} />
                                            <meshBasicMaterial color="#44ff44" transparent opacity={0.3} wireframe />
                                        </mesh>
                                    )}
                                </group>
                            );
                        })}

                        {/* Transport Links (Curves) */}
                        {data.connections?.map((conn, i) => {
                            const src = data.manifold_nodes?.find(n => n.id === conn.source);
                            const tgt = data.manifold_nodes?.find(n => n.id === conn.target);
                            if (!src || !tgt || !animState.transporting) return null;

                            // Curve control point (arc)
                            const mid = [
                                (src.pos[0] + tgt.pos[0]) / 2,
                                (src.pos[1] + tgt.pos[1]) / 2 + 2, // Arc height
                                (src.pos[2] + tgt.pos[2]) / 2
                            ];
                            
                            const curve = new THREE.QuadraticBezierCurve3(
                                new THREE.Vector3(...src.pos),
                                new THREE.Vector3(...mid),
                                new THREE.Vector3(...tgt.pos)
                            );
                            
                            const points = curve.getPoints(20);

                            return (
                                <group key={i}>
                                    <Line points={points} color="#4488ff" lineWidth={3} opacity={0.8} transparent />
                                    {/* Moving particle? */}
                                    <mesh position={mid}>
                                        <sphereGeometry args={[0.1, 8, 8]} />
                                        <meshBasicMaterial color="#ffffff" />
                                    </mesh>
                                </group>
                            );
                        })}
                    </group>
                )}
            </Canvas>
        </div>
        
        {/* Caption/Legend */}
        <div style={{ 
            height: '60px', background: '#111', borderTop: '1px solid #333', 
            display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#888', fontSize: '12px' 
        }}>
            <div>
                <span style={{color: '#00ffff'}}>●</span> Manifold (Logic) &nbsp;&nbsp;
                <span style={{color: '#ffffff'}}>|</span> Fiber (Content) &nbsp;&nbsp;
                <span style={{color: '#4488ff'}}>➜</span> Transport (Association)
            </div>
        </div>
    </div>
  );
};

export default FiberNetV2Demo;
