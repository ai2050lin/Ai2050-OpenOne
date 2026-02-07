
import { Line, OrbitControls, Text } from '@react-three/drei';
import { Canvas } from '@react-three/fiber';
import { useEffect, useState } from 'react';
import * as THREE from 'three';

const FiberNetV2Demo = ({ t }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
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
      fetch(`http://localhost:8888${endpoint}`)
          .then(res => res.json())
          .then(d => {
              if (d.error || d.detail) {
                  console.warn("Backend Error:", d.error || d.detail);
                  // alert("Error: " + (d.error || d.detail)); // Optional: alert user
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

  // Controls
  const handleInject = () => {
      setAnimState(prev => ({ ...prev, injecting: true }));
      setTimeout(() => setAnimState(prev => ({ ...prev, injecting: false })), 2000);
      if (dataMode === 'mock') fetchData(); 
  };

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
        {/* Control Toolbar */}
        <div style={{ 
            position: 'absolute', top: 10, left: 10, zIndex: 10, 
            display: 'flex', gap: '10px', background: 'rgba(0,0,0,0.5)', padding: '10px', borderRadius: '8px' 
        }}>
            <button
                onClick={() => setDataMode(prev => prev === 'mock' ? 'real' : 'mock')}
                style={{ background: '#666', color: 'white', border: '1px solid #999', padding: '8px 12px', cursor: 'pointer', borderRadius: '4px', fontWeight: 'bold' }}
            >
                {dataMode === 'mock' ? '🔄 Switch to Real Data (NFB-RA)' : '🔄 Switch to Mock Demo'}
            </button>
            <div style={{width: 1, background: '#555', margin: '0 5px'}}></div>
            <button 
                onClick={handleInject}
                style={{ background: animState.injecting ? '#44ff44' : '#333', color: 'white', border: '1px solid #666', padding: '8px 12px', cursor: 'pointer', borderRadius: '4px' }}
            >
                1. {t ? t('fibernet.inject', 'Inject Knowledge') : 'Inject Knowledge'} (Efficiency)
            </button>
            <button 
                onClick={handleTransport}
                style={{ background: animState.transporting ? '#4488ff' : '#333', color: 'white', border: '1px solid #666', padding: '8px 12px', cursor: 'pointer', borderRadius: '4px' }}
            >
                2. {t ? t('fibernet.transport', 'Affine Transport') : 'Affine Transport'} (Connectivity)
            </button>
            <button 
                onClick={handleConstraint}
                style={{ background: animState.constraining ? '#ff4444' : '#333', color: 'white', border: '1px solid #666', padding: '8px 12px', cursor: 'pointer', borderRadius: '4px' }}
            >
                3. {t ? t('fibernet.constraint', 'Manifold Constraint') : 'Manifold Constraint'} (Low-Dim)
            </button>
        </div>

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
                        {data.manifold_nodes?.map((node, i) => (
                            <mesh key={node.id} position={node.pos}>
                                <sphereGeometry args={[0.1, 16, 16]} />
                                <meshStandardMaterial color={animState.constraining ? "#ff0000" : "#00ffff"} emissive={animState.constraining ? "#ff0000" : "#0044aa"} />
                                <Text position={[0, -0.3, 0]} fontSize={0.2} color="#aaa">{node.id}</Text>
                            </mesh>
                        ))}

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
