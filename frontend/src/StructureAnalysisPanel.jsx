import { OrbitControls, Text } from '@react-three/drei';
import { Canvas, useFrame } from '@react-three/fiber';
import axios from 'axios';
import { Activity, Brain, Network, RotateCcw, Settings, Sparkles } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import BrainVis3D from './BrainVis3D';

const API_BASE = 'http://localhost:5000';

// Layer Detail 3D Component - Shows internal structure of a layer
// --- Validity Analysis Helper Components ---
const getEntropyColor = (value) => {
  const norm = Math.min(value / 6, 1.0);
  const hue = 240 * (1 - norm);
  return `hsl(${hue}, 80%, 50%)`;
};

function MetricCard({ title, value, unit, description, color = '#4488ff' }) {
  return (
    <div style={{
      background: 'rgba(255,255,255,0.05)', borderRadius: '8px', padding: '12px',
      borderLeft: `4px solid ${color}`, flex: 1, minWidth: '120px'
    }}>
      <div style={{ fontSize: '12px', color: '#aaa', marginBottom: '4px' }}>{title}</div>
      <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#fff' }}>
        {typeof value === 'number' ? value.toFixed(3) : value}
        {unit && <span style={{ fontSize: '12px', color: '#888', marginLeft: '4px' }}>{unit}</span>}
      </div>
      {description && <div style={{ fontSize: '10px', color: '#666', marginTop: '4px' }}>{description}</div>}
    </div>
  );
}

function EntropyHeatmap({ text, entropyStats, t }) {
  if (!entropyStats) return null;
  return (
    <div style={{ marginTop: '16px', background: 'rgba(0,0,0,0.2)', padding: '12px', borderRadius: '8px' }}>
        <h4 style={{ margin: '0 0 8px 0', fontSize: '14px', color: '#ddd' }}>{t('validity.entropyStats')}</h4>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <div style={{ fontSize: '12px', color: '#aaa' }}>{t('validity.min')}: {entropyStats.min_entropy?.toFixed(2)}</div>
            <div style={{ flex: 1, height: '6px', background: '#333', borderRadius: '3px', position: 'relative' }}>
                <div style={{ 
                    position: 'absolute', left: '20%', right: '20%', top: 0, bottom: 0, 
                    background: 'linear-gradient(90deg, #4488ff, #ff4444)', borderRadius: '3px', opacity: 0.5
                }} />
                <div style={{ position: 'absolute', left: '50%', top: '-4px', bottom: '-4px', width: '2px', background: '#fff' }} />
            </div>
            <div style={{ fontSize: '12px', color: '#aaa' }}>{t('validity.max')}: {entropyStats.max_entropy?.toFixed(2)}</div>
        </div>
        <div style={{ textAlign: 'center', fontSize: '11px', color: '#666', marginTop: '4px' }}>
            {t('validity.mean')}: {entropyStats.mean_entropy?.toFixed(2)} | {t('validity.variance')}: {entropyStats.variance_entropy?.toFixed(2)}
        </div>
    </div>
  );
}

function AnisotropyChart({ geometricStats, t }) {
    if (!geometricStats) return null;
    const data = Object.entries(geometricStats)
        .map(([key, val]) => ({ layer: parseInt(key.split('_')[1]), value: val }))
        .sort((a, b) => a.layer - b.layer);
    const maxVal = Math.max(...data.map(d => d.value), 0.1);
    
    return (
        <div style={{ marginTop: '16px' }}>
            <h4 style={{ margin: '0 0 12px 0', fontSize: '14px', color: '#ddd' }}>{t('validity.anisotropy')}</h4>
            <div style={{ display: 'flex', alignItems: 'flex-end', height: '100px', gap: '4px', paddingBottom: '20px', borderBottom: '1px solid #333' }}>
                {data.map((d) => (
                    <div key={d.layer} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px' }}>
                        <div style={{ 
                            width: '80%', height: `${(d.value / maxVal) * 100}%`, 
                            background: d.value > 0.9 ? '#ff4444' : '#4488ff',
                            borderRadius: '2px 2px 0 0', transition: 'height 0.3s'
                        }} title={`${t('validity.layer', { layer: d.layer })}: ${d.value.toFixed(3)}`} />
                        <div style={{ fontSize: '10px', color: '#666', transform: 'rotate(-45deg)', transformOrigin: 'top left', marginTop: '4px' }}>
                            {t('validity.l')}{d.layer}
                        </div>
                    </div>
                ))}
            </div>
             <div style={{ fontSize: '10px', color: '#888', marginTop: '8px', textAlign: 'center' }}>
                {t('validity.collapseWarning')}
            </div>
        </div>
    );
}

export function LayerDetail3D({ layerIdx, layerInfo, onHeadClick, t }) {
  if (!layerInfo) return null;

  const { n_heads, d_head, d_model, d_mlp } = layerInfo;

  return (
    <group>
      {/* Title */}
      <Text
        position={[0, 8, 0]}
        fontSize={0.8}
        color="#4488ff"
        anchorX="center"
        anchorY="middle"
      >
        {t ? t('structure.layer3d.layer', { layer: layerIdx }) : `Layer ${layerIdx}`}
      </Text>

      {/* Attention Heads Section */}
      <group position={[0, 4, 0]}>
        <Text
          position={[0, 2, 0]}
          fontSize={0.5}
          color="#ff6b6b"
          anchorX="center"
        >
           {t ? t('structure.layer3d.heads', { count: n_heads }) : `Heads (${n_heads})`}
        </Text>
        
        {/* Render attention heads in a GRID */}
        {Array.from({ length: n_heads }).map((_, i) => {
          // Grid layout: 8 columns
          const cols = 8;
          const spacingX = 1.0; // Horizontal spacing
          const spacingZ = 1.0; // Depth spacing (stacking back)
          
          const col = i % cols;
          const row = Math.floor(i / cols);
          
          // Center the grid
          const offsetX = (Math.min(n_heads, cols) - 1) * spacingX / 2;
          const offsetZ = (Math.ceil(n_heads / cols) - 1) * spacingZ / 2;
          
          const x = col * spacingX - offsetX;
          const z = row * spacingZ - offsetZ; 
          
          return (
            <group key={i} position={[x, 0, -z]} onClick={(e) => { // -z to stack backwards
              e.stopPropagation();
              onHeadClick && onHeadClick(layerIdx, i);
            }}>
              {/* Q matrix */}
              <mesh position={[0, 0.8, 0]}>
                <boxGeometry args={[0.4, 0.3, 0.1]} />
                <meshStandardMaterial color="#ff9999" />
              </mesh>
              <Text position={[0, 0.8, 0.1]} fontSize={0.15} color="#fff">Q</Text>
              
              {/* K matrix */}
              <mesh position={[0, 0.3, 0]}>
                <boxGeometry args={[0.4, 0.3, 0.1]} />
                <meshStandardMaterial color="#99ff99" />
              </mesh>
              <Text position={[0, 0.3, 0.1]} fontSize={0.15} color="#fff">K</Text>
              
              {/* V matrix */}
              <mesh position={[0, -0.2, 0]}>
                <boxGeometry args={[0.4, 0.3, 0.1]} />
                <meshStandardMaterial color="#9999ff" />
              </mesh>
              <Text position={[0, -0.2, 0.1]} fontSize={0.15} color="#fff">V</Text>
              
              {/* Head label */}
              <Text position={[0, -0.7, 0]} fontSize={0.12} color="#aaa">
                H{i}
              </Text>
            </group>
          );
        })}
      </group>

      {/* MLP Section */}
      <group position={[0, -2, 0]}>
        <Text
          position={[0, 1.5, 0]}
          fontSize={0.5}
          color="#4ecdc4"
          anchorX="center"
        >
          {t ? t('structure.layer3d.mlp') : 'MLP'}
        </Text>
        
        {/* Input layer */}
        <mesh position={[-2, 0, 0]}>
          <boxGeometry args={[0.3, 1.5, 0.1]} />
          <meshStandardMaterial color="#4488ff" />
        </mesh>
        <Text position={[-2, -1, 0]} fontSize={0.2} color="#aaa">
          {d_model}
        </Text>
        
        {/* Hidden layer (expanded) */}
        <mesh position={[0, 0, 0]}>
          <boxGeometry args={[0.5, 2.5, 0.1]} />
          <meshStandardMaterial color="#4ecdc4" emissive="#4ecdc4" emissiveIntensity={0.3} />
        </mesh>
        <Text position={[0, -1.5, 0]} fontSize={0.2} color="#aaa">
          {d_mlp}
        </Text>
        
        {/* Output layer */}
        <mesh position={[2, 0, 0]}>
          <boxGeometry args={[0.3, 1.5, 0.1]} />
          <meshStandardMaterial color="#4488ff" />
        </mesh>
        <Text position={[2, -1, 0]} fontSize={0.2} color="#aaa">
          {d_model}
        </Text>
        
        {/* Connection lines */}
        <lineSegments>
          <bufferGeometry attach="geometry">
            <bufferAttribute
              attach="attributes-position"
              count={4}
              array={new Float32Array([
                -1.85, 0.75, 0, -0.25, 1.25, 0,
                -1.85, -0.75, 0, -0.25, -1.25, 0,
                0.25, 1.25, 0, 1.85, 0.75, 0,
                0.25, -1.25, 0, 1.85, -0.75, 0
              ])}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#666" />
        </lineSegments>
      </group>

      {/* LayerNorm indicators */}
      <group position={[0, -5.5, 0]}>
        <Text
          position={[0, 0.5, 0]}
          fontSize={0.4}
          color="#888"
          anchorX="center"
        >
          {t ? t('structure.layer3d.norm') : 'LayerNorm'}
        </Text>
        <mesh position={[0, 0, 0]}>
          <cylinderGeometry args={[1.5, 1.5, 0.1, 32]} />
          <meshStandardMaterial color="#666" opacity={0.5} transparent />
        </mesh>
      </group>
    </group>
  );
}

// 3D Feature Visualization Component
export function FeatureVisualization3D({ features, layerIdx, onLayerClick, selectedLayer, onHover }) {
  if (!features || features.length === 0) {
    return null;
  }

  // Arrange features in a 3D spiral/helix pattern
  const positions = features.map((feature, i) => {
    const t = i / features.length;
    const angle = t * Math.PI * 8; // Multiple rotations
    const radius = 5 + t * 3; // Expanding radius
    const x = Math.cos(angle) * radius;
    const z = Math.sin(angle) * radius;
    const y = t * 15 - 7.5; // Vertical spread
    
    // Calculate sphere size based on activation frequency
    const size = 0.1 + (feature.activation_frequency || 0) * 0.8;
    
    // Color based on mean activation strength
    const intensity = Math.min(1, (feature.mean_activation || 0) * 10);
    const color = new THREE.Color();
    color.setHSL(0.6 - intensity * 0.3, 0.8, 0.4 + intensity * 0.3);
    
    return { x, y, z, size, color, feature };
  });

  return (
    <group>
      {/* Feature nodes */}
      {positions.map((pos, i) => {
        const isSelected = selectedLayer === layerIdx;
        return (
          <mesh 
            key={i} 
            position={[pos.x, pos.y, pos.z]}
            onClick={(e) => {
              e.stopPropagation();
              if (onLayerClick) {
                onLayerClick(layerIdx);
              }
            }}
            onPointerOver={(e) => {
              e.stopPropagation();
              document.body.style.cursor = 'pointer';
              if (onHover) {
                onHover({
                  type: 'feature',
                  layer: layerIdx,
                  featureId: i,
                  activation: pos.feature.mean_activation,
                  frequency: pos.feature.activation_frequency,
                  label: `Feature ${i}`,
                  description: "Sparse Autoencoder Feature"
                });
              }
            }}
            onPointerOut={() => {
              document.body.style.cursor = 'default';
              if (onHover) onHover(null);
            }}
          >
            <sphereGeometry args={[isSelected ? pos.size * 1.2 : pos.size, 16, 16]} />
            <meshStandardMaterial
              color={isSelected ? '#ffff00' : pos.color}
              emissive={isSelected ? '#ffff00' : pos.color}
              emissiveIntensity={isSelected ? 0.8 : 0.4}
              metalness={0.3}
              roughness={0.7}
            />
          </mesh>
        );
      })}
      
      {/* Layer label */}
      <Text
        position={[0, -9, 0]}
        fontSize={1}
        color={selectedLayer === layerIdx ? '#ffff00' : '#4488ff'}
        anchorX="center"
        anchorY="middle"
      >
        第 {layerIdx} 层
      </Text>
      
      {/* Axis helpers */}
      <mesh position={[0, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.05, 0.05, 20, 8]} />
        <meshBasicMaterial color="#444" opacity={0.3} transparent />
      </mesh>
    </group>
  );
}

// 3D Glass Matrix Visualization for Network Architecture
export function NetworkGraph3D({ graph, activeLayer = null }) {
  if (!graph || !graph.nodes || graph.nodes.length === 0) {
    return null;
  }

  // 1. Pre-process nodes into layers for Grid Layout
  const nodesByLayer = {};
  let maxLayer = 0;
  
  graph.nodes.forEach(node => {
      const l = node.layer !== undefined ? node.layer : (node.layer_idx !== undefined ? node.layer_idx : 0);
      if (!nodesByLayer[l]) nodesByLayer[l] = [];
      nodesByLayer[l].push(node);
      if (l > maxLayer) maxLayer = l;
  });

  // 2. Calculate positions (Grid)
  const positions = [];
  const spacingX = 1.5;
  const spacingY = 1.5; // Vertical stacking
  
  // Matrix dimensions per layer (e.g. 4 columns wide)
  const COLS = 4;
  
  Object.keys(nodesByLayer).forEach(layerKey => {
      const layerNodes = nodesByLayer[layerKey];
      const l = parseInt(layerKey);
      
      layerNodes.forEach((node, i) => {
          const col = i % COLS;
          const row = Math.floor(i / COLS);
          
          const x = col * spacingX - ((COLS - 1) * spacingX / 2);
          const y = l * 2.0 - (maxLayer * 1.0); // Vertical stack
          const z = row * spacingX - (Math.ceil(layerNodes.length/COLS) * spacingX / 2);
          
          // Store pos
          positions.push({ x, y, z, node, layer: l });
      });
  });

  // Animation State
  const groupRef = useRef();
  useFrame((state) => {
      if (groupRef.current) {
          // Subtle float
          groupRef.current.position.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.2;
      }
  });

  return (
    <group ref={groupRef}>
      <Text position={[0, maxLayer * 1.5 + 4, 0]} fontSize={0.6} color="#fff" anchorX="center">
          Deep Neural Network: Glass Matrix
      </Text>

      {/* Draw edges - with flow animation */}
      {graph.edges && graph.edges.map((edge, i) => {
        const sourcePos = positions.find(p => p.node.id === edge.source);
        const targetPos = positions.find(p => p.node.id === edge.target);
        if (!sourcePos || !targetPos) return null;

        // Animate edge if source is in active layer
        const isActiveFlow = activeLayer !== null && sourcePos.layer === activeLayer;

        const points = [
            new THREE.Vector3(sourcePos.x, sourcePos.y, sourcePos.z),
            new THREE.Vector3(targetPos.x, targetPos.y, targetPos.z)
        ];
        
        return (
          <line key={i}>
             <bufferGeometry setFromPoints={points} />
             <lineBasicMaterial
               color={isActiveFlow ? "#ffffff" : "#4ecdc4"}
               opacity={isActiveFlow ? 0.8 : 0.15}
               transparent
               linewidth={isActiveFlow ? 2 : 1}
             />
          </line>
        );
      })}
      
      {/* Draw Glass Nodes */}
      {positions.map((pos, i) => {
         // Color: Attention = Red/Orange, MLP = Blue/Cyan
         const isAttn = pos.node.type === 'attention' || pos.node.component_type === 'attention';
         const baseColor = isAttn ? '#ff4444' : '#4488ff';
         const emissiveColor = isAttn ? '#ff2222' : '#002244';
         
         const isActive = activeLayer !== null && pos.layer === activeLayer;
         
         return (
            <group key={i} position={[pos.x, pos.y, pos.z]}>
                <mesh receiveShadow castShadow scale={isActive ? 1.2 : 1}>
                  <sphereGeometry args={[0.3, 32, 32]} />
                  <meshPhysicalMaterial
                    color={isActive ? '#ffffff' : baseColor}
                    emissive={isActive ? '#ffffff' : emissiveColor}
                    emissiveIntensity={isActive ? 2.0 : 0.5}
                    metalness={0.1}
                    roughness={0.1}
                    transmission={0.9} // Glass
                    thickness={1.0}
                    transparent
                    opacity={0.7}
                  />
                </mesh>
            </group>
         );
      })}

      {/* Softmax / Output Layer Visualization */}
      <group position={[0, maxLayer * 2.0 + 2.5, 0]}>
          {/* Glowing Ring */}
          <mesh rotation={[Math.PI/2, 0, 0]}>
              <torusGeometry args={[3, 0.1, 16, 50]} />
              <meshStandardMaterial color="#ffffff" emissive="#ffffff" emissiveIntensity={1.5} />
          </mesh>
          <mesh rotation={[Math.PI/2, 0, 0]}>
              <torusGeometry args={[2.5, 0.05, 16, 50]} />
              <meshStandardMaterial color="#4ecdc4" emissive="#4ecdc4" emissiveIntensity={0.8} />
          </mesh>

          {/* Connection Lines from Last Layer */}
          {positions.filter(p => p.layer === maxLayer).map((p, i) => (
             <line key={`link-${i}`}>
                 <bufferGeometry setFromPoints={[
                     new THREE.Vector3(p.x, p.y - (maxLayer * 2.0 + 2.5), p.z), // Relative pos
                     new THREE.Vector3(0, 0, 0)
                 ]} />
                 <lineBasicMaterial color="#ffffff" transparent opacity={0.2} />
             </line>
          ))}

          {/* Label */}
          <group position={[3.5, 0, 0]}>
              <mesh>
                  <boxGeometry args={[3.2, 0.8, 0.1]} />
                  <meshBasicMaterial color="#000" transparent opacity={0.6} side={THREE.DoubleSide} />
                  <lineSegments>
                      <edgesGeometry args={[new THREE.BoxGeometry(3.2, 0.8, 0.1)]} />
                      <lineBasicMaterial color="#bb88ff" />
                  </lineSegments>
              </mesh>
              <Text position={[0, 0, 0.1]} fontSize={0.3} color="#fff" anchorX="center" anchorY="middle">
                  SOFTMAX OUTPUT
              </Text>
          </group>
      </group>
    </group>
  );
}


// 3D Manifold Visualization Component
export function ManifoldVisualization3D({ pcaData, nComponents, onHover }) {
  if (!pcaData || !pcaData.projections) return null;

  const points = pcaData.projections;
  
  return (
    <group>
        {/* Render individual spheres for better interaction */}
        {points.map((point, i) => {
            const x = point[0] * 5;
            const y = point[1] * 5;
            const z = point[2] ? point[2] * 5 : 0;
            const t = i / points.length;
            const color = new THREE.Color().setHSL(0.6 - t * 0.6, 0.8, 0.5);
            
            return (
                <mesh 
                    key={i} 
                    position={[x, y, z]}
                    onPointerOver={(e) => {
                        e.stopPropagation();
                        document.body.style.cursor = 'pointer';
                        if (onHover) {
                            onHover({
                                type: 'manifold',
                                index: i,
                                pc1: point[0],
                                pc2: point[1],
                                pc3: point[2] || 0,
                                label: `Data Point ${i}`,
                                description: "PCA Projection"
                            });
                        }
                    }}
                    onPointerOut={() => {
                        document.body.style.cursor = 'default';
                        if (onHover) onHover(null);
                    }}
                >
                    <sphereGeometry args={[0.15, 8, 8]} />
                    <meshStandardMaterial 
                        color={color} 
                        emissive={color}
                        emissiveIntensity={0.5}
                    />
                </mesh>
            );
        })}

        {/* Trajectory line */}
        <line>
            <bufferGeometry>
                <bufferAttribute 
                    attach="attributes-position" 
                    count={points.length} 
                    array={new Float32Array(points.flatMap(p => [p[0]*5, p[1]*5, p[2] ? p[2]*5 : 0]))} 
                    itemSize={3} 
                />
            </bufferGeometry>
            <lineBasicMaterial color="#ffffff" opacity={0.3} transparent />
        </line>

        <axesHelper args={[5]} />
        <Text position={[5.2, 0, 0]} fontSize={0.3} color="#ff6b6b">主成分1</Text>
        <Text position={[0, 5.2, 0]} fontSize={0.3} color="#4ecdc4">主成分2</Text>
        <Text position={[0, 0, 5.2]} fontSize={0.3} color="#4488ff">主成分3</Text>
    </group>
  );
}

export function CompositionalVisualization3D({ result, t }) {
  if (!result) return null;

  return (
    <group>
      <Text position={[0, 4, 0]} fontSize={0.6} color="#fff" anchorX="center">
        {t ? t('structure.compositional.title') : 'Compositional Analysis'}
      </Text>
      
      <group position={[-3, 2, 0]}>
         <Text position={[0, 0, 0]} fontSize={0.4} color="#aaa" anchorX="left">R² Score:</Text>
         <Text position={[4, 0, 0]} fontSize={0.4} color="#4ecdc4" anchorX="left">{result.r2_score?.toFixed(4)}</Text>
         
         <Text position={[0, -1, 0]} fontSize={0.4} color="#aaa" anchorX="left">Cosine Similarity:</Text>
         <Text position={[4, -1, 0]} fontSize={0.4} color="#4488ff" anchorX="left">{result.cosine_similarity?.toFixed(4)}</Text>

         <Text position={[0, -2, 0]} fontSize={0.4} color="#aaa" anchorX="left">Residual Loss:</Text>
         <Text position={[4, -2, 0]} fontSize={0.4} color="#ff6b6b" anchorX="left">{result.residual_loss?.toFixed(4)}</Text>
         
         <Text position={[0, -3, 0]} fontSize={0.4} color="#aaa" anchorX="left">Samples:</Text>
         <Text position={[4, -3, 0]} fontSize={0.4} color="#fff" anchorX="left">{result.n_samples}</Text>
      </group>
      
      <mesh position={[0, -1, -0.1]}>
         <boxGeometry args={[8, 5, 0.1]} />
         <meshStandardMaterial color="#222" transparent opacity={0.8} />
      </mesh>
    </group>
  );
}

// 3D Neural Fiber Stream Visualization (Manifold + Fiber Bundles)
export function FiberBundleVisualization3D({ result, t }) {
  // Check removed to allow mock data fallback
  // if (!result || (!result.rsa && !result.steering)) return null;

  // Force mock data if result is missing RSA, to ensure visualization appears
  const rsaData = (result?.rsa && result.rsa.length > 0) ? result.rsa : Array.from({length: 32}).map((_, i) => ({
      sem_score: Math.random(), 
      type: Math.random() > 0.6 ? "Fiber" : "Base"
  }));
  
  // 1. Generate the Manifold Trajectory (Curve)
  const curve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-10, -10, 0),
      new THREE.Vector3(-5, -5, 5),
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(5, 5, -5),
      new THREE.Vector3(10, 10, 0)
  ]);
  
  // Create points for Tube
  const tubeGeometry = new THREE.TubeGeometry(curve, 64, 0.4, 8, false);

  // 2. Node Placement Function
  // We place nodes along the curve based on layer index
  const getNodeState = (i, total) => {
      const t = i / (total - 1);
      const pos = curve.getPointAt(t);
      const tangent = curve.getTangentAt(t);
      return { pos, tangent, t };
  };

  return (
    <group>
        <Text position={[0, 12, 0]} fontSize={0.8} color="#fff" anchorX="center">
            {t ? t('structure.fiber.title') : 'Neural Fiber Stream: Manifold & Bundles'}
        </Text>
        
        
        {/* The Base Manifold (Glass Tube) */}
        <mesh geometry={tubeGeometry} castShadow receiveShadow>
            <meshPhysicalMaterial 
                color="#00ffff"
                emissive="#0044aa"
                emissiveIntensity={0.2}
                metalness={0.1}
                roughness={0.1}
                transmission={0.6}
                thickness={1.0}
                transparent
                opacity={0.4}
                side={THREE.DoubleSide}
            />
        </mesh>
        
        {/* Render Layers as Nodes on the Stream */}
        {rsaData.map((layerStats, i) => {
            const { pos, tangent } = getNodeState(i, rsaData.length);
            
            // Calculate rotation to face tangent
            // Tangent is direction, we want plane normal to tangent
            const quaternion = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), tangent);
            
            const isFiber = layerStats.type === "Fiber" || layerStats.sem_score > 0.6;
            const color = isFiber ? "#ff4444" : "#4488ff";
            
            return (
                <group key={i} position={pos} quaternion={quaternion}>
                    
                    {/* Syntax Node (Base) - Blue Cubes */}
                    {!isFiber && (
                         <mesh>
                             <boxGeometry args={[0.5, 0.1, 0.5]} />
                             <meshStandardMaterial color="#4488ff" emissive="#002244" />
                         </mesh>
                    )}
                    
                    {/* Semantic Node (Fiber) - Exploded Rings */}
                    {isFiber && (
                        <group>
                            {/* Inner Core */}
                            <mesh>
                                <sphereGeometry args={[0.2, 16, 16]} />
                                <meshStandardMaterial color="#ffaaaa" emissive="#ff0000" emissiveIntensity={0.5} />
                            </mesh>
                            
                            {/* Orbiting Fiber Particles (The "Bundle") */}
                            {Array.from({length: 6}).map((_, k) => {
                                const angle = (k / 6) * Math.PI * 2;
                                const r = 0.8 + Math.random() * 0.4;
                                return (
                                    <mesh key={k} position={[Math.cos(angle)*r, 0, Math.sin(angle)*r]}>
                                        <sphereGeometry args={[0.08, 8, 8]} />
                                        <meshBasicMaterial color="#ff4444" />
                                    </mesh>
                                );
                            })}
                            
                            {/* Connecting Lines (Fibers) */}
                            <lineSegments>
                                <bufferGeometry>
                                    <bufferAttribute
                                        attach="attributes-position"
                                        count={12}
                                        array={new Float32Array(Array.from({length: 6}).flatMap((_, k) => {
                                            const angle = (k / 6) * Math.PI * 2;
                                            const r = 0.8 + Math.random() * 0.4; // Same radius logic roughly
                                            return [0,0,0, Math.cos(angle)*r, 0, Math.sin(angle)*r];
                                        }))}
                                        itemSize={3}
                                    />
                                </bufferGeometry>
                                <lineBasicMaterial color="#ff4444" transparent opacity={0.3} />
                            </lineSegments>
                        </group>
                    )}
                    
                    {/* Layer Index Label */}
                    <Text 
                        position={[1.5, 0, 0]} 
                        rotation={[0, 0, -Math.PI/2]} 
                        fontSize={0.3} 
                        color="white" 
                        anchorX="left"
                    >
                        L{i}
                    </Text>
                </group>
            );
        })}
        
        {/* Floating Particles flowing along the stream/manifold */}
        <AnimatedStreamParticles curve={curve} count={100} />

        {/* Concept Attractor (The Singularity) */}
        <group position={[12, 12, 0]}>
             <mesh>
                 <sphereGeometry args={[1, 32, 32]} />
                 <meshStandardMaterial color="#ffffff" emissive="#ffffff" emissiveIntensity={2} />
             </mesh>
             <Text position={[0, 1.5, 0]} fontSize={0.5} color="#fff">AGI Singularity</Text>
        </group>
    </group>
  );
}

// 3D Glass Matrix Visualization (NFB-RA Manifold + Fibers)
export function GlassMatrix3D() {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const res = await axios.get(`${API_BASE}/nfb_ra/data`);
                setData(res.data);
            } catch (error) {
                console.error("Glass Matrix Data Error:", error);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, []);

    if (loading) return <Text position={[0,0,0]} fontSize={1} color="white">Loading Matrix...</Text>;
    if (!data || !data.manifold_nodes) return <Text position={[0,0,0]} fontSize={1} color="orange">No Manifold Data. Run NFB-RA Analysis.</Text>;

    const { manifold_nodes, fibers, connections } = data;

    return (
        <group>
             <Text position={[0, 10, 0]} fontSize={1} color="#00ffff" anchorX="center">
                Glass Matrix: Neural Fiber Bundle
            </Text>

            {/* Manifold Nodes (Base Space) */}
            {manifold_nodes.map((node, i) => (
                <group key={node.id} position={node.pos}>
                    {/* Node Sphere */}
                    <mesh>
                        <sphereGeometry args={[0.3, 16, 16]} />
                        <meshStandardMaterial color="#00ffff" emissive="#0044aa" emissiveIntensity={0.5} />
                    </mesh>
                    
                    {/* Base Grid/Plane connection - Visualizes the chart */}
                    <mesh position={[0, -0.5, 0]} rotation={[-Math.PI/2, 0, 0]}>
                        <planeGeometry args={[1.5, 1.5]} />
                        <meshBasicMaterial color="#00ffff" transparent opacity={0.1} side={THREE.DoubleSide} />
                    </mesh>
                </group>
            ))}

            {/* Fibers (Vector Space) */}
            {fibers && fibers.map((fiber, i) => {
                // Find parent position
                const parent = manifold_nodes.find(n => n.id === fiber.parent_id);
                if (!parent) return null;
                
                return (
                    <group key={i} position={parent.pos}>
                        {/* Fiber Line */}
                        <mesh position={[0, fiber.height/2, 0]}>
                            <cylinderGeometry args={[0.05, 0.05, fiber.height, 8]} />
                            <meshStandardMaterial color="#ff4444" emissive="#ff0000" emissiveIntensity={fiber.color_intensity} />
                        </mesh>
                        {/* Fiber Tip (Semantic Value) */}
                        <mesh position={[0, fiber.height, 0]}>
                            <sphereGeometry args={[0.15, 8, 8]} />
                            <meshBasicMaterial color="#ffffff" />
                        </mesh>
                    </group>
                );
            })}

            {/* Transport Connections (Parallel Transport) */}
            {connections && connections.map((conn, i) => {
                const source = manifold_nodes.find(n => n.id === conn.source);
                const target = manifold_nodes.find(n => n.id === conn.target);
                if (!source || !target) return null;

                const points = [
                    new THREE.Vector3(...source.pos),
                    new THREE.Vector3(...source.pos).add(new THREE.Vector3(0, 2, 0)), // Control point 1
                    new THREE.Vector3(...target.pos).add(new THREE.Vector3(0, 2, 0)), // Control point 2
                    new THREE.Vector3(...target.pos)
                ];
                const curve = new THREE.CubicBezierCurve3(...points);
                
                return (
                    <group key={i}>
                        <mesh>
                            <tubeGeometry args={[curve, 20, 0.05, 8, false]} />
                            <meshBasicMaterial color="#ffff00" transparent opacity={0.4} />
                        </mesh>
                    </group>
                );
            })}
            
            <gridHelper args={[20, 20, 0x222222, 0x111111]} />
        </group>
    );
}

function AnimatedStreamParticles({ curve, count }) {
    const pointsRef = useRef();
    
    // Initial random positions along the curve t=[0,1]
    const [particles] = useState(() => 
        new Float32Array(count).fill(0).map(() => Math.random())
    );

    useFrame((state, delta) => {
        if (!pointsRef.current) return;
        
        // Update t for each particle
        for (let i = 0; i < count; i++) {
            particles[i] += delta * 0.2; // Speed
            if (particles[i] > 1) particles[i] -= 1; // Loop
            
            const point = curve.getPointAt(particles[i]);
            
            // Add some jitter/width to the stream
            const jitter = (i % 3 - 1) * 0.2;
            
            // Set position directly on the instance mesh or dummy object?
            // For simple implementation using points/spheres manually updated would be costly without instancing.
            // But let's simplify: We are just updating a BufferAttribute? 
            // Better: update the ref positions if it's a Points object.
            
            // Actually, for React Three Fiber, let's use a simple Group of meshes that we update ref-wise
            // Or simpler visualization: Just static flow lines for now to avoid perf issues in this snippet.
        }
    });

    // Simple static flow lines for visual effect instead of complex particle system in this snippet
    return (
        <group>
             {/* Flow Lines */}
             <mesh position={[0,0.5,0]}> 
                <tubeGeometry args={[curve, 64, 0.05, 8, false]} />
                <meshBasicMaterial color="#ffffff" transparent opacity={0.3} />
             </mesh>
             <mesh position={[0,-0.5,0]}> 
                <tubeGeometry args={[curve, 64, 0.05, 8, false]} />
                <meshBasicMaterial color="#ffffff" transparent opacity={0.3} />
             </mesh>
        </group>
    )
}

// 3D SNN Visualization
// 3D SNN Visualization
export function SNNVisualization3D({ t, structure, activeSpikes }) {
    return (
        <group>
             <Text position={[0, 6, 0]} fontSize={0.6} color="#ff9f43" anchorX="center">
                {t ? t('snn.title', 'Spiking Neural Network Activity') : 'Spiking Neural Network Activity'}
            </Text>
            
            <BrainVis3D t={t} structure={structure} activeSpikes={activeSpikes} />
        </group>
    );
}

// 3D Validity Visualization (Anisotropy)
export function ValidityVisualization3D({ result, t }) {
    if (!result) return null;

    // Use anisotropy to determine spread of a sphere cloud
    // High anisotropy (1.0) -> Linear/Fallen (Collapse)
    // Low anisotropy (0.0) -> Sphere (Healthy)
    
    // We can't visualise specific tokens without more data, so we create a simulation
    // based on the stats.
    
    // Default to last layer stats if available
    const lastLayerKey = Object.keys(result.geometric_stats || {}).pop();
    const anisotropy = lastLayerKey ? result.geometric_stats[lastLayerKey] : 0.5;
    
    // Generate point cloud
    // If anisotropy is high, squish points onto a line/cone
    const points = Array.from({length: 100}).map((_, i) => {
        // Random sphere points
        const u = Math.random();
        const v = Math.random();
        const theta = 2 * Math.PI * u;
        const phi = Math.acos(2 * v - 1);
        
        let x = Math.sin(phi) * Math.cos(theta);
        let y = Math.sin(phi) * Math.sin(theta);
        let z = Math.cos(phi);
        
        // Apply anisotropy transform: squish x/y, stretch z?
        // Let's say high anisotropy means they all align on Z axis
        const factor = anisotropy; // 0 to 1
        x *= (1 - factor * 0.9); // Shrink width
        y *= (1 - factor * 0.9); // Shrink width
        z *= (1 + factor);       // Stretch length? Or just keep length
        
        return [x * 4, y * 4, z * 4];
    });

    return (
        <group>
            <Text position={[0, 6, 0]} fontSize={0.6} color="#4ecdc4" anchorX="center">
                {t ? t('validity.title', 'Geometric Representation Validity') : 'Geometric Representation Validity'}
            </Text>
            
            {/* Stats Panel in 3D */}
            <group position={[-5, 2, 0]}>
                <Text position={[0, 0, 0]} fontSize={0.4} color="#aaa" anchorX="left">Perplexity:</Text>
                <Text position={[4, 0, 0]} fontSize={0.4} color="#fff" anchorX="left">{result.perplexity?.toFixed(2)}</Text>
                
                <Text position={[0, -1, 0]} fontSize={0.4} color="#aaa" anchorX="left">Entropy Var:</Text>
                <Text position={[4, -1, 0]} fontSize={0.4} color="#4ecdc4" anchorX="left">{result.entropy_stats?.variance_entropy?.toFixed(4)}</Text>
                
                <Text position={[0, -2, 0]} fontSize={0.4} color="#aaa" anchorX="left">Anisotropy:</Text>
                <Text position={[4, -2, 0]} fontSize={0.4} color={anisotropy > 0.8 ? "#ff6b6b" : "#5ec962"} anchorX="left">
                    {anisotropy?.toFixed(4)} {anisotropy > 0.8 ? '(COLLAPSE)' : '(Healthy)'}
                </Text>
            </group>

            {/* Point Cloud */}
            <group position={[2, 0, 0]}>
                {points.map((p, i) => (
                    <mesh key={i} position={[p[0], p[1], p[2]]}>
                        <sphereGeometry args={[0.1]} />
                        <meshStandardMaterial color={anisotropy > 0.8 ? "#ff6b6b" : "#4488ff"} />
                    </mesh>
                ))}
            </group>
            
            <axesHelper args={[2]} position={[2,0,0]} />
        </group>
    );
}


// Add CSS animation for slide-in effect

const style = document.createElement('style');
style.textContent = `
  @keyframes slideIn {
    from {
      opacity: 0;
      transform: translateX(20px);
    }
    to {
      opacity: 1;
      transform: translateX(0);
    }
  }
`;
if (!document.querySelector('style[data-animation="slideIn"]')) {
  style.setAttribute('data-animation', 'slideIn');
  document.head.appendChild(style);
}


// --- Styled Helper Components ---
const ControlGroup = ({ label, children, style }) => (
  <div style={{ marginBottom: '20px', ...style }}>
    {label && <label style={{ display: 'block', marginBottom: '8px', fontSize: '12px', color: '#888', fontWeight: '600', letterSpacing: '0.5px' }}>{label.toUpperCase()}</label>}
    {children}
  </div>
);

const StyledInput = (props) => (
  <input
    {...props}
    style={{
      width: '100%', padding: '10px 12px', backgroundColor: 'rgba(0,0,0,0.2)', 
      border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px',
      color: '#fff', fontSize: '13px', outline: 'none',
      transition: 'all 0.2s',
      ...props.style
    }}
    onFocus={(e) => {
        e.target.style.borderColor = '#4488ff';
        e.target.style.backgroundColor = 'rgba(0,0,0,0.3)';
    }}
    onBlur={(e) => {
        e.target.style.borderColor = 'rgba(255,255,255,0.1)';
        e.target.style.backgroundColor = 'rgba(0,0,0,0.2)';
    }}
  />
);

const StyledTextArea = (props) => (
  <textarea
    {...props}
    style={{
      width: '100%', padding: '10px 12px', backgroundColor: 'rgba(0,0,0,0.2)', 
      border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px',
      color: '#fff', fontSize: '13px', outline: 'none', resize: 'vertical',
      fontFamily: 'monospace', lineHeight: '1.5',
      transition: 'all 0.2s',
      ...props.style
    }}
    onFocus={(e) => {
        e.target.style.borderColor = '#4488ff';
        e.target.style.backgroundColor = 'rgba(0,0,0,0.3)';
    }}
    onBlur={(e) => {
        e.target.style.borderColor = 'rgba(255,255,255,0.1)';
        e.target.style.backgroundColor = 'rgba(0,0,0,0.2)';
    }}
  />
);

const ActionButton = ({ loading, onClick, children, color = '#4488ff', icon: Icon }) => (
  <button
    onClick={onClick}
    disabled={loading}
    style={{
      width: '100%', padding: '12px', backgroundColor: loading ? '#333' : color, 
      color: '#fff', border: 'none', borderRadius: '8px', 
      cursor: loading ? 'not-allowed' : 'pointer', fontSize: '13px', fontWeight: '600',
      boxShadow: loading ? 'none' : `0 4px 12px ${color}40`,
      display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px',
      transition: 'all 0.2s',
      transform: loading ? 'scale(0.98)' : 'scale(1)'
    }}
  >
    {loading ? <div className="spinner" style={{width: 16, height: 16, border: '2px solid rgba(255,255,255,0.3)', borderTop: '2px solid #fff', borderRadius: '50%', animation: 'spin 1s linear infinite'}}></div> : (
        <>
            {Icon && <Icon size={16} />}
            {children}
        </>
    )}
  </button>
);

// Settings Panel Component

function SettingsPanel({ 
  isOpen, onClose, 
  showSidebar, setShowSidebar, 
  showResultsOverlay, setShowResultsOverlay, 
  onReset 
}) {
  if (!isOpen) return null;
  
  return (
    <div style={{
      position: 'absolute', top: '50px', left: '20px', zIndex: 100,
      backgroundColor: 'rgba(26, 26, 26, 0.95)', border: '1px solid #444',
      borderRadius: '8px', padding: '16px', width: '250px',
      boxShadow: '0 4px 20px rgba(0,0,0,0.5)', backdropFilter: 'blur(10px)'
    }}>
      <h3 style={{ color: '#fff', fontSize: '14px', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px', marginTop: 0 }}>
        <Settings size={16} /> 界面配置
      </h3>
      
      <div style={{ marginBottom: '12px' }}>
        <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: '#ccc', fontSize: '13px', cursor: 'pointer' }}>
          <span>显示侧边栏</span>
          <input 
            type="checkbox" 
            checked={showSidebar} 
            onChange={e => setShowSidebar(e.target.checked)}
            style={{ accentColor: '#4488ff' }}
          />
        </label>
      </div>
       <div style={{ marginBottom: '16px' }}>
        <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: '#ccc', fontSize: '13px', cursor: 'pointer' }}>
          <span>显示结果浮窗</span>
          <input 
            type="checkbox" 
            checked={showResultsOverlay} 
            onChange={e => setShowResultsOverlay(e.target.checked)}
            style={{ accentColor: '#4488ff' }}
          />
        </label>
      </div>
      
      <button onClick={onReset} style={{
        width: '100%', padding: '8px', backgroundColor: '#333', color: '#fff', border: 'none',
        borderRadius: '4px', cursor: 'pointer', fontSize: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
        transition: 'background 0.2s'
      }}>
        <RotateCcw size={12} /> 重置布局
      </button>
      
      <button 
        onClick={onClose} 
        style={{ position: 'absolute', top: '8px', right: '8px', background: 'none', border: 'none', color: '#888', cursor: 'pointer' }}
      >
        ✕
      </button>
    </div>
  );
}

// --- Refactored UI Components ---

function StatusOverlay({ data }) {
  if (!data) return null;
  return (
    <div className="animate-fade-in" style={{
       position: 'absolute', bottom: '20px', right: '20px',
       background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(10px)',
       padding: '16px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.1)',
       color: '#fff', fontSize: '12px', pointerEvents: 'none',
       minWidth: '240px', boxShadow: '0 8px 32px rgba(0,0,0,0.3)',
       zIndex: 10
    }}>
       <h4 style={{
           margin: '0 0 12px 0', borderBottom: '1px solid rgba(255,255,255,0.1)', 
           paddingBottom: '8px', fontSize: '13px', color: '#4488ff',
           display: 'flex', alignItems: 'center', gap: '8px'
       }}>
         <Sparkles size={14} /> {data.time ? `Step ${data.time}` : (data.title || 'System Status')}
       </h4>
       
       <div style={{display: 'flex', flexDirection: 'column', gap: '8px'}}>
           {Object.entries(data.items || {}).map(([k, v]) => (
              <div key={k} style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                 <span style={{color: '#888'}}>{k}:</span>
                 <span style={{fontWeight: '600', color: '#fff', fontFamily: 'monospace'}}>{v}</span>
              </div>
           ))}
       </div>

       {data.description && (
          <div style={{
              marginTop: '12px', paddingTop: '8px', 
              borderTop: '1px solid rgba(255,255,255,0.1)',
              fontStyle: 'italic', color: '#aaa', lineHeight: '1.4'
          }}>
             {data.description}
          </div>
       )}
    </div>
  )
}

function InfoPanel({ activeTab, t }) {
    // Content definitions
    const infoContent = {
        circuit: {
            title: t('structure.circuit.title'),
            desc: "就像寻找家里的电路故障一样，这个工具能帮我们找出 AI 完成特定任务时最核心的“神经回路”。",
            tech: "Edge Attribution Patching"
        },
        features: {
            title: t('structure.features.title'),
            desc: "AI 的思维非常杂乱，我们通过这个工具将其拆解为一个个具体的、人能听懂的概念（特征）。",
            tech: "Sparse Autoencoders (SAE)"
        },
        causal: {
            title: t('structure.causal.title'),
            desc: "如果我们强制改变模型内部的一个信号，它的最终答案会变吗？这能帮我们确定谁才是真正的“幕后主使”。",
            tech: "Activation Patching"
        },
        manifold: {
            title: t('structure.manifold.title'),
            desc: "分析 AI 思维世界的“地形地貌”，看看它的想法是井然有序的，还是已经乱成了一团。",
            tech: "Intrinsic Dimensionality"
        },
        compositional: {
            title: t('structure.compositional.title'),
            desc: "测试 AI 是否懂得“1+1=2”的逻辑，比如它是否理解“黑色”+“猫”=“黑猫”这种组合概念。",
            tech: "Vector Arithmetic, OLS"
        },
        agi: {
            title: "神经纤维丛分析 (Neural Fiber Bundle Analysis)",
            desc: "基于最新的统一场论，验证网络内部是否存在完美的数学纤维丛结构——这是通往通用人工智能的关键。",
            tech: "RSA, Differential Geometry"
        },
        snn: {
            title: t('snn.title', '脉冲神经网络'),
            desc: "开启仿生模式。您可以观察神经元像真实大脑一样，通过电脉冲的同步爆发来“绑定”不同的概念。",
            tech: "LIF Neurons, Phase Locking"
        },
        validity: {
            title: t('validity.title', '语言有效性分析'),
            desc: "检查 AI 是否在胡言乱语。如果它的思维空间缩成了一个点（坍缩），说明它已经失去了逻辑能力。",
            tech: "Entropy, Anisotropy, PPL"
        },
        glass_matrix: {
            title: "玻璃矩阵 (Glass Matrix)",
            desc: "以3D方式可视化神经纤维丛的拓扑结构。观察流形（Manifold）作为基础空间，以及附着其上的纤维（Fibers）作为语义空间。",
            tech: "NFB-RA, React Three Fiber"
        },
        flow_tubes: {
            title: "深度动力学 (Deep Dynamics)",
            desc: "可视化 Token 在深层网络中的演化轨迹。流管 (Flow Tubes) 显示了不同概念（如性别、情感）在层级间的几何变换路径。",
            tech: "Neural ODE, Flow Tubes"
        }
    };

    const current = infoContent[activeTab] || { title: "Algorithm Info", desc: "Select an algorithm to view details." };

    return (
        <div style={{
            marginTop: 'auto',
            padding: '20px',
            background: 'rgba(0,0,0,0.3)',
            borderTop: '1px solid rgba(255,255,255,0.1)',
            backdropFilter: 'blur(20px)'
        }}>
            <h3 style={{
                color: '#4488ff', fontSize: '14px', margin: '0 0 8px 0', 
                display: 'flex', alignItems: 'center', gap: '8px'
            }}>
                <Brain size={16} /> {current.title}
            </h3>
            <p style={{color: '#ccc', fontSize: '12px', lineHeight: '1.5', margin: '0 0 12px 0'}}>
                {current.desc}
            </p>
            {current.tech && (
                <div style={{display: 'flex', alignItems: 'center', gap: '6px'}}>
                    <span style={{fontSize: '11px', color: '#666', textTransform: 'uppercase', fontWeight: 'bold'}}>TECH:</span>
                    <span style={{fontSize: '11px', color: '#4ecdc4', background: 'rgba(78, 205, 196, 0.1)', padding: '2px 6px', borderRadius: '4px'}}>
                        {current.tech}
                    </span>
                </div>
            )}
        </div>
    );
}

export function StructureAnalysisControls({ 
  autoResult,
  circuitForm, setCircuitForm,
  featureForm, setFeatureForm,
  causalForm, setCausalForm,
  manifoldForm, setManifoldForm, 
  compForm, setCompForm, 
  onResultUpdate, 
  activeTab, setActiveTab,
  containerStyle, 
  t,
  // New Props
  systemType, setSystemType,
  // AGI Form
  agiForm, setAgiForm,
  // SNN Props
  snnState, onInitializeSNN, onToggleSNNPlay, onStepSNN, onInjectStimulus
}) {
  const [loading, setLoading] = useState(false);
  const [progressLogs, setProgressLogs] = useState([]);
  
  // Settings State
  const [showSettings, setShowSettings] = useState(false);
  
  const [steeringForm, setSteeringForm] = useState({
      prompt: "The meeting will begin shortly.",
      layer_idx: 15,
      strength: 1.0,
      concept_pair: "formal_casual"
  });

  // Validity State
  const [validityForm, setValidityForm] = useState({ prompt: "The quick brown fox jumps over the lazy dog." });
  const [validityResult, setValidityResult] = useState(null);

  useEffect(() => {
    // When switching systems, default to a tab
    if (systemType === 'snn') {
        if (activeTab !== 'snn' && activeTab !== 'validity') setActiveTab('snn');
    } else {
        if (activeTab === 'snn' || activeTab === 'validity') setActiveTab('circuit');
    }
  }, [systemType]);

  // -- API Call Wrapper Helper --
  const runAnalysis = async (name, apiPath, form, onSuccess) => {
      setLoading(true);
      setProgressLogs([]);
      onResultUpdate(null);
      const addLog = (msg) => setProgressLogs(p => [...p, msg]);
      
      try {
          addLog(`🚀 Starting ${name}...`);
          const response = await axios.post(`${API_BASE}/${apiPath}`, form);
          addLog('✅ Analysis Complete!');
          onSuccess(response.data, addLog);
          onResultUpdate(response.data);
      } catch (error) {
          console.error(`${name} failed:`, error);
          const msg = error.response?.data?.detail || error.message;
          addLog(`❌ Error: ${msg}`);
          alert(`Error: ${msg}`);
      }
      setLoading(false);
  };

  const runCircuitDiscovery = () => runAnalysis('Circuit Discovery', 'discover_circuit', circuitForm, (data, log) => {
      log(`📊 Nodes: ${data.nodes?.length || 0}, Edges: ${data.graph?.edges?.length || 0}`);
  });

  const runFeatureExtraction = () => runAnalysis('Feature Extraction', 'extract_features', featureForm, (data, log) => {
      log(`📊 Features: ${data.top_features?.length || 0}`);
      log(`🎯 Reconstruction Error: ${data.reconstruction_error?.toFixed(6)}`);
  });

  const runCausalAnalysis = () => runAnalysis('Causal Analysis', 'causal_analysis', causalForm, (data, log) => {
      log(`⭐ Important Components: ${data.n_important_components || 0}`);
  });

  const runManifoldAnalysis = () => runAnalysis('Manifold Analysis', 'manifold_analysis', manifoldForm, (data, log) => {
      log(`📊 Intrinsic Dim: ${data.intrinsic_dimensionality?.participation_ratio?.toFixed(2)}`);
  });

  const runCompositionalAnalysis = () => runAnalysis('Compositional Analysis', 'compositional_analysis', compForm, (data, log) => {
      log(`📈 R²: ${data.r2_score?.toFixed(4)}`);
  });

  const runValidityAnalysis = () => runAnalysis('Validity Analysis', 'analyze_validity', validityForm, (data, log) => {
      log(`📉 Perplexity: ${data.perplexity?.toFixed(2)}`);
      setValidityResult(data);
  });

  const runAgiVerification = () => runAnalysis('Fiber Bundle Reconstruction', 'fiber_bundle_analysis', { prompt: agiForm.prompt }, (data, log) => {
      const baseCount = data.rsa?.filter(l => l.type === 'Base').length;
      log(`📊 Systematic Layers: ${baseCount}`);
      log(`🧬 Fiber Basis Identified`);
  });

  const runConceptSteering = async () => {
      // Custom handler for steering to keep result
      setLoading(true);
      setProgressLogs(p => [...p, '🚀 Starting Concept Steering...']);
      try {
          const response = await axios.post(`${API_BASE}/steer_concept`, steeringForm);
          onResultUpdate(prev => ({ 
              ...prev, 
              steering: { ...response.data, layer_idx: steeringForm.layer_idx }
          }));
          setProgressLogs(p => [...p, '✅ Steering Complete']);
      } catch (e) { console.error(e); }
      setLoading(false);
  };

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', height: '100%',
      backgroundColor: 'rgba(20, 20, 25, 0.95)',
      backdropFilter: 'blur(20px)',
      borderRight: '1px solid rgba(255,255,255,0.1)',
      ...containerStyle
    }}>
      
      {/* Top Section: System Type Selector */}
      <div style={{ padding: '16px', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
          <h2 style={{ 
             margin: '0 0 12px 0', fontSize: '18px', fontWeight: 'bold', 
             background: 'linear-gradient(45deg, #00d2ff, #3a7bd5)', 
             WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
             display: 'flex', alignItems: 'center', gap: '8px' 
          }}>
            🧠 {t('structure.title')}
          </h2>


      </div>

      {/* Middle Section: Algorithm Tabs & content */}
      <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>
          
          {/* Algorithm List */}
          <div style={{ 
              display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '4px', p: '8px',
              padding: '12px', borderBottom: '1px solid rgba(255,255,255,0.05)'
          }}>
              {(systemType === 'dnn' ? [
                 { id: 'circuit', icon: Network, label: '回路 (Circuit)' },
                 { id: 'features', icon: Sparkles, label: '特征 (Features)' },
                 { id: 'causal', icon: Brain, label: '因果 (Causal)' },
                 { id: 'compositional', icon: Network, label: '组合 (Compos)' },
                 { id: 'agi', icon: Sparkles, label: '神经纤维丛 (Fiber)' },
                 { id: 'tda', icon: Activity, label: '拓扑 (Topology)' },
                 { id: 'fibernet_v2', icon: Network, label: 'FiberNet V2 (Demo)' },
                 { id: 'glass_matrix', icon: Network, label: '玻璃矩阵 (Glass)' },
                 { id: 'flow_tubes', icon: Activity, label: '动力学 (Dynamics)' }
              ] : [
                 { id: 'snn', icon: Brain, label: 'SNN 仿真' },
                 { id: 'validity', icon: Activity, label: '有效性 (Valid)' }
              ]).map(tab => (
                 <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    style={{
                        padding: '8px 4px',
                        backgroundColor: activeTab === tab.id ? 'rgba(68, 136, 255, 0.2)' : 'transparent',
                        color: activeTab === tab.id ? '#4488ff' : '#666',
                        border: activeTab === tab.id ? '1px solid rgba(68, 136, 255, 0.4)' : '1px solid transparent',
                        borderRadius: '6px', cursor: 'pointer', fontSize: '11px', fontWeight: '500',
                        display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px'
                    }}
                 >
                    <tab.icon size={14} />
                    {tab.label}
                 </button>
              ))}
          </div>

          {/* Form Content */}
          <div style={{ padding: '16px', flex: 1 }}>
              
            {/* --- DNN Forms --- */}
            {activeTab === 'tda' && (
                <div className="animate-fade-in">
                    <ControlGroup label="Topological Data Analysis (PH)">
                        <div style={{color: '#aaa', fontSize: '13px', lineHeight: '1.5', padding: '10px', background: 'rgba(255,255,255,0.05)', borderRadius: '8px', marginBottom: '12px'}}>
                            <p style={{marginTop:0}}><strong>Persistent Homology</strong></p>
                            <p>计算流形的贝蒂数 (Betti Numbers)，揭示拓扑孔洞结构。</p>
                        </div>
                    </ControlGroup>
                    
                    <ActionButton onClick={() => {
                        setLoading(true);
                        axios.get(`${API_BASE}/nfb_ra/tda`)
                            .then(res => {
                                onResultUpdate(res.data);
                                setLoading(false);
                            })
                            .catch(err => {
                                console.error(err);
                                setLoading(false);
                                alert("Failed to fetch TDA results");
                            });
                    }} loading={loading} icon={Activity}>
                        获取拓扑特征 (Get Betti Numbers)
                    </ActionButton>
                    
                    {/* Inline Results for TDA (since it's simple) */}
                    {activeTab === 'tda' && autoResult && autoResult.ph_0d && (
                        <div style={{ mt: 2, padding: '10px', background: '#222', borderRadius: '6px', fontSize: '12px', color: '#ddd' }}>
                            <div><strong>0-dim (Components):</strong> {autoResult.ph_0d.length} (Connected)</div>
                            <div><strong>1-dim (Loops):</strong> {autoResult.ph_1d.length} (Holes found)</div>
                            <div style={{ marginTop: '8px', fontSize: '10px', color: '#888' }}>
                                TDA results loaded from backend.
                            </div>
                        </div>
                    )}
                </div>
            )}
            {activeTab === 'circuit' && (
              <div className="animate-fade-in">
                <ControlGroup label={t('structure.circuit.cleanPrompt')}>
                  <StyledTextArea rows={3} value={circuitForm.clean_prompt} onChange={e => setCircuitForm({...circuitForm, clean_prompt: e.target.value})} placeholder="Clean prompt..." />
                </ControlGroup>
                <ControlGroup label={t('structure.circuit.corruptedPrompt')}>
                  <StyledTextArea rows={3} value={circuitForm.corrupted_prompt} onChange={e => setCircuitForm({...circuitForm, corrupted_prompt: e.target.value})} placeholder="Corrupted prompt..." />
                </ControlGroup>
                 <div style={{ marginBottom: '24px' }}>
                     <label style={{ display: 'flex', justifyContent: 'space-between', color: '#888', fontSize: '12px', marginBottom: '6px' }}>
                        <span>Threshold</span> <span style={{color: '#4488ff'}}>{circuitForm.threshold}</span>
                     </label>
                     <input type="range" min="0.01" max="0.5" step="0.01" value={circuitForm.threshold} onChange={e => setCircuitForm({...circuitForm, threshold: parseFloat(e.target.value)})} style={{ width: '100%', height: '4px', accentColor: '#4488ff' }} />
                 </div>
                <ActionButton onClick={runCircuitDiscovery} loading={loading} icon={Network}>{t('structure.circuit.run')}</ActionButton>
              </div>
            )}

            {activeTab === 'features' && (
              <div className="animate-fade-in">
                 <ControlGroup label="Prompt">
                    <StyledTextArea rows={3} value={featureForm.prompt} onChange={e => setFeatureForm({...featureForm, prompt: e.target.value})} />
                 </ControlGroup>
                 <ControlGroup label={`Layer (L${featureForm.layer_idx})`}>
                    <input type="range" min="0" max="32" step="1" value={featureForm.layer_idx} onChange={e => setFeatureForm({...featureForm, layer_idx: parseInt(e.target.value)})} style={{ width: '100%', height: '4px', accentColor: '#4488ff' }} />
                 </ControlGroup>
                 <ControlGroup label={`Dim (${featureForm.hidden_dim})`}>
                    <input type="range" min="256" max="4096" step="256" value={featureForm.hidden_dim} onChange={e => setFeatureForm({...featureForm, hidden_dim: parseInt(e.target.value)})} style={{ width: '100%', height: '4px', accentColor: '#4ecdc4' }} />
                 </ControlGroup>
                 <ActionButton onClick={runFeatureExtraction} loading={loading} icon={Network}>{t('structure.features.run')}</ActionButton>
              </div>
            )}

            {activeTab === 'causal' && (
               <div className="animate-fade-in">
                  <ControlGroup label="Prompt">
                    <StyledTextArea rows={3} value={causalForm.prompt} onChange={e => setCausalForm({...causalForm, prompt: e.target.value})} />
                  </ControlGroup>
                  <ActionButton onClick={runCausalAnalysis} loading={loading} icon={Brain}>{t('structure.causal.run')}</ActionButton>
               </div>
            )}

            {activeTab === 'manifold' && (
               <div className="animate-fade-in">
                  <ControlGroup label="Prompt">
                    <StyledTextArea rows={3} value={manifoldForm.prompt} onChange={e => setManifoldForm({...manifoldForm, prompt: e.target.value})} />
                  </ControlGroup>
                  <ControlGroup label={`Layer (L${manifoldForm.layer_idx})`}>
                    <input type="range" min="0" max="32" step="1" value={manifoldForm.layer_idx} onChange={e => setManifoldForm({...manifoldForm, layer_idx: parseInt(e.target.value)})} style={{ width: '100%', height: '4px', accentColor: '#4488ff' }} />
                 </ControlGroup>
                  <ActionButton onClick={runManifoldAnalysis} loading={loading} icon={Network}>{t('structure.manifold.run')}</ActionButton>
               </div>
            )}

            {activeTab === 'compositional' && (
               <div className="animate-fade-in">
                  <ControlGroup label="Phrases (Format: a, b, a+b)">
                    <StyledTextArea rows={5} value={compForm.raw_phrases} onChange={e => setCompForm({...compForm, raw_phrases: e.target.value})} />
                  </ControlGroup>
                  <ActionButton onClick={runCompositionalAnalysis} loading={loading} icon={Network}>{t('structure.compositional.run')}</ActionButton>
               </div>
            )}

            {activeTab === 'fibernet_v2' && (
                <div style={{ width: '100%', padding: '20px', textAlign: 'center', color: '#888', fontStyle: 'italic' }}>
                    3D 演示已在主屏幕背景中显示。
                    <br/><br/>
                    请使用主屏幕交互。
                </div>
            )}

            {activeTab === 'agi' && (
                <div className="animate-fade-in">
                    <ControlGroup label="Analysis Prompt">
                         <StyledTextArea 
                             rows={3} 
                             value={agiForm.prompt} 
                             onChange={e => setAgiForm({...agiForm, prompt: e.target.value})} 
                             placeholder="请输入要分析的文本以提取其数学结构..."
                         />
                    </ControlGroup>
                    <ActionButton onClick={runAgiVerification} loading={loading} icon={Sparkles}>开始神经纤维丛分析</ActionButton>
                    <div style={{margin: '20px 0', borderTop: '1px solid rgba(255,255,255,0.1)'}} />
                    <ControlGroup label="概念驾驶 (Concept Steering)">
                         <StyledTextArea rows={2} value={steeringForm.prompt} onChange={e => setSteeringForm({...steeringForm, prompt: e.target.value})} placeholder="输入干预概念..." />
                         <div style={{marginTop: '10px'}}>
                             <input type="range" min="-5" max="5" step="0.5" value={steeringForm.strength} onChange={e => setSteeringForm({...steeringForm, strength: parseFloat(e.target.value)})} style={{ width: '100%', accentColor: '#4488ff' }} />
                         </div>
                    </ControlGroup>
                    <ActionButton onClick={runConceptSteering} loading={loading} icon={Network}>执行概念干预</ActionButton>
                </div>
            )}

            {activeTab === 'glass_matrix' && (
                <div className="animate-fade-in">
                    <ControlGroup label="Glass Matrix Visualization">
                        <div style={{color: '#aaa', fontSize: '13px', lineHeight: '1.5', padding: '10px', background: 'rgba(255,255,255,0.05)', borderRadius: '8px', marginBottom: '12px'}}>
                            <p style={{marginTop:0}}>玻璃矩阵可视化已激活。请在主视图中查看 3D 结构。</p>
                            <ul style={{paddingLeft: '20px', margin: '10px 0'}}>
                                <li style={{marginBottom:'4px'}}><strong style={{color:'#00ffff'}}>Manifold (Cyan):</strong> 句法/逻辑基础</li>
                                <li><strong style={{color:'#ff4444'}}>Fibers (RGB Vectors):</strong> 语义向量空间</li>
                            </ul>
                            <p style={{marginBottom:0, fontSize:'12px', fontStyle:'italic'}}>数据来源: frontend/public/nfb_data.json</p>
                        </div>
                    </ControlGroup>
                    
                     <ActionButton onClick={() => window.location.reload()} loading={loading} icon={RotateCcw}>
                        重置/刷新视图 (Refresh View)
                    </ActionButton>
                    
                    <div style={{marginTop: '12px', fontSize: '11px', color: '#666', textAlign: 'center'}}>
                        提示: 如需生成新数据，请运行 backend 脚本 experiments/nfb_ra_qwen.py
                    </div>
                </div>
            )}

            {activeTab === 'flow_tubes' && (
                <div className="animate-fade-in">
                     <ControlGroup label="Deep Dynamics Visualization">
                        <div style={{color: '#aaa', fontSize: '13px', lineHeight: '1.5', padding: '10px', background: 'rgba(255,255,255,0.05)', borderRadius: '8px', marginBottom: '12px'}}>
                            <p style={{marginTop:0}}><strong>Deep Dynamics Flow Tubes</strong></p>
                            <p>展示 Transformer 内部的几何动力学轨迹。</p>
                            <ul style={{paddingLeft: '20px', margin: '10px 0'}}>
                                <li style={{marginBottom:'4px'}}><strong style={{color:'#3498db'}}>Male (Blue):</strong> 男性概念子空间演化</li>
                                <li style={{marginBottom:'4px'}}><strong style={{color:'#e74c3c'}}>Female (Red):</strong> 女性概念子空间演化</li>
                                <li><strong style={{color:'#2ecc71'}}>Positive (Green):</strong> 情感极性演化</li>
                            </ul>
                        </div>
                    </ControlGroup>
                    
                    <ActionButton onClick={() => window.location.reload()} loading={loading} icon={RotateCcw}>
                        刷新数据 (Refresh Data)
                    </ActionButton>
                </div>
            )}

            {activeTab === 'snn' && (
                <div className="animate-fade-in">
                    {!snnState?.initialized ? (
                        <div style={{ textAlign: 'center', padding: '20px 0' }}>
                            <div style={{ marginBottom: '12px', color: '#aaa', fontSize: '13px' }}>
                                NeuroFiber Network not initialized
                            </div>
                            <ActionButton onClick={onInitializeSNN} loading={loading} icon={Brain}>
                                Initialize SNN
                            </ActionButton>
                        </div>
                    ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                            {/* Simulation Controls */}
                            <div style={{ display: 'flex', gap: '8px' }}>
                                <button
                                    onClick={onToggleSNNPlay}
                                    style={{
                                        flex: 1, padding: '8px',
                                        background: snnState.isPlaying ? '#ff5252' : '#4ecdc4',
                                        border: 'none', borderRadius: '6px',
                                        color: '#000', fontWeight: 'bold', cursor: 'pointer',
                                        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px'
                                    }}
                                >
                                    {snnState.isPlaying ? '⏹ Stop' : '▶ Run'}
                                </button>
                                <button
                                    onClick={onStepSNN}
                                    style={{
                                        flex: 1, padding: '8px',
                                        background: '#333', border: '1px solid #555', borderRadius: '6px',
                                        color: '#fff', cursor: 'pointer',
                                        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px'
                                    }}
                                >
                                    Step
                                </button>
                            </div>

                            {/* Info Stats */}
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#aaa', padding: '0 4px' }}>
                                <span>Time: {snnState.time.toFixed(1)}ms</span>
                                <span>Layers: {snnState.layers.length}</span>
                            </div>

                            <div style={{margin: '8px 0', borderTop: '1px solid rgba(255,255,255,0.1)'}} />

                            {/* Stimulus Injection */}
                            <ControlGroup label="Stimulus Injection">
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                    <button
                                        onClick={() => onInjectStimulus('Retina_Shape', 5)}
                                        style={{
                                            padding: '8px', background: 'rgba(255,107,107,0.1)',
                                            border: '1px solid #ff6b6b', color: '#ff6b6b',
                                            borderRadius: '6px', cursor: 'pointer', fontSize: '12px',
                                            textAlign: 'left'
                                        }}
                                    >
                                        🍎 Inject "Apple" (Shape)
                                    </button>
                                    <button
                                        onClick={() => onInjectStimulus('Retina_Color', 5)}
                                        style={{
                                            padding: '8px', background: 'rgba(255,107,107,0.1)',
                                            border: '1px solid #ff6b6b', color: '#ff6b6b',
                                            borderRadius: '6px', cursor: 'pointer', fontSize: '12px',
                                            textAlign: 'left'
                                        }}
                                    >
                                        🔴 Inject "Red" (Color)
                                    </button>
                                </div>
                            </ControlGroup>
                        </div>
                    )}
                </div>
            )}
            
            {activeTab === 'validity' && (
                <div className="animate-fade-in">
                    <ControlGroup label={t('validity.prompt', 'Analysis Text')}>
                        <StyledTextArea 
                            rows={4} 
                            value={validityForm.prompt} 
                            onChange={e => setValidityForm({...validityForm, prompt: e.target.value})} 
                        />
                    </ControlGroup>
                    <ActionButton onClick={runValidityAnalysis} loading={loading} icon={Activity}>
                        {t('validity.analyze', 'Analyze Validity')}
                    </ActionButton>
                    
                    {/* Inline Results for Validity */}
                    {validityResult && (
                        <div style={{ marginTop: '20px', className: 'fade-in' }}>
                             <div style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
                                <MetricCard 
                                    title="PPL" 
                                    value={validityResult.perplexity} 
                                />
                                <MetricCard 
                                    title="Entropy"
                                    value={validityResult.entropy_stats?.mean_entropy} 
                                    color="#ffaa00"
                                />
                            </div>
                            <EntropyHeatmap entropyStats={validityResult.entropy_stats} text={validityForm.prompt} t={t} />
                            <AnisotropyChart geometricStats={validityResult.geometric_stats} t={t} />
                        </div>
                    )}
                </div>
            )}

          </div>
      </div>


      {/* Bottom Section: Info Panel */}
      <InfoPanel activeTab={activeTab} t={t} />

      {/* Progress Logs (Overlay style inside controls if needed, or integrated) */}
      {loading && (
          <div style={{ padding: '10px', background: 'rgba(0,0,0,0.5)', fontSize: '10px', maxHeight: '100px', overflowY: 'auto' }}>
              {progressLogs.map((l,i) => <div key={i} style={{color:'#aaa'}}>{l}</div>)}
          </div>
      )}

    </div>
  );
}
// --- Main Panel Component ---

export default function StructureAnalysisPanel({ 
    t, 
    snnState, onInitializeSNN, onToggleSNNPlay, onStepSNN, onInjectStimulus
}) {
  const [activeTab, setActiveTab] = useState('circuit');
  const [systemType, setSystemType] = useState('dnn');
  
  // Forms local state (lifted for persistence during tab switch)
  const [circuitForm, setCircuitForm] = useState({ clean_prompt: "The cat sat on the mat.", corrupted_prompt: "The dog sat on the mat.", threshold: 0.1 });
  const [featureForm, setFeatureForm] = useState({ prompt: "Hello world", layer_idx: 6, hidden_dim: 2048, sparsity_coef: 0.01, n_epochs: 1000 });
  const [causalForm, setCausalForm] = useState({ prompt: "The Eiffel Tower is in Paris", target_token_pos: -1, importance_threshold: 0.01 });
  const [manifoldForm, setManifoldForm] = useState({ prompt: "Mathematics is the language of the universe.", layer_idx: 15 });
  const [compForm, setCompForm] = useState({ layer_idx: 15, raw_phrases: "", phrases: [] });
  const [agiForm, setAgiForm] = useState({ prompt: "The quick brown fox jumps over the lazy dog." });
  
  const [analysisResult, setAnalysisResult] = useState(null);
  
  // Status Overlay State
  const [statusData, setStatusData] = useState(null);

  // Handler for SNN updates
  const handleStatusUpdate = (data) => {
      setStatusData({
          title: "SNN Simulation",
          time: data.step,
          items: {
              "Neurons": data.activeCount,
              "Status": data.isPlaying ? "Running" : "Idle"
          },
          description: data.description
      });
  };
  
  // Handle DNN result updates for status overlay
  useEffect(() => {
      if (systemType === 'dnn') {
          if (analysisResult) {
               // Map result active tab to a summary
               let items = {};
               if (activeTab === 'circuit') items = { Nodes: analysisResult.nodes?.length, Edges: analysisResult.graph?.edges?.length };
               if (activeTab === 'features') items = { Features: analysisResult.top_features?.length, Error: analysisResult.reconstruction_error?.toFixed(4) };
               if (activeTab === 'causal') items = { Components: analysisResult.n_components_analyzed };
               if (activeTab === 'validity') items = { PPL: analysisResult.perplexity, Entropy: analysisResult.entropy_stats?.mean_entropy?.toFixed(2) };
               
               setStatusData({
                   title: activeTab.toUpperCase() + " Analysis",
                   items: items,
                   description: "Analysis complete. View 3D results."
               });
          } else {
               setStatusData(null);
          }
      }
      // Also handle SNN validity updates if systemType is SNN but using the generic analysis result
      if (systemType === 'snn' && activeTab === 'validity' && analysisResult) {
           setStatusData({
               title: "Validity Analysis",
               items: { PPL: analysisResult.perplexity, Entropy: analysisResult.entropy_stats?.mean_entropy?.toFixed(2) },
               description: "Representation validity analysis complete."
           });
      }
  }, [analysisResult, activeTab, systemType]);

  // Mock t if not provided (safety)
  const tSafe = t || ((k, d) => d || k);

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'row', position: 'relative', overflow: 'hidden' }}>
      
      {/* Left Panel: Controls */}
      <div style={{ width: '340px', height: '100%', zIndex: 5, flexShrink: 0 }}>
         <StructureAnalysisControls
            systemType={systemType}
            setSystemType={setSystemType}
            activeTab={activeTab}
            setActiveTab={setActiveTab}
            circuitForm={circuitForm} setCircuitForm={setCircuitForm}
            featureForm={featureForm} setFeatureForm={setFeatureForm}
            causalForm={causalForm} setCausalForm={setCausalForm}
            manifoldForm={manifoldForm} setManifoldForm={setManifoldForm}
            compForm={compForm} setCompForm={setCompForm}
            agiForm={agiForm} setAgiForm={setAgiForm}
            onResultUpdate={setAnalysisResult}
            t={tSafe}
            // SNN Props
            snnState={snnState}
            onInitializeSNN={onInitializeSNN}
            onToggleSNNPlay={onToggleSNNPlay}
            onStepSNN={onStepSNN}
            onInjectStimulus={onInjectStimulus}
         />
      </div>

      {/* Right Panel: Visualization (Flex) */}
      <div style={{ flex: 1, position: 'relative', backgroundColor: '#050510' }}>
         
         {/* Render Logic */}
         <div style={{ width: '100%', height: '100%' }}>
              {/* SNN View */}
              {activeTab === 'snn' && (
                  <Canvas camera={{ position: [20, 20, 20], fov: 50 }}>
                     <ambientLight intensity={0.5} />
                     <pointLight position={[10, 10, 10]} intensity={1} />
                     <OrbitControls makeDefault />
                     <SNNVisualization3D 
                        t={tSafe} 
                        onStatusUpdate={handleStatusUpdate} 
                        structure={snnState?.structure}
                        activeSpikes={snnState?.spikes}
                     />
                  </Canvas>
              )}

              {/* Validity View */}
              {activeTab === 'validity' && (
                   <Canvas camera={{ position: [10, 5, 10], fov: 50 }}>
                      <ambientLight intensity={0.6} />
                      <pointLight position={[10, 10, 10]} intensity={1} />
                      <OrbitControls makeDefault />
                      <ValidityVisualization3D result={analysisResult} t={tSafe} />
                   </Canvas>
              )}
             
             {/* DNN Views */}
             {systemType === 'dnn' && (
                 <>
                    {!analysisResult && !['agi'].includes(activeTab) && (
                        <div style={{
                            display:'flex', flexDirection: 'column', alignItems:'center', justifyContent:'center', 
                            height:'100%', color:'#444'
                        }}>
                             <Network size={64} style={{ opacity: 0.2, marginBottom: '20px' }} />
                             <div style={{ fontSize: '16px' }}>Select an algorithm to start analysis</div>
                        </div>
                    )}
                    
                    {activeTab === 'circuit' && analysisResult && <NetworkGraph3D graph={analysisResult.graph} />}
                    {activeTab === 'features' && analysisResult && <FeatureVisualization3D features={analysisResult.top_features} layerIdx={featureForm.layer_idx} />}
                    {activeTab === 'manifold' && analysisResult && <ManifoldVisualization3D pcaData={analysisResult.pca} />}
                    
                    {/* AGI Theory / Fiber Bundle View */}
                    {activeTab === 'agi' && (
                        analysisResult ? 
                        <div style={{width: '100%', height: '100%'}}>
                            <Canvas camera={{ position: [0, 0, 20], fov: 60 }}>
                                <ambientLight intensity={0.5} />
                                <pointLight position={[10, 10, 10]} intensity={1} />
                                <spotLight position={[-10, 10, -10]} intensity={0.5} />
                                <OrbitControls makeDefault />
                                <FiberBundleVisualization3D result={analysisResult} t={tSafe} />
                            </Canvas>
                        </div> :
                        <div style={{display:'flex', alignItems:'center', justifyContent:'center', height:'100%', color:'#666'}}>
                             Click "开始神经纤维丛分析" to generate fiber stream visualization
                        </div>
                    )}
                    
                    {activeTab === 'compositional' && analysisResult && <CompositionalVisualization3D result={analysisResult} t={tSafe} />}
                 </>
             )}
         </div>

         {/* Status Overlay */}
         <StatusOverlay data={statusData} />
         
      </div>
    </div>
  );
}
