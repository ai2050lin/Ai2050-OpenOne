import { OrbitControls, Text } from '@react-three/drei';
import { Canvas } from '@react-three/fiber';
import axios from 'axios';
import { Activity, Brain, Network, RotateCcw, Settings, Sparkles } from 'lucide-react';
import { useEffect, useState } from 'react';
import * as THREE from 'three';
import BrainVis3D from './BrainVis3D';

const API_BASE = 'http://localhost:8888';

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

// Graph visualization component for 3D network topology
export function NetworkGraph3D({ graph }) {
  if (!graph || !graph.nodes || graph.nodes.length === 0) {
    return null;
  }

  // Position nodes in a circular layout by layer
  const positions = [];
  const maxLayer = Math.max(...graph.nodes.map(n => n.layer || 0));
  
  graph.nodes.forEach((node, i) => {
    // Normalize node properties
    const layer = node.layer !== undefined ? node.layer : (node.layer_idx !== undefined ? node.layer_idx : 0);
    const type = node.type || node.component_type || 'unknown';
    
    // Count nodes in this layer for positioning
    const nodesInLayer = graph.nodes.filter(n => {
        const l = n.layer !== undefined ? n.layer : (n.layer_idx !== undefined ? n.layer_idx : 0);
        return l === layer;
    }).length;

    // Calculate index within layer
    const indexInLayer = graph.nodes.filter((n, idx) => {
        if (idx >= i) return false;
        const l = n.layer !== undefined ? n.layer : (n.layer_idx !== undefined ? n.layer_idx : 0);
        return l === layer;
    }).length;
    
    const radius = 8;
    // Spiral layout if too many nodes, or circle if few
    const angle = (indexInLayer / Math.max(nodesInLayer, 1)) * Math.PI * 2;
    
    const x = Math.cos(angle) * radius;
    const z = layer * 3 - (maxLayer * 1.5); 
    const y = Math.sin(angle) * radius;
    
    // Store normalized node for easier access later
    positions.push({ x, y, z, node: { ...node, layer, type } });
  });

  return (
    <group>
      {/* Draw edges */}
      {graph.edges && graph.edges.map((edge, i) => {
        const sourcePos = positions.find(p => p.node.id === edge.source);
        const targetPos = positions.find(p => p.node.id === edge.target);
        
        if (!sourcePos || !targetPos) return null;
        
        const start = new THREE.Vector3(sourcePos.x, sourcePos.y, sourcePos.z);
        const end = new THREE.Vector3(targetPos.x, targetPos.y, targetPos.z);
        const curve = new THREE.LineCurve3(start, end);
        const points = curve.getPoints(50);
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        
        return (
          <line key={i} geometry={geometry}>
            <lineBasicMaterial
              attach="material"
              color="#4488ff"
              linewidth={2}
              opacity={0.6}
              transparent
            />
          </line>
        );
      })}
      
      {/* Draw nodes */}
      {positions.map((pos, i) => (
        <mesh key={i} position={[pos.x, pos.y, pos.z]}>
          <sphereGeometry args={[0.3, 16, 16]} />
          <meshStandardMaterial
            color={pos.node.type === 'attention' ? '#ff6b6b' : '#4ecdc4'}
            emissive={pos.node.type === 'attention' ? '#ff6b6b' : '#4ecdc4'}
            emissiveIntensity={0.3}
          />
        </mesh>
      ))}
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

// 3D Fiber Bundle Visualization
export function FiberBundleVisualization3D({ result, t }) {
  if (!result || (!result.rsa && !result.steering)) return null;

  // Render dummy RSA structure if missing but steering exists
  const rsaData = result.rsa || Array.from({length: 32}).map((_, i) => ({
      sem_score: 0.5,
      type: "Base"
  }));

  return (
    <group>
        <Text position={[0, 6, 0]} fontSize={0.6} color="#fff" anchorX="center">
            {t ? t('structure.fiber.title') : 'Neural Fiber Bundle Structure'}
        </Text>
        
        {/* Base Manifold Representation (Grid) */}
        <mesh rotation={[-Math.PI/2, 0, 0]} position={[0, -2, 0]}>
            <planeGeometry args={[12, 12, 24, 24]} />
            <meshStandardMaterial color="#335588" wireframe opacity={0.3} transparent />
        </mesh>
        
        {/* Render Layers as fibers rising from manifold */}
        {rsaData.map((layerStats, i) => {
            const x = (i - rsaData.length/2) * 0.8;
            const height = layerStats.sem_score * 8; // Height prop to semantic score
            
            // Fiber color based on dominance type
            let color = "#aaaaaa";
            if (layerStats.type === "Base") color = "#4488ff"; // Blue for logic
            if (layerStats.type === "Fiber") color = "#ff6b6b"; // Red for content
            if (layerStats.type === "Mixed") color = "#bb88ff"; // Purple mixed

            return (
                <group key={i} position={[x, -2, 0]}>
                    {/* The fiber line */}
                    <mesh position={[0, height/2, 0]}>
                        <cylinderGeometry args={[0.08, 0.08, height, 8]} />
                        <meshStandardMaterial color={color} opacity={0.8} transparent />
                    </mesh>

                    {/* Node on top */}
                    <mesh position={[0, height, 0]}>
                        <sphereGeometry args={[0.25]} />
                        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.6} />
                    </mesh>
                    
                    {/* Layer Index on base */}
                    <Text position={[0, 0.2, 0.5]} fontSize={0.25} color="#aaa">L{i}</Text>
                </group>
            );
        })}

        {/* Legend */}
        <group position={[5, 4, 0]}>
            <mesh position={[0, 0.5, 0]}>
                <sphereGeometry args={[0.2]} />
                <meshStandardMaterial color="#4488ff" />
            </mesh>
            <Text position={[0.5, 0.5, 0]} fontSize={0.3} color="#aaa" anchorX="left">Base (Logic)</Text>
            
            <mesh position={[0, 0, 0]}>
                <sphereGeometry args={[0.2]} />
                <meshStandardMaterial color="#ff6b6b" />
            </mesh>
            <Text position={[0.5, 0, 0]} fontSize={0.3} color="#aaa" anchorX="left">Fiber (Fact)</Text>
        </group>

        {/* Concept Steering Visualization */}
        {result.steering && (
            <group>
                {(() => {
                    const layerIdx = result.steering.layer_idx || 15;
                    const x = (layerIdx - rsaData.length/2) * 0.8;
                    const height = (rsaData[layerIdx]?.sem_score || 0.5) * 8;
                    const y = -2 + height;
                    
                    return (
                        <group position={[x, y + 1.5, 0]}>
                            {/* Steering Vector Arrow */}
                            <arrowHelper 
                                args={[new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 0), 2, 0xffff00, 0.5, 0.3]} 
                            />
                            <Text position={[0, 2.5, 0]} fontSize={0.4} color="#ffff00" anchorX="center">
                                Steering
                            </Text>
                            <Text position={[0, 3.0, 0]} fontSize={0.3} color="#aaa" anchorX="center">
                                ε={result.steering.strength || 1.0}
                            </Text>
                            
                            {/* Pulse Effect Ring */}
                            <mesh rotation={[Math.PI/2, 0, 0]}>
                                <ringGeometry args={[0.5, 0.6, 32]} />
                                <meshBasicMaterial color="#ffff00" transparent opacity={0.5} side={THREE.DoubleSide} />
                            </mesh>
                        </group>
                    );
                })()}
            </group>
        )}
    </group>
  );
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
            title: "AGI 理论验证",
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

  const runAgiVerification = () => runAnalysis('AGI Theory Verification', 'verify_agi', {}, (data, log) => {
      const baseCount = data.rsa?.filter(l => l.type === 'Base').length;
      log(`📊 Systematic Layers: ${baseCount}`);
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
                 { id: 'circuit', icon: Network, label: 'Circuit' },
                 { id: 'features', icon: Sparkles, label: 'Features' },
                 { id: 'causal', icon: Brain, label: 'Causal' },
                 { id: 'manifold', icon: Network, label: 'Manifold' },
                 { id: 'compositional', icon: Network, label: 'Compos' },
                 { id: 'agi', icon: Sparkles, label: 'AGI Theory' }
              ] : [
                 { id: 'snn', icon: Brain, label: 'Simulation' },
                 { id: 'validity', icon: Activity, label: 'Validity' }
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

            {activeTab === 'agi' && (
                <div className="animate-fade-in">
                    <ActionButton onClick={runAgiVerification} loading={loading} icon={Sparkles}>Verify AGI Theory</ActionButton>
                    <div style={{margin: '20px 0', borderTop: '1px solid rgba(255,255,255,0.1)'}} />
                    <ControlGroup label="Concept Steering">
                         <StyledTextArea rows={2} value={steeringForm.prompt} onChange={e => setSteeringForm({...steeringForm, prompt: e.target.value})} />
                         <div style={{marginTop: '10px'}}>
                             <input type="range" min="-5" max="5" step="0.5" value={steeringForm.strength} onChange={e => setSteeringForm({...steeringForm, strength: parseFloat(e.target.value)})} style={{ width: '100%', accentColor: '#4488ff' }} />
                         </div>
                    </ControlGroup>
                    <ActionButton onClick={runConceptSteering} loading={loading} icon={Network}>Run Steering</ActionButton>
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
                        <FiberBundleVisualization3D result={analysisResult} t={tSafe} /> :
                        <div style={{display:'flex', alignItems:'center', justifyContent:'center', height:'100%', color:'#666'}}>
                             Click "Verify AGI Theory" to generate fiber bundle visualization
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
