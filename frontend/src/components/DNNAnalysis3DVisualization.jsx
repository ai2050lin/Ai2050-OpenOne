import React, { useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Text, MeshTransmissionMaterial, MeshDistortMaterial } from '@react-three/drei';
import * as THREE from 'three';

/**
 * 3D神经元节点组件
 */
function NeuronNode3D({ 
  position, 
  size = 0.8, 
  color = '#4facfe', 
  label = '', 
  activation = 0.5,
  onHover,
  onClick 
}) {
  const [hovered, setHovered] = useState(false);
  
  return (
    <group position={position}>
      <mesh
        scale={hovered ? size * 1.2 : size}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHovered(true);
          onHover?.(label, activation);
        }}
        onPointerOut={(e) => {
          e.stopPropagation();
          setHovered(false);
        }}
        onClick={(e) => {
          e.stopPropagation();
          onClick?.(label, activation);
        }}
      >
        <sphereGeometry args={[1, 32, 32]} />
        <MeshDistortMaterial
          color={hovered ? '#ffffff' : color}
          roughness={0.2}
          metalness={0.8}
          distort={0.2}
          speed={2}
          envMapIntensity={0.5}
        />
      </mesh>
      
      {label && (
        <Text
          position={[0, size + 0.8, 0]}
          fontSize={0.3}
          color={hovered ? '#ffffff' : '#a0c4ff'}
          anchorX="center"
          anchorY="middle"
          visible={hovered}
        >
          {label}
        </Text>
      )}
      
      {/* 激活度指示器 */}
      <mesh position={[0, -size - 0.2, 0]}>
        <cylinderGeometry args={[0.1, 0.1, activation * 1.5, 8]} />
        <meshStandardMaterial 
          color={activation > 0.7 ? '#4ade80' : activation > 0.4 ? '#facc15' : '#f87171'} 
          emissive={activation > 0.7 ? '#4ade80' : activation > 0.4 ? '#facc15' : '#f87171'}
          emissiveIntensity={0.3}
        />
      </mesh>
    </group>
  );
}

/**
 * 3D连接组件
 */
function Connection3D({ 
  start, 
  end, 
  strength = 0.5, 
  animated = false 
}) {
  const midPoint = new THREE.Vector3()
    .addVectors(start, end)
    .multiplyScalar(0.5);
  
  const direction = new THREE.Vector3().subVectors(end, start);
  const length = direction.length();
  
  return (
    <group>
      {/* 主连接线 */}
      <mesh position={midPoint} quaternion={new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize())}>
        <cylinderGeometry args={[0.05 * strength, 0.05 * strength, length, 8]} />
        <MeshTransmissionMaterial
          color="#4facfe"
          roughness={0.1}
          transmission={0.9}
          thickness={0.5}
          emissive="#4facfe"
          emissiveIntensity={strength * 0.3}
        />
      </mesh>
      
      {animated && strength > 0.3 && (
        <group position={midPoint}>
          <mesh>
            <sphereGeometry args={[0.1 + strength * 0.1, 16, 16]} />
            <meshStandardMaterial
              color="#ffffff"
              emissive="#ffffff"
              emissiveIntensity={1}
            />
          </mesh>
        </group>
      )}
    </group>
  );
}

/**
 * DNN编码结构3D可视化组件
 */
function EncodingStructure3D({ data, onNodeClick }) {
  if (!data) return null;
  
  const { layers = 11, activations_per_layer = [] } = data;
  const neurons = [];
  const connections = [];
  
  // 创建层间神经元
  activations_per_layer.forEach((activationCount, layerIdx) => {
    const layerX = (layerIdx - layers / 2) * 2.5;
    const neuronCount = Math.min(activationCount, 50); // 限制每层显示的神经元数量
    const spacing = 0.8;
    const startY = -(neuronCount - 1) * spacing / 2;
    
    for (let i = 0; i < neuronCount; i++) {
      const activation = Math.random();
      const y = startY + i * spacing;
      const z = (Math.random() - 0.5) * 2;
      
      neurons.push({
        id: `layer-${layerIdx}-neuron-${i}`,
        position: [layerX, y, z],
        size: 0.3 + activation * 0.4,
        color: activation > 0.7 ? '#4ade80' : activation > 0.4 ? '#facc15' : '#4facfe',
        label: `L${layerIdx}-N${i}`,
        activation,
        layer: layerIdx
      });
    }
  });
  
  // 创建层间连接
  for (let i = 0; i < neurons.length; i++) {
    for (let j = 0; j < neurons.length; j++) {
      if (neurons[j].layer === neurons[i].layer + 1) {
        const strength = Math.random() * 0.5 + 0.2;
        connections.push({
          start: new THREE.Vector3(...neurons[i].position),
          end: new THREE.Vector3(...neurons[j].position),
          strength,
          animated: strength > 0.6
        });
      }
    }
  }
  
  return (
    <group>
      {neurons.map((neuron) => (
        <NeuronNode3D
          key={neuron.id}
          position={neuron.position}
          size={neuron.size}
          color={neuron.color}
          label={neuron.label}
          activation={neuron.activation}
          onClick={onNodeClick}
        />
      ))}
      
      {connections.map((conn, idx) => (
        <Connection3D
          key={idx}
          start={conn.start}
          end={conn.end}
          strength={conn.strength}
          animated={conn.animated}
        />
      ))}
    </group>
  );
}

/**
 * 注意力模式3D可视化组件
 */
function AttentionPattern3D({ data, onNodeClick }) {
  if (!data) return null;
  
  const { dominantLayers = [5, 10, 15, 20], activePatterns = 45 } = data;
  const nodes = [];
  const connections = [];
  
  // 创建层节点
  dominantLayers.forEach((layer, idx) => {
    const angle = (idx / dominantLayers.length) * Math.PI * 2;
    const radius = 8;
    const x = Math.cos(angle) * radius;
    const y = (idx - dominantLayers.length / 2) * 1.5;
    const z = Math.sin(angle) * radius;
    
    nodes.push({
      id: `attention-layer-${layer}`,
      position: [x, y, z],
      size: 1.2,
      color: '#f472b6',
      label: `L${layer}`,
      activation: 0.8
    });
  });
  
  // 创建层间连接
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const strength = Math.random() * 0.7 + 0.3;
      connections.push({
        start: new THREE.Vector3(...nodes[i].position),
        end: new THREE.Vector3(...nodes[j].position),
        strength,
        animated: true
      });
    }
  }
  
  return (
    <group>
      {nodes.map((node) => (
        <NeuronNode3D
          key={node.id}
          position={node.position}
          size={node.size}
          color={node.color}
          label={node.label}
          activation={node.activation}
          onClick={onNodeClick}
        />
      ))}
      
      {connections.map((conn, idx) => (
        <Connection3D
          key={idx}
          start={conn.start}
          end={conn.end}
          strength={conn.strength}
          animated={conn.animated}
        />
      ))}
    </group>
  );
}

/**
 * 特征提取3D可视化组件
 */
function FeatureExtraction3D({ data, onNodeClick }) {
  if (!data) return null;
  
  const { identifiedFeatures = 389, totalFeatures = 512, confidence = 0.85 } = data;
  const features = [];
  const clusters = Math.ceil(identifiedFeatures / 50);
  
  // 创建特征簇
  for (let i = 0; i < clusters; i++) {
    const angle = (i / clusters) * Math.PI * 2;
    const radius = 6 + Math.random() * 4;
    const x = Math.cos(angle) * radius;
    const y = (i - clusters / 2) * 2;
    const z = Math.sin(angle) * radius;
    
    const featuresInCluster = Math.min(identifiedFeatures - i * 50, 50);
    
    for (let j = 0; j < featuresInCluster; j++) {
      const offsetX = (Math.random() - 0.5) * 2;
      const offsetY = (Math.random() - 0.5) * 2;
      const offsetZ = (Math.random() - 0.5) * 2;
      
      features.push({
        id: `feature-${i}-${j}`,
        position: [x + offsetX, y + offsetY, z + offsetZ],
        size: 0.2 + Math.random() * 0.3,
        color: ['#a78bfa', '#60a5fa', '#34d399', '#fbbf24'][i % 4],
        label: `F${i * 50 + j}`,
        activation: Math.random()
      });
    }
  }
  
  return (
    <group>
      {features.map((feature) => (
        <NeuronNode3D
          key={feature.id}
          position={feature.position}
          size={feature.size}
          color={feature.color}
          label={feature.label}
          activation={feature.activation}
          onClick={onNodeClick}
        />
      ))}
    </group>
  );
}

/**
 * 层间动力学3D可视化组件
 */
function LayerDynamics3D({ data, onNodeClick }) {
  if (!data) return null;
  
  const { criticalLayers = [8, 16, 24, 30], propagationSpeed = 0.65, informationLoss = 0.12 } = data;
  const layers = [];
  const flowParticles = [];
  
  // 创建关键层
  criticalLayers.forEach((layer, idx) => {
    const x = (idx - criticalLayers.length / 2) * 4;
    
    layers.push({
      id: `critical-layer-${layer}`,
      position: [x, 0, 0],
      size: 1.5,
      color: '#f97316',
      label: `L${layer}`,
      activation: 0.9
    });
    
    // 创建流动粒子
    const particleCount = 20;
    for (let i = 0; i < particleCount; i++) {
      const progress = (Date.now() % 5000 + i * 250) / 5000;
      const currentX = x + progress * 4;
      const y = (Math.random() - 0.5) * 3;
      const z = (Math.random() - 0.5) * 3;
      
      if (idx < criticalLayers.length - 1) {
        flowParticles.push({
          id: `flow-${idx}-${i}`,
          position: [currentX, y, z],
          size: 0.15,
          color: '#fb923c',
          label: '',
          activation: 0.7
        });
      }
    }
  });
  
  return (
    <group>
      {layers.map((layer) => (
        <NeuronNode3D
          key={layer.id}
          position={layer.position}
          size={layer.size}
          color={layer.color}
          label={layer.label}
          activation={layer.activation}
          onClick={onNodeClick}
        />
      ))}
      
      {flowParticles.map((particle) => (
        <NeuronNode3D
          key={particle.id}
          position={particle.position}
          size={particle.size}
          color={particle.color}
          label={particle.label}
          activation={particle.activation}
        />
      ))}
    </group>
  );
}

/**
 * 神经元分组3D可视化组件
 */
function NeuronGroups3D({ data, onNodeClick }) {
  if (!data) return null;
  
  const { totalGroups = 12, averageGroupSize = 170, functionalRoles = ['encoding', 'routing', 'output'] } = data;
  const groups = [];
  
  // 创建神经元分组
  functionalRoles.forEach((role, roleIdx) => {
    const groupsInRole = Math.ceil(totalGroups / functionalRoles.length);
    const angleBase = (roleIdx / functionalRoles.length) * Math.PI * 2;
    
    for (let i = 0; i < groupsInRole; i++) {
      const angle = angleBase + (i / groupsInRole) * (Math.PI * 2 / functionalRoles.length);
      const radius = 4 + Math.random() * 4;
      const x = Math.cos(angle) * radius;
      const y = (roleIdx - functionalRoles.length / 2) * 3;
      const z = Math.sin(angle) * radius;
      
      const roleColors = {
        encoding: '#22c55e',
        routing: '#3b82f6',
        output: '#f59e0b',
        attention: '#ec4899',
        memory: '#8b5cf6'
      };
      
      groups.push({
        id: `group-${role}-${i}`,
        position: [x, y, z],
        size: 0.8 + Math.random() * 0.4,
        color: roleColors[role] || '#6b7280',
        label: `${role}-${i}`,
        activation: Math.random()
      });
    }
  });
  
  return (
    <group>
      {groups.map((group) => (
        <NeuronNode3D
          key={group.id}
          position={group.position}
          size={group.size}
          color={group.color}
          label={group.label}
          activation={group.activation}
          onClick={onNodeClick}
        />
      ))}
    </group>
  );
}

/**
 * DNN分析3D可视化主组件
 */
export default function DNNAnalysis3DVisualization({ 
  dimension = 'encoding_structure', 
  data = null,
  onNodeClick 
}) {
  const [hoveredNode, setHoveredNode] = useState(null);
  
  const renderVisualization = () => {
    switch (dimension) {
      case 'encoding_structure':
        return <EncodingStructure3D data={data} onNodeClick={onNodeClick} />;
      case 'attention_patterns':
        return <AttentionPattern3D data={data} onNodeClick={onNodeClick} />;
      case 'feature_extractions':
        return <FeatureExtraction3D data={data} onNodeClick={onNodeClick} />;
      case 'layer_dynamics':
        return <LayerDynamics3D data={data} onNodeClick={onNodeClick} />;
      case 'neuron_groups':
        return <NeuronGroups3D data={data} onNodeClick={onNodeClick} />;
      default:
        return null;
    }
  };
  
  return (
    <div style={{ width: '100%', height: '100%', minHeight: '400px' }}>
      <Canvas shadows>
        <PerspectiveCamera makeDefault position={[0, 0, 20]} fov={50} />
        <OrbitControls enableDamping dampingFactor={0.05} />
        
        <ambientLight intensity={0.4} />
        <pointLight position={[10, 10, 10]} intensity={0.8} />
        <pointLight position={[-10, -10, 10]} intensity={0.3} color="#00d2ff" />
        <pointLight position={[0, 0, -10]} intensity={0.2} color="#ff6b9d" />
        
        <color attach="background" args={['#090b15']} />
        
        {renderVisualization()}
        
        {hoveredNode && (
          <Text
            position={[0, 8, 0]}
            fontSize={0.6}
            color="#ffffff"
            anchorX="center"
            anchorY="middle"
          >
            {hoveredNode.label || ''}
          </Text>
        )}
      </Canvas>
    </div>
  );
}