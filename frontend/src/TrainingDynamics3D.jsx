import { Line, Sphere, Text } from '@react-three/drei';
import axios from 'axios';
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

const API_BASE = 'http://localhost:5000';

export default function TrainingDynamics3D({ t }) {
  const [metrics, setMetrics] = useState({ Transformer: [], FiberNet: [] });
  const [active, setActive] = useState(true);
  const chartGroupRef = useRef();

  useEffect(() => {
    let interval;
    if (active) {
      interval = setInterval(async () => {
        try {
          const res = await axios.get(`${API_BASE}/toy_experiment/metrics`);
          if (res.data.status === 'success') {
            setMetrics(res.data.data);
          }
        } catch (err) {
          console.error("Failed to fetch training metrics:", err);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [active]);

  const renderCurve = (data, color, label, offsetZ) => {
    if (!data || data.length < 2) return null;

    // We only show the last 100 points to avoid clutter
    const windowSize = 100;
    const displayData = data.slice(-windowSize);
    
    const points = displayData.map((d, i) => {
      const x = (i / windowSize) * 20 - 10;
      const y = (d.accuracy / 100) * 8 - 4; // Map 0-100% to -4 to 4
      return new THREE.Vector3(x, y, offsetZ);
    });

    return (
      <group>
        <Line points={points} color={color} lineWidth={2} />
        {points.length > 0 && (
          <Sphere args={[0.2]} position={points[points.length - 1]}>
            <meshStandardMaterial color={color} emissive={color} emissiveIntensity={2} />
          </Sphere>
        )}
        <Text
          position={[11, points[points.length - 1]?.y || 0, offsetZ]}
          fontSize={0.5}
          color={color}
          anchorX="left"
        >
          {label}: {data[data.length - 1].accuracy.toFixed(1)}%
        </Text>
      </group>
    );
  };

  return (
    <group ref={chartGroupRef}>
      <Text position={[0, 6, 0]} fontSize={0.8} color="#fff" anchorX="center">
        {t ? t('training.dynamics.title') || 'Training Dynamics: Curvature vs Accuracy' : 'Training Dynamics: Logic Emergence'}
      </Text>

      {/* Grid Floor */}
      <gridHelper args={[20, 20, 0x444444, 0x222222]} rotation={[Math.PI / 2, 0, 0]} position={[0, 0, -1]} />

      {/* Axes */}
      <Line points={[new THREE.Vector3(-10, -4, 0), new THREE.Vector3(10, -4, 0)]} color="#666" lineWidth={1} /> {/* X-axis */}
      <Line points={[new THREE.Vector3(-10, -4, 0), new THREE.Vector3(-10, 4, 0)]} color="#666" lineWidth={1} /> {/* Y-axis */}
      
      <Text position={[-11, 4, 0]} fontSize={0.3} color="#aaa" anchorX="right">100% Acc</Text>
      <Text position={[-11, -4, 0]} fontSize={0.3} color="#aaa" anchorX="right">0% Acc</Text>
      <Text position={[0, -5, 0]} fontSize={0.4} color="#888">Training Epochs (Time & Curvature Flow)</Text>

      {/* Curves */}
      {renderCurve(metrics.Transformer, "#ff4444", "Transformer", 0)}
      {renderCurve(metrics.FiberNet, "#4488ff", "FiberNet", 1)}

      {/* Logic Consistency indicator (Curvature) */}
      <group position={[12, -4, 0]}>
         <mesh position={[0, 4, 0]}>
            <boxGeometry args={[0.5, 8, 0.1]} />
            <meshBasicMaterial color="#111" />
         </mesh>
         {/* We fake curvature based on lack of accuracy for now, 
             but real FiberNet will report actual holonomy error */}
         {metrics.FiberNet.length > 0 && (
           <mesh position={[0, 4 * (1 - metrics.FiberNet[metrics.FiberNet.length-1].accuracy/100), 0]}>
              <boxGeometry args={[0.6, 0.2, 0.2]} />
              <meshBasicMaterial color="#4488ff" />
           </mesh>
         )}
         <Text position={[0, 8.5, 0]} fontSize={0.3} color="#4488ff" anchorX="center">LOGIC ERROR (Ω)</Text>
      </group>
    </group>
  );
}
