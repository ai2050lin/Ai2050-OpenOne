import { Line, Sphere, Text } from '@react-three/drei';
import axios from 'axios';
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

const API_BASE = 'http://localhost:5002';

export default function TrainingDynamics3D({ t }) {
  const [metrics, setMetrics] = useState({ Transformer: [], FiberNet: [] });
  const [ricciData, setRicciData] = useState([]);
  const [active, setActive] = useState(true);
  const chartGroupRef = useRef();

  useEffect(() => {
    let interval;
    if (active) {
      interval = setInterval(async () => {
        try {
          // 1. Fetch Training Metrics
          const res = await axios.get(`${API_BASE}/toy_experiment/metrics`);
          if (res.data.status === 'success') {
            setMetrics(res.data.data);
          }
          // 2. Fetch Ricci Flow Data
          const ricciRes = await axios.get(`${API_BASE}/toy_experiment/ricci_metrics`);
          if (ricciRes.data.status === 'success') {
            setRicciData(ricciRes.data.data);
          }
        } catch (err) {
          console.error("Failed to fetch training metrics:", err);
        }
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [active]);

  const renderCurve = (data, color, label, offsetZ) => {
    if (!data || data.length < 2) return null;
    const windowSize = 100;
    const displayData = data.slice(-windowSize);
    const points = displayData.map((d, i) => {
      const x = (i / windowSize) * 20 - 10;
      const y = (d.accuracy / 100) * 8 - 4;
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
        <Text position={[11, points[points.length - 1]?.y || 0, offsetZ]} fontSize={0.5} color={color} anchorX="left">
          {label}: {data[data.length - 1].accuracy.toFixed(1)}%
        </Text>
      </group>
    );
  };

  const renderRicciManifold = () => {
    if (!ricciData || ricciData.length === 0) return null;
    const latest = ricciData[ricciData.length - 1];
    const points = latest.manifold.map((y, i) => {
      const x = (i / latest.manifold.length) * 20 - 10;
      return new THREE.Vector3(x, y * 2, -5); // Positioned in background
    });
    return (
      <group position={[0, -2, 0]}>
        <Line points={points} color="#00ffcc" lineWidth={3} transparent opacity={0.6} />
        <Text position={[-10, 4, -5]} fontSize={0.4} color="#00ffcc">Manifold Smoothing (Ricci Flow)</Text>
        <Text position={[11, -1, -5]} fontSize={0.6} color="#00ffcc" anchorX="left">
          Ω (Curvature): {latest.avg_curvature.toFixed(6)}
        </Text>
      </group>
    );
  };

  return (
    <group ref={chartGroupRef}>
      <Text position={[0, 7, 0]} fontSize={0.8} color="#fff" anchorX="center">
        {t ? t('training.dynamics.title') || 'AGI Training Dynamics' : 'AGI Training Dynamics'}
      </Text>

      {/* Grid Floor */}
      <gridHelper args={[20, 20, 0x444444, 0x222222]} rotation={[Math.PI / 2, 0, 0]} position={[0, 0, -1]} />

      {/* Curves */}
      {renderCurve(metrics.Transformer, "#ff4444", "Transformer", 0)}
      {renderCurve(metrics.FiberNet, "#4488ff", "FiberNet", 1)}
      
      {/* Ricci Flow Visualization */}
      {renderRicciManifold()}

      {/* HUD Info */}
      <group position={[-11, -6, 0]}>
        <Text fontSize={0.3} color="#888" anchorX="left">Mode: Non-Abelian Ricci Optimization</Text>
      </group>
    </group>
  );
}
