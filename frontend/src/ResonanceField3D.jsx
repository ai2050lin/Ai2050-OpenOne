
import { Sphere } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import { useMemo, useRef } from 'react';
import * as THREE from 'three';

// --- 核心 Shader：泛几何共振场 (PGRF) ---
const ResonanceShader = {
  uniforms: {
    uTime: { value: 0 },
    uColor: { value: new THREE.Color("#4488ff") },
    uResolution: { value: new THREE.Vector2() },
    uIntensity: { value: 1.0 },
    uCurvature: { value: 0.01 }, 
    uMode: { value: 0.0 }, // 0: Normal, 1: RPT (Steady), 2: Curvature (Vibrant), 3: Circuit (Logical)
    uActiveNodes: { value: [] },   
  },
  vertexShader: `
    varying vec2 vUv;
    varying vec3 vPosition;
    varying vec3 vNormal;
    uniform float uTime;
    uniform float uCurvature;
    uniform float uMode;

    void main() {
      vUv = uv;
      vPosition = position;
      vNormal = normal;
      
      vec3 pos = position;
      // 动力学波动：RPT模式下更平稳，曲率模式下波动更剧烈
      float speed = uMode == 2.0 ? 2.5 : 1.0;
      float amplitude = (0.05 + uCurvature) * (uMode == 1.0 ? 0.3 : 1.0);
      
      float displacement = sin(pos.x * 3.0 + uTime * speed) * cos(pos.y * 3.0 + uTime * speed) * amplitude;
      pos += normal * displacement;
      
      gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
    }
  `,
  fragmentShader: `
    varying vec2 vUv;
    varying vec3 vPosition;
    varying vec3 vNormal;
    uniform float uTime;
    uniform vec3 uColor;
    uniform float uIntensity;
    uniform float uCurvature;
    uniform float uMode;

    void main() {
      // 波动场干涉图样
      float pulse = sin(vPosition.z * 8.0 - uTime * 2.0) * 0.5 + 0.5;
      
      // 颜色模态映射
      vec3 baseColor = uColor;
      if (uMode == 1.0) baseColor = vec3(0.2, 0.6, 1.0); // RPT: Deep Blue
      if (uMode == 2.0) baseColor = vec3(1.0, 0.4, 0.2); // Curvature: Hot Orange
      if (uMode == 3.0) baseColor = vec3(0.4, 1.0, 0.7); // Circuit: Matrix Green
      
      // 熵增视觉表现：曲率大时颜色偏红且波动杂乱
      vec3 finalColor = mix(baseColor, vec3(1.0, 0.2, 0.1), clamp(uCurvature * 15.0, 0.0, 1.0));
      
      float alpha = (0.15 + pulse * 0.25) * uIntensity;
      
      // 边缘增强（Fresnel效应），增强玻璃球体感
      float fresnel = pow(1.0 - dot(normalize(vNormal), vec3(0,0,1)), 3.0);
      alpha += fresnel * 0.6;

      gl_FragColor = vec4(finalColor, alpha);
    }
  `
};

export default function ResonanceField3D({ topologyResults, activeTab }) {
  const meshRef = useRef();
  
  const modeMap = {
    'global_topology': 0.0,
    'rpt': 1.0,
    'curvature': 2.0,
    'circuit': 3.0,
    'causal': 3.0
  };

  const avgCurvature = useMemo(() => {
    if (!topologyResults?.summary) return 0.005;
    const summaries = Object.values(topologyResults.summary);
    if (summaries.length === 0) return 0.005;
    return summaries.reduce((acc, s) => acc + s.avg_ortho_error, 0) / summaries.length;
  }, [topologyResults]);

  useFrame((state) => {
    if (meshRef.current) {
      const material = meshRef.current.material;
      material.uniforms.uTime.value = state.clock.elapsedTime;
      material.uniforms.uCurvature.value = THREE.MathUtils.lerp(
        material.uniforms.uCurvature.value,
        avgCurvature,
        0.05
      );
      material.uniforms.uMode.value = THREE.MathUtils.lerp(
        material.uniforms.uMode.value,
        modeMap[activeTab] || 0.0,
        0.1
      );
    }
  });

  return (
    <group>
      {/* 核心观测场：代表流形背景 */}
      <Sphere ref={meshRef} args={[15, 64, 64]}>
        <shaderMaterial
          attach="material"
          uniforms={THREE.UniformsUtils.clone(ResonanceShader.uniforms)}
          vertexShader={ResonanceShader.vertexShader}
          fragmentShader={ResonanceShader.fragmentShader}
          transparent
          depthWrite={false}
          side={THREE.DoubleSide}
        />
      </Sphere>

      <ambientLight intensity={0.2} />
      <pointLight position={[15, 15, 15]} intensity={1} color="#ffffff" />
      
      <Stars radius={120} depth={60} count={3000} factor={4} saturation={0} fade speed={0.5} />
    </group>
  );
}

function Stars({ radius, depth, count, factor, saturation, fade, speed }) {
  const mesh = useRef();
  const [geo, mat, coords] = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const mat = new THREE.PointsMaterial({ size: 0.1, color: "white", transparent: true, opacity: 0.5 });
    const coords = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
        const r = radius + (Math.random() - 0.5) * depth;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        coords[i * 3] = r * Math.sin(phi) * Math.cos(theta);
        coords[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
        coords[i * 3 + 2] = r * Math.cos(phi);
    }
    geo.setAttribute('position', new THREE.BufferAttribute(coords, 3));
    return [geo, mat, coords];
  }, [count, radius, depth]);

  return <points ref={mesh} geometry={geo} material={mat} />;
}
