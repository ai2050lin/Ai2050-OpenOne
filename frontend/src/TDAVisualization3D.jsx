// TDAVisualization3D.jsx - 拓扑数据分析 3D 可视化组件
// 展示持久同调 (Persistent Homology) 的贝蒂数和条形码图

import { Line, Text } from '@react-three/drei';
import { useMemo } from 'react';

/**
 * 3D Barcode 可视化
 * - 0维: 连通分量 (蓝色)
 * - 1维: 环/孔 (红色)
 */
export default function TDAVisualization3D({ result, t }) {
  // 安全获取数据
  const ph0d = result?.ph_0d || [];
  const ph1d = result?.ph_1d || [];
  const status = result?.status || 'not_computed';

  // 计算贝蒂数
  const betti0 = ph0d.length;
  const betti1 = ph1d.length;

  // 生成条形码数据 (模拟或真实)
  const barcodeData = useMemo(() => {
    const bars0 = ph0d.map((interval, idx) => ({
      birth: interval[0] || 0,
      death: interval[1] === null ? Infinity : (interval[1] || 1),
      dim: 0,
      idx
    }));
    
    const bars1 = ph1d.map((interval, idx) => ({
      birth: interval[0] || 0,
      death: interval[1] === null ? Infinity : (interval[1] || 1),
      dim: 1,
      idx
    }));
    
    return { bars0, bars1 };
  }, [ph0d, ph1d]);

  // 生成流形示意点
  const manifoldPoints = useMemo(() => {
    const points = [];
    // 如果有连通分量，在空间中生成对应的点云
    for (let i = 0; i < betti0; i++) {
      const angle = (i / Math.max(betti0, 1)) * Math.PI * 2;
      const radius = 4 + i * 0.3;
      points.push({
        position: [Math.cos(angle) * radius, 0, Math.sin(angle) * radius],
        color: `hsl(${200 + i * 30}, 70%, 60%)`
      });
    }
    return points;
  }, [betti0]);

  // 生成环结构
  const loopGeometries = useMemo(() => {
    const loops = [];
    for (let i = 0; i < Math.min(betti1, 5); i++) {
      const offsetY = (i - Math.min(betti1, 5) / 2) * 2;
      const radius = 2 + i * 0.5;
      loops.push({
        position: [0, offsetY + 5, 0],
        radius,
        color: `hsl(${0 + i * 40}, 80%, 55%)`
      });
    }
    return loops;
  }, [betti1]);

  // 生成条形码线条
  const barcodeLines = useMemo(() => {
    const lines = [];
    const maxLife = Math.max(
      ...barcodeData.bars0.map(b => b.death === Infinity ? b.birth + 1 : b.death),
      ...barcodeData.bars1.map(b => b.death === Infinity ? b.birth + 1 : b.death),
      1
    );

    // 0维条形码 (左侧)
    barcodeData.bars0.forEach((bar, idx) => {
      const y = -6 + idx * 0.4;
      const x1 = -8 + (bar.birth / maxLife) * 6;
      const x2 = -8 + (Math.min(bar.death, maxLife) / maxLife) * 6;
      lines.push({
        start: [x1, y, 5],
        end: [x2, y, 5],
        color: '#00aaff',
        dim: 0,
        isPersistent: bar.death === Infinity
      });
    });

    // 1维条形码 (右侧)
    barcodeData.bars1.forEach((bar, idx) => {
      const y = -6 + idx * 0.4;
      const x1 = 2 + (bar.birth / maxLife) * 6;
      const x2 = 2 + (Math.min(bar.death, maxLife) / maxLife) * 6;
      lines.push({
        start: [x1, y, 5],
        end: [x2, y, 5],
        color: '#ff4444',
        dim: 1,
        isPersistent: bar.death === Infinity
      });
    });

    return lines;
  }, [barcodeData]);

  if (status === 'not_computed') {
    return (
      <group>
        <Text position={[0, 2, 0]} fontSize={1} color="#666" anchorX="center">
          拓扑特征尚未计算
        </Text>
        <Text position={[0, 0, 0]} fontSize={0.6} color="#444" anchorX="center">
          请点击 "获取拓扑特征" 按钮
        </Text>
      </group>
    );
  }

  if (status === 'error') {
    return (
      <group>
        <Text position={[0, 2, 0]} fontSize={1} color="#ff4444" anchorX="center">
          计算出错
        </Text>
        <Text position={[0, 0, 0]} fontSize={0.5} color="#888" anchorX="center">
          {result?.error || '未知错误'}
        </Text>
      </group>
    );
  }

  return (
    <group>
      {/* 标题 */}
      <Text position={[0, 10, 0]} fontSize={1.2} color="#e056fd" anchorX="center" fontWeight="bold">
        拓扑数据分析 (Persistent Homology)
      </Text>

      {/* 贝蒂数显示 */}
      <group position={[-8, 8, 0]}>
        <Text position={[0, 0, 0]} fontSize={0.8} color="#00aaff" anchorX="left">
          β₀ = {betti0} (连通分量)
        </Text>
      </group>
      <group position={[2, 8, 0]}>
        <Text position={[0, 0, 0]} fontSize={0.8} color="#ff4444" anchorX="left">
          β₁ = {betti1} (环/孔)
        </Text>
      </group>

      {/* 流形可视化区域 - 中心 */}
      <group position={[0, 3, -5]}>
        {/* 底流形平面 */}
        <mesh rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[12, 12, 20, 20]} />
          <meshBasicMaterial color="#1a1a2e" wireframe transparent opacity={0.3} />
        </mesh>

        {/* 连通分量点 */}
        {manifoldPoints.map((point, idx) => (
          <mesh key={`comp-${idx}`} position={point.position}>
            <sphereGeometry args={[0.4, 16, 16]} />
            <meshStandardMaterial color={point.color} emissive={point.color} emissiveIntensity={0.5} />
          </mesh>
        ))}

        {/* 环结构 */}
        {loopGeometries.map((loop, idx) => (
          <group key={`loop-${idx}`} position={loop.position}>
            <mesh rotation={[Math.PI / 2, 0, 0]}>
              <torusGeometry args={[loop.radius, 0.15, 16, 32]} />
              <meshStandardMaterial 
                color={loop.color} 
                emissive={loop.color} 
                emissiveIntensity={0.4}
                transparent
                opacity={0.8}
              />
            </mesh>
          </group>
        ))}

        {/* 标签 */}
        <Text position={[0, -1.5, 6]} fontSize={0.5} color="#888" anchorX="center">
          激活流形 (Activation Manifold)
        </Text>
      </group>

      {/* 条形码图 - 底部 */}
      <group position={[0, -3, 5]}>
        {/* 0维条形码标签 */}
        <Text position={[-5, 2, 0]} fontSize={0.6} color="#00aaff" anchorX="center">
          0维 (Components)
        </Text>
        
        {/* 1维条形码标签 */}
        <Text position={[5, 2, 0]} fontSize={0.6} color="#ff4444" anchorX="center">
          1维 (Loops)
        </Text>

        {/* 渲染条形码线条 */}
        {barcodeLines.map((bar, idx) => (
          <Line
            key={`bar-${idx}`}
            points={[bar.start, bar.end]}
            color={bar.color}
            lineWidth={bar.isPersistent ? 4 : 2}
            transparent
            opacity={bar.isPersistent ? 1 : 0.7}
          />
        ))}

        {/* 无数据提示 */}
        {barcodeLines.length === 0 && (
          <Text position={[0, -2, 0]} fontSize={0.5} color="#666" anchorX="center">
            暂无持久性数据
          </Text>
        )}
      </group>

      {/* 说明文字 */}
      <Text position={[0, -10, 0]} fontSize={0.4} color="#555" anchorX="center" maxWidth={20}>
        持久同调揭示了激活空间的拓扑结构。连通分量(β₀)表示独立的概念簇，环(β₁)表示语义关系的循环依赖。
      </Text>
    </group>
  );
}
