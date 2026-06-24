# 3D可视化客户端方案设计

> **版本**: v1.0 | **日期**: 2026-04-26
> **核心目标**: 测试脚本生成标准数据文件 → 3D客户端读取 → 实时3D可视化
> **技术栈**: React 19 + Three.js 0.182 + @react-three/fiber 9 + @react-three/drei 10 + Vite 7

---

## 一、设计理念

### 1.1 核心原则

1. **数据驱动**: 所有可视化从标准JSON文件读取，无需后端API
2. **文件优先**: 测试脚本输出标准格式JSON → 前端直接读取 → 零配置可视化
3. **交互探索**: 旋转/缩放/点击/悬停/过滤/动画 — 理解数据的每个维度
4. **实时反馈**: 数据文件更新后自动刷新，实验结果即刻可见

### 1.2 数据流架构

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ 测试脚本     │────▶│ 标准JSON数据文件  │────▶│ 3D可视化客户端   │
│ phase_*.py  │     │ vis_*.json       │     │ React+Three.js  │
└─────────────┘     └──────────────────┘     └─────────────────┘
     生成                标准格式                 读取+渲染
```

无需后端服务器！前端直接通过Vite的静态文件服务或`fetch()`读取JSON。

---

## 二、标准数据文件格式规范

### 2.1 顶层结构

每个实验输出一个JSON文件，放在 `results/vis_data/` 目录下：

```json
{
  "schema_version": "1.0",
  "phase": "CCLXIV",
  "experiment": "deep_locking_delta_injection",
  "model": "qwen3",
  "timestamp": "2026-04-26T22:30:00",
  "model_info": {
    "class": "Qwen3ForCausalLM",
    "n_layers": 36,
    "d_model": 2560,
    "n_heads": 20
  },
  "visualizations": [ ... ],   // 可视化对象数组（核心）
  "summary": { ... }           // 人类可读的摘要
}
```

### 2.2 visualization对象类型

每个visualization对象有一个`type`字段，决定3D渲染方式：

#### Type 1: `trajectory` — Token轨迹（最核心）

```json
{
  "type": "trajectory",
  "id": "dog_to_apple_delta",
  "label": "dog→apple 差分注入",
  "token": "apple",
  "source_token": "dog",
  "template": "The {} is",
  "points": [
    {
      "layer": 0,
      "x": 1.23, "y": -0.45, "z": 2.10,   // PCA/UMAP降维后的3D坐标
      "norm": 12.5,
      "cos_with_target": 0.35,
      "cos_with_source": 0.89,
      "delta_cos": 0.95
    },
    {
      "layer": 1,
      "x": 1.15, "y": -0.30, "z": 1.98,
      "norm": 15.2,
      "cos_with_target": 0.40,
      "cos_with_source": 0.82,
      "delta_cos": 0.77
    }
    // ... 逐层数据
  ],
  "color": "#ff6b6b",          // 轨迹颜色
  "correction_layers": [3, 8]  // 纠正层标记
}
```

**3D渲染**: 点+线连接的3D轨迹，颜色渐变表示delta_cos衰减

#### Type 2: `point_cloud` — 语义空间点云

```json
{
  "type": "point_cloud",
  "id": "layer18_concepts",
  "label": "L18 概念空间",
  "layer": 18,
  "points": [
    {
      "token": "apple",
      "category": "fruit",
      "x": 1.23, "y": -0.45, "z": 2.10,
      "norm": 156.3,
      "activation": 0.92    // 概念激活强度
    },
    {
      "token": "dog",
      "category": "animal",
      "x": -2.10, "y": 0.80, "z": -1.50,
      "norm": 148.7,
      "activation": 0.88
    }
    // ... 更多概念
  ],
  "categories": {
    "fruit": "#ff6b6b",
    "animal": "#4ecdc4",
    "vehicle": "#ffe66d",
    "tool": "#a855f7"
  }
}
```

**3D渲染**: 球体点云，大小=norm，颜色=category，悬停显示token名

#### Type 3: `heatmap_3d` — 层×概念×指标热力图

```json
{
  "type": "heatmap_3d",
  "id": "delta_cos_matrix",
  "label": "delta_cos衰减矩阵",
  "x_axis": {"label": "Layer", "values": [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35]},
  "y_axis": {"label": "Concept Pair", "values": ["dog→apple", "cat→hammer", "horse→rice", "eagle→ocean", "shark→desert", "snake→cheese"]},
  "z_axis": {"label": "delta_cos", "range": [0, 1]},
  "cells": [
    {"x": 0, "y": 0, "value": 0.95, "color": "#ff0000"},
    {"x": 1, "y": 0, "value": 0.77, "color": "#ff4444"},
    {"x": 2, "y": 0, "value": 0.53, "color": "#ff8888"}
    // ... 所有(x,y)组合
  ]
}
```

**3D渲染**: 3D柱状图，高度=value，颜色=值映射

#### Type 4: `flow` — 注意力/信息流

```json
{
  "type": "flow",
  "id": "attention_L18",
  "label": "L18 注意力流",
  "layer": 18,
  "source_label": "Token位置",
  "target_label": "Token位置",
  "flows": [
    {
      "source": 0, "target": 3,
      "weight": 0.45,
      "head": 7,
      "color": "#4ecdc4"
    },
    {
      "source": 1, "target": 3,
      "weight": 0.32,
      "head": 7,
      "color": "#4ecdc4"
    }
    // ... 更多注意力权重
  ],
  "node_positions": [
    {"id": 0, "token": "The", "x": -4.0, "y": 0, "z": 0},
    {"id": 1, "token": "dog", "x": -2.0, "y": 0, "z": 0},
    {"id": 2, "token": "is", "x": 0.0, "y": 0, "z": 0},
    {"id": 3, "token": "running", "x": 2.0, "y": 0, "z": 0}
  ]
}
```

**3D渲染**: 弧线连接，粗细=weight，动画=粒子沿弧线流动

#### Type 5: `layer_stack` — 层堆叠模型

```json
{
  "type": "layer_stack",
  "id": "qwen3_full_model",
  "label": "Qwen3 全模型层结构",
  "n_layers": 36,
  "d_model": 2560,
  "layers": [
    {
      "layer": 0,
      "label": "Embedding",
      "function": "lexical",
      "color": "#ff6b6b",
      "metrics": {
        "avg_norm": 12.5,
        "avg_delta_cos": 0.95,
        "switch_rate": 0.83,
        "category_R2": 0.82
      }
    },
    {
      "layer": 18,
      "label": "Template Hotspot",
      "function": "syntactic",
      "color": "#ffe66d",
      "metrics": {
        "avg_norm": 156.3,
        "avg_delta_cos": 0.47,
        "switch_rate": 0.50,
        "category_R2": 0.45
      }
    }
    // ... 所有关键层
  ],
  "trajectories": [
    // 内嵌trajectory对象的ID引用
    "dog_to_apple_delta"
  ]
}
```

**3D渲染**: 水平层板堆叠，层板上显示概念点云，层间有轨迹弧线

---

## 三、3D可视化客户端架构

### 3.1 组件树

```
NeuralVis3DApp                     // 顶层入口
├── DataSourcePanel                // 数据源选择面板
│   ├── FileLoader                 // 加载本地JSON文件
│   ├── DataList                   // 已加载数据文件列表
│   └── DataLoader                 // 从results/目录扫描
├── SceneController                // 场景控制器
│   ├── ViewModeSelector           // 可视化模式选择
│   ├── LayerFilter                // 层过滤器
│   ├── CategoryFilter             // 类别过滤器
│   ├── AnimationControl           // 动画播放/暂停/速度
│   └── CameraPreset               // 相机预设（俯视/侧视/自由）
├── Vis3DCanvas                    // 3D渲染画布（核心）
│   ├── LayerStackRenderer         // 层堆叠渲染器
│   ├── TrajectoryRenderer         // 轨迹渲染器
│   ├── PointCloudRenderer         // 点云渲染器
│   ├── Heatmap3DRenderer          // 3D热力图渲染器
│   ├── FlowRenderer               // 信息流渲染器
│   ├── AxisHelper                 // 坐标轴+标注
│   └── GridHelper                 // 地面网格
├── DetailPanel                    // 详情面板
│   ├── LayerDetail                // 层详情（选中层后显示）
│   ├── TokenDetail                // Token详情（悬停时显示）
│   └── TrajectoryDetail           // 轨迹详情
└── MetricPanel                    // 指标面板
    ├── DeltaCosChart              // delta_cos衰减曲线(2D叠加)
    ├── SwitchRateChart            // switch_rate柱状图
    └── CrossModelCompare          // 跨模型对比
```

### 3.2 核心渲染器设计

#### LayerStackRenderer — 层堆叠

```jsx
// 概念: 水平透明层板从下到上排列，间距=LAYER_GAP
// 每层板显示该层的概念点云
// 层板颜色由功能决定: 词法=红, 语义=蓝, 语法=黄, 决策=紫

function LayerStackRenderer({ layerStack, selectedLayers }) {
  return (
    <group>
      {layerStack.layers
        .filter(l => selectedLayers.includes(l.layer))
        .map((layer, i) => (
          <group key={layer.layer} position={[0, i * LAYER_GAP, 0]}>
            {/* 透明层板 */}
            <mesh>
              <planeGeometry args={[PLANE_SIZE, PLANE_SIZE]} />
              <meshStandardMaterial 
                color={layer.color} 
                transparent 
                opacity={0.08} 
                side={THREE.DoubleSide} 
              />
            </mesh>
            {/* 层标签 */}
            <Text position={[-PLANE_SIZE/2 - 1, 0, 0]} fontSize={0.5}>
              L{layer.layer} - {layer.label}
            </Text>
            {/* 该层的概念点 */}
            {layer.point_cloud && layer.point_cloud.map((pt, j) => (
              <TokenSphere key={j} point={pt} layer={layer.layer} />
            ))}
          </group>
        ))}
    </group>
  );
}
```

#### TrajectoryRenderer — 轨迹渲染

```jsx
// 概念: 3D空间中的曲线+球标记
// 颜色渐变: delta_cos=1(红) → delta_cos=0(蓝)
// 球大小: norm归一化
// 纠正层: 标记为闪烁的环

function TrajectoryRenderer({ trajectory, animated, animationProgress }) {
  const points = trajectory.points;
  const visiblePoints = animated 
    ? points.filter((_, i) => i / points.length <= animationProgress)
    : points;
  
  // 颜色插值: delta_cos 1→0 映射为 红→蓝
  const getColor = (deltaCos) => {
    const r = deltaCos;
    const b = 1 - deltaCos;
    return new THREE.Color(r, 0.2, b);
  };
  
  return (
    <group>
      {/* 轨迹线 */}
      <Line
        points={visiblePoints.map(p => [p.x, p.y, p.z])}
        color="white"
        lineWidth={2}
        transparent
        opacity={0.6}
      />
      {/* 逐层标记球 */}
      {visiblePoints.map((pt, i) => (
        <group key={i} position={[pt.x, pt.y, pt.z]}>
          <mesh>
            <sphereGeometry args={[0.15 + pt.norm * 0.001, 16, 16]} />
            <meshStandardMaterial 
              color={getColor(pt.delta_cos)}
              emissive={getColor(pt.delta_cos)}
              emissiveIntensity={0.5}
            />
          </mesh>
          {/* 纠正层特殊标记 */}
          {trajectory.correction_layers?.includes(pt.layer) && (
            <mesh rotation={[Math.PI/2, 0, 0]}>
              <torusGeometry args={[0.4, 0.05, 8, 32]} />
              <meshStandardMaterial color="#ffff00" emissive="#ffff00" emissiveIntensity={2} />
            </mesh>
          )}
        </group>
      ))}
    </group>
  );
}
```

#### PointCloudRenderer — 点云渲染

```jsx
// 概念: InstancedMesh批量渲染，高效支持1000+点
// 大小 = norm归一化, 颜色 = category
// 悬停: 显示token名+指标tooltip

function PointCloudRenderer({ pointCloud }) {
  const meshRef = useRef();
  const colorMap = pointCloud.categories;
  
  // 使用InstancedMesh批量渲染
  const { count, positions, colors, sizes } = useMemo(() => {
    const n = pointCloud.points.length;
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    const siz = new Float32Array(n);
    
    pointCloud.points.forEach((pt, i) => {
      pos[i*3] = pt.x;
      pos[i*3+1] = pt.y;
      pos[i*3+2] = pt.z;
      
      const c = new THREE.Color(colorMap[pt.category] || '#ffffff');
      col[i*3] = c.r; col[i*3+1] = c.g; col[i*3+2] = c.b;
      
      siz[i] = 0.2 + pt.norm * 0.002;
    });
    
    return { count: n, positions: pos, colors: col, sizes: siz };
  }, [pointCloud]);
  
  return (
    <instancedMesh ref={meshRef} args={[null, null, count]}>
      <sphereGeometry args={[1, 12, 12]} />
      <meshStandardMaterial vertexColors />
    </instancedMesh>
  );
}
```

### 3.3 动画系统

```jsx
// 三种动画模式:
// 1. 逐层传播: 轨迹从L0逐层生长到L35，delta_cos从红渐变到蓝
// 2. 信息流: 粒子沿注意力弧线从source流向target
// 3. 层板脉动: 选中某概念后，各层该概念的激活强度以脉动动画呈现

function AnimationController({ mode, speed, onComplete }) {
  const [progress, setProgress] = useState(0);
  const [playing, setPlaying] = useState(false);
  
  useFrame((state, delta) => {
    if (!playing) return;
    setProgress(prev => {
      const next = prev + delta * speed * 0.1;
      if (next >= 1) {
        setPlaying(false);
        onComplete?.();
        return 1;
      }
      return next;
    });
  });
  
  return { progress, playing, play: () => setPlaying(true), pause: () => setPlaying(false) };
}
```

---

## 四、数据文件生成规范

### 4.1 测试脚本输出接口

每个测试脚本在实验完成后，调用标准化的数据导出函数：

```python
# 在 tests/glm5/vis_data_exporter.py 中定义

import json
import numpy as np
from datetime import datetime
from pathlib import Path

VIS_DATA_DIR = Path("results/vis_data")

def export_trajectory(phase, model, experiment_id, token, source_token, 
                      template, per_layer_data, correction_layers=None,
                      pca_coords=None):
    """导出trajectory类型数据
    
    Args:
        per_layer_data: list of {layer, norm, cos_with_target, cos_with_source, delta_cos}
        pca_coords: list of {x, y, z} — PCA降维后的3D坐标(可选)
    """
    # 如果没有PCA坐标，用指标构造伪3D坐标
    if pca_coords is None:
        pca_coords = []
        for d in per_layer_data:
            x = d["layer"] * 0.5                    # X轴=层号
            y = d["delta_cos"] * 10                  # Y轴=delta_cos
            z = d["cos_with_target"] * 5 - 2.5       # Z轴=cos_with_target
            pca_coords.append({"x": x, "y": y, "z": z})
    
    points = []
    for i, d in enumerate(per_layer_data):
        points.append({
            "layer": d["layer"],
            "x": round(pca_coords[i]["x"], 4),
            "y": round(pca_coords[i]["y"], 4),
            "z": round(pca_coords[i]["z"], 4),
            "norm": round(d["norm"], 2),
            "cos_with_target": round(d["cos_with_target"], 4),
            "cos_with_source": round(d["cos_with_source"], 4),
            "delta_cos": round(d["delta_cos"], 4)
        })
    
    vis = {
        "type": "trajectory",
        "id": f"{source_token}_to_{token}_delta",
        "label": f"{source_token}→{token} 差分注入",
        "token": token,
        "source_token": source_token,
        "template": template,
        "points": points,
        "color": "#ff6b6b",
        "correction_layers": correction_layers or []
    }
    return vis


def export_point_cloud(phase, model, experiment_id, layer, points_data, 
                       categories_colors=None):
    """导出point_cloud类型数据
    
    Args:
        points_data: list of {token, category, x, y, z, norm, activation}
    """
    vis = {
        "type": "point_cloud",
        "id": f"layer{layer}_concepts",
        "label": f"L{layer} 概念空间",
        "layer": layer,
        "points": points_data,
        "categories": categories_colors or {}
    }
    return vis


def export_heatmap_3d(phase, model, experiment_id, x_label, y_label, z_label,
                      x_values, y_values, cells):
    """导出heatmap_3d类型数据"""
    vis = {
        "type": "heatmap_3d",
        "id": experiment_id,
        "label": f"{z_label}矩阵",
        "x_axis": {"label": x_label, "values": x_values},
        "y_axis": {"label": y_label, "values": y_values},
        "z_axis": {"label": z_label, "range": [0, 1]},
        "cells": cells
    }
    return vis


def save_vis_file(phase, model, experiment, visualizations, model_info, summary=None):
    """保存标准可视化数据文件
    
    Args:
        visualizations: list of visualization objects
    """
    VIS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    data = {
        "schema_version": "1.0",
        "phase": phase,
        "experiment": experiment,
        "model": model,
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "model_info": model_info,
        "visualizations": visualizations,
        "summary": summary or {}
    }
    
    filename = f"vis_{phase}_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = VIS_DATA_DIR / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"  [VIS] 可视化数据已保存: {filepath}")
    return filepath
```

### 4.2 测试脚本集成示例

在 `phase_cclxiv_deep_locking.py` 的结果保存部分添加：

```python
from vis_data_exporter import export_trajectory, export_heatmap_3d, save_vis_file

# ... 实验代码 ...

# 导出可视化数据
visualizations = []
for pair_name, res in all_results.items():
    source, target = pair_name.split("→")
    
    # 构造逐层数据
    per_layer = []
    for li_str in sorted(res["per_layer_delta_cos"].keys(), key=int):
        li = int(li_str)
        per_layer.append({
            "layer": li,
            "norm": res["per_layer_norm"].get(li_str, 0),
            "cos_with_target": res["per_layer_cos_with_target"].get(li_str, 0),
            "cos_with_source": res["per_layer_cos_with_source"].get(li_str, 0),
            "delta_cos": res["per_layer_delta_cos"].get(li_str, 0),
        })
    
    vis = export_trajectory(
        phase="CCLXIV", model=args.model, 
        experiment_id=f"{source}_to_{target}_delta",
        token=target, source_token=source,
        template="The {} is",
        per_layer_data=per_layer,
        correction_layers=[cl[0] for cl in res.get("correction_layers", [])]
    )
    visualizations.append(vis)

# 导出热力图
heatmap_cells = []
for i, pair_name in enumerate(all_results.keys()):
    for j, li_str in enumerate(sorted(all_results[pair_name]["per_layer_delta_cos"].keys(), key=int)):
        heatmap_cells.append({
            "x": j, "y": i,
            "value": all_results[pair_name]["per_layer_delta_cos"][li_str],
        })

visualizations.append(export_heatmap_3d(
    phase="CCLXIV", model=args.model,
    experiment_id="delta_cos_matrix",
    x_label="Layer", y_label="Concept Pair", z_label="delta_cos",
    x_values=sorted(list(all_results[list(all_results.keys())[0]]["per_layer_delta_cos"].keys()), key=int),
    y_values=list(all_results.keys()),
    cells=heatmap_cells
))

save_vis_file("CCLXIV", args.model, "deep_locking", visualizations, 
              model_info={"class": model_class, "n_layers": n_layers, "d_model": d_model})
```

---

## 五、前端实现方案

### 5.1 项目结构

```
frontend/src/
├── neural_vis/                      # 3D可视化模块
│   ├── index.jsx                    # 入口: NeuralVis3DApp
│   ├── components/
│   │   ├── DataSourcePanel.jsx      # 数据源面板
│   │   ├── SceneController.jsx      # 场景控制
│   │   ├── DetailPanel.jsx          # 详情面板
│   │   ├── MetricPanel.jsx          # 指标面板
│   │   └── AnimationControl.jsx     # 动画控制
│   ├── renderers/
│   │   ├── LayerStackRenderer.jsx   # 层堆叠渲染器
│   │   ├── TrajectoryRenderer.jsx   # 轨迹渲染器
│   │   ├── PointCloudRenderer.jsx   # 点云渲染器
│   │   ├── Heatmap3DRenderer.jsx    # 3D热力图渲染器
│   │   ├── FlowRenderer.jsx         # 信息流渲染器
│   │   └── SceneHelpers.jsx         # 坐标轴/网格/标注
│   ├── hooks/
│   │   ├── useVisData.js            # 数据加载hook
│   │   ├── useAnimation.js          # 动画控制hook
│   │   └── useInteraction.js        # 交互状态hook
│   ├── utils/
│   │   ├── dataParser.js            # JSON解析+验证
│   │   ├── colorSchemes.js          # 颜色方案
│   │   └── pcaProject.js            # 3D坐标投影(备用)
│   └── styles/
│       └── neural_vis.css           # 样式
├── App.jsx                          # 主入口(添加路由)
└── main.jsx
```

### 5.2 数据加载方式

```javascript
// hooks/useVisData.js

import { useState, useEffect } from 'react';

const VIS_DATA_BASE = '/vis_data';  // Vite静态目录映射

export function useVisData() {
  const [dataFiles, setDataFiles] = useState([]);
  const [activeData, setActiveData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // 方式1: 扫描预定义的数据文件列表
  // (因为浏览器无法直接扫描目录，需要通过manifest文件)
  const loadDataManifest = async () => {
    try {
      const resp = await fetch(`${VIS_DATA_BASE}/manifest.json`);
      const manifest = await resp.json();
      setDataFiles(manifest.files);
    } catch (e) {
      console.warn('No manifest found, using fallback list');
      // 备选: 硬编码的已知数据文件列表
      setDataFiles([
        { name: 'CCLXIV Qwen3 差分注入', path: 'vis_CCLXIV_qwen3_20260426_223000.json' },
        { name: 'CCLXIV GLM4 差分注入', path: 'vis_CCLXIV_glm4_20260426_223500.json' },
      ]);
    }
  };

  // 方式2: 加载单个数据文件
  const loadDataFile = async (filepath) => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(`${VIS_DATA_BASE}/${filepath}`);
      const data = await resp.json();
      // 验证schema
      if (data.schema_version !== '1.0') {
        throw new Error(`Unsupported schema: ${data.schema_version}`);
      }
      setActiveData(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  // 方式3: 从本地文件选择器加载
  const loadLocalFile = (file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target.result);
        if (data.schema_version !== '1.0') {
          throw new Error(`Unsupported schema: ${data.schema_version}`);
        }
        setActiveData(data);
      } catch (err) {
        setError(err.message);
      }
    };
    reader.readAsText(file);
  };

  return { dataFiles, activeData, loading, error, loadDataManifest, loadDataFile, loadLocalFile };
}
```

### 5.3 主渲染组件

```jsx
// neural_vis/index.jsx

import React, { useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Stars, Text } from '@react-three/drei';
import * as THREE from 'three';
import { useVisData } from './hooks/useVisData';
import DataSourcePanel from './components/DataSourcePanel';
import SceneController from './components/SceneController';
import DetailPanel from './components/DetailPanel';
import LayerStackRenderer from './renderers/LayerStackRenderer';
import TrajectoryRenderer from './renderers/TrajectoryRenderer';
import PointCloudRenderer from './renderers/PointCloudRenderer';
import Heatmap3DRenderer from './renderers/Heatmap3DRenderer';
import SceneHelpers from './renderers/SceneHelpers';
import './styles/neural_vis.css';

const LAYER_GAP = 3;      // 层板间距
const PLANE_SIZE = 20;    // 层板大小

export default function NeuralVis3DApp() {
  const { activeData, loading, error, loadDataFile, loadLocalFile, loadDataManifest } = useVisData();
  const [selectedVis, setSelectedVis] = useState(null);
  const [animProgress, setAnimProgress] = useState(1);
  const [viewMode, setViewMode] = useState('layer_stack'); // layer_stack | free
  const [selectedLayers, setSelectedLayers] = useState(null);
  const [hoveredToken, setHoveredToken] = useState(null);

  // 从数据中提取visualizations
  const visualizations = activeData?.visualizations || [];

  // 按类型分组
  const trajectories = visualizations.filter(v => v.type === 'trajectory');
  const pointClouds = visualizations.filter(v => v.type === 'point_cloud');
  const heatmaps = visualizations.filter(v => v.type === 'heatmap_3d');
  const flows = visualizations.filter(v => v.type === 'flow');
  const layerStacks = visualizations.filter(v => v.type === 'layer_stack');

  return (
    <div className="neural-vis-app">
      {/* 左侧面板 */}
      <div className="neural-vis-sidebar">
        <DataSourcePanel 
          onLoadFile={loadDataFile}
          onLoadLocal={loadLocalFile}
          onRefresh={loadDataManifest}
          data={activeData}
        />
        <SceneController
          viewMode={viewMode}
          onViewModeChange={setViewMode}
          nLayers={activeData?.model_info?.n_layers || 36}
          selectedLayers={selectedLayers}
          onLayerFilter={setSelectedLayers}
          animProgress={animProgress}
          onAnimProgress={setAnimProgress}
        />
      </div>

      {/* 中央3D画布 */}
      <div className="neural-vis-canvas">
        <Canvas
          camera={{ position: [30, 20, 30], fov: 50 }}
          gl={{ antialias: true, alpha: true }}
        >
          <PerspectiveCamera makeDefault position={[30, 20, 30]} fov={50} />
          <OrbitControls 
            enableDamping 
            dampingFactor={0.1}
            minDistance={5}
            maxDistance={100}
          />
          
          {/* 灯光 */}
          <ambientLight intensity={0.4} />
          <directionalLight position={[10, 20, 10]} intensity={0.8} />
          <pointLight position={[-10, -10, -10]} intensity={0.3} />
          
          {/* 背景 */}
          <Stars radius={100} depth={50} count={2000} factor={4} fade />
          
          {/* 场景辅助 */}
          <SceneHelpers nLayers={activeData?.model_info?.n_layers} />
          
          {/* 渲染器 */}
          {layerStacks.map(ls => (
            <LayerStackRenderer 
              key={ls.id} 
              layerStack={ls} 
              selectedLayers={selectedLayers}
              layerGap={LAYER_GAP}
              planeSize={PLANE_SIZE}
            />
          ))}
          
          {trajectories.map(traj => (
            <TrajectoryRenderer
              key={traj.id}
              trajectory={traj}
              animated={animProgress < 1}
              animationProgress={animProgress}
              layerGap={LAYER_GAP}
              onHoverToken={setHoveredToken}
            />
          ))}
          
          {pointClouds.map(pc => (
            <PointCloudRenderer
              key={pc.id}
              pointCloud={pc}
              onHoverToken={setHoveredToken}
            />
          ))}
          
          {heatmaps.map(hm => (
            <Heatmap3DRenderer
              key={hm.id}
              heatmap={hm}
            />
          ))}
        </Canvas>
      </div>

      {/* 右侧详情面板 */}
      <div className="neural-vis-detail">
        <DetailPanel
          data={activeData}
          hoveredToken={hoveredToken}
          selectedVis={selectedVis}
          onSelectVis={setSelectedVis}
        />
      </div>
    </div>
  );
}
```

### 5.4 Vite静态文件配置

```javascript
// vite.config.js 添加:
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    // 静态文件映射: /vis_data → results/vis_data
    fs: {
      allow: [
        path.resolve(__dirname),
        path.resolve(__dirname, '../results/vis_data'),
      ],
    },
    proxy: {
      '/vis_data': {
        target: 'http://localhost:5173',
        rewrite: (path) => path.replace('/vis_data', ''),
        // 实际使用时需要通过后端或直接复制文件
      },
    },
  },
});
```

**更简单的方案**: 用Python脚本将 `results/vis_data/` 下的JSON文件生成一个 `manifest.json`，前端直接读取。

---

## 六、优先级与实施计划

### Phase 1: 基础框架（1天）

- [ ] 创建 `tests/glm5/vis_data_exporter.py` 标准导出模块
- [ ] 创建 `results/vis_data/` 目录
- [ ] 修改 `phase_cclxiv_deep_locking.py` 集成导出
- [ ] 创建前端 `neural_vis/` 模块骨架
- [ ] 实现数据加载 hook (`useVisData`)
- [ ] 实现基础3D场景（Canvas + 轨道控制 + 灯光）

### Phase 2: 核心渲染器（1天）

- [ ] 实现 `TrajectoryRenderer` — 轨迹+球标记+纠正层高亮
- [ ] 实现 `PointCloudRenderer` — 点云+类别着色+悬停
- [ ] 实现 `SceneHelpers` — 层标签+坐标轴
- [ ] 实现动画控制（逐层传播动画）

### Phase 3: 层堆叠+交互（1天）

- [ ] 实现 `LayerStackRenderer` — 层板+内嵌点云
- [ ] 实现交互：点击层→展开详情，悬停token→tooltip
- [ ] 实现 `DetailPanel` — 显示选中对象的所有指标
- [ ] 实现层过滤和类别过滤

### Phase 4: 高级可视化（1天）

- [ ] 实现 `Heatmap3DRenderer` — 3D柱状图
- [ ] 实现 `FlowRenderer` — 注意力弧线+粒子动画
- [ ] 实现 `MetricPanel` — 2D指标曲线叠加
- [ ] 跨模型对比模式

### Phase 5: 完善与优化（1天）

- [ ] 回溯集成到已有Phase测试脚本
- [ ] 性能优化（InstancedMesh, LOD）
- [ ] 深色主题UI完善
- [ ] 导出截图/视频功能

---

## 七、颜色方案

### 7.1 层功能颜色

| 功能 | 颜色 | 用途 |
|------|------|------|
| 词法(Lexical) | `#ff6b6b` 珊瑚红 | L0-L9 |
| 语义(Semantic) | `#4ecdc4` 青色 | L6-L18 |
| 语法(Syntactic) | `#ffe66d` 金黄 | L14-L24 |
| 决策(Decision) | `#a855f7` 紫色 | L24+ |
| 纠正(Correction) | `#fbbf24` 琥珀 | 纠正层标记 |

### 7.2 Delta衰减颜色映射

```
delta_cos = 1.0  →  #ef4444 (红: 差分完全保持)
delta_cos = 0.5  →  #f59e0b (橙: 差分半衰减)
delta_cos = 0.0  →  #3b82f6 (蓝: 差分完全消失)
```

### 7.3 类别颜色

```
fruit   = #ff6b6b    animal  = #4ecdc4
vehicle = #ffe66d    tool    = #a855f7
nature  = #34d399    food    = #f97316
person  = #ec4899    abstract= #6366f1
```

---

## 八、与现有系统的集成

### 8.1 入口集成

在 `App.jsx` 中添加路由：

```jsx
import NeuralVis3DApp from './neural_vis';

// 在路由中添加:
<Route path="/neural-vis" element={<NeuralVis3DApp />} />
```

### 8.2 与方案.md中3D可视化规格的对应

| 方案.md §八 | 本设计 | 实现状态 |
|------------|--------|---------|
| 语义空间点云 | `PointCloudRenderer` | Phase 2 |
| 概念向量方向 | `TrajectoryRenderer` 中的箭头 | Phase 2 |
| 层级结构图 | `LayerStackRenderer` | Phase 3 |
| 信息流动态 | `FlowRenderer` | Phase 4 |
| Token轨迹 | `TrajectoryRenderer` | Phase 2 |

### 8.3 与系统方案设计.md §七 3D可视化系统的对应

| §七.2 可视化对象 | 本设计type | 渲染器 |
|-----------------|-----------|--------|
| 语义空间点云 | `point_cloud` | PointCloudRenderer |
| 概念向量方向 | `trajectory` (含correction标记) | TrajectoryRenderer |
| 层级结构 | `layer_stack` | LayerStackRenderer |
| 信息流动态 | `flow` | FlowRenderer |
| Token轨迹 | `trajectory` | TrajectoryRenderer |
| (新增) 热力图 | `heatmap_3d` | Heatmap3DRenderer |
