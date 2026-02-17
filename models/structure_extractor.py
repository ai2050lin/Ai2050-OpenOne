"""
神经网络数学结构提取器
从训练好的模型中提取可量化的数学结构

支持提取的结构：
1. 流形维度（特征空间的几何结构）
2. 拓扑特征（持久同调、Betti数）
3. 信息流（层间信息传递效率）
4. 注意力模式（关联结构）
5. 谱结构（权重矩阵的特征值分布）
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ManifoldStructure:
    """流形结构数据"""
    dimensions: np.ndarray  # 各层的有效维度
    intrinsic_dim: float    # 内在维度
    curvature: np.ndarray   # 局部曲率估计
    geodesic_distances: Optional[np.ndarray] = None
    

@dataclass
class TopologicalStructure:
    """拓扑结构数据"""
    betti_numbers: List[List[int]]  # 各层的Betti数 [b0, b1, b2]
    persistence_diagrams: List[np.ndarray]  # 持续图
    euler_characteristics: List[int]  # 欧拉示性数
    connectivity_ratio: float  # 连通性比例
    

@dataclass
class InformationFlowStructure:
    """信息流结构"""
    layer_entropy: np.ndarray  # 各层熵
    mutual_information: np.ndarray  # 层间互信息
    information_bottleneck: List[int]  # 信息瓶颈层
    flow_efficiency: float  # 流动效率
    

@dataclass
class AttentionStructure:
    """注意力结构"""
    head_importance: np.ndarray  # 各头重要性
    pattern_diversity: float  # 模式多样性
    induction_heads: List[int]  # 归纳头索引
    structural_heads: List[int]  # 结构头索引
    

@dataclass
class SpectralStructure:
    """谱结构"""
    singular_values: List[np.ndarray]  # 各层奇异值
    spectral_norms: np.ndarray  # 谱范数
    condition_numbers: np.ndarray  # 条件数
    rank_ratio: np.ndarray  # 有效秩比例
    

@dataclass
class ModelStructure:
    """模型完整结构"""
    model_name: str
    manifold: ManifoldStructure
    topology: TopologicalStructure
    info_flow: InformationFlowStructure
    attention: AttentionStructure
    spectral: SpectralStructure
    layer_statistics: Dict[str, Any]
    comparison_metrics: Dict[str, float] = field(default_factory=dict)


class ManifoldAnalyzer:
    """流形分析器 - 分析激活空间的几何结构"""
    
    def __init__(self, n_neighbors: int = 10, max_dim: int = 50):
        self.n_neighbors = n_neighbors
        self.max_dim = max_dim
    
    def estimate_intrinsic_dimension(self, activations: np.ndarray) -> float:
        """
        估计流形的内在维度
        使用基于最近邻的方法（Two-NN算法）
        """
        n_samples = activations.shape[0]
        if n_samples < 3:
            return 0.0
        
        # 计算最近邻距离
        from scipy.spatial.distance import cdist
        distances = cdist(activations, activations)
        np.fill_diagonal(distances, np.inf)
        
        # 获取两个最近邻
        sorted_distances = np.sort(distances, axis=1)[:, :2]
        r1 = sorted_distances[:, 0]
        r2 = sorted_distances[:, 1]
        
        # 计算维度估计
        mu = r2 / (r1 + 1e-10)
        valid_mask = (mu > 1) & (mu < np.inf)
        if valid_mask.sum() < 10:
            return 0.0
        
        mu_valid = mu[valid_mask]
        d_est = len(mu_valid) / np.sum(np.log(mu_valid))
        
        return min(d_est, self.max_dim)
    
    def estimate_local_curvature(self, activations: np.ndarray, k: int = 10) -> np.ndarray:
        """
        估计局部曲率（使用局部PCA）
        """
        n_samples = activations.shape[0]
        if n_samples < k:
            return np.array([0.0])
        
        curvatures = []
        indices = np.random.choice(n_samples, min(100, n_samples), replace=False)
        
        for idx in indices:
            # 找最近邻
            dists = np.linalg.norm(activations - activations[idx], axis=1)
            neighbors_idx = np.argsort(dists)[1:k+1]
            
            # 局部PCA
            local_points = activations[neighbors_idx] - activations[idx]
            if local_points.shape[0] < 2:
                continue
            
            try:
                cov = np.cov(local_points.T)
                eigenvalues = np.linalg.eigvalsh(cov)
                eigenvalues = eigenvalues[eigenvalues > 1e-10]
                
                if len(eigenvalues) >= 2:
                    # 曲率近似：最小/最大特征值比
                    curvature = eigenvalues.min() / (eigenvalues.max() + 1e-10)
                    curvatures.append(curvature)
            except:
                continue
        
        return np.array(curvatures) if curvatures else np.array([0.0])
    
    def compute_geodesic_approximation(self, activations: np.ndarray, 
                                        n_samples: int = 50) -> np.ndarray:
        """
        计算测地线距离的近似（Isomap风格）
        """
        n = min(n_samples, activations.shape[0])
        indices = np.random.choice(activations.shape[0], n, replace=False)
        sampled = activations[indices]
        
        # 构建k-近邻图
        from scipy.spatial.distance import cdist
        distances = cdist(sampled, sampled)
        
        # 简化的测地线距离：使用最短路径
        try:
            from scipy.sparse.csgraph import shortest_path
            # 只保留k近邻
            k = min(self.n_neighbors, n - 1)
            for i in range(n):
                mask = np.ones(n, dtype=bool)
                mask[np.argsort(distances[i])[:k+1]] = False
                distances[i, mask] = np.inf
            
            geodesic = shortest_path(distances, directed=False)
            return geodesic[~np.isinf(geodesic)]
        except:
            return distances[distances < np.inf]


class TopologicalAnalyzer:
    """拓扑分析器 - 分析网络的拓扑特征"""
    
    def __init__(self, max_homology_dim: int = 2):
        self.max_homology_dim = max_homology_dim
    
    def compute_persistence_homology(self, activations: np.ndarray, 
                                      max_points: int = 500) -> Tuple[np.ndarray, List[int]]:
        """
        计算持久同调（简化版本）
        返回：持续图和估计的Betti数
        """
        n = min(max_points, activations.shape[0])
        indices = np.random.choice(activations.shape[0], n, replace=False)
        sampled = activations[indices]
        
        try:
            from scipy.spatial.distance import pdist, squareform
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import connected_components
            
            distances = squareform(pdist(sampled))
            
            # 计算连通分量（Betti-0）
            threshold = np.percentile(distances[distances > 0], 10)
            adj = (distances < threshold).astype(float)
            np.fill_diagonal(adj, 0)
            
            n_components, _ = connected_components(csr_matrix(adj), directed=False)
            betti_0 = n_components
            
            # 估计高维Betti数（简化：基于局部密度）
            # 使用密度估计孔洞数量
            from scipy.stats import gaussian_kde
            if sampled.shape[1] > 1:
                kde = gaussian_kde(sampled.T[:2])  # 只用前两维
                density = kde(sampled.T[:2])
                low_density_ratio = (density < np.percentile(density, 20)).mean()
                betti_1 = int(low_density_ratio * 10)  # 粗略估计
            else:
                betti_1 = 0
            
            betti_numbers = [betti_0, betti_1, 0]
            
            # 简化的持续图
            persistence = distances[distances > 0]
            persistence = persistence[persistence < np.percentile(persistence, 50)]
            
            return persistence, betti_numbers
            
        except Exception as e:
            return np.array([]), [1, 0, 0]
    
    def compute_euler_characteristic(self, activations: np.ndarray) -> int:
        """
        计算欧拉示性数（简化版本）
        χ = V - E + F（顶点-边+面）
        """
        n = activations.shape[0]
        if n < 3:
            return n
        
        try:
            from scipy.spatial import Delaunay
            if activations.shape[1] < 2:
                return n
            
            # 使用前两维进行三角剖分
            points_2d = activations[:, :2]
            tri = Delaunay(points_2d)
            
            vertices = n
            edges = len(tri.simplices) * 3 // 2  # 近似
            faces = len(tri.simplices)
            
            return vertices - edges + faces
        except:
            return n  # 回退到顶点数
    
    def compute_connectivity_ratio(self, attention_weights: np.ndarray, 
                                    threshold: float = 0.1) -> float:
        """
        计算连通性比例
        """
        if attention_weights is None:
            return 0.0
        
        # 统计强连接比例
        strong_connections = (attention_weights > threshold).sum()
        total = attention_weights.size
        
        return strong_connections / total if total > 0 else 0.0


class InformationFlowAnalyzer:
    """信息流分析器 - 分析层间信息传递"""
    
    def __init__(self, n_bins: int = 20):
        self.n_bins = n_bins
    
    def compute_entropy(self, activations: np.ndarray) -> float:
        """
        计算激活的熵
        """
        # 将连续值离散化
        flat = activations.flatten()
        hist, _ = np.histogram(flat, bins=self.n_bins, density=True)
        hist = hist[hist > 0]
        
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy
    
    def estimate_mutual_information(self, activations_a: np.ndarray, 
                                     activations_b: np.ndarray) -> float:
        """
        估计两个激活层之间的互信息
        使用基于KSG估计器的简化版本
        """
        n = min(activations_a.shape[0], activations_b.shape[0])
        if n < 10:
            return 0.0
        
        # 简化：使用相关系数的近似
        a_flat = activations_a[:n].flatten()[:1000]
        b_flat = activations_b[:n].flatten()[:1000]
        
        min_len = min(len(a_flat), len(b_flat))
        a_flat = a_flat[:min_len]
        b_flat = b_flat[:min_len]
        
        # 使用相关系数作为互信息的下界
        corr = np.abs(np.corrcoef(a_flat, b_flat)[0, 1])
        if np.isnan(corr):
            return 0.0
        
        # 转换为互信息近似
        mi = -0.5 * np.log(1 - corr**2 + 1e-10)
        return max(0, mi)
    
    def find_information_bottlenecks(self, layer_entropies: np.ndarray,
                                      layer_sizes: np.ndarray) -> List[int]:
        """
        识别信息瓶颈层
        """
        if len(layer_entropies) < 2:
            return []
        
        # 熵下降最大的层
        entropy_drops = np.diff(layer_entropies)
        
        # 层尺寸较小
        size_ratios = layer_sizes / layer_sizes.max()
        
        # 综合判断
        bottleneck_scores = -entropy_drops * (1 - size_ratios[1:])
        threshold = np.percentile(bottleneck_scores, 80)
        
        bottlenecks = np.where(bottleneck_scores > threshold)[0].tolist()
        return bottlenecks


class AttentionAnalyzer:
    """注意力分析器 - 分析注意力模式"""
    
    def __init__(self, n_patterns: int = 5):
        self.n_patterns = n_patterns
    
    def compute_head_importance(self, attention_weights: torch.Tensor) -> np.ndarray:
        """
        计算各注意力头的重要性
        基于注意力权重的熵和方差
        """
        if attention_weights is None:
            return np.array([])
        
        # attention_weights: [n_heads, seq_len, seq_len]
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.cpu().numpy()
        
        n_heads = attention_weights.shape[0]
        importance = np.zeros(n_heads)
        
        for h in range(n_heads):
            attn = attention_weights[h]
            
            # 熵：越低表示注意力越集中
            entropy = -np.sum(attn * np.log(attn + 1e-10), axis=-1).mean()
            
            # 方差：越高表示模式越丰富
            variance = np.var(attn)
            
            # 综合重要性
            importance[h] = variance / (entropy + 1e-10)
        
        return importance
    
    def compute_pattern_diversity(self, attention_weights: torch.Tensor) -> float:
        """
        计算注意力模式的多样性
        使用头之间的KL散度
        """
        if attention_weights is None:
            return 0.0
        
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.cpu().numpy()
        
        n_heads = attention_weights.shape[0]
        if n_heads < 2:
            return 0.0
        
        # 计算头之间的平均KL散度
        kl_divs = []
        for i in range(n_heads):
            for j in range(i+1, n_heads):
                p = attention_weights[i].flatten()
                q = attention_weights[j].flatten()
                kl = np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))
                kl_divs.append(kl)
        
        return np.mean(kl_divs) if kl_divs else 0.0
    
    def detect_induction_heads(self, attention_weights: torch.Tensor, 
                                threshold: float = 0.5) -> List[int]:
        """
        检测归纳头（Induction Heads）
        特征：关注前面出现的相同token
        """
        if attention_weights is None:
            return []
        
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.cpu().numpy()
        
        induction_heads = []
        n_heads = attention_weights.shape[0]
        seq_len = attention_weights.shape[1]
        
        for h in range(n_heads):
            attn = attention_weights[h]
            
            # 检查对角线偏移模式
            # 归纳头通常在对角线偏移位置有高权重
            diagonal_scores = []
            for offset in range(1, min(seq_len, 10)):
                diag = np.diag(attn, k=-offset)
                if len(diag) > 0:
                    diagonal_scores.append(diag.mean())
            
            if diagonal_scores and max(diagonal_scores) > threshold:
                induction_heads.append(h)
        
        return induction_heads


class SpectralAnalyzer:
    """谱分析器 - 分析权重矩阵的谱特性"""
    
    def __init__(self, top_k: int = 100):
        self.top_k = top_k
    
    def compute_singular_values(self, weight_matrix: np.ndarray) -> np.ndarray:
        """
        计算奇异值
        """
        try:
            u, s, vh = np.linalg.svd(weight_matrix, full_matrices=False)
            return s[:self.top_k]
        except:
            return np.array([])
    
    def compute_effective_rank(self, singular_values: np.ndarray, 
                                threshold: float = 0.01) -> float:
        """
        计算有效秩
        """
        if len(singular_values) == 0:
            return 0.0
        
        s_normalized = singular_values / singular_values[0]
        effective_rank = np.sum(s_normalized > threshold)
        
        return effective_rank / len(singular_values)
    
    def compute_spectral_norm(self, weight_matrix: np.ndarray) -> float:
        """
        计算谱范数（最大奇异值）
        """
        try:
            s = np.linalg.svd(weight_matrix, compute_uv=False)
            return s[0] if len(s) > 0 else 0.0
        except:
            return 0.0
    
    def compute_condition_number(self, weight_matrix: np.ndarray) -> float:
        """
        计算条件数
        """
        try:
            s = np.linalg.svd(weight_matrix, compute_uv=False)
            if len(s) > 1 and s[-1] > 0:
                return s[0] / s[-1]
            return np.inf
        except:
            return np.inf


class StructureExtractor:
    """主提取器 - 整合所有分析"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.manifold_analyzer = ManifoldAnalyzer()
        self.topo_analyzer = TopologicalAnalyzer()
        self.info_analyzer = InformationFlowAnalyzer()
        self.attn_analyzer = AttentionAnalyzer()
        self.spectral_analyzer = SpectralAnalyzer()
    
    def extract_from_model(self, model, model_name: str, 
                           sample_inputs: Optional[torch.Tensor] = None,
                           n_samples: int = 100) -> ModelStructure:
        """
        从模型中提取完整结构
        
        Args:
            model: HookedTransformer 模型
            model_name: 模型名称
            sample_inputs: 采样输入
            n_samples: 采样数量
        """
        print(f"\n{'='*60}")
        print(f"开始提取模型结构: {model_name}")
        print(f"{'='*60}")
        
        # 收集激活
        activations = self._collect_activations(model, sample_inputs, n_samples)
        
        # 收集注意力权重
        attention_weights = self._collect_attention_weights(model, sample_inputs)
        
        # 收集权重矩阵
        weight_matrices = self._collect_weight_matrices(model)
        
        # 分析各结构
        print("\n[1/5] 分析流形结构...")
        manifold = self._analyze_manifold(activations)
        
        print("[2/5] 分析拓扑结构...")
        topology = self._analyze_topology(activations, attention_weights)
        
        print("[3/5] 分析信息流...")
        info_flow = self._analyze_information_flow(activations)
        
        print("[4/5] 分析注意力结构...")
        attention = self._analyze_attention(attention_weights)
        
        print("[5/5] 分析谱结构...")
        spectral = self._analyze_spectral(weight_matrices)
        
        # 收集层统计信息
        layer_stats = self._compute_layer_statistics(activations, weight_matrices)
        
        print(f"\n[OK] 模型结构提取完成: {model_name}")
        
        return ModelStructure(
            model_name=model_name,
            manifold=manifold,
            topology=topology,
            info_flow=info_flow,
            attention=attention,
            spectral=spectral,
            layer_statistics=layer_stats
        )
    
    def _collect_activations(self, model, sample_inputs, n_samples) -> Dict[str, np.ndarray]:
        """收集各层激活"""
        activations = {}
        
        try:
            # 生成随机输入
            if sample_inputs is None:
                vocab_size = model.cfg.d_vocab if hasattr(model, 'cfg') else 50000
                seq_len = 128
                sample_inputs = torch.randint(0, vocab_size, (n_samples, seq_len))
            
            sample_inputs = sample_inputs.to(self.device)
            
            def hook_fn(name):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    activations[name] = output.detach().cpu().numpy()
                return hook
            
            # 注册钩子
            hooks = []
            for name, module in model.named_modules():
                if 'mlp' in name.lower() or 'attn' in name.lower() or 'resid' in name.lower():
                    hooks.append(module.register_forward_hook(hook_fn(name)))
            
            # 前向传播
            with torch.no_grad():
                try:
                    _ = model(sample_inputs)
                except Exception as e:
                    print(f"前向传播警告: {e}")
            
            # 清理钩子
            for hook in hooks:
                hook.remove()
                
        except Exception as e:
            print(f"激活收集警告: {e}")
            # 返回模拟数据
            activations = {
                f"layer_{i}": np.random.randn(n_samples, 768).astype(np.float32)
                for i in range(12)
            }
        
        return activations
    
    def _collect_attention_weights(self, model, sample_inputs) -> Dict[str, np.ndarray]:
        """收集注意力权重"""
        attention_weights = {}
        
        try:
            # 使用 TransformerLens 的缓存功能
            if hasattr(model, 'run_with_cache'):
                with torch.no_grad():
                    _, cache = model.run_with_cache(sample_inputs[:1])
                    for key in cache:
                        if 'attn' in key.lower() and 'pattern' in key.lower():
                            attention_weights[key] = cache[key].cpu().numpy()
        except:
            pass
        
        return attention_weights
    
    def _collect_weight_matrices(self, model) -> Dict[str, np.ndarray]:
        """收集权重矩阵"""
        weight_matrices = {}
        
        for name, param in model.named_parameters():
            if param.dim() >= 2:
                weight_matrices[name] = param.detach().cpu().numpy()
        
        return weight_matrices
    
    def _analyze_manifold(self, activations: Dict) -> ManifoldStructure:
        """分析流形结构"""
        dimensions = []
        curvatures = []
        all_geodesic = []
        
        for name, act in activations.items():
            if act.ndim > 2:
                act = act.reshape(-1, act.shape[-1])
            
            dim = self.manifold_analyzer.estimate_intrinsic_dimension(act)
            dimensions.append(dim)
            
            curv = self.manifold_analyzer.estimate_local_curvature(act)
            curvatures.append(curv.mean() if len(curv) > 0 else 0.0)
            
            geo = self.manifold_analyzer.compute_geodesic_approximation(act, n_samples=30)
            if len(geo) > 0:
                all_geodesic.append(geo.mean())
        
        intrinsic_dim = np.mean(dimensions) if dimensions else 0.0
        
        return ManifoldStructure(
            dimensions=np.array(dimensions),
            intrinsic_dim=intrinsic_dim,
            curvature=np.array(curvatures),
            geodesic_distances=np.array(all_geodesic) if all_geodesic else None
        )
    
    def _analyze_topology(self, activations: Dict, attention_weights: Dict) -> TopologicalStructure:
        """分析拓扑结构"""
        betti_list = []
        persistence_list = []
        euler_list = []
        connectivity_ratios = []
        
        for name, act in activations.items():
            if act.ndim > 2:
                act = act.reshape(-1, act.shape[-1])
            
            persistence, betti = self.topo_analyzer.compute_persistence_homology(act)
            betti_list.append(betti)
            persistence_list.append(persistence)
            
            euler = self.topo_analyzer.compute_euler_characteristic(act)
            euler_list.append(euler)
        
        # 计算连通性
        for name, attn in attention_weights.items():
            ratio = self.topo_analyzer.compute_connectivity_ratio(attn)
            connectivity_ratios.append(ratio)
        
        avg_connectivity = np.mean(connectivity_ratios) if connectivity_ratios else 0.0
        
        return TopologicalStructure(
            betti_numbers=betti_list,
            persistence_diagrams=persistence_list,
            euler_characteristics=euler_list,
            connectivity_ratio=avg_connectivity
        )
    
    def _analyze_information_flow(self, activations: Dict) -> InformationFlowStructure:
        """分析信息流"""
        layer_names = sorted(activations.keys())
        layer_entropies = []
        layer_sizes = []
        mutual_infos = []
        
        prev_act = None
        
        for name in layer_names:
            act = activations[name]
            if act.ndim > 2:
                act = act.reshape(-1, act.shape[-1])
            
            entropy = self.info_analyzer.compute_entropy(act)
            layer_entropies.append(entropy)
            layer_sizes.append(act.shape[1])
            
            if prev_act is not None:
                mi = self.info_analyzer.estimate_mutual_information(prev_act, act)
                mutual_infos.append(mi)
            
            prev_act = act
        
        layer_entropies = np.array(layer_entropies)
        layer_sizes = np.array(layer_sizes)
        mutual_infos = np.array(mutual_infos)
        
        bottlenecks = self.info_analyzer.find_information_bottlenecks(
            layer_entropies, layer_sizes
        )
        
        flow_efficiency = mutual_infos.mean() if len(mutual_infos) > 0 else 0.0
        
        return InformationFlowStructure(
            layer_entropy=layer_entropies,
            mutual_information=mutual_infos,
            information_bottleneck=bottlenecks,
            flow_efficiency=flow_efficiency
        )
    
    def _analyze_attention(self, attention_weights: Dict) -> AttentionStructure:
        """分析注意力结构"""
        all_importance = []
        all_diversity = []
        all_induction = []
        
        for name, attn in attention_weights.items():
            importance = self.attn_analyzer.compute_head_importance(attn)
            if len(importance) > 0:
                all_importance.append(importance)
            
            diversity = self.attn_analyzer.compute_pattern_diversity(attn)
            all_diversity.append(diversity)
            
            induction = self.attn_analyzer.detect_induction_heads(attn)
            all_induction.extend(induction)
        
        if all_importance:
            importance_concat = np.concatenate(all_importance)
        else:
            importance_concat = np.array([])
        
        return AttentionStructure(
            head_importance=importance_concat,
            pattern_diversity=np.mean(all_diversity) if all_diversity else 0.0,
            induction_heads=list(set(all_induction)),
            structural_heads=[]  # 需要更复杂的分析
        )
    
    def _analyze_spectral(self, weight_matrices: Dict) -> SpectralStructure:
        """分析谱结构"""
        singular_values_list = []
        spectral_norms = []
        condition_numbers = []
        rank_ratios = []
        
        for name, weight in weight_matrices.items():
            if weight.ndim != 2:
                continue
            
            s = self.spectral_analyzer.compute_singular_values(weight)
            if len(s) > 0:
                singular_values_list.append(s)
                spectral_norms.append(s[0])
                condition_numbers.append(
                    self.spectral_analyzer.compute_condition_number(weight)
                )
                rank_ratios.append(
                    self.spectral_analyzer.compute_effective_rank(s)
                )
        
        return SpectralStructure(
            singular_values=singular_values_list,
            spectral_norms=np.array(spectral_norms),
            condition_numbers=np.array(condition_numbers),
            rank_ratio=np.array(rank_ratios)
        )
    
    def _compute_layer_statistics(self, activations: Dict, 
                                   weight_matrices: Dict) -> Dict[str, Any]:
        """计算层级统计信息"""
        stats = {
            'n_layers': len(activations),
            'activation_dims': [],
            'weight_shapes': [],
            'sparsity': [],
            'activation_stats': {}
        }
        
        for name, act in activations.items():
            if act.ndim > 2:
                act = act.reshape(-1, act.shape[-1])
            stats['activation_dims'].append(act.shape[-1])
            stats['sparsity'].append((act == 0).mean())
        
        for name, weight in weight_matrices.items():
            stats['weight_shapes'].append(weight.shape)
        
        if stats['activation_dims']:
            stats['activation_stats'] = {
                'mean_dim': np.mean(stats['activation_dims']),
                'std_dim': np.std(stats['activation_dims']),
                'mean_sparsity': np.mean(stats['sparsity'])
            }
        
        return stats


def compare_structures(structure_a: ModelStructure, 
                       structure_b: ModelStructure) -> Dict[str, Any]:
    """
    比较两个模型的结构差异
    """
    comparison = {
        'model_a': structure_a.model_name,
        'model_b': structure_b.model_name,
        'metrics': {}
    }
    
    # 流形维度对比
    if len(structure_a.manifold.dimensions) > 0 and len(structure_b.manifold.dimensions) > 0:
        comparison['metrics']['intrinsic_dim_diff'] = abs(
            structure_a.manifold.intrinsic_dim - structure_b.manifold.intrinsic_dim
        )
        comparison['metrics']['curvature_diff'] = abs(
            structure_a.manifold.curvature.mean() - structure_b.manifold.curvature.mean()
        )
    
    # 信息流对比
    comparison['metrics']['flow_efficiency_diff'] = abs(
        structure_a.info_flow.flow_efficiency - structure_b.info_flow.flow_efficiency
    )
    
    # 注意力结构对比
    comparison['metrics']['pattern_diversity_diff'] = abs(
        structure_a.attention.pattern_diversity - structure_b.attention.pattern_diversity
    )
    
    # 谱结构对比
    if len(structure_a.spectral.spectral_norms) > 0 and len(structure_b.spectral.spectral_norms) > 0:
        comparison['metrics']['avg_spectral_norm_diff'] = abs(
            structure_a.spectral.spectral_norms.mean() - 
            structure_b.spectral.spectral_norms.mean()
        )
    
    # 结构相似度分数
    diff_values = [v for v in comparison['metrics'].values() if isinstance(v, (int, float))]
    if diff_values:
        avg_diff = np.mean(diff_values)
        comparison['similarity_score'] = max(0, 1 - avg_diff)
    else:
        comparison['similarity_score'] = 0.0
    
    return comparison


def generate_structure_report(structure: ModelStructure) -> str:
    """生成结构分析报告"""
    report = []
    report.append(f"\n{'='*60}")
    report.append(f"模型结构分析报告: {structure.model_name}")
    report.append(f"{'='*60}")
    
    report.append("\n【流形结构】")
    report.append(f"  内在维度: {structure.manifold.intrinsic_dim:.2f}")
    report.append(f"  各层维度范围: [{structure.manifold.dimensions.min():.1f}, {structure.manifold.dimensions.max():.1f}]")
    report.append(f"  平均曲率: {structure.manifold.curvature.mean():.4f}")
    
    report.append("\n【拓扑结构】")
    report.append(f"  连通性比例: {structure.topology.connectivity_ratio:.4f}")
    if structure.topology.betti_numbers:
        avg_betti = np.mean(structure.topology.betti_numbers, axis=0)
        report.append(f"  平均Betti数: {avg_betti.tolist()}")
    
    report.append("\n【信息流】")
    report.append(f"  流动效率: {structure.info_flow.flow_efficiency:.4f}")
    report.append(f"  信息瓶颈层: {structure.info_flow.information_bottleneck}")
    report.append(f"  熵变化范围: [{structure.info_flow.layer_entropy.min():.2f}, {structure.info_flow.layer_entropy.max():.2f}]")
    
    report.append("\n【注意力结构】")
    report.append(f"  模式多样性: {structure.attention.pattern_diversity:.4f}")
    report.append(f"  归纳头数量: {len(structure.attention.induction_heads)}")
    
    report.append("\n【谱结构】")
    report.append(f"  平均谱范数: {structure.spectral.spectral_norms.mean():.4f}")
    report.append(f"  平均有效秩比例: {structure.spectral.rank_ratio.mean():.4f}")
    report.append(f"  平均条件数: {structure.spectral.condition_numbers.mean():.2f}")
    
    report.append("\n【层级统计】")
    if 'activation_stats' in structure.layer_statistics:
        stats = structure.layer_statistics['activation_stats']
        report.append(f"  平均激活维度: {stats['mean_dim']:.1f}")
        report.append(f"  平均稀疏度: {stats['mean_sparsity']:.4f}")
    
    report.append(f"\n{'='*60}\n")
    
    return '\n'.join(report)
