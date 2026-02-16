import json
import os
import time

import numpy as np


class IntentionalityEngine:
    """
    自主目标管理引擎 (Geodesic Intentionality Engine)
    功能：在语义流形上注入势能场 (Potential Field)，引导推理测地线向目标点汇聚。
    这是实现 AGI 自主规划与目标导向行为的核心组件。
    """
    def __init__(self, manifold_dim=32, learning_rate=0.01):
        self.manifold_dim = manifold_dim
        self.lr = learning_rate
        # 流形度量张量 (简化为单位阵)
        self.metric_g = np.eye(manifold_dim)
        # 目标势能点 (Sink)
        self.targets = {} # {target_id: position_vector}
        # 障碍势能点 (Obstacle)
        self.obstacles = {} # {obs_id: position_vector}

    def add_target(self, target_id, position):
        """设定一个绝对零势能目标点"""
        self.targets[target_id] = np.array(position)
        print(f"[+] 目标设定成功: {target_id} -> Position {position[:5]}...")

    def add_obstacle(self, obs_id, position):
        """设定一个高势能障碍点"""
        self.obstacles[obs_id] = np.array(position)
        print(f"[!] 障碍设定成功: {obs_id} -> Position {position[:5]}...")

    def calculate_potential_gradient(self, current_state):
        """
        优化后的势能梯度计算
        引入目标增强型平方反比引力
        """
        grad = np.zeros(self.manifold_dim)
        
        # 1. 目标增强引力 (Attraction)
        for target_pos in self.targets.values():
            dist_vec = target_pos - current_state
            dist = np.linalg.norm(dist_vec) + 1e-9
            # 使用指数级增强的引力，越接近目标吸引力越大
            grad += (dist_vec / dist) * (1.0 + 1.0/dist)
            
        # 2. 障碍斥力 (Repulsion)
        for obs_pos in self.obstacles.values():
            dist_vec = current_state - obs_pos
            dist = np.linalg.norm(dist_vec) + 1e-9
            if dist < 3.0: # 缩小感应半径，避免全局干扰
                # 强斥力场
                grad += (dist_vec / (dist**4)) * 20.0
                
        # 梯度归一化，确保步长稳定
        norm = np.linalg.norm(grad) + 1e-9
        return grad / norm

    def generate_intentional_trace(self, start_pos, steps=200):
        """
        生成受目标引导的意图轨迹 (增加步数限制)
        """
        print(f"[*] 启动高精度意图推理轨迹生成...")
        trace = [start_pos]
        current_pos = np.array(start_pos)
        
        for step in range(steps):
            grad = self.calculate_potential_gradient(current_pos)
            
            # 动态调整学习率：越接近目标越精细
            dist_to_target = np.min([np.linalg.norm(t - current_pos) for t in self.targets.values()])
            adaptive_lr = self.lr * (0.5 if dist_to_target < 1.0 else 1.0)
            
            current_pos = current_pos + adaptive_lr * grad
            trace.append(current_pos.copy())
            
            # 提前停止检测
            if dist_to_target < 0.1:
                print(f"[+] 目标已锁定，提前结束。")
                break
            
        return np.array(trace)

    def run_intentional_test(self, learning_rate=0.5):
        """运行高强度自主规划测试"""
        self.lr = learning_rate
        # 1. 设定起点与目标点
        start = np.zeros(self.manifold_dim)
        target = np.ones(self.manifold_dim) * 5.0
        self.add_target("AGI_MISSION_COMPLETE", target)
        
        # 2. 设定路径中间的逻辑障碍
        obstacle = np.ones(self.manifold_dim) * 2.5
        obstacle[0] += 0.2
        self.add_obstacle("LOGIC_CONTRADICTION", obstacle)
        
        # 3. 生成轨迹
        trace = self.generate_intentional_trace(start, steps=300)
        
        # 4. 评估指标
        final_dist = np.linalg.norm(trace[-1] - target)
        path_smoothness = np.mean(np.linalg.norm(np.diff(trace, axis=0), axis=1))
        
        results = {
            "start_dist": float(np.linalg.norm(target - start)),
            "final_dist": float(final_dist),
            "alignment_success": bool(final_dist < 0.5),
            "path_energy": float(path_smoothness),
            "status": "GOAL_REACHED_SMOOTHLY" if final_dist < 0.5 else "TARGET_LOST"
        }
        
        return results, trace

if __name__ == "__main__":
    engine = IntentionalityEngine()
    summary, trace = engine.run_intentional_test(learning_rate=0.3)
    print(f"\n--- 自主目标引导实验总结 ---\n{json.dumps(summary, indent=2, ensure_ascii=False)}")
    
    # 导出轨迹数据供可视化参考
    os.makedirs("tempdata", exist_ok=True)
    with open("tempdata/intentionality_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    # 模拟 3D 导出
    np.save("tempdata/intentional_trace.npy", trace)
