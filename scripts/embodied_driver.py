import json
import os
import time

import numpy as np
from agi_core_engine import AGICoreEngine


class EmbodiedDriver:
    """
    具身驱动管理器 (Embodied Driver Manager)
    SHMC 核心组件：将 AGI 意识流转化为具备物理约束的控制信号。
    """
    def __init__(self, agi_engine: AGICoreEngine):
        self.agi = agi_engine
        # 物理状态：[x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        # 动作阻尼 (由 EE 稳定性控制)
        self.damping_base = 0.05
        # 物理限制
        self.max_force = 10.0
        
    def resolve_movement(self):
        """
        基于 AGI 意图引力计算物理推力
        """
        # 1. 获取 GWS 优胜意识块作为方向导引
        intent_vector = self.agi.gws.state[:3] # 取前三维映射为 XYZ 方向
        
        # 2. 计算内稳态阻尼因子
        # 稳定性越高，动作越平滑且精确
        stability = self.agi.emotion.stability
        damping = 1.0 / (stability + 0.1) * self.damping_base
        
        # 3. 合成控制力 (模拟 F = ma)
        raw_force = intent_vector * self.max_force
        
        # 4. 模拟物理方程 (简单二阶集成)
        dt = 0.1
        acceleration = raw_force - (self.state[3:] * damping) # 加入空气阻尼
        self.state[3:] += acceleration * dt # 更新速度
        self.state[:3] += self.state[3:] * dt # 更新位置
        
        # 5. 碰撞反馈检测 (模拟)
        collision = False
        if np.linalg.norm(self.state[:3]) > 5.0: # 假设 5.0 为边界
            collision = True
            self.state[3:] *= -0.5 # 碰撞反弹
            # 反馈给 AGI: 注入负向评价信号 (痛觉模拟)
            self.agi.emotion.stability *= 0.8
            self.agi.emotion.curiosity *= 1.2
            
        report = {
            "pos": self.state[:3].tolist(),
            "vel": self.state[3:].tolist(),
            "force": raw_force.tolist(),
            "damping": float(damping),
            "stability": float(stability),
            "event": "COLLISION_PROTECT" if collision else "SMOOTH_GLIDE"
        }
        return report

    def run_embodied_session(self, steps=50):
        """
        运行具身仿真会话
        """
        print(f"[*] 启动具身控制仿真 (AGI Embodiment Session)...")
        history = []
        for i in range(steps):
            # 模拟 AGI 持续产生意图
            self.agi.run_conscious_step(0, np.random.randn(32) * 0.1)
            report = self.resolve_movement()
            history.append(report)
            
            if i % 10 == 0:
                print(f"[Step {i}] Pos: {report['pos']}, Stability: {report['stability']:.2f}")
                
        return history

if __name__ == "__main__":
    # 初始化统合引擎
    agi_core = AGICoreEngine(manifold_dim=32)
    # 启动具身驱动
    driver = EmbodiedDriver(agi_core)
    
    # 运行仿真测试
    sim_data = driver.run_embodied_session(50)
    
    # 保存仿真结果
    os.makedirs("tempdata", exist_ok=True)
    with open("tempdata/embodied_sim_report.json", "w", encoding="utf-8") as f:
        json.dump(sim_data[-1], f, indent=4, ensure_ascii=False)
    
    print(f"\n--- 物理实感驱动实验总结 ---")
    print(json.dumps(sim_data[-1], indent=2, ensure_ascii=False))
