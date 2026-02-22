import os
import torch
from transformers import GPT2Tokenizer

class MotherEngineService:
    """
    负责加载脱水的物理母体状态 (mother_language_state.pt)
    并响应前端的前向能量崩塌请求
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_path = "tempdata/mother_language_state_v2.pt" # 升级至 V2 物理结晶
        self.is_loaded = False
        self.tokenizer = None
        self.vocab_size = 50257
        self.represent_dim = 3000
        self.dim_l2 = 1024
        
        # 物理张量阵列
        self.W_receptors = None # L1: [vocab, dim_l1]
        self.P_topo = None      # L1拓扑: [dim_l1, dim_l1]
        self.W_L2 = None        # L2耦合: [dim_l1, dim_l2]
        self.P_L2 = None        # L2拓扑: [dim_l2, dim_l2]
        
    def _fallback_tokenizer(self, text):
        words = text.split()
        return [(hash(w) % self.vocab_size) for w in words]
        
    def load_brain_crystal(self):
        """加载 Phase XXXI 多层物理引力图和感受器"""
        print(f"[Mother Engine Service] Waking up Phase XXXI brain from {self.state_path}...")
        try:
            # 加载 Tokenizer
            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)
            except Exception as e:
                print(f"[Mother Engine Service] Tokenizer warning... Error: {e}")

            if not os.path.exists(self.state_path):
                print(f"[Mother Engine Service] File not found: {self.state_path}. Falling back to V1...")
                self.state_path = "tempdata/mother_language_state.pt"
                if not os.path.exists(self.state_path): return False

            state = torch.load(self.state_path, map_location=self.device)
            self.vocab_size = state.get('vocab_size', 50257)
            self.represent_dim = state.get('represent_dim', 3000)
            
            self.W_receptors = state['W_receptors'].to(self.device)
            self.P_topo = state['P_topo'].to(self.device)
            
            # 支持 V2 多层权重
            if 'W_L2' in state:
                self.W_L2 = state['W_L2'].to(self.device)
                self.P_L2 = state['P_L2'].to(self.device)
                self.dim_l2 = state.get('dim_l2', 1024)
                print(f"[Mother Engine Service] Multi-layer Topology active! L1:{self.represent_dim}D -> L2:{self.dim_l2}D")
            
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"[Mother Engine Service] Failed to load crystal: {e}")
            return False

    def generate_energy_spikes(self, prompt_text: str, steps: int = 15):
        """
        核心法则 31：多层流形耦合与非线性阈值门控 (Phase XXXI)
        """
        if not self.is_loaded:
            if not self.load_brain_crystal():
                return {"error": "Physical Brain State could not be loaded."}
                
        # 1. 编码
        if self.tokenizer and hasattr(self.tokenizer, 'encode'):
            prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        else:
            prompt_tokens = self._fallback_tokenizer(prompt_text)
            
        current_energy = torch.zeros(self.vocab_size, device=self.device)
        for tid in prompt_tokens:
            current_energy[tid] += 1.0
            
        generated_ids = []
        energy_flow_traces = []
        
        # 全局相干性锚点 (Global Context Anchor)
        # 用于维持长程语义方向
        context_anchor = torch.zeros(self.represent_dim, device=self.device)
        
        # 推导循环
        for step_idx in range(steps):
            # --- 层级一：感知共振 (L1 Awareness) ---
            # [vocab] -> [dim_l1]
            h1 = torch.mv(self.W_receptors.t(), current_energy)
            h1 = torch.mv(self.P_topo, h1)
            
            # --- 非线性能量门控 (Non-linear Thresholding) ---
            # 只有足够强的信号才能进入高层逻辑抽象层
            # 模拟生物神经元的发放门槛，过滤线性噪声
            tau = 0.02
            h1_gated = torch.where(torch.abs(h1) > tau, h1, torch.zeros_like(h1))
            
            # --- 层级二：逻辑耦合 (L2 Logic) ---
            if self.W_L2 is not None:
                # [dim_l1] -> [dim_l2]
                h2 = torch.mv(self.W_L2.t(), h1_gated)
                h2 = torch.mv(self.P_L2, h2)
                
                # 强化跨层反馈：将 L2 的抽象逻辑落回 L1
                # 形成一种“意图指导感知”的闭环，显著提升连贯性
                logic_feedback = torch.mv(self.W_L2, h2)
                h1 = h1 + 0.3 * logic_feedback # 反馈系数 0.3
            
            # --- 全局锚点融合 (Context Guidance) ---
            h1 = h1 + 0.2 * context_anchor # 锚点引导
            
            # --- 回落词表 (Collapse to Vocab) ---
            # [dim_l1] -> [vocab]
            vocab_resonance = torch.mv(self.W_receptors, h1)
            
            # 动态反馈抑制 (Inhibition)
            # 增加随机相干噪声
            noise = torch.randn_like(vocab_resonance) * 0.015
            vocab_resonance += noise
            
            # 掩码已存在的高激活区，防止语义停滞
            active_mask = current_energy > 0.05
            vocab_resonance[active_mask] *= -0.2
            
            # --- 能量坍塌采样 (Top-K Sampling) ---
            top_k = min(8, self.vocab_size)
            values, indices = torch.topk(vocab_resonance, top_k)
            probs = torch.softmax(values * 6.0, dim=0) # 进一步提升逆温度以锁定逻辑
            
            sample_pos = torch.multinomial(probs, 1).item()
            best_id = indices[sample_pos].item()
            max_energy_val = values[sample_pos].item()
            
            generated_ids.append(best_id)
            
            # 轨迹记录
            energy_flow_traces.append({
                "step": step_idx + 1,
                "token_id": best_id,
                "resonance_energy": max_energy_val,
                "l2_active": self.W_L2 is not None
            })
            
            # --- 记忆状态演化 ---
            # 更新上下文锚点：模拟“思维流动”的残余
            context_anchor = 0.7 * context_anchor + 0.3 * h1
            
            # 词表能量更新
            current_energy[best_id] += 1.8
            current_energy *= 0.5 # 更快的衰减以推进语义流动
            
        # 3. 后处理解码
        generated_text = ""
        try:
            if self.tokenizer:
                generated_text = self.tokenizer.decode(generated_ids)
                for trace in energy_flow_traces:
                    trace["token_str"] = self.tokenizer.decode([trace["token_id"]])
            else:
                generated_text = f"IDs: {generated_ids}"
                for trace in energy_flow_traces:
                    trace["token_str"] = f"H_{trace['token_id']}"
        except:
            generated_text = "[Error during decoding]"
                
        return {
            "prompt": prompt_text,
            "generated_text": generated_text,
            "traces": energy_flow_traces,
            "status": "success",
            "physics_details": {
                "vocab": self.vocab_size,
                "represent_dim": f"L1:{self.represent_dim} -> L2:{self.dim_l2}",
                "non_linear_gate": "Thresholding Tau=0.02"
            }
        }

# 全局单例
mother_engine_service = MotherEngineService()
