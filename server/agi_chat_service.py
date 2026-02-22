import torch
from transformers import GPT2Tokenizer
from datasets import load_dataset
import threading
import os

class AGIChatEngine:
    def __init__(self):
        self.tokenizer = None
        self.P_topo = None
        self.N = 50257
        self.energy_state = None
        self.is_ready = False
        self.status_msg = "Uninitialized"
        
    def initialize_async(self):
        thread = threading.Thread(target=self.initialize)
        thread.daemon = True
        thread.start()

    def initialize(self):
        try:
            self.status_msg = "Loading Tokenizer..."
            print("[AGI Chat Engine] Initializing...")
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            
            self.status_msg = "Building chaotic ocean..."
            k_init = 200
            row_idx = torch.randint(0, self.N, (self.N * k_init,))
            col_idx = torch.randint(0, self.N, (self.N * k_init,))
            indices = torch.stack([row_idx, col_idx])
            values = torch.randn(self.N * k_init) * 0.01 
            self.P_topo = torch.sparse_coo_tensor(indices, values, size=(self.N, self.N)).coalesce()
            
            self.status_msg = "Washing with Offline OpenWebText..."
            print("[AGI Chat Engine] Washing with Local offline stream to build gravity basins...")
            
            temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tempdata")
            files = []
            if os.path.exists(temp_dir):
                files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.startswith("openwebtext_part_") and f.endswith(".txt")]
            
            batch_spike_indices = []
            batch_spike_values = []
            lr_p = 0.05
            decay = 0.001
            threshold = 0.005
            
            step = 0
            # Read first chunk locally to build initial topology
            if files:
                current_file = files[0]
                print(f"[AGI Chat Engine] Found local shards, using base shard: {os.path.basename(current_file)}")
                with open(current_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        text = line.strip()
                        if len(text) < 30: continue
                        
                        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                        if len(token_ids) == 0: continue
                        
                        for i in range(len(token_ids) - 1):
                            src = token_ids[i]
                            dst = token_ids[i+1]
                            batch_spike_indices.append(torch.tensor([[src], [dst]]))
                            batch_spike_values.append(torch.tensor([lr_p * 2.0])) 
                            
                        unique_tokens = list(set(token_ids))
                        if len(unique_tokens) > 1:
                            u_t = torch.tensor(unique_tokens)
                            grid_x, grid_y = torch.meshgrid(u_t, u_t, indexing='ij')
                            spike_indices = torch.stack([grid_x.flatten(), grid_y.flatten()])
                            spike_values = torch.ones(spike_indices.size(1)) * lr_p
                            batch_spike_indices.append(spike_indices)
                            batch_spike_values.append(spike_values)
                            
                        step += 1
                        if step % 300 == 0:
                            self.status_msg = f"Washing... {step}/1000 sentences"
                            if len(batch_spike_indices) > 0:
                                all_i = torch.cat(batch_spike_indices, dim=1)
                                all_v = torch.cat(batch_spike_values, dim=0)
                                self.P_topo = self.P_topo + torch.sparse_coo_tensor(all_i, all_v, size=(self.N, self.N))
                                batch_spike_indices = []
                                batch_spike_values = []
                            
                            self.P_topo = self.P_topo.coalesce()
                            current_vals = self.P_topo._values() * (1.0 - decay)
                            mask = torch.abs(current_vals) > threshold
                            self.P_topo = torch.sparse_coo_tensor(self.P_topo._indices()[:, mask], current_vals[mask], size=(self.N, self.N)).coalesce()
                            
                        if step >= 1000: break
            else:
                print("[AGI Chat Engine] No local shards found. Creating empty ready state.")
                
            self.status_msg = f"Ready ({self.P_topo._nnz():,} synapses)"
            print(f"[AGI Chat Engine] Ready! Network size: {self.P_topo._nnz():,} synapses.")
            self.energy_state = torch.zeros(self.N, dtype=torch.float32)
            self.is_ready = True
        except Exception as e:
            self.status_msg = f"Error: {e}"
            print(f"[AGI Chat Engine] Initialization failed: {e}")
        
    def generate(self, prompt_text, max_new_tokens=15, mem_decay=0.8):
        if not self.is_ready:
            return {"error": f"Engine is currently: {self.status_msg}", "status": "not_ready"}
            
        prompt_ids = self.tokenizer.encode(prompt_text)
        
        # Add to working memory
        for tid in prompt_ids:
            self.energy_state[tid] += 2.0
            
        generated_ids = []
        emitted_set = set(prompt_ids)
        
        tokens_flow = []
        energy_levels = []
        
        # 1. Synaptic Consolidation Threshold: Nodes seen < 5 times are noise (not consolidated)
        # 2. Strong Fractional Suppression: Since noise is pruned, we can safely punish hubs heavily!
        node_degrees = torch.sparse.sum(self.P_topo, dim=1).to_dense()
        consolidated_mask = (node_degrees > 5.0).float()
        degree_penalty = torch.pow(node_degrees.clamp(min=1.0), 0.85)
        
        for _ in range(max_new_tokens):
            active_mask = self.energy_state > 0
            if not active_mask.any():
                break
                
            focus_wave = torch.sparse_coo_tensor(
                torch.nonzero(self.energy_state).t(),
                self.energy_state[active_mask],
                size=(self.N,)
            ).coalesce()
            
            # Raw energy projection
            next_thoughts = torch.sparse.mm(self.P_topo, focus_wave.to_dense().unsqueeze(1)).squeeze()
            
            # 1. Apply Homeostatic Plasticity (strong fractional division)
            next_thoughts = next_thoughts / degree_penalty
            
            # 2. Apply Synaptic Consolidation Mask (prune completely unseen/unconsolidated noise)
            next_thoughts = next_thoughts * consolidated_mask
            
            # 3. Dynamic Noise Pruning: Only keep the Top 100 strongest activated thoughts.
            topk_vals, topk_indices = torch.topk(next_thoughts, min(100, self.N))
            pruned_thoughts = torch.full_like(next_thoughts, -9999.0)
            pruned_thoughts[topk_indices] = topk_vals
            next_thoughts = pruned_thoughts
            
            # Prevent endless loops
            for eid in emitted_set:
                next_thoughts[eid] = -9999.0
                
            # 4. Probabilistic Selection over the surviving attractor basin
            temperature = 0.5
            probs = torch.softmax(next_thoughts / temperature, dim=0)
            best_id = torch.multinomial(probs, 1).item()
            
            generated_ids.append(best_id)
            emitted_set.add(best_id)
            
            word = self.tokenizer.decode([best_id])
            tokens_flow.append(word)
            
            # Boost the latest thought's energy massively so it controls the next step
            # rather than getting washed out by the background global syntax.
            self.energy_state[best_id] += 20.0
            self.energy_state = self.energy_state * mem_decay
            
            # Record top 5 energy nodes for visual
            top_energies, top_indices = torch.topk(self.energy_state, 5)
            energy_levels.append([
                {"word": self.tokenizer.decode([idx.item()]), "energy": float(val.item())}
                for val, idx in zip(top_energies, top_indices) if val.item() > 0
            ])
            
        return {
            "status": "success",
            "prompt": prompt_text,
            "generated_text": "".join(tokens_flow),
            "tokens": tokens_flow,
            "working_memory_flow": energy_levels
        }
        
    def reset_memory(self):
        if self.is_ready:
            self.energy_state = torch.zeros(self.N, dtype=torch.float32)
        return {"status": "success", "message": "Working memory cleared."}

    def start_background_wash(self, max_files=1):
        if not self.is_ready:
            return {"status": "error", "message": "Engine not initialized yet."}
            
        thread = threading.Thread(target=self._wash_local_data, args=(max_files,))
        thread.daemon = True
        thread.start()
        return {"status": "success", "message": "Background washing started."}

    def _wash_local_data(self, max_files):
        temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tempdata")
        if not os.path.exists(temp_dir):
            print(f"[AGI Chat Engine] Tempdir not found: {temp_dir}")
            return
            
        files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.startswith("openwebtext_part_") and f.endswith(".txt")]
        if not files:
            print("[AGI Chat Engine] No openwebtext chunks found in tempdata.")
            return
            
        files = files[:max_files]
        print(f"[AGI Chat Engine] Starting background wash on {len(files)} files...")
        
        lr_p = 0.05
        decay = 0.001
        threshold = 0.005
        total_sentences = 0
        
        for file_path in files:
            try:
                print(f"[AGI Chat Engine] Washing with file: {os.path.basename(file_path)}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    batch_spike_indices = []
                    batch_spike_values = []
                    
                    for line in f:
                        text = line.strip()
                        if len(text) < 30: continue
                        
                        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                        if len(token_ids) == 0: continue
                        
                        for i in range(len(token_ids) - 1):
                            src = token_ids[i]
                            dst = token_ids[i+1]
                            batch_spike_indices.append(torch.tensor([[src], [dst]]))
                            batch_spike_values.append(torch.tensor([lr_p * 2.0]))
                            
                        unique_tokens = list(set(token_ids))
                        if len(unique_tokens) > 1:
                            u_t = torch.tensor(unique_tokens)
                            grid_x, grid_y = torch.meshgrid(u_t, u_t, indexing='ij')
                            spike_indices = torch.stack([grid_x.flatten(), grid_y.flatten()])
                            spike_values = torch.ones(spike_indices.size(1)) * lr_p
                            batch_spike_indices.append(spike_indices)
                            batch_spike_values.append(spike_values)
                            
                        total_sentences += 1
                        
                        if total_sentences % 500 == 0:
                            self.status_msg = f"Background washing... {total_sentences} sentences processed."
                            if len(batch_spike_indices) > 0:
                                all_i = torch.cat(batch_spike_indices, dim=1)
                                all_v = torch.cat(batch_spike_values, dim=0)
                                self.P_topo = self.P_topo + torch.sparse_coo_tensor(all_i, all_v, size=(self.N, self.N))
                                batch_spike_indices = []
                                batch_spike_values = []
                            
                            self.P_topo = self.P_topo.coalesce()
                            current_vals = self.P_topo._values() * (1.0 - decay)
                            mask = torch.abs(current_vals) > threshold
                            self.P_topo = torch.sparse_coo_tensor(self.P_topo._indices()[:, mask], current_vals[mask], size=(self.N, self.N)).coalesce()
                            
                self.status_msg = f"Ready. Last washed {total_sentences} sentences. ({self.P_topo._nnz():,} synapses)"
                print(f"[AGI Chat Engine] Background wash file complete! Network size: {self.P_topo._nnz():,} synapses.")
            except Exception as fe:
                print(f"[AGI Chat Engine] Error parsing {file_path}: {fe}")

agi_chat_engine = AGIChatEngine()
