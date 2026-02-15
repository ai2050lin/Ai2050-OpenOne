
import copy
import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to sys.path to find models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fibernet_v2 import DecoupledFiberNet


class FiberNetService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.vocabs = {}
        self.id2tokens = {}
        
        # Define Vocabs
        self.setup_vocabs()
        
        # Initialize and Train Models
        print("Initializing FiberNet Service...")
        self.train_models()
        print("FiberNet Service Ready.")

    def setup_vocabs(self):
        # English
        en_words = ["<PAD>", "I", "You", "We", "They", "love", "see", "know", "help", "him", "her", "it", "them"]
        self.vocabs["en"] = {w: i for i, w in enumerate(en_words)}
        self.id2tokens["en"] = {i: w for i, w in enumerate(en_words)}
        
        # French
        fr_words = ["<PAD>", "Je", "Tu", "Nous", "Ils", "aime", "vois", "sais", "aide", "le", "la", "lui", "les"]
        self.vocabs["fr"] = {w: i for i, w in enumerate(fr_words)}
        self.id2tokens["fr"] = {i: w for i, w in enumerate(fr_words)}
        
        # Vision (Synthetic Corner) classes
        # ... mapped to tokens for simplicity? Or handled separately?
        # Let's stick to text for this demo first.

    def train_models(self):
        # 1. Train English
        print("Training English FiberNet...")
        model_en = DecoupledFiberNet(vocab_size=len(self.vocabs["en"]), d_model=32, n_layers=2, group_type='circle', max_len=5).to(self.device)
        data_en = self.generate_svo_data(self.vocabs["en"], ["I", "You", "We", "They"], ["love", "see", "know", "help"], ["him", "her", "it", "them"])
        self.train_single(model_en, data_en, epochs=50)
        self.models["en"] = model_en
        
        # 2. Transfer to French (Frozen Logic)
        print("Transferring to French FiberNet...")
        model_fr = DecoupledFiberNet(vocab_size=len(self.vocabs["fr"]), d_model=32, n_layers=2, group_type='circle', max_len=5).to(self.device)
        
        # Copy Logic Weights
        logic_state = model_en.state_dict()
        new_state = model_fr.state_dict()
        for k, v in logic_state.items():
            if "content_embed" not in k and "head" not in k:
                new_state[k] = v
        model_fr.load_state_dict(new_state)
        
        # Train French (Frozen Logic)
        data_fr = self.generate_svo_data(self.vocabs["fr"], ["Je", "Tu", "Nous", "Ils"], ["aime", "vois", "sais", "aide"], ["le", "la", "lui", "les"])
        self.train_single(model_fr, data_fr, epochs=50, freeze_logic=True)
        self.models["fr"] = model_fr

    def generate_svo_data(self, vocab, subjs, verbs, objs, n_samples=200):
        data = []
        for _ in range(n_samples):
            s = random.choice(subjs)
            v = random.choice(verbs)
            o = random.choice(objs)
            ids = [vocab[s], vocab[v], vocab[o]]
            data.append(ids)
        return torch.tensor(data, dtype=torch.long).to(self.device)

    def train_single(self, model, data, epochs=50, freeze_logic=False):
        if freeze_logic:
            params = [p for n, p in model.named_parameters() if "content_embed" in n or "head" in n]
        else:
            params = model.parameters()
        optimizer = optim.Adam(params, lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(epochs):
            optimizer.zero_grad()
            inputs = data[:, :-1]
            targets = data[:, 1:]
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()

    def inference(self, text, lang="en"):
        # Preprocess
        vocab = self.vocabs.get(lang)
        if not vocab: return {"error": "Unsupported language"}
        
        tokens = text.strip().split()
        input_ids = [vocab.get(t, 0) for t in tokens] # 0 = PAD/Unk
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        model = self.models[lang]
        
        # We need to capture Attention. 
        # Since DecoupledFiberNet doesn't return it by default, we need a hook or modified forward.
        # Let's use the same trick as visualization script: wrap or hook.
        # But here we already instantiated DecoupledFiberNet.
        # We can register a forward hook on the layers.
        
        attns = []
        def get_attn_hook(module, input, output):
            # LogicDrivenAttention output is NOT the weights.
            # We need to compute weights inside. 
            # Or we can just monkey-patch the Layer's forward.
            # Actually, easiest is to just re-compute attention given input logic.
            # But we don't have easy access to input logic.
            pass
            
        # Let's run a custom forward pass for inference that exposes attention
        # Retain logic of model
        batch, seq = input_tensor.shape
        positions = torch.arange(seq, device=self.device).unsqueeze(0).expand(batch, -1)
        curr_logic = model.pos_embed(positions)
        curr_memory = model.content_embed(input_tensor)
        
        layer_attns = []
        token_embeddings = [] # Trace visualization
        
        for layer in model.layers:
            # Logic Evolve
            res_l = curr_logic
            curr_logic, _ = layer.logic_attn(curr_logic, curr_logic, curr_logic)
            curr_logic = layer.logic_norm1(res_l + curr_logic)
            res_l = curr_logic
            curr_logic = layer.logic_norm2(res_l + layer.logic_ffn(curr_logic))
            
            # Logic Driven Attention Calculation
            lda = layer.attn
            head_dim_logic = lda.d_logic // lda.nhead
            Q = lda.W_Q(curr_logic).reshape(batch, seq, lda.nhead, head_dim_logic).transpose(1, 2)
            K = lda.W_K(curr_logic).reshape(batch, seq, lda.nhead, head_dim_logic).transpose(1, 2)
            scores = (Q @ K.transpose(-2, -1)) / (head_dim_logic ** 0.5)
            attn_weights = torch.softmax(scores, dim=-1) # [B, Head, Seq, Seq]
            
            # Store for frontend (Average heads? Or return all? Return Head 0 for simplicity)
            layer_attns.append(attn_weights[0, 0].detach().cpu().numpy().tolist())
            
            # Memory Evolve
            head_dim_memory = lda.d_memory // lda.nhead
            V = lda.W_V(curr_memory).reshape(batch, seq, lda.nhead, head_dim_memory).transpose(1, 2)
            transported = attn_weights @ V
            transported = transported.transpose(1, 2).flatten(2)
            transported = lda.W_O(transported)
            
            res_m = curr_memory
            curr_memory = layer.mem_norm1(res_m + transported)
            res_m = curr_memory
            curr_memory = layer.mem_norm2(res_m + layer.mem_ffn(curr_memory))
            
            token_embeddings.append(curr_memory[0].detach().cpu().numpy().tolist())

        output_logits = model.head(curr_memory)
        preds = torch.argmax(output_logits, dim=-1)
        
        # Next token prediction for the last position
        next_token_id = preds[0, -1].item()
        next_token = self.id2tokens[lang].get(next_token_id, "???")
        
        return {
            "tokens": tokens,
            "attention": layer_attns, # List of [Seq, Seq]
            "next_token": next_token,
            "embeddings": token_embeddings # List of [Seq, Dim]
        }

# Global instance
fibernet_service = FiberNetService()
