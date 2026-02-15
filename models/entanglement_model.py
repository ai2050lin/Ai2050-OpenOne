
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fiber_net import ConnectionLayer, FiberBundle


class VisualBranch(nn.Module):
    """
    Manifold: Sequence of SO(3) Rotation Matrices.
    Fiber: Object Identity (e.g., Cube, Sphere).
    """
    def __init__(self, num_objects, d_model=64):
        super().__init__()
        self.d_model = d_model
        
        # Fiber: Object Embedding
        self.object_embedding = nn.Embedding(num_objects, d_model)
        
        # Manifold: Process 3x3 matrices
        self.matrix_encoder = nn.Linear(9, d_model)
        self.rnn = nn.LSTM(d_model, d_model, batch_first=True)
        
        # Connection
        self.connection = ConnectionLayer(d_model, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 100, d_model)) # Max len 100

    def forward(self, rotation_matrices, object_ids):
        # rotation_matrices: [batch, seq, 3, 3]
        # object_ids: [batch] (One object per sequence for simplicity, or seq of them)
        
        b, s, _, _ = rotation_matrices.shape
        
        # 1. Manifold Stream (Geometry/Structure)
        flat_matrices = rotation_matrices.view(b, s, 9)
        m_emb = self.matrix_encoder(flat_matrices)
        m_states, _ = self.rnn(m_emb) # [batch, seq, d_model]
        
        # 2. Fiber Stream (Content)
        # Expand object id to sequence length
        obj_emb = self.object_embedding(object_ids).unsqueeze(1).expand(-1, s, -1) # [batch, seq, d_model]
        
        f_pos = self.pos_embed[:, :s, :]
        f_keys = obj_emb + f_pos
        
        # 3. Transport
        transported = self.connection(m_states, f_keys, obj_emb)
        
        return transported, m_states

class LanguageBranch(nn.Module):
    """
    Standard FiberNet for text.
    Structure: Syntax/Order.
    Fiber: Semantics.
    """
    def __init__(self, vocab_size, d_model=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.LSTM(d_model, d_model, batch_first=True)
        self.connection = ConnectionLayer(d_model, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 100, d_model))
        self.out = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids, initial_state=None):
        b, s = input_ids.shape
        emb = self.embedding(input_ids)
        
        # Manifold (Syntax) - approximated by LSTM over embeddings
        # If initial_state is provided (from Visual Bridge), use it to initialize LSTM
        if initial_state is not None:
            # LSTM expects (h_0, c_0). We usually map manifold state to h_0. 
            # c_0 can be zero or also mapped.
            # Assuming initial_state is [batch, d_model] -> expand to [1, batch, d_model]
            h_0 = initial_state.unsqueeze(0)
            c_0 = torch.zeros_as(h_0)
            m_states, _ = self.rnn(emb, (h_0, c_0))
        else:
            m_states, _ = self.rnn(emb)
        
        # Fiber (Semantics) - the embeddings themselves
        f_pos = self.pos_embed[:, :s, :]
        f_keys = emb + f_pos
        
        transported = self.connection(m_states, f_keys, emb)
        logits = self.out(transported)
        return logits, transported, m_states

class DualFiberNet(nn.Module):
    """
    Cross-Bundle Entanglement Model.
    Co-trains Visual and Language streams with an Entanglement Bridge.
    """
    def __init__(self, num_objects, vocab_size, d_model=64):
        super().__init__()
        self.visual = VisualBranch(num_objects, d_model)
        self.language = LanguageBranch(vocab_size, d_model)
        
        # The Bridge Phi: Maps Visual Manifold Tangent Space to Language Manifold Tangent Space
        self.bridge = nn.Linear(d_model, d_model)
        
    def forward(self, visual_input, lang_input):
        rot_mats, obj_ids = visual_input
        lang_ids = lang_input
        
        # Run Visual Stream
        v_out, v_manifold = self.visual(rot_mats, obj_ids)
        
        # Map Visual State to Language State (Entanglement)
        # We take the *last* visual state to initialize or guide the language generation
        # Or in training, we can force alignment at every step.
        # For simplicity, let's align the *final* states of the sequences.
        bridged_state = self.bridge(v_manifold) # [batch, seq, d_model]
        
        # Run Language Stream
        # During training, we don't necessarily inject, we just compare.
        # But to enforce "Visual guides Language", we could inject.
        # Let's injecting the *first* visual state (start) to init language?
        # Or better: Just compute the alignment loss and let them co-evolve.
        # But for inference, we need injection.
        
        l_out, l_transported, l_manifold = self.language(lang_ids)
        
        return l_out, bridged_state, l_manifold

    def generate_from_visual(self, visual_input, start_token_id, max_len=10):
        """
        Zero-Shot Generation:
        1. Encode Visual Sequence -> Visual Manifold State.
        2. Bridge -> Linked Language Manifold State.
        3. Decode Language from that state.
        """
        rot_mats, obj_ids = visual_input
        v_out, v_manifold = self.visual(rot_mats, obj_ids)
        
        # Use the final visual state to initialize language generation
        # (Assuming the visual sequence is "complete" action)
        final_visual_state = v_manifold[:, -1, :] 
        bridged_init = self.bridge(final_visual_state)
        
        # Auto-regressive decoding
        curr_token = torch.tensor([[start_token_id]], device=rot_mats.device)
        generated = []
        
        # Init LSTM state with bridged vector
        h_0 = bridged_init.unsqueeze(0)
        c_0 = torch.zeros_like(h_0)
        state = (h_0, c_0)
        
        for _ in range(max_len):
            # We need to expose state passing in LanguageBranch.forward or split it
            # For simplicity, let's just use the modified forward which takes init state
            # but that reinits every time if we pass full sequence.
            # We need step-by-step decoding.
            # Let's just implement a simple step here for the demo.
            
            emb = self.language.embedding(curr_token)
            m_out, state = self.language.rnn(emb, state)
            
            # Fiber part
            f_key = emb # Pos enc is tricky in step-by-step, ignore for now or keep generic
            transported = self.language.connection(m_out, f_key, emb)
            logits = self.language.out(transported)
            
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            generated.append(next_token.item())
            curr_token = next_token.unsqueeze(0)
            
            if next_token.item() == 2: # STOP token
                break
                
        return generated

