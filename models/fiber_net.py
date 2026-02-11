
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------------------
# AGI Unified Field Theory Core Components
# -----------------------------------------------------------------------------------------

class ManifoldStream(nn.Module):
    """
    Manifold Stream ($M$): Learns the syntactic/logical skeleton.
    
    Start with a simple RNN/LSTM for the manifold evolution. 
    In the theory, this represents Equation (1): dp/dt = X_u(p)
    """
    def __init__(self, vocab_size, d_manifold, n_layers=2):
        super().__init__()
        self.d_manifold = d_manifold
        self.embedding = nn.Embedding(vocab_size, d_manifold)
        # Using LSTM to capture trajectory/path integration properties
        # UPGRADE v1.5: Bidirectional=True is REQUIRED for Passive Voice detection.
        # At t=0, the model sees "Object", but needs to know if "BY" comes later to decide
        # whether to treat it as a Subject or Object. Global logical context is needed.
        self.rnn = nn.LSTM(d_manifold, d_manifold, n_layers, batch_first=True, bidirectional=True)
        
        # Project back to d_manifold (since bidirectional doubles the output size)
        self.proj = nn.Linear(d_manifold * 2, d_manifold)
        
    def forward(self, x):
        # x: [batch, seq_len] (structure tokens)
        embedded = self.embedding(x)
        # manifold_states: [batch, seq_len, d_manifold * 2]
        rnn_out, _ = self.rnn(embedded)
        
        # Project back to standard dimension for the Interaction Layer
        manifold_states = self.proj(rnn_out)
        
        return manifold_states

class FiberStream(nn.Module):
    """
    Fiber Stream ($F$): Stores semantic content (Knowledge Base).
    
    This acts as the "Atlas" of charts. 
    Ideally, these vectors should be orthogonal to the manifold space.
    """
    def __init__(self, num_concepts, d_fiber):
        super().__init__()
        self.d_fiber = d_fiber
        # The 'Fiber Bundle' - a collection of vectors at rest (canonical forms)
        # We don't train these with gradients in the 'Alice' test (Fast updating)
        self.fiber_memory = nn.Embedding(num_concepts, d_fiber)
        
    def forward(self, concept_ids):
        # concept_ids: [batch, seq_len] (content tokens)
        return self.fiber_memory(concept_ids)

class ConnectionLayer(nn.Module):
    """
    Connection Operator ($\nabla$): Defines how Fibers move over the Manifold.
    This replaces/extends the previous InteractionLayer with rigorous geometric definitions.
    
    Computes the Parallel Transport: F_q = P(p->q) * F_p
    """
    def __init__(self, d_manifold, d_fiber):
        super().__init__()
        # Manifold acts as the "Director" (Query)
        self.W_Q = nn.Linear(d_manifold, d_fiber)
        # Fibers act as the "Resources" (Key/Value)
        self.W_K = nn.Linear(d_fiber, d_fiber)
        self.W_V = nn.Linear(d_fiber, d_fiber)
        
        self.scale = d_fiber ** -0.5
        
        # For capturing the last transport matrix (for curvature analysis)
        self.last_transport_matrix = None
        
    def forward(self, m_states, f_states_k, f_states_v):
        # m_states: [batch, seq, d_m] -> Query (The Manifold asks "What matches this structure?")
        # f_states_k: [batch, seq, d_f] -> Key (Content + Position)
        # f_states_v: [batch, seq, d_f] -> Value (Content Only)
        
        Q = self.W_Q(m_states) # [batch, seq, d_f]
        K = self.W_K(f_states_k) # [batch, seq, d_f]
        V = self.W_V(f_states_v) # [batch, seq, d_f]
        
        # Compute Attention (The Connection Form A_mu)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1) # [batch, seq, seq]
        
        # Save for analysis
        self.last_transport_matrix = attn_weights.detach()
        
        # Transport Values (Content Only)
        # This is the 'Covariant Derivative' step roughly
        transported = torch.matmul(attn_weights, V) # [batch, seq, d_f]
        
        return transported

class CurvatureMonitor:
    """
    Curvature Monitor ($\Omega$): Tracks the holonomy of the connection.
    calculates \Omega_{mu,nu} = [D_mu, D_nu]
    """
    def __init__(self):
        self.history = []
    
    def compute_holonomy_error(self, transport_matrix, identity_target=True):
        """
        Computes deviation from identity (flatness) or orthogonality.
        For a closed loop, Transport should be Identity if curvature is zero.
        Here we approximate local curvature by checking if Transport is orthonormal/diagonal specific.
        """
        # transport_matrix: [batch, seq, seq]
        # In a perfect copy task, this should be close to a Permutation Matrix or Identity
        # We calculate Entropy of the attention rows as a proxy for 'Focus' (Sharpness of transport)
        
        # 1. Sharpness (Low Entropy -> High Confidence Transport)
        entropy = -torch.sum(transport_matrix * torch.log(transport_matrix + 1e-9), dim=-1).mean()
        
        # 2. Orthogonality (Preservation of Fiber Norm) - weak check here
        
        return entropy.item()

    def log(self, transport_matrix):
        error = self.compute_holonomy_error(transport_matrix)
        self.history.append(error)
        return error

class FiberBundle(nn.Module):
    """
    Fiber Bundle ($E$): The Total Space.
    Composes Manifold ($B$), Fiber ($F$), and Connection ($\nabla$).
    """
    def __init__(self, structure_vocab, content_vocab, d_manifold=64, d_fiber=512, max_len=512):
        super().__init__()
        self.manifold = ManifoldStream(structure_vocab, d_manifold)
        self.fiber = FiberStream(content_vocab, d_fiber)
        self.connection = ConnectionLayer(d_manifold, d_fiber)
        self.curvature_monitor = CurvatureMonitor()
        
        # Position embedding for differentiation of identical content at different locs
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_fiber) * 1.0)
        
    def forward(self, structure_ids, content_ids):
        # 1. Base Space Evolution
        m_states = self.manifold(structure_ids)
        
        # 2. Fiber Space Lift
        f_content = self.fiber(content_ids)
        seq_len = structure_ids.size(1)
        f_pos_only = self.pos_embed[:, :seq_len, :]
        f_states_k = f_content + f_pos_only # Key includes position
        
        # 3. Parallel Transport via Connection
        # Note: We use f_pos_only as Key to enforce 'Pattern Matching' not 'Content Matching'
        # This is the "Pure Structural Attention" upgrade
        transported_fiber = self.connection(m_states, f_pos_only, f_content)
        
        # 4. Projection / Readout
        # Project back to Semantic Space (Fiber)
        logits = F.linear(transported_fiber, self.fiber.fiber_memory.weight)
        
        return logits, transported_fiber, m_states
        
# -----------------------------------------------------------------------------------------
# Legacy / Reference Implementations (Kept for compatibility)
# -----------------------------------------------------------------------------------------

class InteractionLayer(ConnectionLayer):
    """Wrapper for backward compatibility"""
    pass

class FiberNet(FiberBundle):
    """Wrapper for backward compatibility, mapping old __init__ signature if needed"""
    def __init__(self, structure_vocab_size, content_vocab_size, d_manifold=64, d_fiber=512, max_len=512):
        super().__init__(structure_vocab_size, content_vocab_size, d_manifold, d_fiber, max_len)
        self.fiber_stream = self.fiber # Alias
        self.conn = self.connection # Alias

# -----------------------------------------------------------------------------------------
# FiberNet V2 Implementation (Advanced)
# -----------------------------------------------------------------------------------------

class ManifoldConstraint(nn.Module):
    def __init__(self, high_dim, low_dim):
        super().__init__()
        self.compress = nn.Linear(high_dim, low_dim)
        self.expand = nn.Linear(low_dim, high_dim)
        self.layer_norm = nn.LayerNorm(low_dim)
        
    def forward(self, x):
        manifold_point = self.compress(x)
        manifold_point = self.layer_norm(manifold_point)
        control_signal = self.expand(manifold_point)
        return control_signal

class AffineTransport(nn.Module):
    def __init__(self, d_manifold, d_fiber, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_fiber // n_heads
        
        self.W_Q = nn.Linear(d_manifold, d_fiber)
        self.W_K = nn.Linear(d_fiber, d_fiber)
        self.W_V = nn.Linear(d_fiber, d_fiber)
        self.W_Shift = nn.Linear(d_manifold, d_fiber)
        self.out = nn.Linear(d_fiber, d_fiber)
        self.scale = self.d_head ** -0.5
        
    def forward(self, manifold_state, pos_embed, fiber_content):
        b, s, _ = manifold_state.shape
        
        # Expand pos_embed to batch size if necessary
        if pos_embed.size(0) != b:
             pos_embed = pos_embed.expand(b, -1, -1)
             
        Q = self.W_Q(manifold_state).view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_K(pos_embed).view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_V(fiber_content).view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        
        transported = torch.matmul(attn, V).transpose(1, 2).contiguous().view(b, s, -1)
        shift = self.W_Shift(manifold_state)
        
        return self.out(transported + shift)

class FiberNetV2(nn.Module):
    def __init__(self, s_vocab, c_vocab, d_manifold=64, d_fiber=512, max_len=512):
        super().__init__()
        self.d_fiber = d_fiber
        self.fiber_mem = nn.Embedding(c_vocab, d_fiber)
        
        self.d_manifold = d_manifold
        self.manifold_embed = nn.Embedding(s_vocab, d_manifold)
        self.manifold_rnn = nn.LSTM(d_manifold, d_manifold, batch_first=True, bidirectional=True)
        self.manifold_proj = nn.Linear(d_manifold*2, d_manifold)
        
        self.constraint = ManifoldConstraint(d_manifold, 8) # Low-Dim Precision
        self.transport = AffineTransport(d_manifold, d_fiber) # Complex Connectivity
        
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_fiber))
        
        self.curvature_monitor = CurvatureMonitor()

    def forward(self, structure, content):
        m_emb = self.manifold_embed(structure)
        m_out, _ = self.manifold_rnn(m_emb)
        m_state = self.manifold_proj(m_out)
        
        m_refined = self.constraint(m_state)
        
        f_content = self.fiber_mem(content)
        seq_len = structure.size(1)
        f_pos = self.pos_embed[:, :seq_len, :]
        
        f_out = self.transport(m_refined, f_pos, f_content)
        
        logits = F.linear(f_out, self.fiber_mem.weight)
        return logits, f_out, m_refined

    def inject_knowledge(self, concept_id, concept_vector):
        with torch.no_grad():
             self.fiber_mem.weight[concept_id] = concept_vector

    def fast_associate(self, query_vector, k=5):
        with torch.no_grad():
            scores = torch.matmul(self.fiber_mem.weight, query_vector)
            vals, inds = torch.topk(scores, k)
            return inds, vals
