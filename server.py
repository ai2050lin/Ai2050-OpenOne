import os

# Set environment variables for model loading
os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import transformer_lens
from agi_verification_api import run_agi_verification, run_concept_steering
from neurofiber_snn import NeuroFiberNetwork
from structure_analyzer import (
    CausalMediation,
    CircuitDiscovery,
    CompositionalAnalysis,
    LanguageValidity,
    ManifoldAnalysis,
    SparseAutoEncoder,
    export_graph_to_json,
)

app = FastAPI(title="TransformerLens 3D API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None

# Storage for structure analysis results
analysis_cache = {}

class PromptRequest(BaseModel):
    prompt: str
    top_k: int = 5

@app.on_event("startup")
async def load_model():
    global model
    import gc

    import torch

    # Try to load the preferred model with safety measures
    # model_name = "gpt2-small"  # Switched to small model to prevent CPU OOM crash
    model_name = "Qwen/Qwen3-4B"  # or "gpt2-small" for faster/safer loading
    use_half_precision = True  # Use float16 to reduce memory usage
    
    print(f"Loading model {model_name}...")
    if use_half_precision and "Qwen" in model_name:
        print("  Using half-precision (float16) to reduce memory usage...")
    
    try:
        if "Qwen" in model_name:
            # Load large model with memory optimizations
            if torch.cuda.is_available():
                # On GPU, use half precision safely
                print("  Loading on GPU with float16...")
                model = transformer_lens.HookedTransformer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if use_half_precision else torch.float32,
                    device="cuda"
                )
            else:
                # On CPU, always use float32 to avoid dtype mismatch errors
                print("  Running on CPU - using float32 to avoid dtype issues...")
                print("  This may take a moment and use ~16GB RAM...")
                
                try:
                    model = transformer_lens.HookedTransformer.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        dtype=torch.float32,  # Use 'dtype' instead of deprecated 'torch_dtype'
                        device="cpu"
                    )
                    print("  Model loaded, converting to float32...")
                    
                    # Explicitly convert all parameters to float32
                    model = model.float()
                    print("  Conversion complete!")
                    
                except (MemoryError, RuntimeError) as mem_error:
                    print(f"  ⚠ Memory error on CPU: {mem_error}")
                    print("  Falling back to gpt2-small...")
                    raise  # Re-raise to trigger fallback
        else:
            # Load small model normally
            model = transformer_lens.HookedTransformer.from_pretrained(model_name)
        
        print(f"✓ Model {model_name} loaded successfully!")
        
        # Clean up unused memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"✗ Error loading {model_name}: {e}")
        print(f"  Error type: {type(e).__name__}")
        print("  Falling back to gpt2-small...")
        
        try:
            # Fallback to small model
            model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")
            print("✓ Fallback model gpt2-small loaded successfully!")
        except Exception as fallback_error:
            print(f"✗ Critical error: Could not load fallback model: {fallback_error}")
            model = None

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/layer_detail/{layer_idx}")
async def get_layer_detail(layer_idx: int):
    """Get detailed information about a specific layer for 3D visualization"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if layer_idx < 0 or layer_idx >= model.cfg.n_layers:
        raise HTTPException(status_code=400, detail=f"Invalid layer index. Must be 0-{model.cfg.n_layers-1}")
    
    return {
        "layer_idx": layer_idx,
        "n_heads": model.cfg.n_heads,
        "d_head": model.cfg.d_head,
        "d_model": model.cfg.d_model,
        "d_mlp": model.cfg.d_mlp,
        "n_layers": model.cfg.n_layers,
    }


class GenerateRequest(BaseModel):
    prompt: str
    num_tokens: int = 10
    temperature: float = 0.7

@app.post("/generate_next")
async def generate_next_tokens(request: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        # Generate tokens
        generated_text = model.generate(
            request.prompt,
            max_new_tokens=request.num_tokens,
            temperature=request.temperature,
            stop_at_eos=False
        )
        
        return {
            "original_prompt": request.prompt,
            "generated_text": generated_text,
            "num_tokens_generated": request.num_tokens
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class LayerDetailsRequest(BaseModel):
    prompt: str
    layer_idx: int

@app.post("/layer_details")
async def get_layer_details(request: LayerDetailsRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        # Run model with cache
        logits, cache = model.run_with_cache(request.prompt)
        tokens = model.to_str_tokens(request.prompt)
        seq_len = len(tokens)
        layer_idx = request.layer_idx
        
        # Get attention patterns for this layer
        attn_pattern = cache[f"blocks.{layer_idx}.attn.hook_pattern"]  # shape: [batch, n_heads, seq_len, seq_len]
        attn_pattern = attn_pattern[0].cpu().detach().numpy()  # Remove batch dimension
        
        # Get MLP activations - try hook_mlp_out first, fallback to hook_post
        try:
            mlp_acts = cache[f"blocks.{layer_idx}.hook_mlp_out"]  # shape: [batch, seq_len, d_model]
        except KeyError:
            try:
                mlp_acts = cache[f"blocks.{layer_idx}.mlp.hook_post"]  # shape: [batch, seq_len, d_mlp]
            except KeyError:
                # If neither exists, use zeros
                mlp_acts = torch.zeros(1, seq_len, model.cfg.d_mlp)
        
        mlp_acts = mlp_acts[0].cpu().detach().numpy()
        
        # Get attention head outputs - use hook_z (per-head output after value projection)
        try:
            attn_out = cache[f"blocks.{layer_idx}.attn.hook_z"]  # shape: [batch, seq_len, n_heads, d_head]
        except KeyError:
            # Fallback: use zeros if not available
            attn_out = torch.zeros(1, seq_len, model.cfg.n_heads, model.cfg.d_head)
        
        attn_out = attn_out[0].cpu().detach().numpy()
        
        # Process attention patterns for each head
        attention_heads = []
        for head_idx in range(model.cfg.n_heads):
            head_pattern = attn_pattern[head_idx].tolist()  # [seq_len, seq_len]
            head_output_stats = {
                "mean": float(attn_out[:, head_idx, :].mean()),
                "std": float(attn_out[:, head_idx, :].std()),
                "max": float(attn_out[:, head_idx, :].max()),
                "min": float(attn_out[:, head_idx, :].min())
            }
            attention_heads.append({
                "head_idx": head_idx,
                "pattern": head_pattern,
                "output_stats": head_output_stats
            })
        
        # MLP activation statistics
        mlp_stats = {
            "mean": float(mlp_acts.mean()),
            "std": float(mlp_acts.std()),
            "max": float(mlp_acts.max()),
            "min": float(mlp_acts.min()),
            "activation_distribution": mlp_acts.mean(axis=0).tolist()[:100]  # First 100 neurons
        }
        
        return {
            "layer_idx": layer_idx,
            "tokens": tokens,
            "seq_len": seq_len,
            "n_heads": model.cfg.n_heads,
            "attention_heads": attention_heads,
            "mlp_stats": mlp_stats
        }
    except Exception as e:
        import traceback
        print(f"[LAYER DETAILS] Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class HeadDetailsRequest(BaseModel):
    prompt: str
    layer_idx: int
    head_idx: int

@app.post("/head_details")
async def get_head_details(request: HeadDetailsRequest):
    """Get detailed Q/K/V/Pattern info for a specific attention head"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        # Run model with cache
        logits, cache = model.run_with_cache(request.prompt)
        tokens = model.to_str_tokens(request.prompt)
        
        l, h = request.layer_idx, request.head_idx
        
        # Extract keys
        # hook_q: [batch, seq, n_heads, d_head]
        q_tensor = cache[f"blocks.{l}.attn.hook_q"]
        k_tensor = cache[f"blocks.{l}.attn.hook_k"]
        v_tensor = cache[f"blocks.{l}.attn.hook_v"]
        
        # Handle GQA (Grouped Query Attention) index mapping
        n_q_heads = model.cfg.n_heads
        n_kv_heads = k_tensor.shape[2] # [batch, seq, n_heads, d_head]
        
        # Calculate correct K/V head index
        if n_kv_heads < n_q_heads:
            kv_idx = int(h * n_kv_heads / n_q_heads)
        else:
            kv_idx = h
            
        q = q_tensor[0, :, h, :].detach().float().cpu().numpy()
        k = k_tensor[0, :, kv_idx, :].detach().float().cpu().numpy()
        v = v_tensor[0, :, kv_idx, :].detach().float().cpu().numpy()
        
        # Output z [batch, seq, n_heads, d_head]
        z = cache[f"blocks.{l}.attn.hook_z"][0, :, h, :].detach().float().cpu().numpy()
        
        # Pattern [batch, n_heads, seq_q, seq_k]
        pattern = cache[f"blocks.{l}.attn.hook_pattern"][0, h, :, :].detach().float().cpu().numpy()
        
        return {
            "layer_idx": l,
            "head_idx": h,
            "tokens": tokens,
            "q": q.tolist(),
            "k": k.tolist(),
            "v": v.tolist(),
            "z": z.tolist(),
            "pattern": pattern.tolist()
        }
    except Exception as e:
        import traceback
        error_msg = f"[HEAD DETAILS] Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        with open("server_error.log", "a") as f:
            f.write(error_msg + "\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_text(request: PromptRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        # Run model with cache
        logits, cache = model.run_with_cache(request.prompt)
        tokens = model.to_str_tokens(request.prompt)
        seq_len = len(tokens)
        n_layers = model.cfg.n_layers
        
        # Logit Lens processing
        # Data structure: [layer][pos] -> {token: str, prob: float}
        results = []
        
        for layer in range(n_layers):
            layer_data = []
            resid = cache[f"blocks.{layer}.hook_resid_post"]
            scaled_resid = model.ln_final(resid)
            layer_logits = model.unembed(scaled_resid)
            layer_probs = torch.softmax(layer_logits[0], dim=-1)
            
            for pos in range(seq_len):
                top_probs, top_indices = torch.topk(layer_probs[pos], 1) # Just top-1 for the 3D grid, can expand to top-k
                layer_data.append({
                    "token": model.to_string(top_indices[0].item()),
                    "prob": top_probs[0].item(),
                    "actual_token": tokens[pos]
                })
            results.append(layer_data)
            
        # Calculate total parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Collect per-layer detailed information
        layer_details = []
        for layer_idx in range(n_layers):
            # Get layer-specific parameters
            attn_params = sum(p.numel() for name, p in model.named_parameters() 
                            if f'blocks.{layer_idx}.attn' in name)
            mlp_params = sum(p.numel() for name, p in model.named_parameters() 
                           if f'blocks.{layer_idx}.mlp' in name)
            
            layer_details.append({
                "layer_idx": layer_idx,
                "attn_params": attn_params,
                "mlp_params": mlp_params,
                "total_params": attn_params + mlp_params,
                "n_heads": model.cfg.n_heads,
                "d_head": model.cfg.d_head,
                "d_mlp": model.cfg.d_mlp
            })
        
        return {
            "tokens": tokens,
            "layers": n_layers,
            "logit_lens": results,
            "model_config": {
                "name": "Qwen/Qwen3-4B",
                "n_layers": n_layers,
                "d_model": model.cfg.d_model,
                "n_heads": model.cfg.n_heads,
                "d_head": model.cfg.d_head,
                "total_params": total_params,
                "vocab_size": model.cfg.d_vocab
            },
            "layer_details": layer_details
        }
    except Exception as e:
        import traceback
        with open("server_error.log", "w") as f:
            f.write(f"Error in analyze_text: {str(e)}\n")
            traceback.print_exc(file=f)
        raise HTTPException(status_code=500, detail=str(e))

# ============ Structure Analysis Endpoints ============

class CircuitDiscoveryRequest(BaseModel):
    clean_prompt: str
    corrupted_prompt: str
    threshold: float = 0.1
    target_token_pos: int = -1  # Position to measure metric on
    target_layer: Optional[int] = None

@app.post("/discover_circuit")
async def discover_circuit(request: CircuitDiscoveryRequest):
    """Discover computational circuit using activation patching"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        # Tokenize inputs
        clean_tokens = model.to_tokens(request.clean_prompt)
        corrupted_tokens = model.to_tokens(request.corrupted_prompt)
        
        # Define metric: logit difference at target position
        def metric_fn(logits):
            target_logits = logits[0, request.target_token_pos]
            # Return max logit as simple metric
            return target_logits.max().item()
        
        # Run circuit discovery
        discoverer = CircuitDiscovery(model, metric_fn)
        nodes, graph = discoverer.discover_circuit(
            clean_tokens, 
            corrupted_tokens, 
            request.threshold,
            target_layer=request.target_layer
        )
        
        # Convert to JSON-serializable format
        circuit_data = {
            "nodes": [
                {
                    "id": str(node),
                    "layer": node.layer_idx,
                    "type": node.component_type,
                    "head": node.head_idx,
                    "importance": node.importance
                }
                for node in nodes
            ],
            "graph": export_graph_to_json(graph),
            "clean_prompt": request.clean_prompt,
            "corrupted_prompt": request.corrupted_prompt
        }
        
        # Cache result
        analysis_id = f"circuit_{len(analysis_cache)}"
        analysis_cache[analysis_id] = circuit_data
        
        return {
            "analysis_id": analysis_id,
            **circuit_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class FeatureExtractionRequest(BaseModel):
    prompt: str
    layer_idx: int
    hidden_dim: int = 2048  # SAE hidden dimension (overcomplete)
    sparsity_coef: float = 1e-3
    n_epochs: int = 50

@app.post("/extract_features")
async def extract_features(request: FeatureExtractionRequest):
    """Extract sparse features from layer activations using SAE"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        # Get activations from target layer
        tokens = model.to_tokens(request.prompt)
        _, cache = model.run_with_cache(tokens)
        
        # Get residual stream activations
        hook_name = f"blocks.{request.layer_idx}.hook_resid_post"
        activations = cache[hook_name]  # [batch, seq, d_model]
        
        # Flatten to [n_samples, d_model]
        batch_size, seq_len, d_model = activations.shape
        activations_flat = activations.reshape(-1, d_model)
        
        # Convert to float32 to avoid dtype mismatch issues
        activations_flat = activations_flat.float()
        
        # Train SAE
        sae = SparseAutoEncoder(
            input_dim=d_model,
            hidden_dim=request.hidden_dim,
            sparsity_coefficient=request.sparsity_coef
        )
        
        if torch.cuda.is_available():
            sae = sae.cuda()
            activations_flat = activations_flat.cuda()
        
        training_history = sae.train_on_activations(
            activations_flat,
            n_epochs=request.n_epochs,
            batch_size=256,
            lr=1e-3
        )
        
        # Get feature statistics
        features, feature_stats = sae.get_feature_activations(activations_flat)
        
        # Reshape features back to [batch, seq, hidden_dim]
        features = features.reshape(batch_size, seq_len, request.hidden_dim)
        
        # Get top active features
        feature_activation_means = features.mean(dim=(0, 1))
        top_k = 20
        top_features_idx = torch.topk(feature_activation_means, k=min(top_k, request.hidden_dim)).indices
        
        result = {
            "layer_idx": request.layer_idx,
            "n_features": request.hidden_dim,
            "training_history": training_history[-10:],  # Last 10 epochs
            "top_features": [
                {
                    **feature_stats[idx.item()],
                    "mean_activation_in_prompt": features[:, :, idx].mean().item()
                }
                for idx in top_features_idx
            ],
            "reconstruction_error": training_history[-1]["reconstruction"],
            "sparsity": training_history[-1]["active_features"]
        }
        
        # Cache SAE model
        analysis_id = f"sae_{len(analysis_cache)}"
        analysis_cache[analysis_id] = {
            "sae_model": sae,
            "layer_idx": request.layer_idx,
            **result
        }
        
        return {
            "analysis_id": analysis_id,
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CausalAnalysisRequest(BaseModel):
    prompt: str
    target_token_pos: int = -1
    importance_threshold: float = 0.01
    target_layer: Optional[int] = None

@app.post("/causal_analysis")
async def causal_analysis(request: CausalAnalysisRequest):
    """Perform causal mediation analysis on components"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    import time
    start_time = time.time()
    print(f"\n[CAUSAL ANALYSIS] Started at {time.strftime('%H:%M:%S')}")
    print(f"  Prompt: {request.prompt}")
    print(f"  Model: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads per layer")
    print(f"  Total components to analyze: {model.cfg.n_layers * model.cfg.n_heads + model.cfg.n_layers}")
    
    try:
        tokens = model.to_tokens(request.prompt)
        print(f"  Tokens: {len(tokens[0])} tokens")
        
        # Define metric function
        def metric_fn(logits):
            target_logits = logits[0, request.target_token_pos]
            return target_logits.max().item()
        
        # Run causal analysis
        print(f"  Starting component importance analysis...")
        analysis_start = time.time()
        causal = CausalMediation(model)
        importance_scores = causal.analyze_component_importance(
            tokens,
            metric_fn,
            ablation_value=0.0,
            target_layer=request.target_layer
        )
        analysis_time = time.time() - analysis_start
        print(f"  ✓ Analysis complete in {analysis_time:.1f}s")
        
        # Build causal graph
        print(f"  Building causal graph...")
        causal_graph = causal.build_causal_graph(
            importance_scores,
            threshold=request.importance_threshold
        )
        print(f"  ✓ Graph built with {len(causal_graph.nodes)} important nodes")
        
        # Sort by importance
        sorted_components = sorted(
            importance_scores.items(),
            key=lambda x: x[1]["direct_effect"],
            reverse=True
        )
        
        result = {
            "prompt": request.prompt,
            "n_components_analyzed": len(importance_scores),
            "n_important_components": len(causal_graph.nodes),
            "top_components": [
                {"name": name, **scores}
                for name, scores in sorted_components[:20]
            ],
            "causal_graph": export_graph_to_json(causal_graph)
        }
        
        # Cache result
        analysis_id = f"causal_{len(analysis_cache)}"
        analysis_cache[analysis_id] = result
        
        total_time = time.time() - start_time
        print(f"[CAUSAL ANALYSIS] ✓ Complete in {total_time:.1f}s")
        
        return {
            "analysis_id": analysis_id,
            **result
        }
    except Exception as e:
        print(f"[CAUSAL ANALYSIS] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/structure_graph/{analysis_id}")
async def get_structure_graph(analysis_id: str):
    """Retrieve cached structure analysis result"""
    if analysis_id not in analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis_cache[analysis_id]


class CircuitDiscoveryRequest(BaseModel):
    clean_prompt: str
    corrupted_prompt: str
    target_token_pos: int = -1
    threshold: float = 0.1
    target_layer: int = -1

@app.post("/discover_circuit")
async def discover_circuit(request: CircuitDiscoveryRequest):
    """Discover circuits based on activation patching"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
        
    try:
        # Basic implementation: 
        # Return a mock/placeholder graph for now to unblock the frontend
        
        graph = {
            "nodes": [
                {"id": "input", "layer": 0, "type": "input", "label": "Input"},
                {"id": "L0H1", "layer": 0, "type": "attention", "label": "L0H1 (Previous Token)"},
                {"id": "L1H4", "layer": 1, "type": "attention", "label": "L1H4 (Induction)"},
                {"id": "L2H0", "layer": 2, "type": "attention", "label": "L2H0"},
                {"id": "output", "layer": 3, "type": "output", "label": "Output"}
            ],
            "edges": [
                {"source": "input", "target": "L0H1", "weight": 0.8},
                {"source": "L0H1", "target": "L1H4", "weight": 0.6},
                {"source": "input", "target": "L1H4", "weight": 0.3},
                {"source": "L1H4", "target": "output", "weight": 0.9}
            ]
        }
        
        return {
            "status": "success",
            "graph": graph,
            "nodes": graph["nodes"],
            "edges": graph["edges"]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class ManifoldAnalysisRequest(BaseModel):
    prompt: str
    layer_idx: int
    n_components: int = 3

@app.post("/manifold_analysis")
async def perform_manifold_analysis(request: ManifoldAnalysisRequest):
    """Analyze the geometric structure of layer representations"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
        
    try:
        tokens = model.to_tokens(request.prompt)
        
        analyzer = ManifoldAnalysis(model)
        
        # Get activations
        activations = analyzer.get_layer_activations(tokens, request.layer_idx)
        
        # Compute Manifold Stats
        pca_result = analyzer.compute_pca(activations, n_components=request.n_components)
        id_result = analyzer.estimate_intrinsic_dimensionality(activations)
        
        result = {
            "layer_idx": request.layer_idx,
            "n_samples": activations.shape[0],
            "input_dim": activations.shape[1],
            "pca": pca_result,
            "intrinsic_dimensionality": id_result
        }
        
        # Cache
        analysis_id = f"manifold_{len(analysis_cache)}"
        analysis_cache[analysis_id] = result
        
        return {
            "analysis_id": analysis_id,
            **result
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class CompositionalAnalysisRequest(BaseModel):
    phrases: List[List[str]] # List of [part1, part2, compound]
    layer_idx: int

@app.post("/compositional_analysis")
async def analyze_compositionality(request: CompositionalAnalysisRequest):
    """Analyze compositionality of phrases"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
        
    try:
        analyzer = CompositionalAnalysis(model)
        
        # Convert list of lists to list of tuples
        phrases_tuples = [tuple(p) for p in request.phrases]
        
        result = analyzer.analyze_compositionality(
            phrases_tuples, 
            request.layer_idx
        )
        
        # Cache
        analysis_id = f"comp_{len(analysis_cache)}"
        analysis_cache[analysis_id] = result
        
        return {
            "analysis_id": analysis_id,
            **result
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))






class ValidityRequest(BaseModel):
    prompt: str
    target_layers: Optional[List[int]] = None

@app.post("/analyze_validity")
async def analyze_validity(request: ValidityRequest):
    """Analyze language validity metrics"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
        
    try:
        analyzer = LanguageValidity(model)
        
        # Determine target layers if not provided (start, middle, end)
        layers = request.target_layers
        if not layers:
            n = model.cfg.n_layers
            layers = [0, n // 2, n - 1]
            
        results = analyzer.analyze_holistic_validity(request.prompt, target_layers=layers)
        
        # Add metadata
        results["analysis_id"] = f"validity_{len(analysis_cache)}"
        results["layers_analyzed"] = layers
        
        # Cache results
        analysis_cache[results["analysis_id"]] = results
        
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify_agi")
async def verify_agi_endpoint():
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return run_agi_verification(model)

class SteeringRequest(BaseModel):
    prompt: str
    layer_idx: int = 15
    strength: float = 1.0
    concept_pair: str = "formal_casual"

@app.post("/steer_concept")
async def steer_concept_endpoint(request: SteeringRequest):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return run_concept_steering(
        model, 
        request.prompt, 
        request.layer_idx, 
        request.strength, 
        request.concept_pair
    )

# ============ SNN Endpoints ============

# Global SNN instance
snn_network = NeuroFiberNetwork()

class SNNInitRequest(BaseModel):
    layers: Dict[str, int] # e.g. {"Retina": 20, "Object": 20}
    connections: List[Dict[str, Any]] # e.g. [{"src": "Retina", "tgt": "Object", "type": "one_to_one"}]

@app.post("/snn/initialize")
async def initialize_snn(request: SNNInitRequest):
    global snn_network
    try:
        snn_network = NeuroFiberNetwork()
        
        # Create layers
        for name, size in request.layers.items():
            snn_network.add_layer(name, size)
            
        # Create connections
        for conn in request.connections:
            src = conn["src"]
            tgt = conn["tgt"]
            ctype = conn.get("type", "one_to_one")
            weight = conn.get("weight", 0.5)
            
            if ctype == "one_to_one":
                snn_network.connect_one_to_one(src, tgt, weight=weight)
            elif ctype == "all_to_all":
                snn_network.connect_all_to_all(src, tgt, weight=weight)
                
        return {
            "status": "initialized", 
            "layers": list(snn_network.neurons.keys()),
            "structure": snn_network.get_structure()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SNNStimulusRequest(BaseModel):
    layer_name: str
    pattern_idx: int
    intensity: float = 1.5

@app.post("/snn/stimulate")
async def inject_stimulus(request: SNNStimulusRequest):
    global snn_network
    try:
        if request.layer_name not in snn_network.neurons:
             raise HTTPException(status_code=404, detail=f"Layer {request.layer_name} not found")
             
        snn_network.inject_stimulus(request.layer_name, request.pattern_idx, request.intensity)
        return {"status": "injected", "time": snn_network.time}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SNNStepRequest(BaseModel):
    steps: int = 1

@app.post("/snn/step")
async def step_snn(request: SNNStepRequest):
    global snn_network
    try:
        # Accumulate unique firing indices over the steps
        history = {name: set() for name in snn_network.neurons}
        
        for _ in range(request.steps):
            step_spikes = snn_network.step_simulation()
            
            # step_spikes is Dict[str, List[int]]
            for layer_name, indices in step_spikes.items():
                for idx in indices:
                    history[layer_name].add(idx)
                    
        # Convert to lists for JSON serialization
        history = {k: list(v) for k, v in history.items()}
        
        return {
            "time": snn_network.time,
            "spikes": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
