"""Check the saved native slices against original checkpoint bytes on CPU."""
from clt_b26 import BLOCK, HEAD, KV, D, OUT, load_run, save
from source_binding_runtime import Runtime
import numpy as np
import torch

pre, _, _ = load_run("discovery")
runtime = Runtime("qwen14")
q = slice(HEAD*128,(HEAD+1)*128)
k = slice(KV*128,(KV+1)*128)
specs = {
    "q": ("q_proj.weight", (q, slice(None))),
    "k": ("k_proj.weight", (k, slice(None))),
    "v": ("v_proj.weight", (k, slice(None))),
    "o": ("o_proj.weight", (slice(None), q)),
    "qnorm": ("q_norm.weight", (slice(None),)),
    "knorm": ("k_norm.weight", (slice(None),)),
}
verified = {}
with np.load(OUT / "native-parameters.npz") as parameters:
    for label,(name,index) in specs.items():
        key = f"model.layers.{BLOCK}.self_attn.{name}"
        layer = runtime.weight(key,"cpu")
        expected = layer[index].contiguous().view(torch.uint16).numpy()
        assert np.array_equal(expected,parameters[label]), key
        verified[label] = dict(tensor=key,n=expected.size,shape=expected.shape)
save("parameter-audit.json",dict(provenance=pre["provenance"],verified=verified,
                                native_sha=D.sha(OUT/"native-parameters.npz")))
print("Verified",sum(item["n"] for item in verified.values()),"native BF16 values")
