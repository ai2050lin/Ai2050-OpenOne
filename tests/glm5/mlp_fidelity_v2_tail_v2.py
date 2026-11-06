"""Preserve native full-sequence layout for exact BF16 unembedding kernels."""
from mlp_fidelity_v2_tail import capability_interval,cluster_interval,operations,metric

def tail_logits(model,residual,y,native_finalnorm_input,positions,keep):
    # Other positions retain actual native values. This is a real single-position
    # output replacement, including the native noncontiguous lm_head input stride.
    full=native_finalnorm_input.clone();batch=full.shape[0];full[:,positions]=(residual+y.to(residual.dtype)).reshape(batch,len(positions),full.shape[-1]);hidden=model.model.norm(full)
    selector=slice(-keep,None) if isinstance(keep,int) else keep
    logits=model.lm_head(hidden[:,selector]);return logits.reshape(-1,logits.shape[-1])
