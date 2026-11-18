"""BF16 realized-norm matching along an observed native-control direction."""
def matched_direction(base,other,target_norm):
    direction=other.float()-base.float();original=float(direction.norm())
    if target_norm==0:return base.clone(),0.,0.,original
    assert original>0,'Zero native-control direction cannot match a nonzero edit'
    direction/=original;amount=target_norm;best=None
    for _ in range(32):
        value=(base.float()+amount*direction).to(base.dtype);actual=float((value.float()-base.float()).norm());error=abs(actual-target_norm)/target_norm
        if best is None or error<best[1]:best=(value,error,actual,original)
        if error<=.02:break
        amount*=target_norm/max(actual,1e-12)
    return best
