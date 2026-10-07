# -*- coding: utf-8 -*-
"""F3 gate precheck: reachability + power.
Library + CLI. Import and call before any run.
"""
import math


def check_reachable(gate_val, lo, hi):
    """gate must lie in the constructible
    reachable domain [lo, hi]."""
    return lo <= gate_val <= hi


def mde_binomial(n, p0=0.2, alpha=0.05,
                 power=0.8):
    """minimal detectable effect for a
    binomial flip-rate gate."""
    z = {0.05: 1.959964, 0.01: 2.326348}[
        alpha]
    # normal-approx two-sample one-shot
    sd = math.sqrt(p0 * (1 - p0) / n)
    return z * sd / power


def precheck(gate_val, lo, hi, n, p0=0.2,
             alpha=0.05, power=0.8):
    m = mde_binomial(n, p0, alpha, power)
    ok = check_reachable(gate_val, lo, hi)
    return {'reachable': ok,
            'mde': m,
            'gate_lt_mde': gate_val < m,
            'verdict': ('RUN' if ok and
                        gate_val >= m else
                        'REJECT_OR_RAISE_N')}
