#!/usr/bin/env python3
"""Algebraic self-test for the Phase1081 route difference-in-differences."""

from __future__ import annotations

import numpy as np


def route(active: dict[tuple[int, int], np.ndarray], duplicate):
    active_left = 0.5 * (active[(0, 0)] + active[(1, 0)])
    active_right = 0.5 * (active[(1, 1)] + active[(0, 1)])
    duplicate_left = 0.5 * (duplicate[(0, 0)] + duplicate[(1, 0)])
    duplicate_right = 0.5 * (duplicate[(0, 1)] + duplicate[(1, 1)])
    content_left = 0.5 * (active_left + duplicate_right)
    content_right = 0.5 * (active_right + duplicate_left)
    return {
        "active": active_left - active_right,
        "duplicate": duplicate_left - duplicate_right,
        "content": content_left - content_right,
    }


def main() -> None:
    rng = np.random.default_rng(1081)
    active = {(m, q): rng.normal(size=17) for m in (0, 1) for q in (0, 1)}
    duplicate = {
        (m, q): rng.normal(size=17) for m in (0, 1) for q in (0, 1)
    }
    result = route(active, duplicate)
    expected = 0.5 * (result["active"] - result["duplicate"])
    assert np.allclose(result["content"], expected, atol=1e-12)

    identical = route(active, active)
    assert np.allclose(identical["content"], 0.0, atol=1e-12)

    offset = rng.normal(size=17)
    shifted_active = {key: value + offset for key, value in active.items()}
    shifted_duplicate = {key: value + offset for key, value in duplicate.items()}
    shifted = route(shifted_active, shifted_duplicate)
    for field in result:
        assert np.allclose(result[field], shifted[field], atol=1e-12)

    print({
        "phase": 1081,
        "status": "route_formula_selftest_passed",
        "content_scale": "0.5*(active_route-duplicate_route)",
    })


if __name__ == "__main__":
    main()
