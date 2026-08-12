#!/usr/bin/env python3
"""Post-hoc template diagnostics for Phase1093; never upgrade frozen gates."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1092_natural_bilingual_attribute_finalize as analysis
import phase1092_posthoc_template_robustness as engine
import phase1093_independent_relation_protocol as protocol


def main() -> None:
    analysis.protocol = protocol
    engine.protocol = protocol
    engine.analysis = analysis
    engine.main()
    path = protocol.OUT_ROOT / "analysis" / "posthoc_template_robustness.json"
    result = protocol.read_json(path)
    result.pop("posthoc_digest", None)
    result["schema_version"] = "phase1093_posthoc_template_robustness.v1"
    result["interpretation"] = (
        "Template-specific comparisons were run after the frozen P1-P10 "
        "analysis. They diagnose aggregation sensitivity and cannot upgrade evidence."
    )
    result["posthoc_digest"] = protocol.digest(result)
    protocol.write_json(path, result)


if __name__ == "__main__":
    main()
