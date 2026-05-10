"""Run Phase 85 experiments and save results to files."""
import sys
import os
sys.path.insert(0, "d:/Ai2050/TransformerLens-Project")
os.chdir("d:/Ai2050/TransformerLens-Project")

import io
import contextlib

# Redirect stdout to capture output
from tests.glm5.ccml_phase85_representation_dynamics import (
    exp_a_value_transport, exp_b_representation_trajectory,
    exp_c_causal_intervention, exp_d_recursive_rewriting
)

experiments = {
    "a": exp_a_value_transport,
    "b": exp_b_representation_trajectory,
    "c": exp_c_causal_intervention,
    "d": exp_d_recursive_rewriting,
}

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=str, required=True, choices=["a", "b", "c", "d"])
args = parser.parse_args()

# Capture output
output_buffer = io.StringIO()
with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
    experiments[args.exp]()

output = output_buffer.getvalue()

# Save to file
output_file = f"tests/glm5_temp/phase85_exp_{args.exp}_output.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(output)

print(f"Saved to {output_file}, length: {len(output)} chars", file=sys.stderr)
