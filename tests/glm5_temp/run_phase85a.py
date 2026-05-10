"""Quick wrapper to run Phase 85 exp A and save output."""
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "tests/glm5/ccml_phase85_representation_dynamics.py", "--exp", "a"],
    capture_output=True, text=True, cwd="d:/Ai2050/TransformerLens-Project",
    encoding="utf-8", errors="replace"
)

with open("tests/glm5_temp/phase85_exp_a_output.txt", "w", encoding="utf-8") as f:
    f.write("=== STDOUT ===\n")
    f.write(result.stdout)
    f.write("\n=== STDERR ===\n")
    f.write(result.stderr)

print(f"Return code: {result.returncode}")
print(f"Output length: {len(result.stdout)} chars")
