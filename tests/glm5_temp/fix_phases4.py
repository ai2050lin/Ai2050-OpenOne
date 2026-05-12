#!/usr/bin/env python3
"""Fix Phase 142-144: extract from git by line number and insert before Phase 145"""
import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Step 1: Extract from git commit d551e1d4
print("Extracting Phase 142-144 from git commit d551e1d4...")
result = subprocess.run(['git', 'show', 'd551e1d4:research/glm5/docs/AGI_GLM5_MEMO.md'], 
                      capture_output=True)
git_content = result.stdout.decode('utf-8', errors='replace')
git_lines = git_content.split('\n')

# Phase 142 at 0-indexed 67229 (line 67230)
# Extract from line 67229 to end of file (since Phase 144 is the last)
# But we only need Phase 142-144, not the "Phase 145 future directions" section
# Let's find Phase 144 content and its end

# Find all Phase 14x headers by line content
phase_starts = {}
for i, line in enumerate(git_lines):
    if line == '## Phase 142: 局部几何与运输联络 [2026-05-12 18:00]':
        phase_starts[142] = i
    elif line == '## Phase 143: 传播几何 — 从"语义流形"到"约束传播系统" [2026-05-12 20:15]':
        phase_starts[143] = i
    elif line == '## Phase 144: 约束传播系统 — "局部光滑分层结构"的发现 [2026-05-12 22:30]':
        phase_starts[144] = i

print(f"Phase starts: {phase_starts}")

if 142 not in phase_starts or 144 not in phase_starts:
    print("ERROR: Could not find Phase 142 or 144")
    sys.exit(1)

# Phase 144 runs from its start to the end of the file
# We need Phase 142-144, i.e., from phase_starts[142] to end
# But exclude the "Phase 145-148 future directions" at the very end if it's just planning

# Let's extract from Phase 142 to end of file
start = phase_starts[142]
# Include blank line before if exists
if start > 0 and git_lines[start - 1].strip() == '':
    start = start - 1

# End: end of file (Phase 144 is the last content)
missing = '\n'.join(git_lines[start:])
print(f"Phase 142-144 content: {len(git_lines) - start} lines, {len(missing)} chars")

# Step 2: Read current file
with open('research/glm5/docs/AGI_GLM5_MEMO.md', 'r', encoding='utf-8') as f:
    current_content = f.read()

current_lines = current_content.split('\n')
print(f"Current file: {len(current_lines)} lines")

# Find Phase 141 end and Phase 145 start in current file
phase141_idx = None
phase145_idx = None
for i, line in enumerate(current_lines):
    if '## Phase 141:' in line and phase141_idx is None:
        phase141_idx = i
    if '## Phase 145:' in line and phase145_idx is None:
        phase145_idx = i

print(f"Current: Phase 141 at line {phase141_idx+1}, Phase 145 at line {phase145_idx+1}")

# Find end of Phase 141 content (last non-blank before Phase 145)
phase141_end = phase145_idx
for i in range(phase145_idx - 1, phase141_idx, -1):
    if current_lines[i].strip() != '':
        phase141_end = i + 1
        break

print(f"Phase 141 content ends at line {phase141_end+1}")

# Show what's between
between = current_lines[phase141_end:phase145_idx]
print(f"Content between Phase 141 and 145: {len(between)} lines")
for i, line in enumerate(between[:8]):
    print(f"  {line[:80]}")

# Step 3: Reconstruct - keep before Phase 141 end + insert Phase 142-144 + keep Phase 145+
before = '\n'.join(current_lines[:phase141_end])
after = '\n'.join(current_lines[phase145_idx:])

# Ensure proper spacing
if not before.endswith('\n'):
    before += '\n'
if not missing.endswith('\n'):
    missing += '\n'

new_content = before + missing + after

with open('research/glm5/docs/AGI_GLM5_MEMO.md', 'w', encoding='utf-8') as f:
    f.write(new_content)

new_lines = new_content.split('\n')
print(f"\nDone! New file: {len(new_lines)} lines (was {len(current_lines)})")

# Verify Phase 14x headers
print("\nPhase 14x headers in new file:")
for i, line in enumerate(new_lines):
    if line.startswith('## Phase 14') and 'Phase 14' in line and '四' not in line:
        print(f"  L{i+1}: {line[:80]}")
