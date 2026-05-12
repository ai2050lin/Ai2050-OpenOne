#!/usr/bin/env python3
"""Recover Phase 123-144 from git commit d551e1d4"""
import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Step 1: Extract Phase 123-144 from git commit
print("Extracting Phase 123-144 from git commit d551e1d4...")
result = subprocess.run(['git', 'show', 'd551e1d4:research/glm5/docs/AGI_GLM5_MEMO.md'], 
                      capture_output=True)
git_content = result.stdout.decode('utf-8', errors='replace')
git_lines = git_content.split('\n')

# Find Phase 123 start and Phase 145 start in git version
phase123_start = None
phase145_start = None
for i, line in enumerate(git_lines):
    if 'Phase 123:' in line and phase123_start is None:
        phase123_start = i
    if 'Phase 145:' in line and phase145_start is None:
        phase145_start = i

print(f"Git version: Phase 123 at line {phase123_start+1}, Phase 145 at line {phase145_start+1}")

# Extract the missing content (Phase 123 to just before Phase 145)
# We need lines from phase123_start to phase145_start-1
# But also include the blank line before Phase 123
start_line = phase123_start
if start_line > 0 and git_lines[start_line - 1].strip() == '':
    start_line = start_line - 1  # Include the blank line before

missing_content = '\n'.join(git_lines[start_line:phase145_start])
print(f"Missing content length: {len(missing_content)} chars, {phase145_start - start_line} lines")

# Step 2: Read current file
print("Reading current MEMO file...")
with open('research/glm5/docs/AGI_GLM5_MEMO.md', 'r', encoding='utf-8') as f:
    current_content = f.read()

current_lines = current_content.split('\n')
print(f"Current file: {len(current_lines)} lines")

# Step 3: Find Phase 122 end and Phase 145 start in current file
phase122_line = None
phase145_line = None
for i, line in enumerate(current_lines):
    if 'Phase 122:' in line and phase122_line is None:
        phase122_line = i
    if 'Phase 145:' in line and phase145_line is None:
        phase145_line = i

print(f"Current: Phase 122 at line {phase122_line+1}, Phase 145 at line {phase145_line+1}")

# Step 4: Find the end of Phase 122 content (just before Phase 145)
# In current file, Phase 122 content runs from phase122_line to phase145_line-1
# We need to replace everything from end of Phase 122 to Phase 145

# Actually, find the boundary: the last non-empty line before Phase 145 in current file
# that belongs to Phase 122's content
phase122_end = phase145_line
# Go back to find the actual end of Phase 122 content (skip blank lines before Phase 145)
for i in range(phase145_line - 1, phase122_line, -1):
    if current_lines[i].strip() != '':
        phase122_end = i + 1
        break

print(f"Phase 122 content ends at line {phase122_end+1}")

# Step 5: Reconstruct file
# Keep lines 0 to phase122_end (Phase 122 content)
# Insert missing Phase 123-144 content
# Keep Phase 145 to end
before = '\n'.join(current_lines[:phase122_end])
after = '\n'.join(current_lines[phase145_line:])

# Ensure proper spacing
if not before.endswith('\n'):
    before += '\n'
if not missing_content.startswith('\n'):
    missing_content = '\n' + missing_content
if not missing_content.endswith('\n'):
    missing_content += '\n'

new_content = before + missing_content + after

with open('research/glm5/docs/AGI_GLM5_MEMO.md', 'w', encoding='utf-8') as f:
    f.write(new_content)

# Verify
verify_lines = new_content.split('\n')
print(f"\nDone! New file: {len(verify_lines)} lines (was {len(current_lines)})")

# Check Phase headers in recovered section
phase_count = 0
for i, line in enumerate(verify_lines):
    if line.startswith('## Phase 1') and ('Phase 12' in line or 'Phase 13' in line or 'Phase 14' in line):
        phase_count += 1
        if phase_count <= 5 or 'Phase 14' in line:
            print(f"  L{i+1}: {line[:80]}")

print(f"\nRecovered {phase_count} Phase sections (123-144)")
