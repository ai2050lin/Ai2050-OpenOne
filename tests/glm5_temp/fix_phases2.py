#!/usr/bin/env python3
"""Fix Phase 142-144: extract from git and insert after Phase 141"""
import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Step 1: Extract Phase 142-144 from git commit d551e1d4
print("Extracting Phase 142-144 from git commit d551e1d4...")
result = subprocess.run(['git', 'show', 'd551e1d4:research/glm5/docs/AGI_GLM5_MEMO.md'], 
                      capture_output=True)
git_content = result.stdout.decode('utf-8', errors='replace')
git_lines = git_content.split('\n')

# Find Phase 142 start and Phase 145 start in git version
phase142_start = None
phase145_start = None
for i, line in enumerate(git_lines):
    if '## Phase 142:' in line and phase142_start is None:
        phase142_start = i
    if '## Phase 145:' in line and phase145_start is None:
        phase145_start = i

print(f"Git: Phase 142 at line {phase142_start+1}, Phase 145 at line {phase145_start+1}")

# Extract Phase 142-144 content (include blank line before)
start = phase142_start
if start > 0 and git_lines[start - 1].strip() == '':
    start = start - 1

missing_142_144 = '\n'.join(git_lines[start:phase145_start])
print(f"Phase 142-144 content: {phase145_start - start} lines, {len(missing_142_144)} chars")

# Step 2: Read current file
print("Reading current MEMO file...")
with open('research/glm5/docs/AGI_GLM5_MEMO.md', 'r', encoding='utf-8') as f:
    current_content = f.read()

current_lines = current_content.split('\n')
print(f"Current file: {len(current_lines)} lines")

# Step 3: Find Phase 141 end and Phase 145 start in current file
phase141_line = None
phase145_line = None
for i, line in enumerate(current_lines):
    if '## Phase 141:' in line and phase141_line is None:
        phase141_line = i
    if '## Phase 145:' in line and phase145_line is None:
        phase145_line = i

print(f"Current: Phase 141 at line {phase141_line+1}, Phase 145 at line {phase145_line+1}")

# Find where Phase 141 content ends (last non-blank line before Phase 145)
phase141_end = phase145_line
for i in range(phase145_line - 1, phase141_line, -1):
    if current_lines[i].strip() != '':
        phase141_end = i + 1
        break

print(f"Phase 141 content ends at line {phase141_end+1}")

# Step 4: Also fix duplicate Phase 121-122
# Find the duplicate Phase 121 in the recovered section
dup_phase121 = None
for i in range(phase141_line + 1, phase145_line):
    if '## Phase 121:' in current_lines[i]:
        dup_phase121 = i
        print(f"Duplicate Phase 121 found at line {i+1}")
        break

# Step 5: Reconstruct file
# Strategy: 
# 1. Keep everything up to Phase 141's end
# 2. Insert Phase 142-144
# 3. Skip to Phase 145

# But we also need to remove the duplicate Phase 121-122 section
# The original Phase 121-122 is around L63183, the duplicate is around L63362

# Let's check if the content between Phase 141 end and Phase 145 is just duplicates
between_content = '\n'.join(current_lines[phase141_end:phase145_line])
print(f"Content between Phase 141 end and Phase 145: {len(between_content)} chars")

# Check if it's mostly Phase 121-122 duplicate
has_dup = False
for i in range(phase141_end, phase145_line):
    if '## Phase 121:' in current_lines[i] or '## Phase 122:' in current_lines[i]:
        has_dup = True
        break

if has_dup:
    print("Found duplicate Phase 121/122 between Phase 141 and 145 - will remove")
    # Find the first '## Phase 121:' after Phase 122 in original section
    original_phase122 = None
    for i, line in enumerate(current_lines):
        if '## Phase 122:' in line:
            original_phase122 = i
            break
    
    # The duplicate Phase 121 starts the section that was incorrectly placed
    # Remove from after Phase 122's real content to Phase 145
    # Actually, let's just rebuild from scratch with the correct structure
    
    # Keep lines 0 to phase141_end (all content including Phase 141)
    before = '\n'.join(current_lines[:phase141_end])
    after = '\n'.join(current_lines[phase145_line:])
    
    # Ensure proper spacing
    if not before.endswith('\n'):
        before += '\n'
    if not missing_142_144.endswith('\n'):
        missing_142_144 += '\n'
    
    new_content = before + missing_142_144 + after
    
    with open('research/glm5/docs/AGI_GLM5_MEMO.md', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    new_lines = new_content.split('\n')
    print(f"\nDone! New file: {len(new_lines)} lines (was {len(current_lines)})")
else:
    # Just insert Phase 142-144
    before = '\n'.join(current_lines[:phase141_end])
    after = '\n'.join(current_lines[phase145_line:])
    
    if not before.endswith('\n'):
        before += '\n'
    if not missing_142_144.endswith('\n'):
        missing_142_144 += '\n'
    
    new_content = before + missing_142_144 + after
    
    with open('research/glm5/docs/AGI_GLM5_MEMO.md', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    new_lines = new_content.split('\n')
    print(f"\nDone! New file: {len(new_lines)} lines (was {len(current_lines)})")

# Verify Phase 14x headers
print("\nPhase 14x headers in new file:")
for i, line in enumerate(new_lines):
    if line.startswith('## Phase 14') and 'Phase 14' in line:
        print(f"  L{i+1}: {line[:80]}")
