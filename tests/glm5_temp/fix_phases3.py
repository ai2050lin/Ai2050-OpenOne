#!/usr/bin/env python3
"""Fix Phase 142-144: extract from git by line number and insert"""
import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Step 1: Extract from git commit d551e1d4 using known line numbers
print("Extracting Phase 142-144 from git commit d551e1d4...")
result = subprocess.run(['git', 'show', 'd551e1d4:research/glm5/docs/AGI_GLM5_MEMO.md'], 
                      capture_output=True)
git_content = result.stdout.decode('utf-8', errors='replace')
git_lines = git_content.split('\n')

# Phase 142 starts at line 67230 (0-indexed: 67229)
# Phase 145 starts at line 67218 is wrong... let me recalculate
# From earlier: Phase 145 at line 67218 in git, but Phase 142 at line 67230?
# That can't be right. Let me search again.

# Find Phase 142 and 145 by searching for the text 'Phase 142' and 'Phase 145'
phase142_idx = None
phase145_idx = None
for i, line in enumerate(git_lines):
    if 'Phase 142' in line and '## Phase' in line and phase142_idx is None:
        phase142_idx = i
    if 'Phase 145' in line and '## Phase' in line and phase145_idx is None:
        phase145_idx = i

print(f"Phase 142 at 0-indexed line: {phase142_idx}")
print(f"Phase 145 at 0-indexed line: {phase145_idx}")

if phase142_idx is None or phase145_idx is None:
    # Try broader search
    for i in range(67200, min(68000, len(git_lines))):
        if 'Phase 142' in git_lines[i]:
            print(f"  Found 'Phase 142' at line {i+1}: {git_lines[i][:80]}")
        if 'Phase 145' in git_lines[i]:
            print(f"  Found 'Phase 145' at line {i+1}: {git_lines[i][:80]}")

# Extract Phase 142-144 (include blank line before)
if phase142_idx is not None and phase145_idx is not None:
    start = phase142_idx
    if start > 0 and git_lines[start - 1].strip() == '':
        start = start - 1
    
    missing = '\n'.join(git_lines[start:phase145_idx])
    print(f"Phase 142-144: {phase145_idx - start} lines, {len(missing)} chars")
    
    # Step 2: Read current file
    with open('research/glm5/docs/AGI_GLM5_MEMO.md', 'r', encoding='utf-8') as f:
        current_content = f.read()
    
    current_lines = current_content.split('\n')
    print(f"Current file: {len(current_lines)} lines")
    
    # Find Phase 141 and Phase 145 in current file
    phase141_idx = None
    phase145_idx_cur = None
    for i, line in enumerate(current_lines):
        if 'Phase 141' in line and '## Phase' in line and phase141_idx is None:
            phase141_idx = i
        if 'Phase 145' in line and '## Phase' in line and phase145_idx_cur is None:
            phase145_idx_cur = i
    
    print(f"Current: Phase 141 at line {phase141_idx+1}, Phase 145 at line {phase145_idx_cur+1}")
    
    # Find end of Phase 141 content
    phase141_end = phase145_idx_cur
    for i in range(phase145_idx_cur - 1, phase141_idx, -1):
        if current_lines[i].strip() != '':
            phase141_end = i + 1
            break
    
    print(f"Phase 141 content ends at line {phase141_end+1}")
    
    # Check what's between Phase 141 end and Phase 145
    between = current_lines[phase141_end:phase145_idx_cur]
    print(f"Between Phase 141 and 145: {len(between)} lines")
    # Show first few lines
    for i, line in enumerate(between[:5]):
        print(f"  {line[:80]}")
    
    # Step 3: Reconstruct
    before = '\n'.join(current_lines[:phase141_end])
    after = '\n'.join(current_lines[phase145_idx_cur:])
    
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
    
    # Verify
    print("\nPhase 14x headers:")
    for i, line in enumerate(new_lines):
        if line.startswith('## Phase 14') and 'Phase 14' in line and not line.startswith('## Phase 14C') and not line.startswith('## 四'):
            print(f"  L{i+1}: {line[:80]}")
else:
    print("ERROR: Could not find Phase 142 or 145 in git version")
