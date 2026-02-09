import os
import re

file_path = 'AGI_RESEARCH_MEMO.md'
if not os.path.exists(file_path):
    print("File not found.")
    exit(1)

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    original = line
    
    # Skip code blocks (simple heuristic: toggle flag)
    # We won't implement full state machine here, assuming lists are main issue.
    
    # 1. Fix list marker spacing: "*   " or "-   " -> "* " or "- "
    # Regex: Start of line, optional whitespace, marker (* or - or +), 2 or more spaces, text...
    # We want to keep the leading whitespace length, keep the marker, reduce marker-text gap to 1 space.
    
    match = re.match(r'^(\s*[-*+])\s{2,}(.*)', line)
    if match:
        prefix = match.group(1)
        content = match.group(2)
        line = f"{prefix} {content}\n"
    
    # 2. Fix indentation: 4 spaces -> 2 spaces for nested lists
    # Only if it looks like a list item
    # Check if it starts with 4 spaces and a marker
    if line.startswith("    * ") or line.startswith("    - ") or line.startswith("    + "):
        line = "  " + line.lstrip()
    
    # Also fix 8 spaces -> 4 spaces?
    if line.startswith("        * ") or line.startswith("        - "):
        line = "    " + line.lstrip()

    new_lines.append(line)

# Pass 2: Ensure blank lines around headings
final_lines = []
for i, line in enumerate(new_lines):
    is_heading = re.match(r'^\s*#{1,6}\s', line)
    
    # Add blank line BEFORE heading if not present
    if is_heading and i > 0:
        prev_line = final_lines[-1]
        if prev_line.strip() != "":
            final_lines.append("\n")
            
    final_lines.append(line)
    
    # Add blank line AFTER heading if not present
    if is_heading and i < len(new_lines) - 1:
        next_line = new_lines[i+1]
        # We can't check next_line easily here without lookahead.
        # But we can just append a newline if the next line is not empty?
        # Actually, let's just ensure we append a newline if the current line is a heading 
        # and the *original* next line wasn't empty. 
        # Simplest is to just ensure the formatting is consistent.
        # But inserting lines changes indices.
        pass

# Re-run pass 2 properly
final_lines_2 = []
for i, line in enumerate(final_lines):
    final_lines_2.append(line)
    is_heading = re.match(r'^\s*#{1,6}\s', line)
    if is_heading and i < len(final_lines) - 1:
        if final_lines[i+1].strip() != "":
             final_lines_2.append("\n")

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(final_lines_2)

print("Fixed markdown errors.")
