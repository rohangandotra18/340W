import os
import re

def fix_fstrings(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Find f-strings and remove f if they don't contain {
    # This is a simple regex that matches f"..." or f'...' with no { inside
    content = re.sub(r'f"([^"\{}]*)"', r'"\1"', content)
    content = re.sub(r"f'([^'\{}]*)'", r"'\1'", content)
    
    with open(filepath, 'w') as f:
        f.write(content)

for root, _, files in os.walk('.'):
    if 'venv' in root:
        continue
    for file in files:
        if file.endswith('.py'):
            fix_fstrings(os.path.join(root, file))
