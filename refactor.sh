#!/bin/bash
set -e

echo ">>> Starting Refactor: release/ -> src/latexify/..."

# 1. Prepare Directories
echo ">>> Creating directory structure..."
mkdir -p src/latexify
mkdir -p scripts

# 2. Move Scripts (Decoupling)
# Check if release/scripts exists before moving
if [ -d "release/scripts" ]; then
    echo ">>> Moving scripts to top-level scripts/..."
    # Move contents of release/scripts to scripts/
    mv release/scripts/* scripts/ 2>/dev/null || true
    # Remove the now empty release/scripts dir
    rmdir release/scripts 2>/dev/null || true
fi

# 3. Move Core Library
echo ">>> Moving core library to src/latexify/..."
# Move everything remaining in release/ to src/latexify/
# We use rsync to handle potential conflicts or complex moves gracefully, then remove source
rsync -av --remove-source-files release/ src/latexify/
# Clean up empty release dir
find release -depth -type d -empty -delete

# 4. Python Script for Smart Refactoring
# We use python for safer regex and file walking than complex sed commands
cat <<EOF > refactor_code.py
import os
import re
from pathlib import Path

# Mappings for text replacement
REPLACEMENTS = [
    (r'from release\b', 'from latexify'),
    (r'import release\b', 'import latexify'),
    (r'release.core', 'latexify.core'),
    (r'release.pipeline', 'latexify.pipeline'),
    (r'release.models', 'latexify.models'),
    (r'release.agents', 'latexify.agents'),
    (r'release.utils', 'latexify.utils'),
    (r'release.tools', 'latexify.tools'),
    # Update path constants often found in configs
    (r'release/inputs', 'src/latexify/inputs'),
    (r'release/samples', 'src/latexify/samples'),
]

DIRS_TO_SCAN = ['src', 'scripts', 'tests', 'apps', 'backend']
FILES_TO_SCAN = ['run_release.py', 'pyproject.toml', 'Makefile']

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        for pattern, repl in REPLACEMENTS:
            content = re.sub(pattern, repl, content)
        
        # Special handling for the ROOT/RELEASE_DIR constants usually found in config.py or main scripts
        # If we see 'RELEASE_DIR = ROOT / "release"', we change it to point to src/latexify
        # Also fixing broken paths in bootstrap_env.py and others that relied on structure
        content = content.replace('ROOT / "release"', 'ROOT / "src" / "latexify"')
        content = content.replace('RELEASE_DIR = Path(__file__).resolve().parents[1]', 'RELEASE_DIR = Path(__file__).resolve().parents[2] / "src" / "latexify"') 
        
        if content != original_content:
            print(f"Refactoring: {filepath}")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
    except UnicodeDecodeError:
        pass # Skip binary files

# Scan directories
for directory in DIRS_TO_SCAN:
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py') or file.endswith('.toml') or file.endswith('.md') or file.endswith('.json'):
                    process_file(os.path.join(root, file))

# Scan individual root files
for fname in FILES_TO_SCAN:
    if os.path.exists(fname):
        process_file(fname)

EOF

echo ">>> Executing code refactoring..."
python3 refactor_code.py
rm refactor_code.py

# 5. Rename Entry Point
if [ -f "run_release.py" ]; then
    echo ">>> Renaming run_release.py to run_latexify.py..."
    mv run_release.py run_latexify.py
fi

# 6. Update pyproject.toml for src layout
echo ">>> Updating pyproject.toml..."
# Use python to strictly modify toml if possible, or carefully append/replace
cat <<EOF > update_toml.py
import re

toml_file = 'pyproject.toml'
try:
    with open(toml_file, 'r') as f:
        content = f.read()

    # Replace py-modules = [] with package discovery
    if 'py-modules = []' in content:
        content = content.replace('py-modules = []', '[tool.setuptools.packages.find]\nwhere = ["src"]')
    else:
        # Fallback or append if setuptools section exists
        if '[tool.setuptools]' in content:
             content = content.replace('[tool.setuptools]', '[tool.setuptools]\npackages = ["latexify"]\npackage-dir = {"" = "src"}')

    with open(toml_file, 'w') as f:
        f.write(content)
except FileNotFoundError:
    print("pyproject.toml not found, skipping update.")
EOF
python3 update_toml.py
rm update_toml.py

echo ">>> Refactoring Complete."
echo "New Structure:"
echo "  src/latexify/  (Core Library)"
echo "  scripts/       (Operational Scripts)"
echo "  run_latexify.py (Entry Point)"
echo "  pyproject.toml (Updated)"
