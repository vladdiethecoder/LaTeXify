#!/usr/bin/env python3
"""
Generalized dependency checker and installer for LaTeXify.
Checks for system binaries and Python packages.
"""
import shutil
import sys
import subprocess
import platform
import os
from typing import List, NamedTuple, Optional

class Dependency(NamedTuple):
    name: str
    check_cmd: List[str]
    install_hint_debian: str
    install_hint_redhat: str
    install_hint_arch: str
    install_hint_mac: str

SYSTEM_DEPENDENCIES = [
    Dependency(
        "pdflatex",
        ["pdflatex", "--version"],
        "sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended",
        "sudo dnf install texlive-scheme-basic texlive-collection-latexextra",
        "sudo pacman -S texlive-core",
        "brew install basictex (or install MacTeX)"
    ),
    Dependency(
        "gs",
        ["gs", "--version"],
        "sudo apt-get install ghostscript",
        "sudo dnf install ghostscript",
        "sudo pacman -S ghostscript",
        "brew install ghostscript"
    ),
    Dependency(
        "pdf2svg",
        ["pdf2svg", "--help"], # pdf2svg often doesn't have --version
        "sudo apt-get install pdf2svg",
        "sudo dnf install pdf2svg",
        "sudo pacman -S pdf2svg",
        "brew install pdf2svg"
    ),
    Dependency(
        "chktex",
        ["chktex", "--version"],
        "sudo apt-get install chktex",
        "sudo dnf install chktex",
        "sudo pacman -S chktex",
        "brew install chktex"
    )
]

def check_dependency(dep: Dependency) -> bool:
    """Returns True if dependency is found, False otherwise."""
    cmd_name = dep.check_cmd[0]
    if not shutil.which(cmd_name):
        return False
    try:
        subprocess.run(
            dep.check_cmd, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL, 
            check=False
        )
        return True
    except FileNotFoundError:
        return False

def get_distro_id() -> str:
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("ID="):
                    return line.strip().split("=")[1].strip('"')
    except FileNotFoundError:
        pass
    return "unknown"

def main():
    missing = []
    print("[ensure_dependencies] Checking system dependencies...")
    
    for dep in SYSTEM_DEPENDENCIES:
        found = check_dependency(dep)
        status = "OK" if found else "MISSING"
        print(f"  - {dep.name}: {status}")
        if not found:
            missing.append(dep)

    if missing:
        print("\n[ensure_dependencies] ERROR: Missing required system dependencies.")
        print("Please install them using the following commands (or equivalent for your OS):")
        
        os_name = platform.system().lower()
        is_mac = "darwin" in os_name
        distro = get_distro_id()
        
        for dep in missing:
            if is_mac:
                hint = dep.install_hint_mac
            elif distro in ["fedora", "rhel", "centos", "almalinux", "rocky"]:
                hint = dep.install_hint_redhat
            elif distro in ["arch", "manjaro"]:
                hint = dep.install_hint_arch
            else: # Default to debian/ubuntu
                hint = dep.install_hint_debian
                
            print(f"\n  {dep.name}:\n    {hint}")
            
        sys.exit(1)
    
    print("\n[ensure_dependencies] All system dependencies satisfied.")
    sys.exit(0)

if __name__ == "__main__":
    main()
