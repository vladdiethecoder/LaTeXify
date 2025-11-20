
import sys
import os
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from release.tools.dependency_installer import ensure_release_dependencies, DependencyInstallError

def main():
    print("Checking release dependencies...")
    try:
        results = ensure_release_dependencies(auto_install_python=True)
        print("All dependencies satisfied:")
        for res in results:
            print(f"  [OK] {res.name}: {res.details}")
    except DependencyInstallError as e:
        print("\nDEPENDENCY ERROR:")
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

