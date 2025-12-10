import importlib.metadata
import importlib.util
import sys
import re
import shutil
from pathlib import Path

# Configuration
REQ_FILE_PATHS = [
    Path("pvp-mockup-be/requirements.txt"),
    Path("requirements.txt"), # Fallback if run from inside the folder
    Path("/app/requirements.txt") # Fallback for container paths
]

def log(msg, status="INFO"):
    colors = {"INFO": "\033[94m", "SUCCESS": "\033[92m", "ERROR": "\033[91m", "RESET": "\033[0m"}
    print(f"{colors.get(status, '')}[{status}] {msg}{colors['RESET']}")

def get_installed_packages():
    """Returns a set of all installed distribution names (normalized)."""
    return {dist.metadata["Name"].lower().replace("_", "-") for dist in importlib.metadata.distributions()}

def parse_requirements(path):
    """Parses requirements.txt, handling git URLs and version specifiers."""
    if not path.exists():
        return None
    
    with open(path, 'r', encoding='utf-16') as f:
        lines = [l.strip() for l in f.readlines()]
    
    packages = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
            
        # Handle Git URLs (e.g., -e git+https://...?egg=pvp_core_lib)
        if "git+" in line and "egg=" in line:
            pkg_name = line.split("egg=")[-1]
        # Handle local directory installs (e.g., -e .)
        elif line.startswith("-e"):
            continue 
        # Handle standard packages (e.g., pandas==2.0.0)
        else:
            # Regex to split on ==, >=, <=, <, >, or whitespace
            pkg_name = re.split(r"[=<>!~;]", line)[0].strip()
            
        packages.append(pkg_name.lower().replace("_", "-"))
        
    return packages

def check_dependencies():
    log("Checking Backend Dependencies...")
    
    # 1. Find requirements.txt
    req_path = next((p for p in REQ_FILE_PATHS if p.exists()), None)
    if not req_path:
        log("Could not find requirements.txt in expected paths.", "ERROR")
        return False
    
    log(f"Using requirements file: {req_path}")
    required = parse_requirements(req_path)
    
    if not required:
        print(f"Requirements not found!")
        return False

    installed = get_installed_packages()
    
    missing = []
    for req in required:
        # Special handling for pvp-core-lib if installed in editable mode
        if req == "pvp-core-lib":
            if "pvp-core-lib" in installed:
                continue
        
        if req not in installed:
            missing.append(req)
            
    if missing:
        log(f"Missing Packages: {', '.join(missing)}", "ERROR")
        return False
    else:
        log("All requirements.txt packages are installed.", "SUCCESS")

    # 2. Check Critical Imports (Runtime check)
    log("Verifying imports...")
    critical_modules = ["uvicorn", "fastapi", "pvp_core_lib", "sqlalchemy"]
    all_imports_ok = True
    
    for mod in critical_modules:
        if importlib.util.find_spec(mod) is None:
            log(f"Module '{mod}' is NOT importable (Installation broken?)", "ERROR")
            all_imports_ok = False
        else:
            log(f"Module '{mod}' found.", "SUCCESS")

    # 3. Check Uvicorn Executable (Path check)
    if shutil.which("uvicorn") is None:
        log("'uvicorn' executable not found in PATH (CMD ['uvicorn'] will fail)", "ERROR")
        log("Recommendation: Use 'python -m uvicorn' in your Dockerfile", "INFO")
        # We don't fail here if the module is importable, as python -m works
    
    return all_imports_ok

if __name__ == "__main__":
    if check_dependencies():
        log("Backend Environment is Ready!", "SUCCESS")
        sys.exit(0)
    else:
        log("Backend Environment Checks Failed.", "ERROR")
        sys.exit(1)