import sys
import subprocess
from typing import List, Tuple

REQUIRED_PACKAGES = {
    'numpy': '>=1.19.0',
    'pandas': '>=1.0.0',
    'yfinance': '>=0.2.0',
    'plotly': '>=5.0.0',
    'Flask': '>=3.0.0',
    'waitress': '>=3.0.0',
    'scikit-learn': '>=0.24.0',
    'nltk': '>=3.6.0',
    'beautifulsoup4': '>=4.9.0',
    'requests': '>=2.25.0',
    'joblib': '>=1.0.0',
    'ta': '>=0.10.0',
}

OPTIONAL_PACKAGES = {
    'tensorflow': '>=2.0.0',
    'transformers': '>=4.0.0',
    'torch': '>=1.0.0',
}

def check_package(package_name, required_version):
    try:
        import importlib.metadata
        installed_version = importlib.metadata.version(package_name)
        return True, f"  [OK] {package_name} {installed_version}"
    except importlib.metadata.PackageNotFoundError:
        return False, f"  [MISSING] {package_name} not found"
    except Exception as e:
        return False, f"  [ERROR] {package_name}: {e}"

def install_package(package_name, version_spec):
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            f"{package_name}{version_spec}"
        ])
        return True
    except subprocess.CalledProcessError:
        return False

def check_dependencies(auto_install=False):
    print("Checking dependencies...")
    print("-" * 50)

    all_satisfied = True
    missing_packages = []

    print("Required packages:")
    for package, version in REQUIRED_PACKAGES.items():
        satisfied, message = check_package(package, version)
        print(message)
        if not satisfied:
            all_satisfied = False
            missing_packages.append((package, version))

    print("\nOptional packages (app works without these):")
    for package, version in OPTIONAL_PACKAGES.items():
        satisfied, message = check_package(package, version)
        print(message + (" (optional)" if not satisfied else ""))

    if not all_satisfied and auto_install:
        print("\nInstalling missing required packages...")
        for package, version in missing_packages:
            print(f"Installing {package}{version}...")
            if install_package(package, version):
                print(f"Successfully installed {package}")
            else:
                print(f"Failed to install {package}")
                return False

        all_satisfied = all(check_package(p, v)[0] for p, v in REQUIRED_PACKAGES.items())

    print("-" * 50)
    if all_satisfied:
        print("All required dependencies are satisfied!")
    else:
        print("\nMissing dependencies. Install with:")
        for package, version in missing_packages:
            print(f"  pip install {package}{version}")

    return all_satisfied

if __name__ == "__main__":
    auto_install = "--install" in sys.argv
    if check_dependencies(auto_install):
        print("\nYou can now run: python app.py")
    else:
        print("\nPlease install missing dependencies before running.")
        sys.exit(1)
