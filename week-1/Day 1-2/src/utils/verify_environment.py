#!/usr/bin/env python3
"""
Environment Verification Script for NumPy Mastery Module

This script verifies that all required dependencies are installed
and available for the NumPy Learning Platform.
"""

import sys
import subprocess
import importlib.util

def check_package(package_name, import_name=None, version_check=None):
    """
    Check if a package is installed and optionally verify version.

    Args:
        package_name (str): Name of the package to check
        import_name (str): Import name if different from package_name
        version_check (str): Minimum version required

    Returns:
        tuple: (success: bool, version: str or None, error: str or None)
    """
    import_name = import_name or package_name

    try:
        # Try to import the package
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown')

        # Check version if specified
        if version_check and version != 'Unknown':
            from packaging.version import parse
            if parse(version) < parse(version_check):
                return False, version, f"Version {version} < {version_check}"

        return True, version, None

    except ImportError as e:
        return False, None, f"Not installed: {e}"
    except Exception as e:
        return False, None, f"Error importing: {e}"


def verify_environment():
    """
    Verify all required packages for the NumPy Mastery module.
    """
    print("=" * 60)
    print("NumPy Mastery - Environment Verification")
    print("=" * 60)

    # Python version check
    python_version = sys.version
    print(f"Python: {python_version}")
    print()

    # Required packages with their import names and version checks
    packages = [
        ("numpy", "numpy", "1.20.0"),
        ("matplotlib", "matplotlib", "3.5.0"),
        ("pillow", "PIL", "9.0.0"),
        ("ipython", "IPython", None),
        ("pytest", "pytest", None),
    ]

    all_passed = True
    results = []

    for package_name, import_name, min_version in packages:
        success, version, error = check_package(package_name, import_name, min_version)

        if success:
            status = "PASS"
            results.append(f"{package_name:12} {version:10} {status}")
        else:
            status = "FAIL"
            results.append(f"{package_name:12} {'N/A':10} {status}")
            if error:
                results.append(f"{'':12} {'':10}  └─ {error}")
            all_passed = False

    # Print results
    for line in results:
        try:
            print(line)
        except UnicodeEncodeError:
            # Fallback to ASCII encoding
            print(line.encode('ascii', 'ignore').decode('ascii'))

    print()
    print("-" * 60)

    if all_passed:
        print("Environment verification PASSED!")
        print("All required packages are installed and compatible.")
        return 0
    else:
        print("Environment verification FAILED!")
        print("Some packages are missing or incompatible.")
        print()
        print("To install missing packages, run:")
        print("  pip install numpy matplotlib pillow ipython pytest")
        return 1


if __name__ == "__main__":
    try:
        # Try to import packaging for version comparison
        import packaging.version  # noqa: F401
    except ImportError:
        print("Warning: 'packaging' module not available. Version checks disabled.")
        print()

    exit_code = verify_environment()
    sys.exit(exit_code)