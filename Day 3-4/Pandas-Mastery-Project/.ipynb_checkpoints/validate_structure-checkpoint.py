#!/usr/bin/env python3
"""
Simplified validation script for the Pandas Mastery Project notebooks.
This script checks basic file structure and imports without requiring full package installation.
"""

import sys
import os
from pathlib import Path


def validate_project_structure():
    """Validate basic project structure."""
    print("Validating project structure...")

    # Check required directories
    required_dirs = ["notebooks", "data", "utils"]
    all_good = True

    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"[OK] {dir_name}/ directory exists")
        else:
            print(f"[MISSING] {dir_name}/ directory missing")
            all_good = False

    # Check required notebooks
    required_notebooks = [
        "pandas_fundamentals.ipynb",
        "real_dataset_analysis.ipynb",
        "COVID19_Analysis.ipynb"
    ]

    for nb_name in required_notebooks:
        nb_path = Path("notebooks") / nb_name
        if nb_path.exists():
            print(f"[OK] {nb_name} exists")
        else:
            print(f"[MISSING] {nb_name} missing")
            all_good = False

    # Check utils module
    if Path("utils/utils.py").exists():
        print("[OK] utils/utils.py exists")
    else:
        print("[MISSING] utils/utils.py missing")
        all_good = False

    return all_good


def validate_basic_imports():
    """Validate that we can import basic libraries."""
    print("\nValidating basic imports...")

    try:
        # Try to import pandas (this may fail if not installed system-wide)
        import pandas as pd
        print("[OK] pandas can be imported")

        # Test basic pandas functionality
        test_series = pd.Series([1, 2, 3])
        test_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        print("[OK] Basic pandas operations work")

        return True
    except ImportError:
        print("! pandas not available, but this is acceptable for structure validation")
        return True  # Allow continuation for structural validation


def validate_notebook_structure():
    """Validate that notebooks have proper structure."""
    print("\nValidating notebook structure...")

    notebooks = ["pandas_fundamentals.ipynb", "real_dataset_analysis.ipynb", "COVID19_Analysis.ipynb"]
    all_good = True

    for nb_name in notebooks:
        nb_path = Path("notebooks") / nb_name
        if nb_path.exists():
            try:
                with open(nb_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '"cells"' in content and '"nbformat"' in content:
                        print(f"[OK] {nb_name} has valid notebook structure")
                    else:
                        print(f"[INVALID] {nb_name} may not be a valid notebook")
                        all_good = False
            except Exception as e:
                print(f"[ERROR] Could not read {nb_name}: {e}")
                all_good = False
        else:
            print(f"[MISSING] {nb_name} does not exist")
            all_good = False

    return all_good


def validate_readme_quickstart():
    """Validate that documentation files exist."""
    print("\nValidating documentation...")

    docs_exist = True

    if Path("README.md").exists():
        print("[OK] README.md exists")
    else:
        print("[MISSING] README.md missing")
        docs_exist = False

    if Path("quickstart.md").exists():
        print("[OK] quickstart.md exists")
    else:
        print("[MISSING] quickstart.md missing")
        docs_exist = False

    return docs_exist


def main():
    """Run all validation checks."""
    print("Starting validation of Pandas Mastery Project structure...")
    print("="*60)

    all_passed = True

    all_passed = validate_project_structure() and all_passed
    all_passed = validate_basic_imports() and all_passed
    all_passed = validate_notebook_structure() and all_passed
    all_passed = validate_readme_quickstart() and all_passed

    print("\n" + "="*60)
    if all_passed:
        print("SUCCESS: All structural validations passed! The Pandas Mastery Project is properly set up.")
        print("\nNext steps:")
        print("1. Install dependencies: uv pip install pandas matplotlib seaborn jupyter")
        print("2. Open Jupyter Notebook: jupyter notebook")
        print("3. Navigate to the notebooks/ directory")
        print("4. Start with pandas_fundamentals.ipynb")
        print("5. Progress through real_dataset_analysis.ipynb")
        print("6. Complete the project with COVID19_Analysis.ipynb")
        return 0
    else:
        print("ERROR: Some validations failed. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())