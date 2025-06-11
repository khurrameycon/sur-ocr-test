# bank_statement_processor/setup_project.py
"""
Script to set up the project structure for bank statement processing
"""
import os
from pathlib import Path

def create_project_structure():
    """Create the complete project directory structure"""
    
    # Define the directory structure
    directories = [
        "bank_statement_processor",
        "bank_statement_processor/src",
        "bank_statement_processor/src/core",
        "bank_statement_processor/src/models",
        "bank_statement_processor/src/utils",
        "bank_statement_processor/src/config",
        "bank_statement_processor/data",
        "bank_statement_processor/data/input",
        "bank_statement_processor/data/output",
        "bank_statement_processor/data/templates",
        "bank_statement_processor/tests",
        "bank_statement_processor/logs"
    ]
    
    # Create directories
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create __init__.py files for Python packages
    init_files = [
        "bank_statement_processor/__init__.py",
        "bank_statement_processor/src/__init__.py",
        "bank_statement_processor/src/core/__init__.py",
        "bank_statement_processor/src/models/__init__.py",
        "bank_statement_processor/src/utils/__init__.py",
        "bank_statement_processor/src/config/__init__.py",
        "bank_statement_processor/tests/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"Created __init__.py: {init_file}")

if __name__ == "__main__":
    create_project_structure()
    print("\n✅ Project structure created successfully!")
    print("\nNext steps:")
    print("1. Run this script to create the directory structure")
    print("2. Copy your bank statement images to data/input/")
    print("3. We'll start implementing the core components")