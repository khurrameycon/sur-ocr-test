# bank_statement_processor/run_setup.py
"""
Setup and run script for bank statement processor
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def setup_project():
    """Set up the project structure and dependencies"""
    print("🚀 Setting up Bank Statement Processor")
    print("=" * 50)
    
    # Create project structure
    print("1. Creating project structure...")
    try:
        exec(open("setup_project.py").read())
        print("   ✅ Project structure created")
    except FileNotFoundError:
        print("   ⚠️  setup_project.py not found, creating directories manually...")
        directories = [
            "bank_statement_processor/src/core",
            "bank_statement_processor/src/models", 
            "bank_statement_processor/src/utils",
            "bank_statement_processor/src/config",
            "bank_statement_processor/data/input",
            "bank_statement_processor/data/output",
            "bank_statement_processor/data/templates",
            "bank_statement_processor/tests",
            "bank_statement_processor/logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"     Created: {directory}")
    
    # Check if Surya is available
    print("\n2. Checking Surya OCR availability...")
    try:
        import sys
        sys.path.append("../")  # Assuming surya is in parent directory
        import surya
        print("   ✅ Surya OCR is available")
    except ImportError:
        print("   ⚠️  Surya OCR not found")
        print("   💡 Make sure Surya is installed and accessible")
        print("   📖 Installation guide: https://github.com/VikParuchuri/surya")
    
    # Install additional requirements
    print("\n3. Installing additional requirements...")
    if Path("requirements.txt").exists():
        success = run_command(
            f"{sys.executable} -m pip install -r requirements.txt",
            "Installing requirements"
        )
        if not success:
            print("   ⚠️  Some packages might not have installed correctly")
    else:
        print("   ℹ️  requirements.txt not found, skipping...")
    
    print("\n✅ Setup completed!")

def run_tests():
    """Run the test suite"""
    print("\n🧪 Running tests...")
    print("=" * 30)
    
    if Path("test_processor.py").exists():
        success = run_command(
            f"{sys.executable} test_processor.py",
            "Running test suite"
        )
        return success
    else:
        print("❌ test_processor.py not found")
        return False

def show_usage_guide():
    """Show usage guide"""
    print("\n📖 Usage Guide")
    print("=" * 30)
    print("""
🏦 Bank Statement Processor - Quick Start

1. SETUP (one-time):
   • Run this script: python run_setup.py
   • Make sure Surya OCR is installed and working

2. PREPARE YOUR DATA:
   • Copy bank statement files to: data/input/
   • Supported formats: PDF, JPG, PNG, TIFF, BMP
   • Supported banks: BNI, Mandiri, OCBC, Danamon

3. PROCESS STATEMENTS:
   • Run: python test_processor.py
   • Or use the processor programmatically:
   
     from src.core.processor import BankStatementProcessor
     processor = BankStatementProcessor()
     result = processor.process_statement("statement.pdf")

4. CHECK RESULTS:
   • Results saved to: data/output/
   • JSON format with bounding boxes
   • Confidence scores included
   • LLM-ready markdown format

5. MONITOR PROCESSING:
   • Check logs in: logs/
   • Monitor confidence scores
   • Review validation errors

🎯 GOAL: Achieve 90%+ accuracy in field extraction

📋 NEXT PHASE:
   • Implement actual field extraction logic
   • Add bank-specific parsing rules
   • Optimize confidence scoring
   • Add validation and error handling
    """)

def check_environment():
    """Check if environment is ready"""
    print("🔍 Environment Check")
    print("=" * 30)
    
    checks = []
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python {python_version.major}.{python_version.minor} (OK)")
        checks.append(True)
    else:
        print(f"❌ Python {python_version.major}.{python_version.minor} (Need 3.8+)")
        checks.append(False)
    
    # Check required directories
    required_dirs = ["data/input", "data/output", "src/core"]
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"✅ Directory {directory} (OK)")
            checks.append(True)
        else:
            print(f"❌ Directory {directory} (Missing)")
            checks.append(False)
    
    # Check for input files
    input_dir = Path("data/input")
    if input_dir.exists():
        files = list(input_dir.glob("*"))
        if files:
            print(f"✅ Found {len(files)} files in input directory")
            checks.append(True)
        else:
            print("⚠️  No files in input directory")
            print("   💡 Add your bank statements to data/input/")
            checks.append(False)
    
    success_rate = sum(checks) / len(checks) * 100
    print(f"\n📊 Environment Status: {success_rate:.0f}% ready")
    
    return all(checks)

def main():
    """Main function"""
    print("🏦 Bank Statement Processor - Setup & Run")
    print("=" * 60)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Bank Statement Processor Setup")
    parser.add_argument("--setup", action="store_true", help="Run setup")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--check", action="store_true", help="Check environment")
    parser.add_argument("--guide", action="store_true", help="Show usage guide")
    
    args = parser.parse_args()
    
    if args.setup or not any(vars(args).values()):
        setup_project()
    
    if args.check:
        check_environment()
    
    if args.test:
        run_tests()
    
    if args.guide or not any(vars(args).values()):
        show_usage_guide()
    
    print("\n" + "=" * 60)
    print("🎉 Ready to process bank statements!")

if __name__ == "__main__":
    main()