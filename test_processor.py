# bank_statement_processor/test_processor.py
"""
Test script for the bank statement processor
"""
import sys
from pathlib import Path

# Add the src directory to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from core.processor import BankStatementProcessor
from config.settings import settings
import logging

# Set up logging for testing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_basic_functionality():
    """Test basic processor functionality"""
    print("🧪 Testing Bank Statement Processor")
    print("=" * 50)
    
    try:
        # Initialize processor
        print("1. Initializing processor...")
        processor = BankStatementProcessor()
        print("   ✅ Processor initialized successfully")
        
        # Check if input directory exists and has files
        input_dir = settings.INPUT_DIR
        print(f"2. Checking input directory: {input_dir}")
        
        if not input_dir.exists():
            print(f"   ⚠️  Input directory doesn't exist. Creating: {input_dir}")
            input_dir.mkdir(parents=True, exist_ok=True)
        
        # Find supported files
        supported_files = []
        for ext in settings.SUPPORTED_FORMATS:
            supported_files.extend(list(input_dir.glob(f"*{ext}")))
        
        print(f"   📁 Found {len(supported_files)} supported files")
        
        if supported_files:
            print("   Files found:")
            for file_path in supported_files[:5]:  # Show first 5 files
                print(f"     - {file_path.name}")
            
            # Test processing one file
            test_file = supported_files[0]
            print(f"\n3. Testing processing of: {test_file.name}")
            
            result = processor.process_statement(test_file)
            
            if result.success:
                print("   ✅ Processing completed successfully!")
                print(f"   📊 Bank type detected: {result.statement.bank_type}")
                print(f"   📄 Pages processed: {result.statement.metadata.pages_processed}")
                print(f"   ⏱️  Processing time: {result.statement.metadata.processing_time_seconds:.2f}s")
                print(f"   🎯 Overall confidence: {result.statement.metadata.overall_confidence:.2f}")
            else:
                print("   ❌ Processing failed:")
                for error in result.errors:
                    print(f"     - {error}")
        
        else:
            print("   📋 No files found for testing")
            print("   💡 To test the processor:")
            print(f"     1. Copy your bank statement images to: {input_dir}")
            print("     2. Run this script again")
            print(f"     3. Supported formats: {', '.join(settings.SUPPORTED_FORMATS)}")
        
        print(f"\n4. Output directory: {settings.OUTPUT_DIR}")
        print(f"   📝 Check this directory for processing results")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_utilities():
    """Test utility functions"""
    print("\n🔧 Testing Utility Functions")
    print("=" * 50)
    
    try:
        from utils.text_utils import TextProcessor
        from utils.image_utils import ImagePreprocessor
        
        # Test text processor
        print("1. Testing Text Processor...")
        text_processor = TextProcessor()
        
        # Test date extraction
        sample_text = "Statement period: 01/07/2023 to 31/07/2023"
        dates = text_processor.extract_dates(sample_text)
        print(f"   📅 Extracted {len(dates)} dates from sample text")
        
        # Test amount extraction
        sample_amounts = "Debit: 1,234,567.89 Credit: 987,654.32"
        amounts = text_processor.extract_amounts(sample_amounts)
        print(f"   💰 Extracted {len(amounts)} amounts from sample text")
        
        # Test account number extraction
        sample_account = "Account Number: 1667942876"
        accounts = text_processor.extract_account_numbers(sample_account)
        print(f"   🏦 Extracted {len(accounts)} account numbers from sample text")
        
        print("   ✅ Text processor working correctly")
        
        # Test image preprocessor
        print("2. Testing Image Preprocessor...")
        image_preprocessor = ImagePreprocessor()
        print("   ✅ Image preprocessor initialized successfully")
        
    except Exception as e:
        print(f"❌ Utility test failed: {e}")

def demonstrate_usage():
    """Demonstrate how to use the processor"""
    print("\n📖 Usage Examples")
    print("=" * 50)
    
    print("""
# Basic usage example:
from core.processor import BankStatementProcessor

# Initialize processor
processor = BankStatementProcessor()

# Process a single file
result = processor.process_statement(Path("statement.pdf"))

if result.success:
    statement = result.statement
    print(f"Bank: {statement.bank_type}")
    print(f"Account: {statement.account_info.account_number.value}")
    print(f"Transactions: {len(statement.transactions)}")

# Process multiple files
results = processor.process_batch(
    input_dir=Path("data/input"),
    output_dir=Path("data/output")
)

# Check results
for result in results:
    if result.success:
        print(f"✅ {result.statement.metadata.file_name}")
    else:
        print(f"❌ Failed: {result.errors}")
    """)

def setup_demo_environment():
    """Set up demo environment with sample data"""
    print("\n🏗️  Setting Up Demo Environment")
    print("=" * 50)
    
    try:
        # Create all necessary directories
        directories = [
            settings.INPUT_DIR,
            settings.OUTPUT_DIR,
            settings.TEMPLATES_DIR,
            settings.LOGS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"   📁 Created/verified: {directory}")
        
        # Create a sample configuration file
        config_file = current_dir / "config.example.json"
        sample_config = {
            "processing": {
                "ocr_confidence_threshold": 0.7,
                "enable_preprocessing": True,
                "batch_size": 4
            },
            "banks": {
                "BNI": {
                    "currency": "IDR",
                    "date_formats": ["DD/MM/YYYY", "DD-MM-YYYY"],
                    "language": "indonesian"
                },
                "MANDIRI": {
                    "currency": "IDR", 
                    "date_formats": ["DD/MM/YYYY"],
                    "language": "indonesian"
                }
            }
        }
        
        import json
        with open(config_file, 'w') as f:
            json.dump(sample_config, f, indent=2)
        print(f"   ⚙️  Created sample config: {config_file}")
        
        # Create a README file
        readme_file = current_dir / "README.md"
        readme_content = """# Bank Statement Processor

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place your bank statement files in `data/input/`

3. Run the processor:
   ```bash
   python test_processor.py
   ```

## Supported Banks
- BNI (Bank Negara Indonesia)
- Mandiri
- OCBC NISP
- Danamon

## Supported Formats
- PDF files
- Image files (JPG, PNG, TIFF, BMP)

## Output
- JSON files with extracted data
- Bounding box information
- Confidence scores
- LLM-ready markdown format

## Directory Structure
```
bank_statement_processor/
├── data/
│   ├── input/          # Place your statement files here
│   ├── output/         # Processed results
│   └── templates/      # Bank-specific templates
├── src/
│   ├── core/          # Main processing logic
│   ├── models/        # Data models
│   ├── utils/         # Utility functions
│   └── config/        # Configuration
└── logs/              # Processing logs
```
"""
        
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        print(f"   📚 Created README: {readme_file}")
        
        print("\n✅ Demo environment setup complete!")
        print("\n📋 Next Steps:")
        print("1. Copy your bank statement images to data/input/")
        print("2. Run: python test_processor.py")
        print("3. Check results in data/output/")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")

def main():
    """Main test function"""
    print("🚀 Bank Statement Processor - Test Suite")
    print("=" * 60)
    
    # Setup demo environment first
    setup_demo_environment()
    
    # Test utilities
    test_utilities()
    
    # Test basic functionality
    test_basic_functionality()
    
    # Show usage examples
    demonstrate_usage()
    
    print("\n" + "=" * 60)
    print("✨ Test suite completed!")
    print("\n📌 Remember to:")
    print("1. Place your bank statement files in data/input/")
    print("2. Check the logs/ directory for detailed processing logs")
    print("3. Review results in data/output/")
    print("\n🎯 Goal: Achieve 90%+ accuracy in field extraction")

if __name__ == "__main__":
    main()