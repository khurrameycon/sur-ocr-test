# bank_statement_processor/src/core/processor.py
"""
Main bank statement processor using Surya OCR
"""
import os
import sys
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import json

# Add the parent directory to sys.path to import surya
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
surya_path = project_root / "surya"  # Assuming surya is in the same parent directory
sys.path.insert(0, str(surya_path))

try:
    from surya.models import load_predictors
    from surya.common.surya.schema import TaskNames
    from surya.input.processing import convert_if_not_rgb
except ImportError as e:
    print(f"Error importing Surya: {e}")
    print(f"Make sure Surya is installed and accessible from: {surya_path}")
    sys.exit(1)

from ..models.bank_schemas import (
    BankStatement, ProcessingResult, ProcessingMetadata, 
    BankType, FieldConfidence, BoundingBox, ExtractedField
)
from ..config.settings import settings, BANK_CONFIGS
from ..utils.image_utils import ImagePreprocessor
from ..utils.text_utils import TextProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.LOGS_DIR / 'processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BankStatementProcessor:
    """Main processor for bank statements using Surya OCR"""
    
    def __init__(self):
        """Initialize the processor with Surya models"""
        logger.info("Initializing Bank Statement Processor...")
        
        # Initialize Surya predictors
        try:
            self.predictors = load_predictors()
            logger.info("✅ Surya predictors loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load Surya predictors: {e}")
            raise
        
        # Initialize utility classes
        self.image_preprocessor = ImagePreprocessor()
        self.text_processor = TextProcessor()
        
        logger.info("🚀 Bank Statement Processor initialized successfully")
    
    def process_statement(
        self, 
        file_path: Path, 
        bank_type: Optional[str] = None
    ) -> ProcessingResult:
        """
        Process a single bank statement file
        
        Args:
            file_path: Path to the statement file
            bank_type: Optional bank type hint (BNI, MANDIRI, OCBC, DANAMON)
            
        Returns:
            ProcessingResult with extracted data
        """
        start_time = time.time()
        logger.info(f"🔄 Processing statement: {file_path}")
        
        try:
            # Validate file
            if not file_path.exists():
                return ProcessingResult(
                    success=False,
                    errors=[f"File not found: {file_path}"]
                )
            
            # Load and preprocess image
            images = self._load_images(file_path)
            if not images:
                return ProcessingResult(
                    success=False,
                    errors=["No valid images found in file"]
                )
            
            # Detect bank type if not provided
            if not bank_type:
                bank_type = self._detect_bank_type(images[0])
                logger.info(f"🏦 Detected bank type: {bank_type}")
            
            # Process with Surya OCR
            ocr_results = self._run_surya_ocr(images)
            
            # Extract structured data
            statement = self._extract_statement_data(
                ocr_results, bank_type, file_path
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            statement.metadata.processing_time_seconds = processing_time
            
            # Validate extracted data
            validation_errors = self._validate_statement(statement)
            statement.validation_errors = validation_errors
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(statement)
            statement.metadata.overall_confidence = overall_confidence
            
            logger.info(f"✅ Processing completed in {processing_time:.2f}s with confidence {overall_confidence:.2f}")
            
            return ProcessingResult(
                success=True,
                statement=statement,
                errors=validation_errors,
                confidence_report=self._generate_confidence_report(statement)
            )
            
        except Exception as e:
            logger.error(f"❌ Error processing statement: {e}")
            return ProcessingResult(
                success=False,
                errors=[str(e)]
            )
    
    def process_batch(
        self, 
        input_dir: Path, 
        output_dir: Path
    ) -> List[ProcessingResult]:
        """
        Process multiple statement files in batch
        
        Args:
            input_dir: Directory containing statement files
            output_dir: Directory to save results
            
        Returns:
            List of ProcessingResult objects
        """
        logger.info(f"📂 Processing batch from: {input_dir}")
        
        # Find all supported files
        files = []
        for ext in settings.SUPPORTED_FORMATS:
            files.extend(input_dir.glob(f"*{ext}"))
        
        logger.info(f"📄 Found {len(files)} files to process")
        
        results = []
        for file_path in files:
            result = self.process_statement(file_path)
            results.append(result)
            
            # Save individual result
            if result.success:
                self._save_result(result, output_dir, file_path.stem)
        
        # Save batch summary
        self._save_batch_summary(results, output_dir)
        
        return results
    
    def _load_images(self, file_path: Path) -> List[Image.Image]:
        """Load images from file (supports PDF and image formats)"""
        try:
            if file_path.suffix.lower() == '.pdf':
                # Handle PDF files
                return self.image_preprocessor.pdf_to_images(file_path)
            else:
                # Handle image files
                image = Image.open(file_path)
                image = convert_if_not_rgb([image])[0]
                return [image]
        except Exception as e:
            logger.error(f"Error loading images from {file_path}: {e}")
            return []
    
    def _detect_bank_type(self, image: Image.Image) -> str:
        """Detect bank type from image using OCR"""
        try:
            # Run quick OCR on the top portion of the image
            top_crop = image.crop((0, 0, image.width, image.height // 3))
            
            # Use Surya OCR for text detection and recognition
            task_names = [TaskNames.ocr_with_boxes]
            ocr_results = self.predictors["recognition"](
                [top_crop],
                task_names=task_names,
                det_predictor=self.predictors["detection"]
            )
            
            # Extract text and look for bank indicators
            text_lines = [line.text.upper() for line in ocr_results[0].text_lines]
            full_text = " ".join(text_lines)
            
            # Check for bank indicators
            for bank in BANK_CONFIGS.keys():
                if bank in full_text:
                    return bank
            
            # Fallback detection based on patterns
            if "BANK NEGARA INDONESIA" in full_text or "BNI" in full_text:
                return "BNI"
            elif "MANDIRI" in full_text:
                return "MANDIRI"
            elif "OCBC" in full_text:
                return "OCBC"
            elif "DANAMON" in full_text:
                return "DANAMON"
            
            logger.warning("Could not detect bank type, defaulting to BNI")
            return "BNI"
            
        except Exception as e:
            logger.error(f"Error detecting bank type: {e}")
            return "BNI"
    
    def _run_surya_ocr(self, images: List[Image.Image]) -> Dict[str, Any]:
        """Run Surya OCR on images"""
        logger.info("🔍 Running Surya OCR...")
        
        try:
            # Preprocess images
            processed_images = []
            for img in images:
                processed_img = self.image_preprocessor.enhance_image(img)
                processed_images.append(processed_img)
            
            # Run layout detection
            layout_results = self.predictors["layout"](processed_images)
            
            # Run text detection and recognition
            task_names = [TaskNames.ocr_with_boxes] * len(processed_images)
            recognition_results = self.predictors["recognition"](
                processed_images,
                task_names=task_names,
                det_predictor=self.predictors["detection"],
                math_mode=False,  # Bank statements don't typically have math
                return_words=True
            )
            
            # Run table recognition for transaction tables
            table_results = self.predictors["table_rec"](processed_images)
            
            return {
                "layout": layout_results,
                "recognition": recognition_results,
                "tables": table_results,
                "images": processed_images
            }
            
        except Exception as e:
            logger.error(f"Error running Surya OCR: {e}")
            raise
    
    def _extract_statement_data(
        self, 
        ocr_results: Dict[str, Any], 
        bank_type: str, 
        file_path: Path
    ) -> BankStatement:
        """Extract structured data from OCR results"""
        logger.info(f"📊 Extracting statement data for {bank_type}")
        
        # This is a placeholder - we'll implement the actual extraction logic
        # in the next phase. For now, create a basic structure.
        
        from ..models.bank_schemas import (
            AccountInfo, StatementPeriod, ExtractedField, 
            FieldConfidence, ProcessingMetadata
        )
        
        # Create dummy data for now - will be replaced with actual extraction
        account_info = AccountInfo(
            account_number=ExtractedField(
                value="PLACEHOLDER_ACCOUNT",
                confidence=FieldConfidence(value=0.0, source="placeholder")
            ),
            account_holder=ExtractedField(
                value="PLACEHOLDER_HOLDER",
                confidence=FieldConfidence(value=0.0, source="placeholder")
            ),
            bank_name=bank_type
        )
        
        statement_period = StatementPeriod(
            start_date=ExtractedField(
                value="PLACEHOLDER_START",
                confidence=FieldConfidence(value=0.0, source="placeholder")
            ),
            end_date=ExtractedField(
                value="PLACEHOLDER_END",
                confidence=FieldConfidence(value=0.0, source="placeholder")
            )
        )
        
        metadata = ProcessingMetadata(
            file_name=file_path.name,
            file_size=file_path.stat().st_size,
            pages_processed=len(ocr_results["images"]),
            overall_confidence=0.0,
            ocr_model_version="surya-v1"
        )
        
        return BankStatement(
            bank_type=BankType(bank_type),
            account_info=account_info,
            statement_period=statement_period,
            metadata=metadata,
            raw_ocr_data=ocr_results
        )
    
    def _validate_statement(self, statement: BankStatement) -> List[str]:
        """Validate extracted statement data"""
        errors = []
        
        # Add validation logic here
        # For now, return empty list
        
        return errors
    
    def _calculate_overall_confidence(self, statement: BankStatement) -> float:
        """Calculate overall confidence score"""
        # Placeholder - will implement proper confidence calculation
        return 0.0
    
    def _generate_confidence_report(self, statement: BankStatement) -> Dict[str, float]:
        """Generate detailed confidence report"""
        # Placeholder - will implement detailed confidence reporting
        return {}
    
    def _save_result(self, result: ProcessingResult, output_dir: Path, filename: str):
        """Save processing result to file"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON result
        json_path = output_dir / f"{filename}_result.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result.dict(), f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"💾 Result saved to: {json_path}")
    
    def _save_batch_summary(self, results: List[ProcessingResult], output_dir: Path):
        """Save batch processing summary"""
        summary = {
            "total_files": len(results),
            "successful": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "average_confidence": sum(
                r.statement.metadata.overall_confidence 
                for r in results if r.success and r.statement
            ) / max(1, sum(1 for r in results if r.success)),
            "processing_timestamp": time.time()
        }
        
        summary_path = output_dir / "batch_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📋 Batch summary saved to: {summary_path}")

# Example usage and testing
if __name__ == "__main__":
    processor = BankStatementProcessor()
    print("🎉 Bank Statement Processor is ready!")
    print("\nNext steps:")
    print("1. Place your bank statement images in the data/input/ directory")
    print("2. Run the processor on your files")
    print("3. Check results in data/output/ directory")