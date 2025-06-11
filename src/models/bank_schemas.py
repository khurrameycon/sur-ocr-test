# bank_statement_processor/src/models/bank_schemas.py
"""
Data models for bank statement processing
"""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum
import re

class BankType(str, Enum):
    BNI = "BNI"
    MANDIRI = "MANDIRI" 
    OCBC = "OCBC"
    DANAMON = "DANAMON"

class FieldConfidence(BaseModel):
    """Confidence score for extracted fields"""
    value: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    source: str = Field(..., description="Source of confidence (OCR, validation, etc.)")

class BoundingBox(BaseModel):
    """Bounding box coordinates for extracted text"""
    x1: float = Field(..., description="Left coordinate")
    y1: float = Field(..., description="Top coordinate") 
    x2: float = Field(..., description="Right coordinate")
    y2: float = Field(..., description="Bottom coordinate")
    page: int = Field(default=1, description="Page number")
    
    @validator('x2')
    def x2_greater_than_x1(cls, v, values):
        if 'x1' in values and v <= values['x1']:
            raise ValueError('x2 must be greater than x1')
        return v
    
    @validator('y2')  
    def y2_greater_than_y1(cls, v, values):
        if 'y1' in values and v <= values['y1']:
            raise ValueError('y2 must be greater than y1')
        return v

class ExtractedField(BaseModel):
    """Base model for extracted fields with confidence and location"""
    value: str = Field(..., description="Extracted value")
    confidence: FieldConfidence = Field(..., description="Confidence metrics")
    bbox: Optional[BoundingBox] = Field(None, description="Bounding box location")
    raw_text: Optional[str] = Field(None, description="Original OCR text before cleaning")

class AccountInfo(BaseModel):
    """Account information from statement header"""
    account_number: ExtractedField = Field(..., description="Bank account number")
    account_holder: ExtractedField = Field(..., description="Account holder name")
    account_type: Optional[ExtractedField] = Field(None, description="Account type")
    bank_name: str = Field(..., description="Bank name")
    branch: Optional[ExtractedField] = Field(None, description="Bank branch")
    currency: str = Field(default="IDR", description="Account currency")

class StatementPeriod(BaseModel):
    """Statement period information"""
    start_date: ExtractedField = Field(..., description="Statement start date")
    end_date: ExtractedField = Field(..., description="Statement end date")
    statement_date: Optional[ExtractedField] = Field(None, description="Statement generation date")

class TransactionEntry(BaseModel):
    """Individual transaction entry"""
    date: ExtractedField = Field(..., description="Transaction date")
    description: ExtractedField = Field(..., description="Transaction description")
    reference: Optional[ExtractedField] = Field(None, description="Transaction reference/ID")
    debit: Optional[ExtractedField] = Field(None, description="Debit amount")
    credit: Optional[ExtractedField] = Field(None, description="Credit amount")
    balance: Optional[ExtractedField] = Field(None, description="Running balance")
    transaction_type: Optional[str] = Field(None, description="Transaction type classification")
    
    @validator('debit', 'credit')
    def validate_amounts(cls, v):
        """Validate that amount values are numeric"""
        if v is not None:
            try:
                # Remove common formatting and convert to float
                cleaned = re.sub(r'[,\s]', '', v.value)
                float(cleaned)
            except ValueError:
                raise ValueError(f"Invalid amount format: {v.value}")
        return v

class BalanceSummary(BaseModel):
    """Balance summary information"""
    opening_balance: Optional[ExtractedField] = Field(None, description="Opening balance")
    closing_balance: Optional[ExtractedField] = Field(None, description="Closing balance")
    total_debits: Optional[ExtractedField] = Field(None, description="Total debits")
    total_credits: Optional[ExtractedField] = Field(None, description="Total credits")

class ProcessingMetadata(BaseModel):
    """Metadata about the processing"""
    processing_timestamp: datetime = Field(default_factory=datetime.now)
    file_name: str = Field(..., description="Original file name")
    file_size: int = Field(..., description="File size in bytes")
    pages_processed: int = Field(..., description="Number of pages processed")
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall processing confidence")
    processing_time_seconds: Optional[float] = Field(None, description="Processing time")
    ocr_model_version: Optional[str] = Field(None, description="OCR model version used")

class BankStatement(BaseModel):
    """Complete bank statement model"""
    # Core information
    bank_type: BankType = Field(..., description="Bank type")
    account_info: AccountInfo = Field(..., description="Account information")
    statement_period: StatementPeriod = Field(..., description="Statement period")
    
    # Transaction data
    transactions: List[TransactionEntry] = Field(default=[], description="List of transactions")
    balance_summary: Optional[BalanceSummary] = Field(None, description="Balance summary")
    
    # Processing metadata
    metadata: ProcessingMetadata = Field(..., description="Processing metadata")
    
    # Raw data for debugging
    raw_ocr_data: Optional[Dict[str, Any]] = Field(None, description="Raw OCR output")
    validation_errors: List[str] = Field(default=[], description="Validation errors found")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ProcessingResult(BaseModel):
    """Result of processing a bank statement"""
    success: bool = Field(..., description="Whether processing was successful")
    statement: Optional[BankStatement] = Field(None, description="Extracted statement data")
    errors: List[str] = Field(default=[], description="Processing errors")
    warnings: List[str] = Field(default=[], description="Processing warnings")
    confidence_report: Dict[str, float] = Field(default={}, description="Confidence scores by field type")

# Output format models for different use cases
class LLMReadyOutput(BaseModel):
    """Simplified output optimized for LLM consumption"""
    bank: str
    account_number: str
    account_holder: str
    statement_period: str
    currency: str
    opening_balance: Optional[float] = None
    closing_balance: Optional[float] = None
    total_transactions: int
    total_debits: Optional[float] = None
    total_credits: Optional[float] = None
    transactions_summary: str
    confidence_score: float
    
    class Config:
        schema_extra = {
            "example": {
                "bank": "BNI",
                "account_number": "1667942876",
                "account_holder": "DUTAGARUDA PIRANTI PRIMA, PT",
                "statement_period": "July 1-31, 2023",
                "currency": "IDR",
                "opening_balance": 112115472.00,
                "closing_balance": 461282862.00,
                "total_transactions": 18,
                "total_debits": 2636330829.00,
                "total_credits": 3623050593.00,
                "transactions_summary": "18 transactions including transfers, payments, and fees",
                "confidence_score": 0.94
            }
        }