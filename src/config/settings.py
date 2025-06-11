# bank_statement_processor/src/config/settings.py
"""
Configuration settings for bank statement processing
"""
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseSettings, validator
import os

class Settings(BaseSettings):
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    INPUT_DIR: Path = DATA_DIR / "input"
    OUTPUT_DIR: Path = DATA_DIR / "output"
    TEMPLATES_DIR: Path = DATA_DIR / "templates"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    
    # OCR Settings
    OCR_CONFIDENCE_THRESHOLD: float = 0.7
    OCR_HIGH_RES_DPI: int = 300
    OCR_STANDARD_DPI: int = 150
    
    # Processing settings
    MIN_FIELD_CONFIDENCE: float = 0.8
    ENABLE_PREPROCESSING: bool = True
    BATCH_SIZE: int = 4
    
    # Supported image formats
    SUPPORTED_FORMATS: List[str] = [".jpg", ".jpeg", ".png", ".pdf", ".tiff", ".bmp"]
    
    # Bank-specific settings
    SUPPORTED_BANKS: List[str] = ["BNI", "MANDIRI", "OCBC", "DANAMON"]
    
    # Field extraction settings
    REQUIRED_FIELDS: List[str] = [
        "account_number",
        "account_holder", 
        "statement_period",
        "transactions"
    ]
    
    # Indonesian language settings
    LANGUAGE_CODES: List[str] = ["id", "en"]  # Indonesian and English
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    @validator('PROJECT_ROOT', 'DATA_DIR', 'INPUT_DIR', 'OUTPUT_DIR', 'TEMPLATES_DIR', 'LOGS_DIR')
    def create_directories(cls, v):
        """Ensure directories exist"""
        v.mkdir(parents=True, exist_ok=True)
        return v

# Global settings instance
settings = Settings()

# Bank-specific configuration
BANK_CONFIGS = {
    "BNI": {
        "currency": "IDR",
        "date_formats": ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"],
        "amount_patterns": [r"[\d,]+\.\d{2}", r"[\d.]+,\d{2}"],
        "table_headers": ["Date", "Description", "Debit", "Credit", "Balance"],
        "language": "id"
    },
    "MANDIRI": {
        "currency": "IDR", 
        "date_formats": ["%d/%m/%Y", "%d/%m/%y"],
        "amount_patterns": [r"[\d,]+\.\d{2}", r"[\d.]+,\d{2}"],
        "table_headers": ["Tanggal", "Keterangan", "Debit", "Kredit", "Saldo"],
        "language": "id"
    },
    "OCBC": {
        "currency": "IDR",
        "date_formats": ["%d/%m/%Y", "%d-%m-%Y"],
        "amount_patterns": [r"[\d,]+\.\d{2}"],
        "table_headers": ["Transaction Date", "Description", "Debit", "Credit", "Balance"],
        "language": "en"
    },
    "DANAMON": {
        "currency": "IDR",
        "date_formats": ["%d/%m/%Y", "%d-%m-%Y"],
        "amount_patterns": [r"[\d,]+\.\d{2}", r"[\d.]+,\d{2}"],
        "table_headers": ["TGL", "KETERANGAN", "DEBITS", "CREDITS", "BALANCE"],
        "language": "id"
    }
}

# Confidence thresholds for different field types
FIELD_CONFIDENCE_THRESHOLDS = {
    "account_number": 0.95,
    "account_holder": 0.90,
    "amounts": 0.92,
    "dates": 0.88,
    "descriptions": 0.85,
    "balances": 0.95
}

print(f"📁 Configuration loaded - Project root: {settings.PROJECT_ROOT}")