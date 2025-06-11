# bank_statement_processor/src/utils/text_utils.py
"""
Text processing utilities for bank statement data extraction
"""
import re
import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from dateutil import parser
import unicodedata
from decimal import Decimal, InvalidOperation

logger = logging.getLogger(__name__)

class TextProcessor:
    """Text processing and extraction utilities"""
    
    def __init__(self):
        # Common patterns for Indonesian bank statements
        self.date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # DD/MM/YYYY or DD-MM-YYYY
            r'\b\d{1,2}\s+\w+\s+\d{4}\b',          # DD Month YYYY
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # YYYY/MM/DD
        ]
        
        self.amount_patterns = [
            r'[\d,]+\.\d{2}',          # 1,234.56
            r'[\d.]+,\d{2}',           # 1.234,56 (European format)
            r'\d+\.\d{2}',             # 1234.56
            r'\d+,\d{2}',              # 1234,56
        ]
        
        self.account_patterns = [
            r'\b\d{10,20}\b',          # Account numbers (10-20 digits)
            r'\b\d{3,4}[-\s]\d{3,4}[-\s]\d{3,8}\b',  # Formatted account numbers
        ]
        
        # Indonesian month names
        self.indonesian_months = {
            'januari': '01', 'februari': '02', 'maret': '03', 'april': '04',
            'mei': '05', 'juni': '06', 'juli': '07', 'agustus': '08',
            'september': '09', 'oktober': '10', 'november': '11', 'desember': '12',
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
            'jun': '06', 'jul': '07', 'aug': '08', 'agu': '08',
            'sep': '09', 'oct': '10', 'okt': '10', 'nov': '11', 'dec': '12', 'des': '12'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_dates(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract dates from text
        
        Args:
            text: Text containing dates
            
        Returns:
            List of dictionaries with date information
        """
        dates = []
        
        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_str = match.group()
                parsed_date = self._parse_date(date_str)
                
                if parsed_date:
                    dates.append({
                        'raw': date_str,
                        'parsed': parsed_date,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': self._calculate_date_confidence(date_str)
                    })
        
        return dates
    
    def extract_amounts(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract monetary amounts from text
        
        Args:
            text: Text containing amounts
            
        Returns:
            List of dictionaries with amount information
        """
        amounts = []
        
        for pattern in self.amount_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                amount_str = match.group()
                parsed_amount = self._parse_amount(amount_str)
                
                if parsed_amount is not None:
                    amounts.append({
                        'raw': amount_str,
                        'parsed': float(parsed_amount),
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': self._calculate_amount_confidence(amount_str)
                    })
        
        return amounts
    
    def extract_account_numbers(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract account numbers from text
        
        Args:
            text: Text containing account numbers
            
        Returns:
            List of dictionaries with account number information
        """
        accounts = []
        
        for pattern in self.account_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                account_str = match.group()
                
                # Clean account number
                cleaned_account = re.sub(r'[-\s]', '', account_str)
                
                # Validate account number
                if self._is_valid_account_number(cleaned_account):
                    accounts.append({
                        'raw': account_str,
                        'cleaned': cleaned_account,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': self._calculate_account_confidence(account_str)
                    })
        
        return accounts
    
    def extract_transaction_descriptions(self, text: str) -> List[str]:
        """
        Extract and clean transaction descriptions
        
        Args:
            text: Text containing transaction descriptions
            
        Returns:
            List of cleaned transaction descriptions
        """
        # Common transaction keywords in Indonesian
        transaction_keywords = [
            'transfer', 'trf', 'pay', 'payment', 'kredit', 'debit',
            'tarik', 'setor', 'biaya', 'fee', 'bunga', 'interest',
            'saldo', 'balance', 'top-up', 'topup', 'pembelian',
            'purchase', 'pembayaran'
        ]
        
        descriptions = []
        lines = text.split('\n')
        
        for line in lines:
            line = self.clean_text(line)
            
            # Check if line contains transaction keywords
            if any(keyword.lower() in line.lower() for keyword in transaction_keywords):
                # Further clean the description
                cleaned_desc = self._clean_transaction_description(line)
                if cleaned_desc and len(cleaned_desc) > 3:
                    descriptions.append(cleaned_desc)
        
        return descriptions
    
    def identify_header_fields(self, text: str) -> Dict[str, str]:
        """
        Identify header fields like account holder, account number, etc.
        
        Args:
            text: Header text from statement
            
        Returns:
            Dictionary of identified header fields
        """
        fields = {}
        
        # Account holder patterns
        holder_patterns = [
            r'(?:nama|name|account holder)[:\s]+([A-Z\s,\.]+)',
            r'([A-Z][A-Z\s,\.]+(?:PT|CV|TBIK|PERSERO))',  # Company names
            r'PT\s+([A-Z\s,\.]+)',  # PT companies
        ]
        
        for pattern in holder_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields['account_holder'] = self.clean_text(match.group(1))
                break
        
        # Account number
        accounts = self.extract_account_numbers(text)
        if accounts:
            fields['account_number'] = accounts[0]['cleaned']
        
        # Statement period
        dates = self.extract_dates(text)
        if len(dates) >= 2:
            fields['period_start'] = dates[0]['parsed'].strftime('%Y-%m-%d')
            fields['period_end'] = dates[-1]['parsed'].strftime('%Y-%m-%d')
        
        return fields
    
    def parse_table_row(self, row_text: str, bank_type: str) -> Dict[str, Any]:
        """
        Parse a table row into structured transaction data
        
        Args:
            row_text: Text from a table row
            bank_type: Type of bank (affects parsing logic)
            
        Returns:
            Dictionary with parsed transaction fields
        """
        # Split the row by common delimiters
        parts = re.split(r'\s{2,}|\t', row_text.strip())
        parts = [self.clean_text(part) for part in parts if part.strip()]
        
        transaction = {}
        
        # Extract date (usually first column)
        if parts:
            dates = self.extract_dates(parts[0])
            if dates:
                transaction['date'] = dates[0]['parsed']
        
        # Extract amounts (usually last few columns)
        amounts = []
        for part in parts:
            part_amounts = self.extract_amounts(part)
            amounts.extend(part_amounts)
        
        if amounts:
            # Typically: debit, credit, balance or just amount, balance
            if len(amounts) >= 3:
                transaction['debit'] = amounts[0]['parsed']
                transaction['credit'] = amounts[1]['parsed']
                transaction['balance'] = amounts[2]['parsed']
            elif len(amounts) == 2:
                transaction['amount'] = amounts[0]['parsed']
                transaction['balance'] = amounts[1]['parsed']
            else:
                transaction['amount'] = amounts[0]['parsed']
        
        # Extract description (middle columns)
        if len(parts) > 2:
            # Combine middle parts as description
            description_parts = []
            for part in parts[1:-2]:  # Skip first (date) and last 2 (amounts)
                if not self.extract_amounts(part) and not self.extract_dates(part):
                    description_parts.append(part)
            
            if description_parts:
                transaction['description'] = ' '.join(description_parts)
        
        return transaction
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object"""
        try:
            # Handle Indonesian month names
            date_str_clean = date_str.lower()
            for indo_month, month_num in self.indonesian_months.items():
                if indo_month in date_str_clean:
                    date_str_clean = date_str_clean.replace(indo_month, month_num)
            
            # Try different date formats
            date_formats = [
                '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y',
                '%Y/%m/%d', '%Y-%m-%d', '%Y.%m.%d',
                '%d/%m/%y', '%d-%m-%y', '%d.%m.%y',
                '%d %m %Y', '%d %m %y'
            ]
            
            for fmt in date_formats:
                try:
                    return datetime.strptime(date_str_clean, fmt)
                except ValueError:
                    continue
            
            # Fallback to dateutil parser
            return parser.parse(date_str, dayfirst=True)
            
        except Exception as e:
            logger.debug(f"Could not parse date '{date_str}': {e}")
            return None
    
    def _parse_amount(self, amount_str: str) -> Optional[Decimal]:
        """Parse amount string to Decimal"""
        try:
            # Remove currency symbols and spaces
            cleaned = re.sub(r'[^\d,.-]', '', amount_str)
            
            # Handle different decimal separators
            if ',' in cleaned and '.' in cleaned:
                # Determine which is decimal separator
                if cleaned.rfind(',') > cleaned.rfind('.'):
                    # Comma is decimal separator
                    cleaned = cleaned.replace('.', '').replace(',', '.')
                else:
                    # Dot is decimal separator
                    cleaned = cleaned.replace(',', '')
            elif ',' in cleaned:
                # Check if comma is thousand separator or decimal
                comma_pos = cleaned.rfind(',')
                after_comma = cleaned[comma_pos + 1:]
                if len(after_comma) == 2 and after_comma.isdigit():
                    # Comma is decimal separator
                    cleaned = cleaned.replace(',', '.')
                else:
                    # Comma is thousand separator
                    cleaned = cleaned.replace(',', '')
            
            return Decimal(cleaned)
            
        except (InvalidOperation, ValueError) as e:
            logger.debug(f"Could not parse amount '{amount_str}': {e}")
            return None
    
    def _is_valid_account_number(self, account: str) -> bool:
        """Validate account number format"""
        # Basic validation: should be numeric and reasonable length
        return account.isdigit() and 8 <= len(account) <= 20
    
    def _calculate_date_confidence(self, date_str: str) -> float:
        """Calculate confidence score for date extraction"""
        confidence = 0.7  # Base confidence
        
        # Increase confidence for standard formats
        if re.match(r'\d{2}/\d{2}/\d{4}', date_str):
            confidence += 0.2
        
        # Decrease confidence for ambiguous formats
        if len(date_str.split()) > 3:
            confidence -= 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_amount_confidence(self, amount_str: str) -> float:
        """Calculate confidence score for amount extraction"""
        confidence = 0.8  # Base confidence
        
        # Increase confidence for clear decimal format
        if re.match(r'[\d,]+\.\d{2}$', amount_str):
            confidence += 0.15
        
        # Decrease confidence for unusual formats
        if amount_str.count('.') > 1 or amount_str.count(',') > 1:
            confidence -= 0.2
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_account_confidence(self, account_str: str) -> float:
        """Calculate confidence score for account number extraction"""
        confidence = 0.75  # Base confidence
        
        # Increase confidence for clean numeric format
        if account_str.replace('-', '').replace(' ', '').isdigit():
            confidence += 0.2
        
        # Adjust based on length
        length = len(account_str.replace('-', '').replace(' ', ''))
        if 10 <= length <= 16:
            confidence += 0.1
        elif length < 8 or length > 20:
            confidence -= 0.3
        
        return min(1.0, max(0.0, confidence))
    
    def _clean_transaction_description(self, description: str) -> str:
        """Clean transaction description text"""
        # Remove common prefixes/suffixes
        prefixes_to_remove = ['trf/', 'pay/', 'trx/', 'ref:']
        suffixes_to_remove = ['fee', 'charge']
        
        cleaned = description.lower()
        
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
        
        for suffix in suffixes_to_remove:
            if cleaned.endswith(suffix):
                cleaned = cleaned[:-len(suffix)]
        
        # Remove extra characters
        cleaned = re.sub(r'[|]+', ' | ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip().title()