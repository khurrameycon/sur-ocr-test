# bank_statement_processor/src/utils/image_utils.py
"""
Image processing utilities for bank statement processing
"""
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import List, Tuple, Optional
import logging
from pathlib import Path
import fitz  # PyMuPDF for PDF processing

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Image preprocessing utilities for better OCR results"""
    
    def __init__(self):
        self.target_dpi = 300
        self.max_width = 2048
        self.max_height = 2048
    
    def enhance_image(self, image: Image.Image) -> Image.Image:
        """
        Enhance image quality for better OCR results
        
        Args:
            image: PIL Image to enhance
            
        Returns:
            Enhanced PIL Image
        """
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large
            image = self._resize_if_needed(image)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            # Apply slight denoising
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            return image
            
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return image
    
    def _resize_if_needed(self, image: Image.Image) -> Image.Image:
        """Resize image if it exceeds maximum dimensions"""
        width, height = image.size
        
        if width <= self.max_width and height <= self.max_height:
            return image
        
        # Calculate new size maintaining aspect ratio
        ratio = min(self.max_width / width, self.max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def correct_skew(self, image: Image.Image) -> Image.Image:
        """
        Correct skew in scanned documents
        
        Args:
            image: PIL Image to correct
            
        Returns:
            Skew-corrected PIL Image
        """
        try:
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Find edges
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Find lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                angles = []
                for rho, theta in lines[:, 0]:
                    angle = theta * 180 / np.pi
                    # Focus on nearly horizontal lines
                    if 85 <= angle <= 95:
                        angles.append(angle - 90)
                
                if angles:
                    # Use median angle for correction
                    skew_angle = np.median(angles)
                    
                    # Only correct if skew is significant
                    if abs(skew_angle) > 0.5:
                        logger.info(f"Correcting skew angle: {skew_angle:.2f} degrees")
                        
                        # Rotate image
                        center = (image.width // 2, image.height // 2)
                        image = image.rotate(-skew_angle, center=center, fillcolor='white')
            
            return image
            
        except Exception as e:
            logger.error(f"Error correcting skew: {e}")
            return image
    
    def remove_noise(self, image: Image.Image) -> Image.Image:
        """
        Remove noise from image using morphological operations
        
        Args:
            image: PIL Image to denoise
            
        Returns:
            Denoised PIL Image
        """
        try:
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply morphological opening to remove small noise
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            
            # Convert back to PIL
            cleaned_pil = Image.fromarray(cleaned).convert('RGB')
            
            return cleaned_pil
            
        except Exception as e:
            logger.error(f"Error removing noise: {e}")
            return image
    
    def binarize_image(self, image: Image.Image) -> Image.Image:
        """
        Convert image to binary (black and white) for better OCR
        
        Args:
            image: PIL Image to binarize
            
        Returns:
            Binary PIL Image
        """
        try:
            # Convert to grayscale
            gray = image.convert('L')
            
            # Convert to numpy array
            img_array = np.array(gray)
            
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Convert back to PIL
            binary_pil = Image.fromarray(binary).convert('RGB')
            
            return binary_pil
            
        except Exception as e:
            logger.error(f"Error binarizing image: {e}")
            return image
    
    def pdf_to_images(self, pdf_path: Path, dpi: int = 300) -> List[Image.Image]:
        """
        Convert PDF pages to images
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for conversion
            
        Returns:
            List of PIL Images, one per page
        """
        try:
            images = []
            
            # Open PDF
            doc = fitz.open(pdf_path)
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Convert page to image
                mat = fitz.Matrix(dpi/72, dpi/72)  # Scale factor for DPI
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("ppm")
                image = Image.open(io.BytesIO(img_data))
                
                images.append(image)
                logger.info(f"Converted PDF page {page_num + 1} to image")
            
            doc.close()
            return images
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            return []
    
    def preprocess_for_tables(self, image: Image.Image) -> Image.Image:
        """
        Specialized preprocessing for table detection
        
        Args:
            image: PIL Image containing tables
            
        Returns:
            Preprocessed PIL Image optimized for table detection
        """
        try:
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Enhance table lines
            kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            # Detect horizontal lines
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_horizontal)
            
            # Detect vertical lines  
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_vertical)
            
            # Combine lines
            table_structure = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            # Enhance the original image with detected structure
            enhanced = cv2.addWeighted(gray, 0.8, table_structure, 0.2, 0.0)
            
            # Convert back to PIL
            enhanced_pil = Image.fromarray(enhanced).convert('RGB')
            
            return enhanced_pil
            
        except Exception as e:
            logger.error(f"Error preprocessing for tables: {e}")
            return image
    
    def extract_table_regions(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """
        Extract table regions from image
        
        Args:
            image: PIL Image containing tables
            
        Returns:
            List of bounding boxes (x1, y1, x2, y2) for detected tables
        """
        try:
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Find table structure
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            
            # Apply morphological operations to find table structure
            binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            # Find horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
            
            # Combine lines to form table structure
            table_structure = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            # Find contours
            contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area and aspect ratio
            table_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Filter by minimum area and reasonable aspect ratio
                if area > 5000 and w > 100 and h > 50:
                    table_regions.append((x, y, x + w, y + h))
            
            return table_regions
            
        except Exception as e:
            logger.error(f"Error extracting table regions: {e}")
            return []

import io  # Add this import at the top