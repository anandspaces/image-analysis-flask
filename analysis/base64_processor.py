import base64
import colorsys
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
from collections import Counter
from typing import Tuple, List, Dict
import cv2

class Base64ImageProcessor:
    """Enhanced image processor for base64 operations"""
    
    @staticmethod
    def decode_base64_image(base64_string: str) -> Tuple[Image.Image, str]:
        """Decode base64 string to PIL Image object"""
        try:
            # Handle data URL format (data:image/png;base64,...)
            if base64_string.startswith('data:image'):
                # Extract the base64 part after the comma
                header, encoded = base64_string.split(',', 1)
                # Extract format from header
                format_type = header.split('/')[1].split(';')[0].upper()
            else:
                # Pure base64 string without data URL prefix
                encoded = base64_string
                format_type = 'UNKNOWN'
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(encoded)
            
            # Create PIL Image from bytes
            image = Image.open(BytesIO(image_bytes))
            
            # Determine format if unknown
            if format_type == 'UNKNOWN':
                format_type = image.format or 'PNG'
            
            return image, format_type
            
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {str(e)}")
    
    @staticmethod
    def encode_image_to_base64(image: Image.Image, format_type: str = 'PNG', quality: int = 95) -> str:
        """Encode PIL Image to base64 string with data URL prefix"""
        try:
            # Create buffer to hold image bytes
            buffer = BytesIO()
            
            # Handle different formats
            if format_type.upper() == 'JPEG':
                # Convert to RGB if necessary for JPEG
                if image.mode in ('RGBA', 'LA'):
                    # Create white background for transparency
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                    image = background
                elif image.mode != 'RGB':
                    image = image.convert('RGB')
                
                image.save(buffer, format='JPEG', quality=quality, optimize=True)
                mime_type = 'image/jpeg'
                
            elif format_type.upper() == 'PNG':
                image.save(buffer, format='PNG', optimize=True)
                mime_type = 'image/png'
                
            elif format_type.upper() == 'WEBP':
                image.save(buffer, format='WEBP', quality=quality, optimize=True)
                mime_type = 'image/webp'
                
            else:
                # Default to PNG for unsupported formats
                image.save(buffer, format='PNG', optimize=True)
                mime_type = 'image/png'
            
            # Encode to base64
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Return with data URL prefix
            return f"data:{mime_type};base64,{img_str}"
            
        except Exception as e:
            raise ValueError(f"Failed to encode image to base64: {str(e)}")
    
    @staticmethod
    def convert_to_grayscale(image: Image.Image) -> Image.Image:
        """Convert image to grayscale"""
        return image.convert('L').convert('RGB')  # Convert back to RGB for consistency
    
    @staticmethod
    def detect_edges(image: Image.Image, threshold1: int = 100, threshold2: int = 200) -> Image.Image:
        """Detect edges using Canny edge detection"""
        try:
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, threshold1, threshold2)
            
            # Convert back to RGB format
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            
            # Convert back to PIL Image
            return Image.fromarray(edges_rgb)
            
        except Exception as e:
            # Fallback to PIL edge detection if OpenCV fails
            return image.convert('L').filter(ImageFilter.FIND_EDGES).convert('RGB')
    
    @staticmethod
    def analyze_dominant_colors(image: Image.Image, num_colors: int = 5) -> Tuple[Image.Image, List[Dict]]:
        """Analyze dominant colors in the image"""
        try:
            # Resize image for faster processing
            temp_image = image.copy()
            temp_image.thumbnail((150, 150), Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary
            if temp_image.mode != 'RGB':
                temp_image = temp_image.convert('RGB')
            
            # Get pixel data
            pixels = list(temp_image.getdata())
            
            # Simple color quantization
            color_counts = Counter(pixels)
            dominant_colors = color_counts.most_common(num_colors)
            
            # Create color info
            color_info = []
            total_pixels = len(pixels)
            
            for i, (color, count) in enumerate(dominant_colors):
                percentage = (count / total_pixels) * 100
                
                # Convert RGB to HSV for additional info
                h, s, v = colorsys.rgb_to_hsv(color[0]/255, color[1]/255, color[2]/255)
                
                color_info.append({
                    "rank": i + 1,
                    "rgb": color,
                    "hex": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                    "percentage": round(percentage, 2),
                    "hue": round(h * 360, 1),
                    "saturation": round(s * 100, 1),
                    "brightness": round(v * 100, 1)
                })
            
            return image, color_info
            
        except Exception as e:
            # Return original image with error info
            return image, [{"error": f"Color analysis failed: {str(e)}"}]
    
    @staticmethod
    def enhance_image(image: Image.Image, enhancement_type: str = 'auto') -> Image.Image:
        """Enhance image quality"""
        try:
            enhanced_image = image.copy()
            
            if enhancement_type == 'brightness':
                enhancer = ImageEnhance.Brightness(enhanced_image)
                enhanced_image = enhancer.enhance(1.2)
                
            elif enhancement_type == 'contrast':
                enhancer = ImageEnhance.Contrast(enhanced_image)
                enhanced_image = enhancer.enhance(1.3)
                
            elif enhancement_type == 'sharpness':
                enhancer = ImageEnhance.Sharpness(enhanced_image)
                enhanced_image = enhancer.enhance(1.5)
                
            elif enhancement_type == 'color':
                enhancer = ImageEnhance.Color(enhanced_image)
                enhanced_image = enhancer.enhance(1.2)
                
            elif enhancement_type == 'auto':
                # Apply multiple enhancements
                enhancer = ImageEnhance.Contrast(enhanced_image)
                enhanced_image = enhancer.enhance(1.1)
                
                enhancer = ImageEnhance.Sharpness(enhanced_image)
                enhanced_image = enhancer.enhance(1.2)
                
                enhancer = ImageEnhance.Color(enhanced_image)
                enhanced_image = enhancer.enhance(1.1)
            
            return enhanced_image
            
        except Exception as e:
            # Return original image if enhancement fails
            return image
