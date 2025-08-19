import base64
from io import BytesIO
from PIL import Image, ImageOps
from werkzeug.utils import secure_filename
from typing import Dict, Any, Tuple
from config.config import Config

class ImageProcessor:
    """Professional image processing utilities"""
    
    @staticmethod
    def validate_image(file) -> Tuple[bool, str]:
        """Validate uploaded image file"""
        if not file or not file.filename:
            return False, "No file provided"
        
        if not ImageProcessor.allowed_file(file.filename):
            return False, f"Invalid file type. Allowed: {', '.join(Config.ALLOWED_EXTENSIONS)}"
        
        try:
            # Check if file is actually an image
            Image.open(file).verify()
            file.seek(0)  # Reset file pointer
            return True, "Valid image"
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"
    
    @staticmethod
    def allowed_file(filename: str) -> bool:
        """Check if filename has allowed extension"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    
    @staticmethod
    def process_image(file, max_size: Tuple[int, int] = None) -> str:
        """Process and optimize image, return base64 encoded data URL"""
        if max_size is None:
            max_size = Config.MAX_IMAGE_SIZE
            
        try:
            image = Image.open(file)
            
            # Convert to RGB if necessary
            if image.mode not in ('RGB', 'RGBA'):
                image = image.convert('RGB')
            
            # Auto-rotate based on EXIF data
            image = ImageOps.exif_transpose(image)
            
            # Resize if too large
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Optimize and encode
            buffer = BytesIO()
            format_type = 'PNG' if image.mode == 'RGBA' else 'JPEG'
            quality = Config.DEFAULT_JPEG_QUALITY if format_type == 'JPEG' else None
            
            image.save(buffer, format=format_type, quality=quality, optimize=True)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/{format_type.lower()};base64,{img_str}"
            
        except Exception as e:
            raise Exception(f"Image processing failed: {str(e)}")
    
    @staticmethod
    def get_image_metadata(file) -> Dict[str, Any]:
        """Extract image metadata"""
        try:
            image = Image.open(file)
            file.seek(0)  # Reset file pointer
            
            metadata = {
                "filename": secure_filename(file.filename),
                "format": image.format,
                "mode": image.mode,
                "size": image.size,
                "width": image.width,
                "height": image.height,
                "has_transparency": image.mode in ('RGBA', 'LA') or 'transparency' in image.info
            }
            
            # EXIF data if available
            if hasattr(image, '_getexif') and image._getexif():
                metadata["has_exif"] = True
            
            return metadata
            
        except Exception as e:
            return {"error": f"Failed to extract metadata: {str(e)}"}
