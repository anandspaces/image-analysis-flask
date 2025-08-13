from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from functools import wraps
from datetime import datetime
from io import BytesIO
from PIL import Image, ImageOps
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, List, Any
from PIL import ImageEnhance, ImageFilter
import os, logging, uuid, base64, threading, time, cv2, colorsys, numpy as np

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff', 'svg'}
    API_RATE_LIMIT = 100  # requests per hour per IP
    LOG_LEVEL = logging.INFO

class ImageAnalysisAPI:
    """Professional Image Analysis API Handler"""
    
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENAI_KEY"),
        )
        self.rate_limiter = {}
        
    def analyze_general(self, image_data: str, custom_prompt: str = None) -> Dict[str, Any]:
        """General image analysis"""
        prompt = custom_prompt or "Provide a detailed analysis of this image including objects, people, setting, colors, mood, and any notable features."
        return self._make_ai_request(image_data, prompt, "general")
    
    def analyze_objects(self, image_data: str) -> Dict[str, Any]:
        """Object detection and identification"""
        prompt = """Identify and list all objects in this image. For each object, provide:
        1. Object name
        2. Location/position in image
        3. Confidence level
        4. Color and material if visible
        5. Size relative to other objects
        Format as a structured list."""
        return self._make_ai_request(image_data, prompt, "objects")
    
    def analyze_scene(self, image_data: str) -> Dict[str, Any]:
        """Scene understanding and context"""
        prompt = """Analyze the scene in this image including:
        1. Location type (indoor/outdoor, specific venue)
        2. Time of day and lighting conditions
        3. Weather conditions if outdoor
        4. Overall atmosphere and mood
        5. Activities taking place
        6. Architectural or natural features"""
        return self._make_ai_request(image_data, prompt, "scene")
    
    def analyze_text(self, image_data: str) -> Dict[str, Any]:
        """OCR and text extraction"""
        prompt = """Extract and transcribe all text visible in this image. Include:
        1. All readable text exactly as shown
        2. Text location and formatting
        3. Language if not English
        4. Font style and size if notable
        5. Text purpose (signs, labels, documents, etc.)"""
        return self._make_ai_request(image_data, prompt, "text")
    
    def analyze_people(self, image_data: str) -> Dict[str, Any]:
        """People analysis (respecting privacy)"""
        prompt = """Analyze people in this image (no identification, general description only):
        1. Number of people
        2. Age groups (child, adult, elderly)
        3. Gender presentation
        4. Clothing and accessories
        5. Activities and poses
        6. Facial expressions and emotions
        7. Interactions between people"""
        return self._make_ai_request(image_data, prompt, "people")
    
    def analyze_technical(self, image_data: str) -> Dict[str, Any]:
        """Technical image analysis"""
        prompt = """Provide technical analysis of this image:
        1. Image quality and resolution assessment
        2. Lighting analysis (exposure, shadows, highlights)
        3. Composition and framing
        4. Color balance and saturation
        5. Potential camera settings used
        6. Image defects or artifacts
        7. Suggestions for improvement"""
        return self._make_ai_request(image_data, prompt, "technical")
    
    def analyze_safety(self, image_data: str) -> Dict[str, Any]:
        """Safety and content moderation"""
        prompt = """Analyze this image for safety and content concerns:
        1. Content appropriateness rating
        2. Any safety hazards visible
        3. Violence or dangerous activities
        4. Adult content indicators
        5. Potential harmful substances
        6. Overall safety assessment
        Be objective and factual."""
        return self._make_ai_request(image_data, prompt, "safety")
    
    def analyze_comprehensive(self, image_data: str) -> Dict[str, Any]:
        """Comprehensive multi-aspect analysis"""
        prompt = """Provide a comprehensive analysis of this image covering:
        
        VISUAL CONTENT:
        - Objects and their properties
        - People and their activities
        - Scene setting and context
        - Text and signage
        
        TECHNICAL ASPECTS:
        - Image quality and composition
        - Lighting and color analysis
        - Camera perspective and framing
        
        CONTEXTUAL ANALYSIS:
        - Purpose and intent of image
        - Cultural or historical elements
        - Emotional tone and atmosphere
        - Potential use cases
        
        INSIGHTS:
        - Key findings and observations
        - Notable patterns or anomalies
        - Recommendations or suggestions"""
        return self._make_ai_request(image_data, prompt, "comprehensive")
    
    def _make_ai_request(self, image_data: str, prompt: str, analysis_type: str) -> Dict[str, Any]:
        """Make request to AI API with error handling and retry logic"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://api.imageanalyzer.pro",
                        "X-Title": "Professional Image Analysis API",
                    },
                    model="google/gemini-2.5-flash",
                    max_tokens=2000,
                    temperature=0.1,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional image analysis AI. Provide accurate, detailed, and structured responses. Be objective and thorough."
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": image_data}}
                            ]
                        }
                    ]
                )
                
                return {
                    "success": True,
                    "analysis_type": analysis_type,
                    "result": completion.choices[0].message.content,
                    "model": "google/gemini-2.5-flash",
                    "timestamp": datetime.utcnow().isoformat(),
                    "tokens_used": completion.usage.total_tokens if hasattr(completion, 'usage') else None
                }
                
            except Exception as e:
                app.logger.error(f"AI request attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    return {
                        "success": False,
                        "error": f"Analysis failed after {max_retries} attempts: {str(e)}",
                        "analysis_type": analysis_type,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                time.sleep(retry_delay * (2 ** attempt))

class ImageProcessor:
    """Professional image processing utilities"""
    
    @staticmethod
    def validate_image(file) -> tuple[bool, str]:
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
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    
    @staticmethod
    def process_image(file, max_size: tuple = (2048, 2048)) -> str:
        """Process and optimize image, return base64 encoded data URL"""
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
            quality = 95 if format_type == 'JPEG' else None
            
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

class RateLimiter:
    """Rate limiting for API endpoints"""
    
    def __init__(self):
        self.requests = {}
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_requests, daemon=True)
        self.cleanup_thread.start()
    
    def is_allowed(self, ip: str, limit: int = Config.API_RATE_LIMIT) -> bool:
        current_time = time.time()
        hour_ago = current_time - 3600
        
        if ip not in self.requests:
            self.requests[ip] = []
        
        # Remove old requests
        self.requests[ip] = [req_time for req_time in self.requests[ip] if req_time > hour_ago]
        
        # Check if under limit
        if len(self.requests[ip]) < limit:
            self.requests[ip].append(current_time)
            return True
        
        return False
    
    def _cleanup_old_requests(self):
        while True:
            time.sleep(300)  # Cleanup every 5 minutes
            current_time = time.time()
            hour_ago = current_time - 3600
            
            for ip in list(self.requests.keys()):
                self.requests[ip] = [req_time for req_time in self.requests[ip] if req_time > hour_ago]
                if not self.requests[ip]:
                    del self.requests[ip]

class Base64ImageProcessor:
    """Enhanced image processor for base64 operations"""
    
    @staticmethod
    def decode_base64_image(base64_string: str) -> tuple[Image.Image, str]:
        """
        Decode base64 string to PIL Image object
        
        Args:
            base64_string: Base64 encoded image string (with or without data URL prefix)
            
        Returns:
            tuple: (PIL Image object, original format)
            
        Raises:
            ValueError: If base64 string is invalid or not an image
        """
        try:
            # Handle data URL format (data:image/png;base64,...)
            if base64_string.startswith('data:image'):
                # Extract the base64 part after the comma
                header, encoded = base64_string.split(',', 1)
                # Extract format from header (e.g., 'png' from 'data:image/png;base64')
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
        """
        Encode PIL Image to base64 string with data URL prefix
        
        Args:
            image: PIL Image object
            format_type: Output format ('PNG', 'JPEG', 'WEBP')
            quality: JPEG quality (1-100), ignored for PNG
            
        Returns:
            str: Base64 encoded image with data URL prefix
        """
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
        """
        Detect edges using Canny edge detection
        
        Args:
            image: PIL Image object
            threshold1: First threshold for edge detection
            threshold2: Second threshold for edge detection
            
        Returns:
            PIL Image with detected edges
        """
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
    def analyze_dominant_colors(image: Image.Image, num_colors: int = 5) -> tuple[Image.Image, List[Dict]]:
        """
        Analyze dominant colors in the image
        
        Args:
            image: PIL Image object
            num_colors: Number of dominant colors to extract
            
        Returns:
            tuple: (Original image, List of dominant color info)
        """
        try:
            # Resize image for faster processing
            temp_image = image.copy()
            temp_image.thumbnail((150, 150), Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary
            if temp_image.mode != 'RGB':
                temp_image = temp_image.convert('RGB')
            
            # Get pixel data
            pixels = list(temp_image.getdata())
            
            # Use k-means clustering to find dominant colors
            from collections import Counter
            
            # Simple color quantization (alternative to k-means for simplicity)
            # Group similar colors
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
        """
        Enhance image quality
        
        Args:
            image: PIL Image object
            enhancement_type: Type of enhancement ('brightness', 'contrast', 'sharpness', 'color', 'auto')
            
        Returns:
            Enhanced PIL Image
        """
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

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
CORS(app, origins=["*"])

# Initialize components
analyzer = ImageAnalysisAPI()
processor = ImageProcessor()
base64_processor = Base64ImageProcessor()
rate_limiter = RateLimiter()

# Setup logging
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create upload directory
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

def require_rate_limit(f):
    """Rate limiting decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        if not rate_limiter.is_allowed(client_ip):
            return jsonify({
                "success": False,
                "error": "Rate limit exceeded. Maximum 100 requests per hour.",
                "error_code": "RATE_LIMIT_EXCEEDED"
            }), 429
        return f(*args, **kwargs)
    return decorated_function

def log_request(f):
    """Request logging decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        app.logger.info(f"[{request_id}] {request.method} {request.endpoint} - {request.remote_addr}")
        
        try:
            result = f(*args, **kwargs)
            duration = time.time() - start_time
            app.logger.info(f"[{request_id}] Completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            app.logger.error(f"[{request_id}] Failed in {duration:.2f}s: {str(e)}")
            raise
            
    return decorated_function

# API Routes

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Image Analysis API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route('/api/v1/analyze/general', methods=['POST'])
@require_rate_limit
@log_request
def analyze_general():
    """General image analysis endpoint"""
    return _process_analysis_request(analyzer.analyze_general)

@app.route('/api/v1/analyze/objects', methods=['POST'])
@require_rate_limit
@log_request
def analyze_objects():
    """Object detection endpoint"""
    return _process_analysis_request(analyzer.analyze_objects)

@app.route('/api/v1/analyze/scene', methods=['POST'])
@require_rate_limit
@log_request
def analyze_scene():
    """Scene analysis endpoint"""
    return _process_analysis_request(analyzer.analyze_scene)

@app.route('/api/v1/analyze/text', methods=['POST'])
@require_rate_limit
@log_request
def analyze_text():
    """Text extraction (OCR) endpoint"""
    return _process_analysis_request(analyzer.analyze_text)

@app.route('/api/v1/analyze/people', methods=['POST'])
@require_rate_limit
@log_request
def analyze_people():
    """People analysis endpoint"""
    return _process_analysis_request(analyzer.analyze_people)

@app.route('/api/v1/analyze/technical', methods=['POST'])
@require_rate_limit
@log_request
def analyze_technical():
    """Technical image analysis endpoint"""
    return _process_analysis_request(analyzer.analyze_technical)

@app.route('/api/v1/analyze/safety', methods=['POST'])
@require_rate_limit
@log_request
def analyze_safety():
    """Safety and content moderation endpoint"""
    return _process_analysis_request(analyzer.analyze_safety)

@app.route('/api/v1/analyze/comprehensive', methods=['POST'])
@require_rate_limit
@log_request
def analyze_comprehensive():
    """Comprehensive analysis endpoint"""
    return _process_analysis_request(analyzer.analyze_comprehensive)

@app.route('/api/v1/analyze/batch', methods=['POST'])
@require_rate_limit
@log_request
def analyze_batch():
    """Batch analysis with multiple analysis types"""
    try:
        image_data, metadata, error = _extract_image_data()
        if error:
            return error
        
        # Get requested analysis types
        analysis_types = request.form.getlist('analysis_types') or ['general']
        custom_prompt = request.form.get('custom_prompt')
        
        results = {}
        for analysis_type in analysis_types:
            if analysis_type == 'general':
                results[analysis_type] = analyzer.analyze_general(image_data, custom_prompt)
            elif analysis_type == 'objects':
                results[analysis_type] = analyzer.analyze_objects(image_data)
            elif analysis_type == 'scene':
                results[analysis_type] = analyzer.analyze_scene(image_data)
            elif analysis_type == 'text':
                results[analysis_type] = analyzer.analyze_text(image_data)
            elif analysis_type == 'people':
                results[analysis_type] = analyzer.analyze_people(image_data)
            elif analysis_type == 'technical':
                results[analysis_type] = analyzer.analyze_technical(image_data)
            elif analysis_type == 'safety':
                results[analysis_type] = analyzer.analyze_safety(image_data)
            elif analysis_type == 'comprehensive':
                results[analysis_type] = analyzer.analyze_comprehensive(image_data)
            else:
                results[analysis_type] = {
                    "success": False,
                    "error": f"Unknown analysis type: {analysis_type}"
                }
        
        return jsonify({
            "success": True,
            "analysis_types": analysis_types,
            "results": results,
            "metadata": metadata,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Batch analysis failed: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Batch analysis failed: {str(e)}",
            "error_code": "BATCH_ANALYSIS_FAILED"
        }), 500


# API Routes for Base64 Image Processing

@app.route('/api/v1/process/grayscale', methods=['POST'])
@require_rate_limit
@log_request
def process_grayscale():
    """Convert image to grayscale via base64"""
    return _process_base64_image(
        process_func=base64_processor.convert_to_grayscale,
        process_name="grayscale"
    )

@app.route('/api/v1/process/edges', methods=['POST'])
@require_rate_limit
@log_request
def process_edges():
    """Detect edges in image via base64"""
    def edge_detection(image):
        threshold1 = int(request.form.get('threshold1', 100))
        threshold2 = int(request.form.get('threshold2', 200))
        return base64_processor.detect_edges(image, threshold1, threshold2)
    
    return _process_base64_image(
        process_func=edge_detection,
        process_name="edge_detection"
    )

@app.route('/api/v1/process/colors', methods=['POST'])
@require_rate_limit
@log_request
def process_colors():
    """Analyze dominant colors in image via base64"""
    try:
        # Get base64 input
        base64_input = request.json.get('image_base64') if request.is_json else request.form.get('image_base64')
        
        if not base64_input:
            return jsonify({
                "success": False,
                "error": "No base64 image data provided. Include 'image_base64' parameter.",
                "error_code": "NO_BASE64_INPUT"
            }), 400
        
        # Decode base64 image
        try:
            image, original_format = base64_processor.decode_base64_image(base64_input)
        except ValueError as e:
            return jsonify({
                "success": False,
                "error": str(e),
                "error_code": "INVALID_BASE64"
            }), 400
        
        # Get number of colors to analyze
        num_colors = int(request.form.get('num_colors', 5)) if not request.is_json else int(request.json.get('num_colors', 5))
        
        # Analyze dominant colors
        processed_image, color_info = base64_processor.analyze_dominant_colors(image, num_colors)
        
        # Encode result back to base64
        output_format = request.form.get('output_format', original_format) if not request.is_json else request.json.get('output_format', original_format)
        result_base64 = base64_processor.encode_image_to_base64(processed_image, output_format)
        
        return jsonify({
            "success": True,
            "process_type": "color_analysis",
            "input_format": original_format,
            "output_format": output_format,
            "image_base64": result_base64,
            "dominant_colors": color_info,
            "metadata": {
                "num_colors_analyzed": num_colors,
                "image_size": image.size,
                "processed_in_memory": True
            },
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Color analysis failed: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Color analysis failed: {str(e)}",
            "error_code": "COLOR_ANALYSIS_FAILED"
        }), 500

@app.route('/api/v1/process/enhance', methods=['POST'])
@require_rate_limit
@log_request
def process_enhance():
    """Enhance image quality via base64"""
    def enhancement(image):
        enhancement_type = request.form.get('enhancement_type', 'auto') if not request.is_json else request.json.get('enhancement_type', 'auto')
        return base64_processor.enhance_image(image, enhancement_type)
    
    return _process_base64_image(
        process_func=enhancement,
        process_name="enhancement"
    )

@app.route('/api/v1/process/custom', methods=['POST'])
@require_rate_limit
@log_request
def process_custom():
    """Custom image processing via base64 with multiple operations"""
    try:
        # Get base64 input
        base64_input = request.json.get('image_base64') if request.is_json else request.form.get('image_base64')
        
        if not base64_input:
            return jsonify({
                "success": False,
                "error": "No base64 image data provided. Include 'image_base64' parameter.",
                "error_code": "NO_BASE64_INPUT"
            }), 400
        
        # Get processing operations
        operations = request.json.get('operations', ['grayscale']) if request.is_json else request.form.getlist('operations') or ['grayscale']
        
        # Decode base64 image
        try:
            image, original_format = base64_processor.decode_base64_image(base64_input)
        except ValueError as e:
            return jsonify({
                "success": False,
                "error": str(e),
                "error_code": "INVALID_BASE64"
            }), 400
        
        # Apply operations in sequence
        processed_image = image.copy()
        applied_operations = []
        
        for operation in operations:
            try:
                if operation == 'grayscale':
                    processed_image = base64_processor.convert_to_grayscale(processed_image)
                    applied_operations.append("grayscale")
                    
                elif operation == 'edges':
                    threshold1 = int(request.form.get('threshold1', 100)) if not request.is_json else int(request.json.get('threshold1', 100))
                    threshold2 = int(request.form.get('threshold2', 200)) if not request.is_json else int(request.json.get('threshold2', 200))
                    processed_image = base64_processor.detect_edges(processed_image, threshold1, threshold2)
                    applied_operations.append("edge_detection")
                    
                elif operation == 'enhance':
                    enhancement_type = request.form.get('enhancement_type', 'auto') if not request.is_json else request.json.get('enhancement_type', 'auto')
                    processed_image = base64_processor.enhance_image(processed_image, enhancement_type)
                    applied_operations.append("enhancement")
                    
            except Exception as e:
                app.logger.warning(f"Operation {operation} failed: {str(e)}")
        
        # Encode result back to base64
        output_format = request.form.get('output_format', original_format) if not request.is_json else request.json.get('output_format', original_format)
        result_base64 = base64_processor.encode_image_to_base64(processed_image, output_format)
        
        return jsonify({
            "success": True,
            "process_type": "custom_processing",
            "input_format": original_format,
            "output_format": output_format,
            "image_base64": result_base64,
            "applied_operations": applied_operations,
            "metadata": {
                "requested_operations": operations,
                "image_size": image.size,
                "processed_in_memory": True
            },
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Custom processing failed: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Custom processing failed: {str(e)}",
            "error_code": "CUSTOM_PROCESSING_FAILED"
        }), 500

def _process_base64_image(process_func, process_name: str):
    """
    Helper function to process base64 images
    
    Args:
        process_func: Function to apply to the image
        process_name: Name of the processing operation
        
    Returns:
        JSON response with processed image
    """
    try:
        # Get base64 input from either JSON or form data
        base64_input = request.json.get('image_base64') if request.is_json else request.form.get('image_base64')
        
        if not base64_input:
            return jsonify({
                "success": False,
                "error": "No base64 image data provided. Include 'image_base64' parameter.",
                "error_code": "NO_BASE64_INPUT"
            }), 400
        
        # Step 1: Decode base64 string into image object
        try:
            image, original_format = base64_processor.decode_base64_image(base64_input)
            app.logger.info(f"Successfully decoded base64 image: {original_format}, Size: {image.size}")
        except ValueError as e:
            return jsonify({
                "success": False,
                "error": str(e),
                "error_code": "INVALID_BASE64"
            }), 400
        
        # Step 2: Apply image processing transformation
        try:
            processed_image = process_func(image)
            app.logger.info(f"Successfully applied {process_name} processing")
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Image processing failed: {str(e)}",
                "error_code": "PROCESSING_FAILED"
            }), 500
        
        # Step 3: Encode processed image back to base64
        try:
            output_format = request.form.get('output_format', original_format) if not request.is_json else request.json.get('output_format', original_format)
            quality = int(request.form.get('quality', 95)) if not request.is_json else int(request.json.get('quality', 95))
            
            result_base64 = base64_processor.encode_image_to_base64(processed_image, output_format, quality)
            app.logger.info(f"Successfully encoded result to base64: {output_format}")
        except ValueError as e:
            return jsonify({
                "success": False,
                "error": f"Base64 encoding failed: {str(e)}",
                "error_code": "ENCODING_FAILED"
            }), 500
        
        # Step 4: Return the processed base64 string
        return jsonify({
            "success": True,
            "process_type": process_name,
            "input_format": original_format,
            "output_format": output_format,
            "image_base64": result_base64,
            "metadata": {
                "image_size": processed_image.size,
                "processed_in_memory": True,
                "quality": quality if output_format.upper() == 'JPEG' else None
            },
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Base64 image processing failed: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Processing failed: {str(e)}",
            "error_code": "PROCESSING_ERROR"
        }), 500
def _process_analysis_request(analysis_function):
    """Helper function to process analysis requests"""
    try:
        image_data, metadata, error = _extract_image_data()
        if error:
            return error
        
        custom_prompt = request.form.get('custom_prompt')
        
        # Call appropriate analysis function
        if custom_prompt and analysis_function == analyzer.analyze_general:
            result = analysis_function(image_data, custom_prompt)
        else:
            result = analysis_function(image_data)
        
        response_data = {
            **result,
            "metadata": metadata,
            "request_timestamp": datetime.utcnow().isoformat()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        app.logger.error(f"Analysis request failed: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Analysis failed: {str(e)}",
            "error_code": "ANALYSIS_FAILED"
        }), 500

def _extract_image_data():
    """Extract and process image data from request"""
    try:
        image_data = None
        metadata = {}
        
        # Check for image URL
        if 'image_url' in request.form and request.form['image_url'].strip():
            image_url = request.form['image_url'].strip()
            
            # Validate URL
            if not (image_url.startswith('http://') or image_url.startswith('https://')):
                return None, None, jsonify({
                    "success": False,
                    "error": "Invalid URL format. Must start with http:// or https://",
                    "error_code": "INVALID_URL"
                }), 400
            
            image_data = image_url
            metadata = {
                "source": "url",
                "url": image_url,
                "processed": False
            }
        
        # Check for uploaded file
        elif 'image' in request.files:
            file = request.files['image']
            
            # Validate file
            valid, message = processor.validate_image(file)
            if not valid:
                return None, None, jsonify({
                    "success": False,
                    "error": message,
                    "error_code": "INVALID_IMAGE"
                }), 400
            
            # Process image
            image_data = processor.process_image(file)
            metadata = {
                "source": "upload",
                "processed": True,
                **processor.get_image_metadata(file)
            }
        
        else:
            return None, None, jsonify({
                "success": False,
                "error": "No image provided. Include 'image' file or 'image_url' parameter.",
                "error_code": "NO_IMAGE"
            }), 400
        
        return image_data, metadata, None
        
    except Exception as e:
        app.logger.error(f"Image extraction failed: {str(e)}")
        return None, None, jsonify({
            "success": False,
            "error": f"Image processing failed: {str(e)}",
            "error_code": "IMAGE_PROCESSING_FAILED"
        }), 500

# Error handlers

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({
        "success": False,
        "error": f"File too large. Maximum size: {Config.MAX_CONTENT_LENGTH // (1024*1024)}MB",
        "error_code": "FILE_TOO_LARGE"
    }), 413

@app.errorhandler(429)
def rate_limit_exceeded(e):
    return jsonify({
        "success": False,
        "error": "Rate limit exceeded. Please try again later.",
        "error_code": "RATE_LIMIT_EXCEEDED"
    }), 429

@app.errorhandler(500)
def internal_error(e):
    app.logger.error(f"Internal server error: {str(e)}")
    return jsonify({
        "success": False,
        "error": "Internal server error. Please try again later.",
        "error_code": "INTERNAL_ERROR"
    }), 500

if __name__ == '__main__':
    app.logger.info("Starting Professional Image Analysis API...")
    app.run(
        debug=os.getenv('FLASK_ENV') == 'development',
        host='0.0.0.0',
        port=int(os.getenv('PORT')),
        threaded=True
    )