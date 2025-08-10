from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from functools import wraps
import os
import logging
from datetime import datetime
import uuid
import hashlib
import base64
from io import BytesIO
from PIL import Image, ImageOps
import requests
from openai import OpenAI
from dotenv import load_dotenv
import threading
import time
import json
from typing import Dict, List, Optional, Any

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

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
CORS(app, origins=["*"])

# Initialize components
analyzer = ImageAnalysisAPI()
processor = ImageProcessor()
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