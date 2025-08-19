from analysis.image_analyzer import ImageAnalysisAPI
from analysis.base64_processor import Base64ImageProcessor
from processing.image_processor import ImageProcessor
from utils.rate_limiter import RateLimiter

class ServiceManager:
    """Centralized service management"""
    analyzer = None
    processor = None
    base64_processor = None
    rate_limiter = None
    
    @classmethod
    def init_app(cls, app):
        """Initialize all services"""
        cls.analyzer = ImageAnalysisAPI()
        cls.processor = ImageProcessor()
        cls.base64_processor = Base64ImageProcessor()
        cls.rate_limiter = RateLimiter()
        
        # Store reference in app context for access in views
        app.service_manager = cls
