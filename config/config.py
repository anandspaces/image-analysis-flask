import os
import logging
from dotenv import load_dotenv

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
    
    # API Configuration
    OPENAI_KEY = os.getenv("OPENAI_KEY")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    AI_MODEL = "google/gemini-2.5-flash"
    AI_MAX_TOKENS = 2000
    AI_TEMPERATURE = 0.1
    AI_MAX_RETRIES = 3
    
    # Image Processing
    MAX_IMAGE_SIZE = (2048, 2048)
    DEFAULT_JPEG_QUALITY = 95

class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = logging.DEBUG

class ProductionConfig(Config):
    DEBUG = False
    LOG_LEVEL = logging.WARNING

class TestingConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = False

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}