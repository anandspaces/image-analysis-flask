from flask import jsonify
from config.config import Config

def register_error_handlers(app):
    """Register all error handlers"""
    
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