import time
import uuid
import logging
from functools import wraps
from flask import request, jsonify, current_app
from .rate_limiter import RateLimiter

# Global rate limiter instance
rate_limiter = RateLimiter()

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
        
        current_app.logger.info(f"[{request_id}] {request.method} {request.endpoint} - {request.remote_addr}")
        
        try:
            result = f(*args, **kwargs)
            duration = time.time() - start_time
            current_app.logger.info(f"[{request_id}] Completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            current_app.logger.error(f"[{request_id}] Failed in {duration:.2f}s: {str(e)}")
            raise
            
    return decorated_function
