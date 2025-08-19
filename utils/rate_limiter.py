import time
import threading
from typing import Dict, List
from config.config import Config

class RateLimiter:
    """Rate limiting for API endpoints"""
    
    def __init__(self):
        self.requests: Dict[str, List[float]] = {}
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_requests, daemon=True)
        self.cleanup_thread.start()
    
    def is_allowed(self, ip: str, limit: int = Config.API_RATE_LIMIT) -> bool:
        """Check if request is allowed under rate limit"""
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
        """Background cleanup of old requests"""
        while True:
            time.sleep(300)  # Cleanup every 5 minutes
            current_time = time.time()
            hour_ago = current_time - 3600
            
            for ip in list(self.requests.keys()):
                self.requests[ip] = [req_time for req_time in self.requests[ip] if req_time > hour_ago]
                if not self.requests[ip]:
                    del self.requests[ip]
