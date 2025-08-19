import time
import logging
from datetime import datetime
from openai import OpenAI
from typing import Dict, Any
from config.config import Config

class ImageAnalysisAPI:
    """Professional Image Analysis API Handler"""
    
    def __init__(self):
        self.client = OpenAI(
            base_url=Config.OPENROUTER_BASE_URL,
            api_key=Config.OPENAI_KEY,
        )
        self.logger = logging.getLogger(__name__)
        
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
        max_retries = Config.AI_MAX_RETRIES
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://api.imageanalyzer.pro",
                        "X-Title": "Professional Image Analysis API",
                    },
                    model=Config.AI_MODEL,
                    max_tokens=Config.AI_MAX_TOKENS,
                    temperature=Config.AI_TEMPERATURE,
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
                    "model": Config.AI_MODEL,
                    "timestamp": datetime.utcnow().isoformat(),
                    "tokens_used": completion.usage.total_tokens if hasattr(completion, 'usage') else None
                }
                
            except Exception as e:
                self.logger.error(f"AI request attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    return {
                        "success": False,
                        "error": f"Analysis failed after {max_retries} attempts: {str(e)}",
                        "analysis_type": analysis_type,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                time.sleep(retry_delay * (2 ** attempt))
