from flask import request, jsonify, current_app
from services.service_manager import ServiceManager

def extract_image_data():
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
            processor = ServiceManager.processor
            
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
        current_app.logger.error(f"Image extraction failed: {str(e)}")
        return None, None, jsonify({
            "success": False,
            "error": f"Image processing failed: {str(e)}",
            "error_code": "IMAGE_PROCESSING_FAILED"
        }), 500
