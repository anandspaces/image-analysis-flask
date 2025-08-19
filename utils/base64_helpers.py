from flask import request, jsonify, current_app
from datetime import datetime
from services.service_manager import ServiceManager

def process_base64_image(process_func, process_name: str):
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
        
        base64_processor = ServiceManager.base64_processor
        
        # Step 1: Decode base64 string into image object
        try:
            image, original_format = base64_processor.decode_base64_image(base64_input)
            current_app.logger.info(f"Successfully decoded base64 image: {original_format}, Size: {image.size}")
        except ValueError as e:
            return jsonify({
                "success": False,
                "error": str(e),
                "error_code": "INVALID_BASE64"
            }), 400
        
        # Step 2: Apply image processing transformation
        try:
            processed_image = process_func(image)
            current_app.logger.info(f"Successfully applied {process_name} processing")
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
            current_app.logger.info(f"Successfully encoded result to base64: {output_format}")
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
        current_app.logger.error(f"Base64 image processing failed: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Processing failed: {str(e)}",
            "error_code": "PROCESSING_ERROR"
        }), 500
