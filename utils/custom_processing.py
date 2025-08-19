from flask import request, jsonify, current_app
from datetime import datetime
from services.service_manager import ServiceManager

def process_custom_operations():
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
        
        base64_processor = ServiceManager.base64_processor
        
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
        
        operation_handlers = {
            'grayscale': lambda img: base64_processor.convert_to_grayscale(img),
            'edges': lambda img: _handle_edge_detection(img, base64_processor),
            'enhance': lambda img: _handle_enhancement(img, base64_processor)
        }
        
        for operation in operations:
            try:
                if operation in operation_handlers:
                    processed_image = operation_handlers[operation](processed_image)
                    applied_operations.append(operation if operation != 'edges' else 'edge_detection')
            except Exception as e:
                current_app.logger.warning(f"Operation {operation} failed: {str(e)}")
        
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
        current_app.logger.error(f"Custom processing failed: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Custom processing failed: {str(e)}",
            "error_code": "CUSTOM_PROCESSING_FAILED"
        }), 500

def _handle_edge_detection(image, processor):
    """Handle edge detection with parameters"""
    threshold1 = int(request.form.get('threshold1', 100)) if not request.is_json else int(request.json.get('threshold1', 100))
    threshold2 = int(request.form.get('threshold2', 200)) if not request.is_json else int(request.json.get('threshold2', 200))
    return processor.detect_edges(image, threshold1, threshold2)

def _handle_enhancement(image, processor):
    """Handle image enhancement with parameters"""
    enhancement_type = request.form.get('enhancement_type', 'auto') if not request.is_json else request.json.get('enhancement_type', 'auto')
    return processor.enhance_image(image, enhancement_type)
