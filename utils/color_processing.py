from flask import request, jsonify, current_app
from datetime import datetime
from services.service_manager import ServiceManager

def process_color_analysis():
    """Analyze dominant colors in image via base64"""
    try:
        # Get base64 input
        base64_input = request.json.get('image_base64') if request.is_json else request.form.get('image_base64')
        
        if not base64_input:
            return jsonify({
                "success": False,
                "error": "No base64 image data provided. Include 'image_base64' parameter.",
                "error_code": "NO_BASE64_INPUT"
            }), 400
        
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
        
        # Get number of colors to analyze
        num_colors = int(request.form.get('num_colors', 5)) if not request.is_json else int(request.json.get('num_colors', 5))
        
        # Analyze dominant colors
        processed_image, color_info = base64_processor.analyze_dominant_colors(image, num_colors)
        
        # Encode result back to base64
        output_format = request.form.get('output_format', original_format) if not request.is_json else request.json.get('output_format', original_format)
        result_base64 = base64_processor.encode_image_to_base64(processed_image, output_format)
        
        return jsonify({
            "success": True,
            "process_type": "color_analysis",
            "input_format": original_format,
            "output_format": output_format,
            "image_base64": result_base64,
            "dominant_colors": color_info,
            "metadata": {
                "num_colors_analyzed": num_colors,
                "image_size": image.size,
                "processed_in_memory": True
            },
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Color analysis failed: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Color analysis failed: {str(e)}",
            "error_code": "COLOR_ANALYSIS_FAILED"
        }), 500
