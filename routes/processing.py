from flask import Blueprint, request, jsonify
from datetime import datetime
from services import ServiceManager
from utils.decorators import require_rate_limit, log_request
from utils.base64_helpers import process_base64_image

processing_bp = Blueprint('processing', __name__)

@processing_bp.route('/grayscale', methods=['POST'])
@require_rate_limit
@log_request
def process_grayscale():
    """Convert image to grayscale via base64"""
    return process_base64_image(
        process_func=ServiceManager.base64_processor.convert_to_grayscale,
        process_name="grayscale"
    )

@processing_bp.route('/edges', methods=['POST'])
@require_rate_limit
@log_request
def process_edges():
    """Detect edges in image via base64"""
    def edge_detection(image):
        threshold1 = int(request.form.get('threshold1', 100))
        threshold2 = int(request.form.get('threshold2', 200))
        return ServiceManager.base64_processor.detect_edges(image, threshold1, threshold2)
    
    return process_base64_image(
        process_func=edge_detection,
        process_name="edge_detection"
    )

@processing_bp.route('/colors', methods=['POST'])
@require_rate_limit
@log_request
def process_colors():
    """Analyze dominant colors in image via base64"""
    from utils.color_processing import process_color_analysis
    return process_color_analysis()

@processing_bp.route('/enhance', methods=['POST'])
@require_rate_limit
@log_request
def process_enhance():
    """Enhance image quality via base64"""
    def enhancement(image):
        enhancement_type = request.form.get('enhancement_type', 'auto') if not request.is_json else request.json.get('enhancement_type', 'auto')
        return ServiceManager.base64_processor.enhance_image(image, enhancement_type)
    
    return process_base64_image(
        process_func=enhancement,
        process_name="enhancement"
    )

@processing_bp.route('/custom', methods=['POST'])
@require_rate_limit
@log_request
def process_custom():
    """Custom image processing via base64 with multiple operations"""
    from utils.custom_processing import process_custom_operations
    return process_custom_operations()
