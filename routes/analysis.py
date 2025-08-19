from flask import Blueprint, request, jsonify
from datetime import datetime
from services.service_manager import ServiceManager
from utils.decorators import require_rate_limit, log_request
from utils.request_helpers import extract_image_data

analysis_bp = Blueprint('analysis', __name__)

@analysis_bp.route('/general', methods=['POST'])
@require_rate_limit
@log_request
def analyze_general():
    """General image analysis endpoint"""
    return _process_analysis_request(ServiceManager.analyzer.analyze_general)

@analysis_bp.route('/objects', methods=['POST'])
@require_rate_limit
@log_request
def analyze_objects():
    """Object detection endpoint"""
    return _process_analysis_request(ServiceManager.analyzer.analyze_objects)

@analysis_bp.route('/scene', methods=['POST'])
@require_rate_limit
@log_request
def analyze_scene():
    """Scene analysis endpoint"""
    return _process_analysis_request(ServiceManager.analyzer.analyze_scene)

@analysis_bp.route('/text', methods=['POST'])
@require_rate_limit
@log_request
def analyze_text():
    """Text extraction (OCR) endpoint"""
    return _process_analysis_request(ServiceManager.analyzer.analyze_text)

@analysis_bp.route('/people', methods=['POST'])
@require_rate_limit
@log_request
def analyze_people():
    """People analysis endpoint"""
    return _process_analysis_request(ServiceManager.analyzer.analyze_people)

@analysis_bp.route('/technical', methods=['POST'])
@require_rate_limit
@log_request
def analyze_technical():
    """Technical image analysis endpoint"""
    return _process_analysis_request(ServiceManager.analyzer.analyze_technical)

@analysis_bp.route('/safety', methods=['POST'])
@require_rate_limit
@log_request
def analyze_safety():
    """Safety and content moderation endpoint"""
    return _process_analysis_request(ServiceManager.analyzer.analyze_safety)

@analysis_bp.route('/comprehensive', methods=['POST'])
@require_rate_limit
@log_request
def analyze_comprehensive():
    """Comprehensive analysis endpoint"""
    return _process_analysis_request(ServiceManager.analyzer.analyze_comprehensive)

@analysis_bp.route('/batch', methods=['POST'])
@require_rate_limit
@log_request
def analyze_batch():
    """Batch analysis with multiple analysis types"""
    try:
        image_data, metadata, error = extract_image_data()
        if error:
            return error
        
        # Get requested analysis types
        analysis_types = request.form.getlist('analysis_types') or ['general']
        custom_prompt = request.form.get('custom_prompt')
        
        results = {}
        analyzer = ServiceManager.analyzer
        
        analysis_map = {
            'general': lambda: analyzer.analyze_general(image_data, custom_prompt),
            'objects': lambda: analyzer.analyze_objects(image_data),
            'scene': lambda: analyzer.analyze_scene(image_data),
            'text': lambda: analyzer.analyze_text(image_data),
            'people': lambda: analyzer.analyze_people(image_data),
            'technical': lambda: analyzer.analyze_technical(image_data),
            'safety': lambda: analyzer.analyze_safety(image_data),
            'comprehensive': lambda: analyzer.analyze_comprehensive(image_data)
        }
        
        for analysis_type in analysis_types:
            if analysis_type in analysis_map:
                results[analysis_type] = analysis_map[analysis_type]()
            else:
                results[analysis_type] = {
                    "success": False,
                    "error": f"Unknown analysis type: {analysis_type}"
                }
        
        return jsonify({
            "success": True,
            "analysis_types": analysis_types,
            "results": results,
            "metadata": metadata,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Batch analysis failed: {str(e)}",
            "error_code": "BATCH_ANALYSIS_FAILED"
        }), 500

def _process_analysis_request(analysis_function):
    """Helper function to process analysis requests"""
    try:
        image_data, metadata, error = extract_image_data()
        if error:
            return error
        
        custom_prompt = request.form.get('custom_prompt')
        
        # Call appropriate analysis function
        if custom_prompt and analysis_function == ServiceManager.analyzer.analyze_general:
            result = analysis_function(image_data, custom_prompt)
        else:
            result = analysis_function(image_data)
        
        response_data = {
            **result,
            "metadata": metadata,
            "request_timestamp": datetime.utcnow().isoformat()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Analysis failed: {str(e)}",
            "error_code": "ANALYSIS_FAILED"
        }), 500
