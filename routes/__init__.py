from .health import health_bp
from .analysis import analysis_bp
from .processing import processing_bp

def register_blueprints(app):
    """Register all application blueprints"""
    app.register_blueprint(health_bp)
    app.register_blueprint(analysis_bp, url_prefix='/api/v1/analyze')
    app.register_blueprint(processing_bp, url_prefix='/api/v1/process')
