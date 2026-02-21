"""FastAPI server for ESP32 traffic light control"""
from fastapi import FastAPI
from .models import PhaseDecision


# Global engine instance
engine = None


def create_app():
    """Create FastAPI application"""
    app = FastAPI(title="CF-MADRL Traffic Control")
    
    @app.get("/get_phase", response_model=PhaseDecision)
    def get_phase():
        """Get next phase decision for ESP32"""
        if engine is None:
            return PhaseDecision(phase=0, duration=30, yellow_required=False)
        
        current_phase = engine.last_phase or 0
        metrics = engine.get_real_metrics()
        
        if metrics is None:
            return PhaseDecision(phase=current_phase, duration=30, yellow_required=False)
        
        queues, waits = metrics
        phase, duration, yellow_required = engine.get_action(queues, waits, current_phase)
        
        return PhaseDecision(phase=phase, duration=duration, yellow_required=yellow_required)
    
    @app.get("/health")
    def health():
        """Health check"""
        return {"status": "ok"}
    
    return app


def set_engine(inference_engine):
    """Set the inference engine instance"""
    global engine
    engine = inference_engine


def run_server(port=8000):
    """Start API server"""
    import uvicorn
    app = create_app()
    print(f"\n✓ API Server on http://0.0.0.0:{port}")
    print(f"✓ ESP32 calls: GET http://<pi-ip>:{port}/get_phase\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
