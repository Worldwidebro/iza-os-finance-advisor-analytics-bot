#!/usr/bin/env python3
"""
IZA OS Finance Advisor Analytics Bot
Main application entry point for financial analytics and intelligence.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Add src to path
sys.path.append(str(Path(__file__).parent))

from analytics.revenue_analytics import RevenueAnalyzer
from analytics.risk_assessment import RiskAnalyzer
from analytics.predictive_models import FinancialPredictor
from data.collectors.financial_data_collector import FinancialDataCollector
from api.endpoints.analytics import router as analytics_router
from api.endpoints.health import router as health_router
from config.analytics_config import AnalyticsConfig
from utils.logging_config import setup_logging


# Global instances
revenue_analyzer: RevenueAnalyzer = None
risk_analyzer: RiskAnalyzer = None
financial_predictor: FinancialPredictor = None
data_collector: FinancialDataCollector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    global revenue_analyzer, risk_analyzer, financial_predictor, data_collector
    
    # Startup
    logging.info("🚀 Starting IZA OS Finance Advisor Analytics Bot...")
    
    try:
        # Initialize configuration
        config = AnalyticsConfig()
        
        # Initialize core components
        data_collector = FinancialDataCollector(config)
        revenue_analyzer = RevenueAnalyzer(config, data_collector)
        risk_analyzer = RiskAnalyzer(config, data_collector)
        financial_predictor = FinancialPredictor(config, data_collector)
        
        # Start data collection
        await data_collector.start()
        
        # Initialize AI models
        await financial_predictor.load_models()
        
        logging.info("✅ Finance Analytics Bot initialized successfully")
        
    except Exception as e:
        logging.error(f"❌ Failed to initialize Finance Analytics Bot: {e}")
        raise
    
    yield
    
    # Shutdown
    logging.info("🛑 Shutting down Finance Analytics Bot...")
    
    if data_collector:
        await data_collector.stop()
    
    logging.info("✅ Finance Analytics Bot shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    # Setup logging
    setup_logging()
    
    app = FastAPI(
        title="IZA OS Finance Advisor Analytics Bot",
        description="Advanced AI-powered financial analytics and intelligence bot",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(analytics_router, prefix="/api/v1")
    app.include_router(health_router, prefix="/api/v1")
    
    # WebSocket connections
    active_connections: Dict[str, WebSocket] = {}
    
    @app.websocket("/ws/analytics")
    async def websocket_analytics(websocket: WebSocket):
        """WebSocket endpoint for real-time analytics updates."""
        await websocket.accept()
        connection_id = f"analytics_{id(websocket)}"
        active_connections[connection_id] = websocket
        
        try:
            while True:
                # Send real-time analytics data
                if revenue_analyzer:
                    analytics_data = await revenue_analyzer.get_realtime_analytics()
                    await websocket.send_json(analytics_data)
                
                await asyncio.sleep(1)  # Update every second
                
        except WebSocketDisconnect:
            active_connections.pop(connection_id, None)
            logging.info(f"WebSocket connection {connection_id} disconnected")
    
    @app.websocket("/ws/alerts")
    async def websocket_alerts(websocket: WebSocket):
        """WebSocket endpoint for financial alerts and notifications."""
        await websocket.accept()
        connection_id = f"alerts_{id(websocket)}"
        active_connections[connection_id] = websocket
        
        try:
            while True:
                # Send financial alerts
                if risk_analyzer:
                    alerts = await risk_analyzer.get_active_alerts()
                    if alerts:
                        await websocket.send_json(alerts)
                
                await asyncio.sleep(5)  # Check alerts every 5 seconds
                
        except WebSocketDisconnect:
            active_connections.pop(connection_id, None)
            logging.info(f"WebSocket connection {connection_id} disconnected")
    
    @app.get("/")
    async def root():
        """Root endpoint with service information."""
        return {
            "service": "IZA OS Finance Advisor Analytics Bot",
            "version": "1.0.0",
            "status": "operational",
            "description": "Advanced AI-powered financial analytics and intelligence bot",
            "endpoints": {
                "analytics": "/api/v1/analytics",
                "health": "/api/v1/health",
                "websocket_analytics": "/ws/analytics",
                "websocket_alerts": "/ws/alerts"
            }
        }
    
    return app


def main():
    """Main entry point."""
    app = create_app()
    
    # Run the application
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()
