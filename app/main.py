from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.config import settings
from app.process_manager import process_manager
from app.routes import instances, websockets


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the application"""
    # Startup: Auto-restart instances that were running
    print("Starting Solar Host...")
    print(f"API Key configured: {settings.api_key[:4]}...")
    await process_manager.auto_restart_running_instances()
    print("Solar Host started successfully")
    
    yield
    
    # Shutdown: Clean up
    print("Shutting down Solar Host...")


app = FastAPI(
    title="Solar Host",
    description="Process manager for llama-server instances",
    version="1.0.0",
    lifespan=lifespan,
    swagger_ui_parameters={"persistAuthorization": True}
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Key authentication middleware
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    """Verify API key for all requests except health check and OpenAPI docs"""
    # Allow access to health check, docs, and OpenAPI schema
    public_paths = ["/health", "/", "/docs", "/redoc", "/openapi.json"]
    if request.url.path in public_paths:
        return await call_next(request)
    
    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key != settings.api_key:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Invalid or missing API key"}
        )
    
    return await call_next(request)


# Include routers
app.include_router(instances.router)
app.include_router(websockets.router)


# Customize OpenAPI schema to add security
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    from fastapi.openapi.utils import get_openapi
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        }
    }
    
    # Apply security to all paths except public ones
    public_paths = ["/health", "/", "/docs", "/redoc", "/openapi.json"]
    for path, path_item in openapi_schema["paths"].items():
        if path not in public_paths:
            for operation in path_item.values():
                if isinstance(operation, dict):
                    operation["security"] = [{"APIKeyHeader": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "solar-host",
        "version": "1.0.0"
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "solar-host",
        "version": "1.0.0",
        "description": "Process manager for llama-server instances"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )

