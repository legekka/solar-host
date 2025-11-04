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
    # Allow CORS preflight requests (OPTIONS) without authentication
    if request.method == "OPTIONS":
        return await call_next(request)
    
    # Allow access to health check, docs, and OpenAPI schema
    public_paths = ["/health", "/", "/docs", "/redoc", "/openapi.json"]
    if request.url.path in public_paths:
        return await call_next(request)
    
    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key != settings.api_key:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Invalid or missing API key"},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            }
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


app.openapi = custom_openapi  # type: ignore[method-assign]


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


@app.get("/memory")
async def get_memory():
    """Get GPU/RAM memory usage"""
    from fastapi import HTTPException
    from app.memory_monitor import get_memory_info
    from app.models import MemoryInfo
    
    memory_info = get_memory_info()
    if not memory_info:
        raise HTTPException(
            status_code=503,
            detail="Memory information not available"
        )
    # Coerce types explicitly to satisfy typing
    try:
        used_gb = float(memory_info.get("used_gb"))  # type: ignore[arg-type]
        total_gb = float(memory_info.get("total_gb"))  # type: ignore[arg-type]
        percent = float(memory_info.get("percent"))  # type: ignore[arg-type]
        memory_type = str(memory_info.get("memory_type"))
    except Exception:
        # Fallback values if coercion fails
        used_gb = 0.0
        total_gb = 0.0
        percent = 0.0
        memory_type = "RAM"
    
    return MemoryInfo(used_gb=used_gb, total_gb=total_gb, percent=percent, memory_type=memory_type)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )

