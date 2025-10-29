from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class InstanceStatus(str, Enum):
    """Status of a llama-server instance"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    FAILED = "failed"
    STOPPING = "stopping"


class InstanceConfig(BaseModel):
    """Configuration for a llama-server instance"""
    model: str = Field(..., description="Path to the GGUF model file")
    alias: str = Field(..., description="Model alias (e.g., gpt-oss:120b)")
    threads: int = Field(default=1, description="Number of threads")
    n_gpu_layers: int = Field(default=999, description="Number of GPU layers")
    temp: float = Field(default=1.0, description="Temperature")
    top_p: float = Field(default=1.0, description="Top-p sampling")
    top_k: int = Field(default=0, description="Top-k sampling")
    min_p: float = Field(default=0.0, description="Min-p sampling")
    ctx_size: int = Field(default=131072, description="Context size")
    chat_template_file: Optional[str] = Field(default=None, description="Path to Jinja chat template")
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: Optional[int] = Field(default=None, description="Port (auto-assigned if not specified)")
    api_key: str = Field(..., description="API key for this llama-server instance")


class Instance(BaseModel):
    """Runtime instance information"""
    id: str
    config: InstanceConfig
    status: InstanceStatus = InstanceStatus.STOPPED
    port: Optional[int] = None
    pid: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0


class InstanceCreate(BaseModel):
    """Request to create a new instance"""
    config: InstanceConfig


class InstanceUpdate(BaseModel):
    """Request to update an instance config"""
    config: InstanceConfig


class LogMessage(BaseModel):
    """Log message with sequence number"""
    seq: int
    timestamp: str
    line: str


class InstanceResponse(BaseModel):
    """Response model for instance operations"""
    instance: Instance
    message: str

