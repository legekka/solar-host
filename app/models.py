from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone
from enum import Enum


class InstanceStatus(str, Enum):
    """Status of a llama-server instance"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    FAILED = "failed"
    STOPPING = "stopping"
class InstancePhase(str, Enum):
    """Fine-grained runtime phase of an active request"""
    IDLE = "idle"
    PREFILL = "prefill"
    GENERATING = "generating"



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
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0

    # Ephemeral runtime fields (not persisted to disk)
    busy: bool = Field(default=False, exclude=True)
    prefill_progress: Optional[float] = Field(default=None, exclude=True)
    active_slots: int = Field(default=0, exclude=True)


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


class InstanceRuntimeState(BaseModel):
    """Ephemeral runtime state for an instance"""
    instance_id: str
    busy: bool
    phase: InstancePhase = InstancePhase.IDLE
    prefill_progress: Optional[float] = None
    active_slots: int = 0
    # Optional contextual metrics
    slot_id: Optional[int] = None
    task_id: Optional[int] = None
    prefill_prompt_tokens: Optional[int] = None
    generated_tokens: Optional[int] = None
    decode_tps: Optional[float] = None
    decode_ms_per_token: Optional[float] = None
    checkpoint_index: Optional[int] = None
    checkpoint_total: Optional[int] = None
    timestamp: str


class InstanceStateEvent(BaseModel):
    """State change event used for WebSocket streaming of runtime state"""
    seq: int
    timestamp: str
    type: str = "instance_state"
    data: InstanceRuntimeState


class InstanceResponse(BaseModel):
    """Response model for instance operations"""
    instance: Instance
    message: str


class MemoryInfo(BaseModel):
    """Memory usage information"""
    used_gb: float = Field(..., description="Used memory in GB")
    total_gb: float = Field(..., description="Total memory in GB")
    percent: float = Field(..., description="Usage percentage")
    memory_type: str = Field(..., description="Type of memory (VRAM or RAM)")


class GenerationMetrics(BaseModel):
    """Per-generation token usage and timing metrics parsed from llama-server logs."""
    instance_id: str
    slot_id: Optional[int] = None
    task_id: Optional[int] = None

    # Token usage
    prompt_tokens: Optional[int] = None
    generated_tokens: Optional[int] = None

    # Decode performance
    decode_tps: Optional[float] = None
    decode_ms_per_token: Optional[float] = None

    # Timestamps
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

