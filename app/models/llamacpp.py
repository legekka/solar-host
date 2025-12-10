"""LlamaCpp backend configuration models."""

from pydantic import BaseModel, Field
from typing import Optional, Literal


class LlamaCppConfig(BaseModel):
    """Configuration for a llama.cpp server instance."""

    backend_type: Literal["llamacpp"] = Field(
        default="llamacpp", description="Backend type identifier"
    )
    model: str = Field(..., description="Path to the GGUF model file")
    alias: str = Field(..., description="Model alias (e.g., gpt-oss:120b)")
    threads: int = Field(default=1, description="Number of threads")
    n_gpu_layers: int = Field(default=999, description="Number of GPU layers")
    temp: float = Field(default=1.0, description="Temperature")
    top_p: float = Field(default=1.0, description="Top-p sampling")
    top_k: int = Field(default=0, description="Top-k sampling")
    min_p: float = Field(default=0.0, description="Min-p sampling")
    ctx_size: int = Field(default=131072, description="Context size")
    chat_template_file: Optional[str] = Field(
        default=None, description="Path to Jinja chat template"
    )
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: Optional[int] = Field(
        default=None, description="Port (auto-assigned if not specified)"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for this instance (defaults to host API key if not set)",
    )
    special: bool = Field(
        default=False, description="Enable llama-server --special flag"
    )
