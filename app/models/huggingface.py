"""HuggingFace backend configuration models."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Literal


class HuggingFaceCausalConfig(BaseModel):
    """Configuration for a HuggingFace AutoModelForCausalLM instance."""

    model_config = ConfigDict(protected_namespaces=())

    backend_type: Literal["huggingface_causal"] = Field(
        default="huggingface_causal", description="Backend type identifier"
    )
    model_id: str = Field(
        ...,
        description="HuggingFace model ID or local path (e.g., 'meta-llama/Llama-2-7b-hf')",
    )
    alias: str = Field(..., description="Model alias (e.g., llama2:7b)")
    device: str = Field(
        default="auto", description="Device to run on: auto, cuda, mps (Mac), cpu"
    )
    dtype: str = Field(
        default="auto", description="Data type: auto, float16, bfloat16, float32"
    )
    max_length: int = Field(default=4096, description="Maximum sequence length")
    trust_remote_code: bool = Field(
        default=False, description="Whether to trust remote code from HuggingFace"
    )
    use_flash_attention: bool = Field(
        default=True, description="Use Flash Attention 2 if available"
    )
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: Optional[int] = Field(
        default=None, description="Port (auto-assigned if not specified)"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for this instance (defaults to host API key if not set)",
    )


class HuggingFaceClassificationConfig(BaseModel):
    """Configuration for a HuggingFace AutoModelForSequenceClassification instance."""

    model_config = ConfigDict(protected_namespaces=())

    backend_type: Literal["huggingface_classification"] = Field(
        default="huggingface_classification", description="Backend type identifier"
    )
    model_id: str = Field(..., description="HuggingFace model ID or local path")
    alias: str = Field(..., description="Model alias (e.g., classifier:deberta)")
    device: str = Field(
        default="auto", description="Device to run on: auto, cuda, mps (Mac), cpu"
    )
    dtype: str = Field(
        default="auto", description="Data type: auto, float16, bfloat16, float32"
    )
    max_length: int = Field(
        default=512, description="Maximum sequence length for classification"
    )
    labels: Optional[List[str]] = Field(
        default=None, description="Optional label names mapping (index -> label name)"
    )
    trust_remote_code: bool = Field(
        default=False, description="Whether to trust remote code from HuggingFace"
    )
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: Optional[int] = Field(
        default=None, description="Port (auto-assigned if not specified)"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for this instance (defaults to host API key if not set)",
    )


class HuggingFaceEmbeddingConfig(BaseModel):
    """Configuration for a HuggingFace embedding model instance using AutoModel."""

    model_config = ConfigDict(protected_namespaces=())

    backend_type: Literal["huggingface_embedding"] = Field(
        default="huggingface_embedding", description="Backend type identifier"
    )
    model_id: str = Field(
        ...,
        description="HuggingFace model ID or local path (e.g., 'sentence-transformers/all-MiniLM-L6-v2')",
    )
    alias: str = Field(..., description="Model alias (e.g., embed:minilm)")
    device: str = Field(
        default="auto", description="Device to run on: auto, cuda, mps (Mac), cpu"
    )
    dtype: str = Field(
        default="auto", description="Data type: auto, float16, bfloat16, float32"
    )
    max_length: int = Field(
        default=512, description="Maximum sequence length for embeddings"
    )
    normalize_embeddings: bool = Field(
        default=True, description="L2 normalize output embedding vectors"
    )
    trust_remote_code: bool = Field(
        default=False, description="Whether to trust remote code from HuggingFace"
    )
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: Optional[int] = Field(
        default=None, description="Port (auto-assigned if not specified)"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for this instance (defaults to host API key if not set)",
    )
