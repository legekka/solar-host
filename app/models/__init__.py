"""Models package for solar-host with multi-backend support."""

from typing import Union, Annotated
from pydantic import Field

# Import base models first (no dependencies on config types)
from app.models.base import (
    BackendType,
    InstanceStatus,
    InstancePhase,
    Instance,
    InstanceCreate,
    InstanceUpdate,
    LogMessage,
    InstanceRuntimeState,
    InstanceStateEvent,
    InstanceResponse,
    MemoryInfo,
    GenerationMetrics,
)

# Import config models
from app.models.llamacpp import LlamaCppConfig
from app.models.huggingface import (
    HuggingFaceCausalConfig,
    HuggingFaceClassificationConfig,
)

# Create the discriminated union type for InstanceConfig
InstanceConfig = Annotated[
    Union[LlamaCppConfig, HuggingFaceCausalConfig, HuggingFaceClassificationConfig],
    Field(discriminator="backend_type"),
]

__all__ = [
    # Enums
    "BackendType",
    "InstanceStatus",
    "InstancePhase",
    # Config types
    "InstanceConfig",
    "LlamaCppConfig",
    "HuggingFaceCausalConfig",
    "HuggingFaceClassificationConfig",
    # Instance models
    "Instance",
    "InstanceCreate",
    "InstanceUpdate",
    "InstanceResponse",
    # Runtime models
    "LogMessage",
    "InstanceRuntimeState",
    "InstanceStateEvent",
    "GenerationMetrics",
    # Other
    "MemoryInfo",
]
