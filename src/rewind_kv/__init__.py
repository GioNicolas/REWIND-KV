"""RewindKV: rollback de KV-Cache com Âncoras de Falha para modelos de raciocínio."""

from .controller import AnchorStyle, GenerationResult, Mode, RewindEvent, RewindKVGenerator, RewindScope, rewind_and_inject
from .detector import FailureSignal, detect_failure

__all__ = [
    "AnchorStyle",
    "FailureSignal",
    "GenerationResult",
    "Mode",
    "RewindEvent",
    "RewindKVGenerator",
    "RewindScope",
    "detect_failure",
    "rewind_and_inject",
]
