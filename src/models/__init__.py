from .base import BaseVLM

# Optional runners — import may fail if deps missing (e.g. qwen, openai)
QwenRunner = Qwen3Runner = LLaVARunner = Sa2VARunner = None
SpatialReasonerRunner = None
DeepSeekVLRunner = GPTRunner = GeminiRunner = None

try:
    from .qwen import QwenRunner
except Exception:
    pass
try:
    from .qwen3 import Qwen3Runner
except Exception:
    pass
try:
    from .llava import LLaVARunner
except Exception:
    pass
try:
    from .sa2va import Sa2VARunner
except Exception:
    pass
try:
    from .deepseek_vl import DeepSeekVLRunner
except Exception:
    pass
try:
    from .spatial_reasoner import SpatialReasonerRunner
except Exception:
    pass
try:
    from .internvl2 import InternVL2Runner
except Exception:
    InternVL2Runner = None  # type: ignore
try:
    from .qwen2_vl import Qwen2VLRunner
except Exception:
    Qwen2VLRunner = None  # type: ignore
try:
    from .gpt import GPTRunner
except Exception:
    pass
try:
    from .gemini import GeminiRunner
except Exception:
    pass

__all__ = [
    "BaseVLM", "QwenRunner", "Qwen3Runner", "LLaVARunner", "Sa2VARunner",
    "SpatialReasonerRunner", "InternVL2Runner", "Qwen2VLRunner",
    "DeepSeekVLRunner", "GPTRunner", "GeminiRunner",
]
