"""
InternVL2 inference (HuggingFace OpenGVLab/InternVL2-*).
Uses remote code `model.chat(tokenizer, pixel_values, question, generation_config)`.
See: https://huggingface.co/OpenGVLab/InternVL2-8B
"""
from __future__ import annotations

import logging
import warnings
from typing import Optional

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image: Image.Image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _patch_internvl_language_model_for_transformers_4_50(model: torch.nn.Module) -> None:
    """
    InternVL2 remote code calls `language_model.generate(...)`.
    From transformers v4.50, `PreTrainedModel` no longer inherits `GenerationMixin`, so
    `InternLM2ForCausalLM` may lack `.generate` (HF warning + AttributeError at runtime).

    Fix: rebind the LM instance to a tiny subclass that adds `GenerationMixin` (same pattern as
    ms-swift / VLMEvalKit discussions for InternVL2 + recent transformers).
    """
    lm = getattr(model, "language_model", None)
    if lm is None:
        return
    if callable(getattr(lm, "generate", None)):
        return
    try:
        from transformers.generation.utils import GenerationMixin
    except ImportError:
        warnings.warn(
            "InternVL2: cannot import GenerationMixin; upgrade transformers or use transformers<4.50."
        )
        return

    base_cls = lm.__class__
    if base_cls.__name__.endswith("_InternVL2GenPatch"):
        return

    patched_cls = type(
        f"{base_cls.__name__}_InternVL2GenPatch",
        (base_cls, GenerationMixin),
        {},
    )
    try:
        lm.__class__ = patched_cls
    except Exception as e:
        warnings.warn(
            f"InternVL2: failed to patch language_model for GenerationMixin: {e}. "
            "Try: pip install 'transformers>=4.37.2,<4.50' for this checkpoint."
        )
        return
    logger.info(
        "InternVL2: patched language_model %s with GenerationMixin for transformers>=4.50 compatibility.",
        base_cls.__name__,
    )


def _pixel_values_from_pil(image: Image.Image, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    transform = _build_transform(input_size)
    images = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    tensors = [transform(im) for im in images]
    return torch.stack(tensors)


class InternVL2Runner:
    """InternVL2 chat model (single image)."""

    def __init__(
        self,
        model_id: str = "OpenGVLab/InternVL2-8B",
        device: Optional[str] = None,
        input_size: int = 448,
        max_num_tiles: int = 12,
        use_flash_attn: bool = False,
        **kwargs,
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.device = device
        self.input_size = input_size
        self.max_num_tiles = max_num_tiles

        load_kw = dict(
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **kwargs,
        )
        if use_flash_attn and device == "cuda":
            try:
                import flash_attn  # noqa: F401

                load_kw["use_flash_attn"] = True
            except ImportError:
                pass

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(model_id, **load_kw).eval()
        _patch_internvl_language_model_for_transformers_4_50(self.model)
        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
        self._dtype = torch.bfloat16 if device == "cuda" else torch.float32

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        top_k: int = 0,
        top_p: float = 0.0,
        **kwargs,
    ) -> str:
        pixel_values = _pixel_values_from_pil(
            image.convert("RGB"),
            input_size=self.input_size,
            max_num=self.max_num_tiles,
        ).to(dtype=self._dtype)
        if self.device == "cuda" and torch.cuda.is_available():
            pixel_values = pixel_values.cuda()

        question = f"<image>\n{prompt}"
        generation_config = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
        )
        if temperature > 0:
            generation_config["temperature"] = temperature
        if top_k and top_k > 0:
            generation_config["top_k"] = top_k
        if top_p and top_p > 0:
            generation_config["top_p"] = top_p

        with torch.inference_mode():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
                history=None,
                return_history=False,
            )
        if isinstance(response, (list, tuple)):
            response = response[0]
        return (response or "").strip()
