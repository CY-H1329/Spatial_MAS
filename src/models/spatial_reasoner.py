from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from PIL import Image

from .base import BaseVLM


class SpatialReasonerRunner(BaseVLM):
    """
    GPU runner for SpatialReasoner (Qwen2.5-VL backbone).

    Note: This runner is intended for inference in the MAS pipeline (single image input).
    Multi-view tasks (e.g. MindCube) are tiled into one image upstream.
    """

    def __init__(
        self,
        model_id: str = "ccvl/SpatialReasoner",
        processor_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        bf16: bool = True,
    ):
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except Exception as e:
            raise ImportError(
                "SpatialReasonerRunner requires transformers with Qwen2.5-VL support. "
                "Install/upgrade transformers, then retry."
            ) from e

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if bf16 else torch.float32

        self.processor = AutoProcessor.from_pretrained(processor_id, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def generate(self, image: Image.Image, prompt: str, **kwargs) -> str:
        max_new_tokens = int(kwargs.get("max_new_tokens", kwargs.get("max_tokens", 256)))
        do_sample = bool(kwargs.get("do_sample", False))

        im = image.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": im},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs: Dict[str, Any] = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        inputs.pop("token_type_ids", None)

        with torch.inference_mode():
            gen_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)

        trimmed = gen_ids[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

