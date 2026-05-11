"""
Qwen2-VL (HF) — distinct from Qwen2.5-VL (`src/models/qwen.py`).
Default: Qwen/Qwen2-VL-7B-Instruct
"""
from __future__ import annotations

from typing import Optional

from PIL import Image
import torch
from transformers import AutoProcessor

try:
    from transformers import Qwen2VLForConditionalGeneration
except ImportError:
    Qwen2VLForConditionalGeneration = None


class Qwen2VLRunner:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: Optional[str] = None,
        **kwargs,
    ):
        if Qwen2VLForConditionalGeneration is None:
            raise ImportError(
                "Qwen2-VL requires a recent transformers with Qwen2VLForConditionalGeneration. "
                "Try: pip install -U transformers>=4.45"
            )
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        device_map = "auto" if device == "cuda" and torch.cuda.is_available() else device
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device_map,
            trust_remote_code=True,
            **kwargs,
        )
        self.model.eval()
        self.device = device

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
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image.convert("RGB")],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        if top_k and top_k > 0:
            gen_kwargs["top_k"] = top_k
        if top_p and top_p > 0:
            gen_kwargs["top_p"] = top_p
        gen_kwargs.update({k: v for k, v in kwargs.items() if v is not None})

        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        in_len = inputs["input_ids"].shape[1]
        answer = self.processor.decode(out[0][in_len:], skip_special_tokens=True)
        return answer.strip()
