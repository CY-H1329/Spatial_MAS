"""
Qwen2.5-VL inference for STVQA-7K.
"""
from typing import Optional
from PIL import Image
import torch
from transformers import AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None  # fallback below


class QwenRunner:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: Optional[str] = None,
        **kwargs,
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        if Qwen2_5_VLForConditionalGeneration is not None:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                trust_remote_code=True,
                **kwargs,
            )
        else:
            from transformers import AutoModelForVision2Seq
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                trust_remote_code=True,
                **kwargs,
            )
        # Avoid requiring `accelerate` (device_map="auto") in constrained environments.
        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
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
        # Qwen2.5-VL chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            **kwargs,
        )
        if temperature > 0:
            if top_k and top_k > 0:
                gen_kwargs["top_k"] = top_k
            if top_p and top_p > 0:
                gen_kwargs["top_p"] = top_p
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        answer = self.processor.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return answer.strip()
