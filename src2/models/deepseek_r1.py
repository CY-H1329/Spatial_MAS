"""
DeepSeek-R1 runner (text-only reasoning model) for Final Reasoning Agent.

DeepSeek-R1 (deepseek-ai/DeepSeek-R1) is a 671B MoE text model.
It does NOT support image input — only text (SharedMemory + query).

Recommended deployment:
  - Serve with vLLM or SGLang and call via OpenAI-compatible API.
  - Or use a distilled variant locally:
      deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
      deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
      deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
      deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
"""
from typing import Optional


class DeepSeekR1Runner:
    """Text-only runner for DeepSeek-R1 via OpenAI-compatible API."""

    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model_name: str = "deepseek-r1",
        timeout: int = 120,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("DeepSeekR1Runner requires openai. pip install openai")

        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model_name = model_name
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: float = 0.0,
        **kwargs,
    ) -> str:
        """Generate text from a text-only prompt (no image)."""
        create_kwargs = dict(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.timeout,
        )
        if temperature > 0 and top_p and top_p > 0:
            create_kwargs["top_p"] = top_p
        response = self.client.chat.completions.create(**create_kwargs)
        return (response.choices[0].message.content or "").strip()


class DeepSeekR1LocalRunner:
    """Local runner for DeepSeek-R1 distilled variants."""

    def __init__(
        self,
        model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        device: Optional[str] = None,
        torch_dtype: Optional[str] = "bfloat16",
        **kwargs,
    ):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = getattr(torch, torch_dtype) if torch_dtype else torch.bfloat16

        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        load_kwargs = dict(
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            **{k: v for k, v in kwargs.items() if k not in ("device_map",)},
        )
        # No device_map — run without accelerate
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        self.model = self.model.to(device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: float = 0.0,
        **kwargs,
    ) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
        )
        if temperature > 0 and top_p and top_p > 0:
            gen_kwargs["top_p"] = top_p
        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
