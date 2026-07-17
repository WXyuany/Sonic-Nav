#!/usr/bin/env python3
"""Minimal local OpenAI-compatible server for Qwen2.5-VL shadow inference."""
from __future__ import annotations

import argparse
import asyncio
import base64
from io import BytesIO
import os
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import uvicorn

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local Qwen2.5-VL model through /v1/chat/completions.")
    parser.add_argument("--model", default=os.environ.get("QWEN_VL_LOCAL_MODEL", DEFAULT_MODEL))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--device", choices=["cuda", "auto", "cpu"], default=os.environ.get("QWEN_VL_DEVICE", "cuda"))
    parser.add_argument("--gpu-memory-gib", type=float, default=float(os.environ.get("QWEN_VL_GPU_MEMORY_GIB", "0")), help="GPU memory budget for --device auto; 0 derives a conservative budget from free memory.")
    return parser.parse_args()


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    max_tokens: int | None = None
    temperature: float | None = None


class LocalQwenService:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.processor = AutoProcessor.from_pretrained(args.model)
        load_args: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
            "low_cpu_mem_usage": True,
        }
        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("--device cuda requires an available CUDA GPU")
            self.device = torch.device("cuda:0")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, **load_args).to(self.device).eval()
        elif args.device == "auto":
            if not torch.cuda.is_available():
                raise RuntimeError("--device auto requires CUDA; use --device cpu when no GPU is available")
            budget = float(args.gpu_memory_gib) or max(1.0, torch.cuda.mem_get_info(0)[0] / (1024**3) * 0.80)
            load_args.update({"device_map": "auto", "max_memory": {0: f"{budget:.1f}GiB", "cpu": "48GiB"}})
            self.device = torch.device("cuda:0")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, **load_args).eval()
        else:
            self.device = torch.device("cpu")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, **load_args).to(self.device).eval()
        self.lock = asyncio.Semaphore(max(1, int(args.max_concurrent)))

    async def generate(self, request: ChatRequest) -> str:
        async with self.lock:
            return await asyncio.to_thread(self._generate_sync, request)

    def _generate_sync(self, request: ChatRequest) -> str:
        messages = _normalize_messages(request.messages)
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = [item["image"] for message in messages for item in message["content"] if item.get("type") == "image"]
        inputs = self.processor(text=[prompt], images=images or None, padding=True, return_tensors="pt").to(self.device)
        token_limit = min(max(1, int(request.max_tokens or self.args.max_new_tokens)), int(self.args.max_new_tokens))
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=token_limit, do_sample=False)
        completion = generated[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(completion, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def create_app(service: LocalQwenService) -> FastAPI:
    app = FastAPI(title="Local Qwen2.5-VL", version="v0")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "model": service.args.model, "device": str(service.device), "device_policy": service.args.device}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": service.args.model, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest) -> dict[str, Any]:
        try:
            content = await service.generate(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "id": "local-qwen-vl",
            "object": "chat.completion",
            "model": service.args.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        }

    return app


def _normalize_messages(raw_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for message in raw_messages:
        role = str(message.get("role") or "user")
        raw_content = message.get("content")
        if isinstance(raw_content, str):
            content = [{"type": "text", "text": raw_content}]
        elif isinstance(raw_content, list):
            content = []
            for item in raw_content:
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type") or "")
                if item_type == "text":
                    content.append({"type": "text", "text": str(item.get("text") or "")})
                elif item_type == "image_url":
                    image_url = item.get("image_url")
                    url = image_url.get("url") if isinstance(image_url, dict) else image_url
                    content.append({"type": "image", "image": _data_url_image(str(url or ""))})
        else:
            content = []
        if content:
            messages.append({"role": role, "content": content})
    if not messages:
        raise ValueError("messages must contain text or image_url content")
    return messages


def _data_url_image(value: str) -> Image.Image:
    prefix = "data:image/"
    if not value.startswith(prefix) or "," not in value:
        raise ValueError("local Qwen-VL server currently requires data:image URLs")
    try:
        encoded = value.split(",", 1)[1]
        return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
    except Exception as exc:
        raise ValueError("invalid base64 image_url") from exc


def main() -> None:
    args = parse_args()
    service = LocalQwenService(args)
    uvicorn.run(create_app(service), host=args.host, port=int(args.port), log_level="info")


if __name__ == "__main__":
    main()
