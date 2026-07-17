#!/usr/bin/env python3
"""Local HTTP service for task-conditioned Grounding DINO detections."""
from __future__ import annotations

import argparse
import asyncio
import base64
from io import BytesIO
import os
import traceback
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import uvicorn
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve local Grounding DINO through /v1/detections.")
    parser.add_argument("--model", default=os.environ.get("GROUNDING_DINO_LOCAL_MODEL", DEFAULT_MODEL))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--max-concurrent", type=int, default=1)
    return parser.parse_args()


class DetectionRequest(BaseModel):
    image_url: str
    labels: list[str]
    box_threshold: float = 0.25
    text_threshold: float = 0.20


class GroundingDinoService:
    def __init__(self, args: argparse.Namespace):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the local Grounding DINO service")
        self.args = args
        self.device = torch.device("cuda:0")
        self.processor = AutoProcessor.from_pretrained(args.model)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(self.device).eval()
        self.lock = asyncio.Semaphore(max(1, int(args.max_concurrent)))

    async def detect(self, request: DetectionRequest) -> list[dict[str, Any]]:
        async with self.lock:
            return await asyncio.to_thread(self._detect_sync, request)

    def _detect_sync(self, request: DetectionRequest) -> list[dict[str, Any]]:
        labels = [str(label).strip() for label in request.labels if str(label).strip()]
        if not labels:
            raise ValueError("labels must not be empty")
        image = _data_url_image(request.image_url)
        text_query = " . ".join(labels) + " ."
        inputs = self.processor(images=image, text=text_query, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=max(0.0, min(1.0, float(request.box_threshold))),
            text_threshold=max(0.0, min(1.0, float(request.text_threshold))),
            target_sizes=[image.size[::-1]],
        )[0]
        boxes = results.get("boxes")
        scores = results.get("scores")
        detected_labels = results.get("labels")
        if boxes is None or scores is None or detected_labels is None:
            return []
        return [
            {
                "label": str(label),
                "bbox": [float(value) for value in box.tolist()],
                "confidence": float(score),
            }
            for box, score, label in zip(boxes, scores, detected_labels)
        ]


def create_app(service: GroundingDinoService) -> FastAPI:
    app = FastAPI(title="Local Grounding DINO", version="v0")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "model": service.args.model, "device": str(service.device)}

    @app.post("/v1/detections")
    async def detections(request: DetectionRequest) -> dict[str, Any]:
        try:
            return {"detections": await service.detect(request)}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            print(traceback.format_exc(), flush=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def _data_url_image(value: str) -> Image.Image:
    if not value.startswith("data:image/") or "," not in value:
        raise ValueError("image_url must be a base64 data:image URL")
    try:
        return Image.open(BytesIO(base64.b64decode(value.split(",", 1)[1]))).convert("RGB")
    except Exception as exc:
        raise ValueError("invalid image_url") from exc


def main() -> None:
    args = parse_args()
    service = GroundingDinoService(args)
    uvicorn.run(create_app(service), host=args.host, port=int(args.port), log_level="info")


if __name__ == "__main__":
    main()
