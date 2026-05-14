# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
HappyHorse (DashScope) text-to-video service.

Handles async task creation, polling, and result download for
DashScope HappyHorse video generation API.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger

from pixelle_video.config.schema import (
    HappyHorseConfig,
    HAPPYHORSE_SUPPORTED_RATIOS,
)


def _dimensions_to_ratio(width: int, height: int) -> str:
    """Convert pixel dimensions to the closest supported ratio string."""
    if width <= 0 or height <= 0:
        return "16:9"
    from math import gcd
    g = gcd(width, height)
    raw = f"{width // g}:{height // g}"
    if raw in HAPPYHORSE_SUPPORTED_RATIOS:
        return raw
    # Find closest supported ratio by aspect ratio difference
    target = width / height
    best, best_diff = "16:9", float("inf")
    for r in HAPPYHORSE_SUPPORTED_RATIOS:
        rw, rh = r.split(":")
        diff = abs(int(rw) / int(rh) - target)
        if diff < best_diff:
            best, best_diff = r, diff
    return best


def _clamp_duration(duration: Optional[int | float], min_d: int = 3, max_d: int = 15) -> int:
    """Clamp duration to HappyHorse supported range."""
    if duration is None:
        return 5
    return max(min_d, min(max_d, int(round(duration))))


class HappyHorseVideoService:
    """DashScope HappyHorse text-to-video provider."""

    def __init__(self, config: HappyHorseConfig) -> None:
        self._cfg = config

    # -- public interface ---------------------------------------------------

    async def generate(
        self,
        prompt: str,
        *,
        width: int = 1280,
        height: int = 720,
        duration: Optional[int | float] = None,
        resolution: Optional[str] = None,
        watermark: Optional[bool] = None,
        seed: Optional[int] = None,
        model: Optional[str] = None,
    ) -> dict:
        """
        Generate a video from text prompt.

        Returns dict with keys: url, duration, task_id
        """
        api_key = self._cfg.effective_api_key
        if not api_key:
            raise ValueError("HappyHorse API key not configured. Set happyhorse.api_key in config.yaml or DASHSCOPE_API_KEY env var.")

        region = self._cfg.effective_region
        workspace_id = self._cfg.effective_workspace_id
        if region == "eu-central-1" and not workspace_id:
            raise ValueError("HappyHorse eu-central-1 region requires workspace_id. Set happyhorse.workspace_id or DASHSCOPE_WORKSPACE_ID env var.")

        base_url = self._cfg.base_url
        ratio = _dimensions_to_ratio(width, height)
        dur = _clamp_duration(duration, 3, 15)
        res = resolution or self._cfg.default_resolution
        wm = watermark if watermark is not None else self._cfg.watermark
        mdl = model or self._cfg.default_model

        logger.info(
            f"HappyHorse: creating task model={mdl} ratio={ratio} "
            f"duration={dur}s resolution={res} watermark={wm}"
        )

        task_id = await self._create_task(
            base_url=base_url,
            api_key=api_key,
            model=mdl,
            prompt=prompt,
            ratio=ratio,
            duration=dur,
            resolution=res,
            watermark=wm,
            seed=seed,
        )
        logger.info(f"HappyHorse: task created task_id={task_id}")

        result = await self._poll_until_done(
            base_url=base_url,
            api_key=api_key,
            task_id=task_id,
        )

        video_url = result.get("video_url", "")
        if not video_url:
            raise RuntimeError(f"HappyHorse task {task_id} succeeded but no video URL in result: {result}")

        return {
            "url": video_url,
            "duration": dur,
            "task_id": task_id,
        }

    # -- internal -----------------------------------------------------------

    async def _create_task(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        prompt: str,
        ratio: str,
        duration: int,
        resolution: str,
        watermark: bool,
        seed: Optional[int],
    ) -> str:
        """Create a DashScope async video generation task."""
        url = f"{base_url}/api/v1/services/aigc/video-generation/video-synthesis"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }
        input_params: dict = {
            "prompt": prompt,
        }
        parameters: dict = {
            "ratio": ratio,
            "duration": duration,
            "resolution": resolution,
            "watermark": watermark,
        }
        if seed is not None:
            parameters["seed"] = seed

        body = {
            "model": model,
            "input": input_params,
            "parameters": parameters,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        # DashScope async response: {"output": {"task_id": "...", "task_status": "PENDING"}, ...}
        task_id = data.get("output", {}).get("task_id", "")
        if not task_id:
            raise RuntimeError(f"HappyHorse: no task_id in response: {data}")
        return task_id

    async def _poll_until_done(
        self,
        *,
        base_url: str,
        api_key: str,
        task_id: str,
    ) -> dict:
        """Poll task status until SUCCEEDED, FAILED, or timeout."""
        url = f"{base_url}/api/v1/tasks/{task_id}"
        headers = {"Authorization": f"Bearer {api_key}"}
        interval = self._cfg.poll_interval_seconds
        deadline = time.monotonic() + self._cfg.timeout_seconds

        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                status = data.get("output", {}).get("task_status", "UNKNOWN")
                logger.debug(f"HappyHorse poll: task_id={task_id} status={status}")

                if status == "SUCCEEDED":
                    return self._extract_result(data)
                if status in ("FAILED", "CANCELED"):
                    msg = data.get("output", {}).get("message", "unknown error")
                    raise RuntimeError(f"HappyHorse task {task_id} failed: {msg}")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"HappyHorse task {task_id} timed out after {self._cfg.timeout_seconds}s (last status={status})")

                await asyncio.sleep(interval)

    @staticmethod
    def _extract_result(data: dict) -> dict:
        """Extract video URL from successful DashScope response."""
        output = data.get("output", {})
        # Try nested video_url first
        video_url = output.get("video_url", "")
        if not video_url:
            # Some models return results list
            results = output.get("results", [])
            if results and isinstance(results, list):
                video_url = results[0].get("url", "") or results[0].get("video_url", "")
        return {"video_url": video_url, "raw_output": output}
