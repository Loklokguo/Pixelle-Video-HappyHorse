# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for HappyHorse (DashScope) video generation service."""

from __future__ import annotations

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from pixelle_video.config.schema import HappyHorseConfig, HAPPYHORSE_REGION_ENDPOINTS
from pixelle_video.services.happyhorse_service import (
    HappyHorseVideoService,
    _dimensions_to_ratio,
    _clamp_duration,
)


# -- Ratio conversion ---------------------------------------------------------

class TestDimensionsToRatio:
    def test_exact_16_9(self):
        assert _dimensions_to_ratio(1920, 1080) == "16:9"

    def test_exact_9_16(self):
        assert _dimensions_to_ratio(1080, 1920) == "9:16"

    def test_exact_1_1(self):
        assert _dimensions_to_ratio(1080, 1080) == "1:1"

    def test_exact_4_3(self):
        assert _dimensions_to_ratio(1440, 1080) == "4:3"

    def test_close_to_16_9(self):
        assert _dimensions_to_ratio(1280, 720) == "16:9"

    def test_close_to_9_16(self):
        assert _dimensions_to_ratio(720, 1280) == "9:16"

    def test_zero_width_returns_default(self):
        assert _dimensions_to_ratio(0, 1080) == "16:9"

    def test_closest_match_4_5(self):
        assert _dimensions_to_ratio(864, 1080) == "4:5"


# -- Duration clamping ---------------------------------------------------------

class TestClampDuration:
    def test_normal_value(self):
        assert _clamp_duration(7) == 7

    def test_below_min(self):
        assert _clamp_duration(1) == 3

    def test_above_max(self):
        assert _clamp_duration(20) == 15

    def test_none_returns_default(self):
        assert _clamp_duration(None) == 5

    def test_float_rounds(self):
        assert _clamp_duration(7.6) == 8

    def test_boundary_3(self):
        assert _clamp_duration(3) == 3

    def test_boundary_15(self):
        assert _clamp_duration(15) == 15


# -- Config --------------------------------------------------------------------

class TestHappyHorseConfig:
    def test_default_values(self):
        cfg = HappyHorseConfig()
        assert cfg.region == "cn-beijing"
        assert cfg.default_model == "happyhorse-1.0-t2v"
        assert cfg.default_resolution == "720P"
        assert cfg.watermark is True
        assert cfg.poll_interval_seconds == 15
        assert cfg.timeout_seconds == 600

    def test_is_configured_false_when_empty(self):
        cfg = HappyHorseConfig(api_key="")
        assert cfg.is_configured is False

    def test_is_configured_true_when_set(self):
        cfg = HappyHorseConfig(api_key="sk-test")
        assert cfg.is_configured is True

    def test_is_configured_uses_env_var(self):
        """Fix 5: is_configured must check DASHSCOPE_API_KEY env var."""
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "env-key"}):
            cfg = HappyHorseConfig(api_key="")
            assert cfg.is_configured is True

    def test_base_url_cn_beijing(self):
        cfg = HappyHorseConfig(region="cn-beijing")
        assert cfg.base_url == "https://dashscope.aliyuncs.com"

    def test_base_url_ap_southeast(self):
        cfg = HappyHorseConfig(region="ap-southeast-1")
        assert cfg.base_url == "https://dashscope-intl.aliyuncs.com"

    def test_base_url_us_east(self):
        cfg = HappyHorseConfig(region="us-east-1")
        assert cfg.base_url == "https://dashscope-us.aliyuncs.com"

    def test_base_url_eu_central_with_workspace(self):
        cfg = HappyHorseConfig(region="eu-central-1", workspace_id="ws-123")
        assert cfg.base_url == "https://ws-123.eu-central-1.maas.aliyuncs.com"

    def test_base_url_eu_central_without_workspace_falls_back(self):
        cfg = HappyHorseConfig(region="eu-central-1", workspace_id="")
        assert cfg.base_url == "https://dashscope.aliyuncs.com"

    def test_base_url_uses_effective_region_from_env(self):
        """Fix 5: base_url must use effective_region (env var aware)."""
        with patch.dict("os.environ", {"DASHSCOPE_REGION": "us-east-1"}):
            cfg = HappyHorseConfig(region="cn-beijing")
            assert cfg.base_url == "https://dashscope-us.aliyuncs.com"

    def test_base_url_eu_central_uses_effective_workspace_from_env(self):
        """Fix 5: base_url eu-central-1 uses effective_workspace_id."""
        with patch.dict("os.environ", {"DASHSCOPE_WORKSPACE_ID": "env-ws"}):
            cfg = HappyHorseConfig(region="eu-central-1", workspace_id="")
            assert cfg.base_url == "https://env-ws.eu-central-1.maas.aliyuncs.com"

    def test_env_var_override_api_key(self):
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "env-key"}):
            cfg = HappyHorseConfig(api_key="config-key")
            assert cfg.effective_api_key == "env-key"

    def test_env_var_override_region(self):
        with patch.dict("os.environ", {"DASHSCOPE_REGION": "us-east-1"}):
            cfg = HappyHorseConfig(region="cn-beijing")
            assert cfg.effective_region == "us-east-1"


# -- Service -------------------------------------------------------------------

class TestHappyHorseVideoService:
    def _make_service(self, **overrides) -> HappyHorseVideoService:
        defaults = {"api_key": "sk-test", "region": "cn-beijing"}
        defaults.update(overrides)
        return HappyHorseVideoService(HappyHorseConfig(**defaults))

    def test_raises_when_no_api_key(self):
        svc = self._make_service(api_key="")
        with pytest.raises(ValueError, match="API key not configured"):
            asyncio.run(svc.generate(prompt="test"))

    def test_raises_when_eu_central_no_workspace(self):
        svc = self._make_service(region="eu-central-1", workspace_id="")
        with pytest.raises(ValueError, match="workspace_id"):
            asyncio.run(svc.generate(prompt="test"))

    @pytest.mark.asyncio
    async def test_request_body_format(self):
        """Fix 4: Verify DashScope body uses input/parameters separation."""
        svc = self._make_service()

        captured_body = {}

        create_response = MagicMock()
        create_response.status_code = 200
        create_response.json.return_value = {
            "output": {"task_id": "task-body", "task_status": "PENDING"}
        }
        create_response.raise_for_status = MagicMock()

        poll_succeeded = MagicMock()
        poll_succeeded.status_code = 200
        poll_succeeded.json.return_value = {
            "output": {
                "task_id": "task-body",
                "task_status": "SUCCEEDED",
                "video_url": "https://example.com/v.mp4",
            }
        }
        poll_succeeded.raise_for_status = MagicMock()

        mock_client = AsyncMock()

        async def capture_post(url, json=None, headers=None):
            captured_body.update(json)
            return create_response

        mock_client.post = AsyncMock(side_effect=capture_post)
        mock_client.get = AsyncMock(return_value=poll_succeeded)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("pixelle_video.services.happyhorse_service.httpx.AsyncClient", return_value=mock_client):
            await svc.generate(
                prompt="a cat", width=1920, height=1080,
                duration=7, resolution="1080P", watermark=False, seed=42,
            )

        # Verify input only has prompt
        assert captured_body["input"] == {"prompt": "a cat"}
        # Verify parameters has the rest
        params = captured_body["parameters"]
        assert params["ratio"] == "16:9"
        assert params["duration"] == 7
        assert params["resolution"] == "1080P"
        assert params["watermark"] is False
        assert params["seed"] == 42
        assert "prompt" not in params

    @pytest.mark.asyncio
    async def test_task_flow_pending_to_succeeded(self):
        """Test full task lifecycle: create -> poll PENDING -> poll RUNNING -> poll SUCCEEDED."""
        svc = self._make_service()

        create_response = MagicMock()
        create_response.status_code = 200
        create_response.json.return_value = {
            "output": {"task_id": "task-123", "task_status": "PENDING"}
        }
        create_response.raise_for_status = MagicMock()

        poll_pending = MagicMock()
        poll_pending.status_code = 200
        poll_pending.json.return_value = {
            "output": {"task_id": "task-123", "task_status": "PENDING"}
        }
        poll_pending.raise_for_status = MagicMock()

        poll_running = MagicMock()
        poll_running.status_code = 200
        poll_running.json.return_value = {
            "output": {"task_id": "task-123", "task_status": "RUNNING"}
        }
        poll_running.raise_for_status = MagicMock()

        poll_succeeded = MagicMock()
        poll_succeeded.status_code = 200
        poll_succeeded.json.return_value = {
            "output": {
                "task_id": "task-123",
                "task_status": "SUCCEEDED",
                "video_url": "https://example.com/video.mp4",
            }
        }
        poll_succeeded.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=create_response)
        mock_client.get = AsyncMock(
            side_effect=[poll_pending, poll_running, poll_succeeded]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("pixelle_video.services.happyhorse_service.httpx.AsyncClient", return_value=mock_client):
            result = await svc.generate(prompt="a cat playing", width=1280, height=720)

        assert result["url"] == "https://example.com/video.mp4"
        assert result["task_id"] == "task-123"

    @pytest.mark.asyncio
    async def test_task_flow_failed(self):
        svc = self._make_service()

        create_response = MagicMock()
        create_response.status_code = 200
        create_response.json.return_value = {
            "output": {"task_id": "task-456", "task_status": "PENDING"}
        }
        create_response.raise_for_status = MagicMock()

        poll_failed = MagicMock()
        poll_failed.status_code = 200
        poll_failed.json.return_value = {
            "output": {
                "task_id": "task-456",
                "task_status": "FAILED",
                "message": "Invalid prompt",
            }
        }
        poll_failed.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=create_response)
        mock_client.get = AsyncMock(return_value=poll_failed)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("pixelle_video.services.happyhorse_service.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(RuntimeError, match="failed"):
                await svc.generate(prompt="bad prompt")

    @pytest.mark.asyncio
    async def test_task_flow_unknown_status(self):
        """UNKNOWN status should be treated as pending and keep polling (then timeout)."""
        svc = self._make_service(timeout_seconds=60, poll_interval_seconds=5)

        create_response = MagicMock()
        create_response.status_code = 200
        create_response.json.return_value = {
            "output": {"task_id": "task-unk", "task_status": "PENDING"}
        }
        create_response.raise_for_status = MagicMock()

        poll_unknown = MagicMock()
        poll_unknown.status_code = 200
        poll_unknown.json.return_value = {
            "output": {"task_id": "task-unk", "task_status": "UNKNOWN"}
        }
        poll_unknown.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=create_response)
        mock_client.get = AsyncMock(return_value=poll_unknown)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        call_count = 0
        original_monotonic = time.monotonic

        def fake_monotonic():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return original_monotonic()
            return original_monotonic() + 999

        with patch("pixelle_video.services.happyhorse_service.httpx.AsyncClient", return_value=mock_client):
            with patch("pixelle_video.services.happyhorse_service.time.monotonic", side_effect=fake_monotonic):
                with pytest.raises(TimeoutError, match="timed out"):
                    await svc.generate(prompt="unknown status")

    @pytest.mark.asyncio
    async def test_task_flow_timeout(self):
        svc = self._make_service(timeout_seconds=60, poll_interval_seconds=5)

        create_response = MagicMock()
        create_response.status_code = 200
        create_response.json.return_value = {
            "output": {"task_id": "task-789", "task_status": "PENDING"}
        }
        create_response.raise_for_status = MagicMock()

        poll_pending = MagicMock()
        poll_pending.status_code = 200
        poll_pending.json.return_value = {
            "output": {"task_id": "task-789", "task_status": "PENDING"}
        }
        poll_pending.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=create_response)
        mock_client.get = AsyncMock(return_value=poll_pending)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        call_count = 0
        original_monotonic = time.monotonic

        def fake_monotonic():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return original_monotonic()
            return original_monotonic() + 999

        with patch("pixelle_video.services.happyhorse_service.httpx.AsyncClient", return_value=mock_client):
            with patch("pixelle_video.services.happyhorse_service.time.monotonic", side_effect=fake_monotonic):
                with pytest.raises(TimeoutError, match="timed out"):
                    await svc.generate(prompt="slow prompt")

    def test_extract_result_nested_video_url(self):
        data = {
            "output": {
                "task_id": "t1",
                "task_status": "SUCCEEDED",
                "video_url": "https://example.com/v1.mp4",
            }
        }
        result = HappyHorseVideoService._extract_result(data)
        assert result["video_url"] == "https://example.com/v1.mp4"

    def test_extract_result_from_results_list(self):
        data = {
            "output": {
                "task_id": "t2",
                "task_status": "SUCCEEDED",
                "results": [{"url": "https://example.com/v2.mp4"}],
            }
        }
        result = HappyHorseVideoService._extract_result(data)
        assert result["video_url"] == "https://example.com/v2.mp4"


# -- Provider routing (media.py) -----------------------------------------------

class TestProviderRouting:
    """Fix 3: Verify media_provider=happyhorse short-circuits before ComfyUI workflow resolution."""

    @pytest.mark.asyncio
    async def test_happyhorse_skips_workflow_resolve(self):
        """When media_provider=happyhorse, _resolve_workflow should NOT be called."""
        from pixelle_video.services.media import MediaService

        mock_core = MagicMock()
        svc = MediaService.__new__(MediaService)
        svc._resolve_workflow = MagicMock(side_effect=AssertionError("Should not be called"))
        svc.core = mock_core

        mock_hh_result = {"url": "https://example.com/hh.mp4", "duration": 5, "task_id": "t1"}

        with patch.object(svc, "_call_happyhorse", return_value=MagicMock(
            media_type="video", url="https://example.com/hh.mp4", duration=5
        )) as mock_hh:
            result = await svc(
                prompt="test",
                media_type="video",
                media_provider="happyhorse",
                width=1280,
                height=720,
                duration=7.0,
            )
            mock_hh.assert_called_once()
            call_kwargs = mock_hh.call_args
            assert call_kwargs.kwargs["prompt"] == "test"
            assert call_kwargs.kwargs["duration"] == 7.0

        # _resolve_workflow must not have been called
        svc._resolve_workflow.assert_not_called()

    @pytest.mark.asyncio
    async def test_comfyui_still_calls_resolve_workflow(self):
        """Default media_provider=comfyui should still resolve workflow."""
        from pixelle_video.services.media import MediaService

        mock_core = MagicMock()
        mock_kit = AsyncMock()
        mock_kit.execute = AsyncMock(return_value=MagicMock(
            status="completed", images=["img.png"], videos=[]
        ))
        mock_core._get_or_create_comfykit = AsyncMock(return_value=mock_kit)

        svc = MediaService.__new__(MediaService)
        svc._resolve_workflow = MagicMock(return_value={
            "key": "selfhost/image_flux.json",
            "source": "selfhost",
            "path": "/path/to/workflow.json",
        })
        svc.core = mock_core

        with patch("pixelle_video.services.media.MediaResult") as MockResult:
            MockResult.return_value = MagicMock(media_type="image", url="img.png")
            await svc(
                prompt="test",
                media_type="image",
                media_provider="comfyui",
                width=512,
                height=512,
            )
            svc._resolve_workflow.assert_called_once()
