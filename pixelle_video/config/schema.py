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
Configuration schema with Pydantic models

Single source of truth for all configuration defaults and validation.
"""
from typing import Optional
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM configuration"""
    api_key: str = Field(default="", description="LLM API Key")
    base_url: str = Field(default="", description="LLM API Base URL")
    model: str = Field(default="", description="LLM Model Name")


class TTSLocalConfig(BaseModel):
    """Local TTS configuration (Edge TTS)"""
    voice: str = Field(default="zh-CN-YunjianNeural", description="Edge TTS voice ID")
    speed: float = Field(default=1.2, ge=0.5, le=2.0, description="Speech speed multiplier (0.5-2.0)")


class TTSComfyUIConfig(BaseModel):
    """ComfyUI TTS configuration"""
    default_workflow: Optional[str] = Field(default=None, description="Default TTS workflow (optional)")


class TTSSubConfig(BaseModel):
    """TTS-specific configuration (under comfyui.tts)"""
    inference_mode: str = Field(default="local", description="TTS inference mode: 'local' or 'comfyui'")
    local: TTSLocalConfig = Field(default_factory=TTSLocalConfig, description="Local TTS (Edge TTS) configuration")
    comfyui: TTSComfyUIConfig = Field(default_factory=TTSComfyUIConfig, description="ComfyUI TTS configuration")
    
    # Backward compatibility: keep default_workflow at top level
    @property
    def default_workflow(self) -> Optional[str]:
        """Get default workflow (for backward compatibility)"""
        return self.comfyui.default_workflow


class ImageSubConfig(BaseModel):
    """Image-specific configuration (under comfyui.image)"""
    default_workflow: Optional[str] = Field(default=None, description="Default image workflow (optional)")
    prompt_prefix: str = Field(
        default="Minimalist black-and-white matchstick figure style illustration, clean lines, simple sketch style",
        description="Prompt prefix for all image generation"
    )


class VideoSubConfig(BaseModel):
    """Video-specific configuration (under comfyui.video)"""
    default_workflow: Optional[str] = Field(default=None, description="Default video workflow (optional)")
    prompt_prefix: str = Field(
        default="Minimalist black-and-white matchstick figure style illustration, clean lines, simple sketch style",
        description="Prompt prefix for all video generation"
    )


class ComfyUIConfig(BaseModel):
    """ComfyUI configuration (includes global settings and service-specific configs)"""
    comfyui_url: str = Field(default="http://127.0.0.1:8188", description="ComfyUI Server URL")
    comfyui_api_key: Optional[str] = Field(default=None, description="ComfyUI API Key (optional)")
    runninghub_api_key: Optional[str] = Field(default=None, description="RunningHub API Key (optional)")
    runninghub_concurrent_limit: int = Field(default=1, ge=1, le=10, description="RunningHub concurrent execution limit (1-10)")
    runninghub_instance_type: Optional[str] = Field(default=None, description="RunningHub instance type (optional, set to 'plus' for 48GB VRAM)")
    tts: TTSSubConfig = Field(default_factory=TTSSubConfig, description="TTS-specific configuration")
    image: ImageSubConfig = Field(default_factory=ImageSubConfig, description="Image-specific configuration")
    video: VideoSubConfig = Field(default_factory=VideoSubConfig, description="Video-specific configuration")


class TemplateConfig(BaseModel):
    """Template configuration"""
    default_template: str = Field(
        default="1080x1920/default.html",
        description="Default frame template path"
    )


# HappyHorse / DashScope text-to-video configuration
HAPPYHORSE_REGION_ENDPOINTS: dict[str, str] = {
    "cn-beijing": "https://dashscope.aliyuncs.com",
    "ap-southeast-1": "https://dashscope-intl.aliyuncs.com",
    "us-east-1": "https://dashscope-us.aliyuncs.com",
}

HAPPYHORSE_SUPPORTED_RATIOS: list[str] = [
    "16:9", "9:16", "1:1", "4:3", "3:4", "4:5", "5:4",
]


class HappyHorseConfig(BaseModel):
    """HappyHorse (DashScope) text-to-video configuration"""
    api_key: str = Field(default="", description="DashScope API Key (or set DASHSCOPE_API_KEY env var)")
    region: str = Field(default="cn-beijing", description="DashScope region: cn-beijing, ap-southeast-1, us-east-1, eu-central-1")
    workspace_id: str = Field(default="", description="Workspace ID (required for eu-central-1)")
    default_model: str = Field(default="happyhorse-1.0-t2v", description="Default HappyHorse model")
    default_resolution: str = Field(default="720P", description="Default resolution: 720P or 1080P")
    watermark: bool = Field(default=True, description="Add watermark to generated videos")
    poll_interval_seconds: int = Field(default=15, ge=5, le=60, description="Polling interval in seconds")
    timeout_seconds: int = Field(default=600, ge=60, le=3600, description="Maximum wait time in seconds")
    concurrent_limit: int = Field(default=1, ge=1, le=10, description="Concurrent task limit")

    @property
    def base_url(self) -> str:
        """Resolve DashScope endpoint from effective region (env var aware)."""
        region = self.effective_region
        ws = self.effective_workspace_id
        if region == "eu-central-1" and ws:
            return f"https://{ws}.eu-central-1.maas.aliyuncs.com"
        return HAPPYHORSE_REGION_ENDPOINTS.get(region, HAPPYHORSE_REGION_ENDPOINTS["cn-beijing"])

    @property
    def is_configured(self) -> bool:
        """Check if HappyHorse has minimum required config (env var aware)."""
        return bool(self.effective_api_key and self.effective_api_key.strip())

    @property
    def effective_api_key(self) -> str:
        """Get API key, allowing env var override."""
        import os
        return os.environ.get("DASHSCOPE_API_KEY", self.api_key)

    @property
    def effective_region(self) -> str:
        """Get region, allowing env var override."""
        import os
        return os.environ.get("DASHSCOPE_REGION", self.region)

    @property
    def effective_workspace_id(self) -> str:
        """Get workspace ID, allowing env var override."""
        import os
        return os.environ.get("DASHSCOPE_WORKSPACE_ID", self.workspace_id)


class PixelleVideoConfig(BaseModel):
    """Pixelle-Video main configuration"""
    project_name: str = Field(default="Pixelle-Video", description="Project name")
    llm: LLMConfig = Field(default_factory=LLMConfig)
    comfyui: ComfyUIConfig = Field(default_factory=ComfyUIConfig)
    happyhorse: HappyHorseConfig = Field(default_factory=HappyHorseConfig)
    template: TemplateConfig = Field(default_factory=TemplateConfig)
    
    def is_llm_configured(self) -> bool:
        """Check if LLM is properly configured"""
        return bool(
            self.llm.api_key and self.llm.api_key.strip() and
            self.llm.base_url and self.llm.base_url.strip() and
            self.llm.model and self.llm.model.strip()
        )
    
    def validate_required(self) -> bool:
        """Validate required configuration"""
        return self.is_llm_configured()
    
    def to_dict(self) -> dict:
        """Convert to dictionary (for backward compatibility)"""
        return self.model_dump()

