from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class OutputFormat(str, Enum):
    SRT = "srt"
    VTT = "vtt"
    JSON = "json"


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class CaptionSource(str, Enum):
    AUDIO = "audio"
    VISION = "vision"
    COMBINED = "combined"


class TaskRequest(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    video_url: Optional[str] = Field(
        default=None,
        description="Public URL of the video to caption",
    )
    video_path: Optional[str] = Field(
        default=None,
        description="Local filesystem path to the video file",
    )
    language: str = Field(
        default="en",
        min_length=2,
        max_length=10,
        description="BCP-47 language code for transcription and captioning",
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.SRT,
        description="Caption output format",
    )
    enable_audio: bool = Field(
        default=True,
        description="Transcribe audio track via faster-whisper",
    )
    enable_vision: bool = Field(
        default=True,
        description="Generate visual captions via Claude Vision",
    )
    model: str = Field(
        default="claude-opus-4-7",
        description="Anthropic model ID to use for vision captioning",
    )
    max_frames: int = Field(
        default=10,
        ge=1,
        le=120,
        description="Maximum video frames to sample for vision captioning",
    )

    @model_validator(mode="after")
    def require_video_source(self) -> TaskRequest:
        if not self.video_url and not self.video_path:
            raise ValueError("One of video_url or video_path must be provided")
        if self.video_url and self.video_path:
            raise ValueError("Provide only one of video_url or video_path, not both")
        return self

    @field_validator("video_url")
    @classmethod
    def validate_url(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.startswith(("http://", "https://")):
            raise ValueError("video_url must start with http:// or https://")
        return v


class TaskResult(BaseModel):
    segment_id: int = Field(ge=0, description="Zero-based index of this caption segment")
    start_time: float = Field(ge=0.0, description="Segment start time in seconds")
    end_time: float = Field(description="Segment end time in seconds")
    text: str = Field(min_length=1, description="Caption text for this segment")
    source: CaptionSource = Field(
        default=CaptionSource.COMBINED,
        description="Which pipeline produced this caption",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Model confidence score (0–1) when available",
    )

    @model_validator(mode="after")
    def end_after_start(self) -> TaskResult:
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
        return self

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class CaptionResponse(BaseModel):
    task_id: str
    status: TaskStatus
    video_source: str = Field(description="Resolved URL or path of the processed video")
    captions: list[TaskResult] = Field(default_factory=list)
    duration: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Total video duration in seconds",
    )
    processing_time: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Wall-clock processing time in seconds",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message when status is FAILED",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def error_requires_failed_status(self) -> CaptionResponse:
        if self.error and self.status != TaskStatus.FAILED:
            raise ValueError("error field is only valid when status is FAILED")
        return self


class ResultsOutput(BaseModel):
    task_id: str
    video_source: str
    output_format: OutputFormat
    language: str
    total_segments: int = Field(ge=0)
    duration: float = Field(ge=0.0, description="Total video duration in seconds")
    captions: list[TaskResult]
    formatted_output: str = Field(
        description="Caption content serialised in the requested output_format"
    )
    model_used: str = Field(description="Anthropic model ID used for vision captioning")
    enable_audio: bool
    enable_vision: bool
    processing_time: float = Field(ge=0.0, description="Total processing time in seconds")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def segment_count_matches(self) -> ResultsOutput:
        if self.total_segments != len(self.captions):
            raise ValueError(
                f"total_segments ({self.total_segments}) does not match "
                f"len(captions) ({len(self.captions)})"
            )
        return self
