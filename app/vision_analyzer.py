import logging
import os
import threading
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/model_cache")
MAX_VIDEO_FRAMES = 10

_CAPTION_PROMPTS = {
    "formal": (
        "Write a single formal, professional caption for a video described as: \"{summary}\". "
        "Use objective language suitable for a corporate or academic context. "
        "Output only the caption text, no explanation."
    ),
    "sarcastic": (
        "Write a single sarcastic, ironic caption for a video described as: \"{summary}\". "
        "Be dry and deadpan as if you've seen it all before. "
        "Output only the caption text, no explanation."
    ),
    "humorous_tech": (
        "Write a single funny caption for a video described as: \"{summary}\". "
        "Use software engineering or tech culture humor — jokes about bugs, deployments, "
        "stack traces, or developer life. "
        "Output only the caption text, no explanation."
    ),
    "humorous_non_tech": (
        "Write a single funny, lighthearted caption for a video described as: \"{summary}\". "
        "Keep it accessible and witty for a general audience with no technical background. "
        "Output only the caption text, no explanation."
    ),
}

_FALLBACK_CAPTIONS = {
    "formal": "Video content unavailable due to model initialization failure.",
    "sarcastic": "Oh great, the model broke. Truly shocking.",
    "humorous_tech": "Model crashed. Have you tried turning it off and on again?",
    "humorous_non_tech": "Oops! Something went sideways — but hey, at least the video tried.",
}


class VisionAnalyzer:
    _instance: "VisionAnalyzer | None" = None
    _instance_lock: threading.Lock = threading.Lock()
    _load_lock: threading.Lock = threading.Lock()

    def __new__(cls, model_id: str = MODEL_ID) -> "VisionAnalyzer":
        with cls._instance_lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance.model_id = model_id
                instance.model = None
                instance.processor = None
                instance._load_failed = False
                cls._instance = instance
        return cls._instance

    def load_model(self) -> None:
        with self._load_lock:
            if self.model is not None or self._load_failed:
                return

            try:
                logger.info("Loading processor from %s (cache: %s)", self.model_id, MODEL_CACHE_DIR)
                self.processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    cache_dir=MODEL_CACHE_DIR,
                    trust_remote_code=True,
                )

                logger.info("Loading model weights...")
                t0 = time.perf_counter()
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    cache_dir=MODEL_CACHE_DIR,
                    trust_remote_code=True,
                )
                self.model.eval()
                logger.info("Model weights loaded in %.2fs", time.perf_counter() - t0)

                device = next(self.model.parameters()).device
                logger.info("Model ready on device: %s", device)

                self._warmup()
            except Exception as exc:
                logger.error("Model loading failed: %s", exc, exc_info=True)
                self.model = None
                self.processor = None
                self._load_failed = True

    def _warmup(self) -> None:
        logger.info("Running warmup inference to prime ROCm kernels...")
        t0 = time.perf_counter()
        try:
            self._generate_text("Hello", max_new_tokens=4)
        except Exception as exc:
            logger.warning("Warmup inference failed (non-fatal): %s", exc)
        logger.info("Warmup complete in %.2fs", time.perf_counter() - t0)

    def _generate_text(self, prompt: str, max_new_tokens: int = 120) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        logger.debug("Text generation: %.2fs (max_new_tokens=%d)", time.perf_counter() - t0, max_new_tokens)

        trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
        result = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()

        del inputs, output_ids, trimmed
        return result

    def analyze_video(self, video_path: str) -> str:
        if self._load_failed:
            logger.warning("Model unavailable; skipping video analysis.")
            return ""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        video_path = str(Path(video_path).resolve())

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{video_path}",
                        "max_pixels": 360 * 420,
                        "nframes": MAX_VIDEO_FRAMES,
                    },
                    {
                        "type": "text",
                        "text": (
                            "Describe what is happening in this video concisely in 2-3 sentences, "
                            "focusing on the main subject, action, and setting."
                        ),
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        del image_inputs, video_inputs

        t0 = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
            )
        logger.info("Video analysis inference: %.2fs", time.perf_counter() - t0)

        trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
        result = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()

        del inputs, output_ids, trimmed
        return result

    def generate_captions(self, summary: str, styles: list[str] | None = None) -> dict[str, str]:
        requested = set(styles) if styles else set(_CAPTION_PROMPTS)
        unknown = requested - set(_CAPTION_PROMPTS)
        if unknown:
            logger.warning("Unknown styles requested (will be skipped): %s", sorted(unknown))

        if self._load_failed:
            logger.warning("Model unavailable; returning fallback captions.")
            return {k: v for k, v in _FALLBACK_CAPTIONS.items() if k in requested}
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        t0 = time.perf_counter()
        captions = {
            style: self._generate_text(prompt.format(summary=summary))
            for style, prompt in _CAPTION_PROMPTS.items()
            if style in requested
        }
        logger.info("Caption generation (%d styles): %.2fs", len(captions), time.perf_counter() - t0)
        return captions
