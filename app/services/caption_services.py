import logging
from vision_analyzer import VisionAnalyzer

logger = logging.getLogger(__name__)


def get_analyzer() -> VisionAnalyzer:
    analyzer = VisionAnalyzer()
    if analyzer.model is None and not analyzer._load_failed:
        analyzer.load_model()
    return analyzer


def process_video(video_path: str, styles: list[str] | None = None) -> dict[str, str]:
    analyzer = get_analyzer()
    summary = analyzer.analyze_video(video_path)
    return analyzer.generate_captions(summary, styles=styles)
