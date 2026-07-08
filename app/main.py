import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from downloader import download_video, cleanup_download
from services.caption_services import process_video, get_analyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

_PROJECT_ROOT = Path(__file__).parent.parent
INPUT_PATH = str(_PROJECT_ROOT / "input" / "tasks.json")
OUTPUT_PATH = str(_PROJECT_ROOT / "output" / "results.json")


def process_task(task: dict) -> dict:
    task_id = task.get("task_id", "<unknown>")
    video_source = task.get("video_url")
    if not video_source:
        raise ValueError("task missing required field 'video_url'")
    styles = task.get("styles") or None

    video_path = download_video(video_source)
    try:
        captions = process_video(str(video_path), styles=styles)
    finally:
        cleanup_download(video_path)

    return {"task_id": task_id, "captions": captions}


def main():
    get_analyzer()

    tasks_file = Path(INPUT_PATH)
    if not tasks_file.exists():
        raise FileNotFoundError(f"Tasks file not found: {INPUT_PATH}")

    try:
        with tasks_file.open() as f:
            tasks = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"tasks.json is not valid JSON: {exc}") from exc

    if not isinstance(tasks, list):
        raise TypeError(f"tasks.json must contain a JSON array, got {type(tasks).__name__}")

    output_dir = Path(OUTPUT_PATH).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for task in tasks:
        if not isinstance(task, dict):
            results.append({"task_id": "<unknown>", "error": f"task entry must be a JSON object, got {type(task).__name__}"})
            continue
        task_id = task.get("task_id", "<unknown>")
        try:
            result = process_task(task)
            results.append(result)
        except Exception as e:
            results.append({"task_id": task_id, "error": str(e)})

    tmp_path = Path(OUTPUT_PATH).with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_path, OUTPUT_PATH)

    print(f"Done. {len(results)} task(s) written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
