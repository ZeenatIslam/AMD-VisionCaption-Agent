def generate_captions(video_summary: str) -> dict[str, str]:
    return {
        "formal": f"A professional overview: {video_summary}",
        "sarcastic": f"Oh wow, groundbreaking stuff: {video_summary}. Truly never seen before.",
        "humorous_tech": f"404: Boredom not found. Behold this stack of awesomeness: {video_summary}",
        "humorous_non_tech": f"So basically someone did a thing and it was pretty neat: {video_summary}",
    }
