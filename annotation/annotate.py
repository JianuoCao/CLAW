#!/usr/bin/env python3
"""
Post-hoc annotator for recorded trajectory output directories.

Reads `command_log.jsonl` produced by compose_trajectory.py and generates
`annotation.json` with diverse natural-language descriptions of the motion.

Usage:
    python annotate.py output/20260314_180948/
    python annotate.py output/20260314_180948/ --overwrite
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotation.templates import compose_annotations


def load_command_log(output_dir: Path) -> list[dict]:
    log_path = output_dir / "command_log.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"command_log.jsonl not found in {output_dir}")
    events = []
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def reconstruct_segments(events: list[dict], data_fps: int = 50) -> list[dict]:
    """
    Rebuild segment dicts from command_log.jsonl events.
    The log is written by compose_trajectory.py with 'segment_start' events.
    """
    segment_starts = [e for e in events if e.get("event") == "segment_start"]
    session_end    = next((e for e in events if e.get("event") == "session_end"), None)

    segments = []
    for i, ev in enumerate(segment_starts):
        t_start = ev["t"]
        if i + 1 < len(segment_starts):
            t_end = segment_starts[i + 1]["t"]
        elif session_end:
            t_end = session_end["t"]
        else:
            t_end = t_start + ev.get("duration", 0.0)

        duration = t_end - t_start
        frame_start = int(round(t_start * data_fps))
        frame_end   = int(round(t_end   * data_fps))

        # Prefer the canonical internal key logged since the mode-expansion update;
        # fall back to normalising the display name for older recordings.
        mode = ev.get("mode") or _normalize_mode(ev.get("mode_name", "walk"))
        seg = {
            "idx":          i,
            "t_start":      round(t_start, 3),
            "t_end":        round(t_end,   3),
            "frame_start":  frame_start,
            "frame_end":    frame_end,
            "mode":         mode,
            "movement":     ev.get("movement", "forward"),
            "turn_deg":     ev.get("turn_deg"),
            "turn_dir":     ev.get("turn_dir"),
            "duration_sec": round(duration, 3),
            "speed":        float(ev.get("speed", -1.0)),
        }
        segments.append(seg)

    return segments


def _normalize_mode(mode_name: str) -> str:
    """Map display mode name → internal key used by templates.py.

    Used as a fallback for recordings made before the internal 'mode' key was
    added to command_log.jsonl.  New recordings carry the key directly.
    """
    _DISPLAY_TO_KEY = {
        "slow walk":       "slow_walk",
        "walk":            "walk",
        "run":             "run",
        "happy":           "happy",
        "stealth":         "stealth",
        "injured":         "injured",
        "squat":           "squat",
        "kneel (two)":     "kneel_two",
        "kneel (one)":     "kneel_one",
        "hand crawl":      "hand_crawl",
        "elbow crawl":     "elbow_crawl",
        "idle boxing":     "idle_boxing",
        "walk boxing":     "walk_boxing",
        "left jab":        "left_jab",
        "right jab":       "right_jab",
        "random punches":  "random_punches",
        "left hook":       "left_hook",
        "right hook":      "right_hook",
        "careful":         "careful",
        "object carrying": "object_carrying",
        "crouch":          "crouch",
        "happy dance":     "happy_dance",
        "zombie":          "zombie",
        "point":           "point",
        "scared":          "scared",
    }
    key = mode_name.lower().strip()
    return _DISPLAY_TO_KEY.get(key, "walk")


def annotate_recording(
    output_dir: Path | str,
    segments: list[dict] | None = None,
    data_fps: int = 50,
    recipe_name: str | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Generate annotation.json for a trajectory output directory.

    Args:
        output_dir:  Path to the recording directory.
        segments:    Pre-built segment list (from compose_trajectory.py).
                     If None, reconstructed from command_log.jsonl.
        data_fps:    Recording frame rate (default 50 Hz).
        recipe_name: Name of the trajectory recipe.
        overwrite:   Overwrite existing annotation.json if present.

    Returns:
        Path to the written annotation.json.
    """
    output_dir = Path(output_dir)
    annotation_path = output_dir / "annotation.json"

    if annotation_path.exists() and not overwrite:
        print(f"annotation.json already exists at {annotation_path}. Use --overwrite to replace.")
        return annotation_path

    if segments is None:
        segments = reconstruct_segments(load_command_log(output_dir), data_fps=data_fps)

    if not segments:
        print("Warning: no segments found; annotation will be empty.")

    # Try to read recipe name from metadata.json if not provided
    if recipe_name is None:
        meta_path = output_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            recipe_name = meta.get("recipe_name") or meta.get("mode", "unknown")
        else:
            recipe_name = "unknown"

    # Total duration
    total_duration = segments[-1]["t_end"] if segments else 0.0

    # Find trajectory file
    traj_file = None
    for candidate in ("trajectory_measured.csv", "trajectory_target.csv"):
        if (output_dir / candidate).exists():
            traj_file = candidate
            break

    annotation = {
        "schema_version":    "1.0",
        "recipe_name":       recipe_name,
        "total_duration_sec": round(total_duration, 3),
        "data_fps":          data_fps,
        "trajectory_file":   traj_file,
        "segments":          segments,
        **compose_annotations(segments),
    }

    annotation_path.write_text(json.dumps(annotation, indent=2), encoding="utf-8")
    print(f"Saved annotation.json → {annotation_path}")
    return annotation_path


def main():
    parser = argparse.ArgumentParser(description="Post-hoc trajectory annotator.")
    parser.add_argument("output_dir", help="Recording output directory containing command_log.jsonl")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing annotation.json")
    parser.add_argument("--fps", type=int, default=50, help="Data recording FPS (default: 50)")
    args = parser.parse_args()

    annotate_recording(
        output_dir=Path(args.output_dir),
        data_fps=args.fps,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
