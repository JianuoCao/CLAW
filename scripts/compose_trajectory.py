#!/usr/bin/env python3
"""
YAML recipe executor for composing multi-segment robot trajectories.

Each recipe defines a sequence of motion segments. The script executes each
segment via ZMQ planner commands, records the trajectory CSV, and generates
diverse text annotations automatically.

Usage:
    python compose_trajectory.py recipe.yaml
    python compose_trajectory.py recipe.yaml --output-dir output/my_traj
    python compose_trajectory.py recipe.yaml --no-record-debug

Recipe YAML schema
------------------
name: "run_forward_turn_left_walk_back"
planner_hz: 20           # optional, default 20
init_wait: 8.0           # optional, default 8.0 s
control_port: 5556       # optional
debug_url: "tcp://127.0.0.1:5557"   # optional
segments:
  - {mode: run,       direction: forward,  duration: 3.0}
  - {mode: walk,      turn: left,  angle: 90, duration: 2.0}
  - {mode: slow_walk, direction: backward, duration: 1.5}

Segment fields:
  mode:       run | walk | slow_walk | happy | stealth | injured | squat |
              kneel_two | kneel_one | hand_crawl | elbow_crawl | idle_boxing |
              walk_boxing | left_jab | right_jab | random_punches | left_hook |
              right_hook | careful | object_carrying | crouch | happy_dance |
              zombie | point | scared
  direction:  forward | backward | strafe_left | strafe_right  (omit when using turn)
  turn:       left | right  (mutually exclusive with direction)
  angle:      degrees to rotate (only with turn, default 90)
  duration:   seconds
  speed:      m/s override (optional; omit or -1 to use mode default)
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# Re-use ZMQ utilities and recording infrastructure from virtual_joystick.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from virtual_joystick import (
    DebugRecorder,
    MODE_IDLE,
    MODE_SLOW_WALK, MODE_WALK, MODE_RUN,
    MODE_HAPPY, MODE_STEALTH, MODE_INJURED,
    MODE_SQUAT, MODE_KNEEL_TWO, MODE_KNEEL_ONE,
    MODE_HAND_CRAWL, MODE_ELBOW_CRAWL,
    MODE_IDLE_BOXING, MODE_WALK_BOXING,
    MODE_LEFT_JAB, MODE_RIGHT_JAB, MODE_RANDOM_PUNCHES,
    MODE_LEFT_HOOK, MODE_RIGHT_HOOK,
    MODE_CAREFUL, MODE_OBJECT_CARRYING, MODE_CROUCH,
    MODE_HAPPY_DANCE, MODE_ZOMBIE, MODE_POINT, MODE_SCARED,
    build_command_message,
    build_planner_message,
    save_recording,
)
from annotate import annotate_recording

import zmq


# ---------------------------------------------------------------------------
# Segment spec
# ---------------------------------------------------------------------------

_MODE_MAP = {
    # Locomotion
    "slow_walk":       (MODE_SLOW_WALK,       "Slow Walk"),
    "walk":            (MODE_WALK,            "Walk"),
    "run":             (MODE_RUN,             "Run"),
    "happy":           (MODE_HAPPY,           "Happy"),
    "stealth":         (MODE_STEALTH,         "Stealth"),
    "injured":         (MODE_INJURED,         "Injured"),
    # Squat / Ground
    "squat":           (MODE_SQUAT,           "Squat"),
    "kneel_two":       (MODE_KNEEL_TWO,       "Kneel (Two)"),
    "kneel_one":       (MODE_KNEEL_ONE,       "Kneel (One)"),
    "hand_crawl":      (MODE_HAND_CRAWL,      "Hand Crawl"),
    "elbow_crawl":     (MODE_ELBOW_CRAWL,     "Elbow Crawl"),
    # Boxing
    "idle_boxing":     (MODE_IDLE_BOXING,     "Idle Boxing"),
    "walk_boxing":     (MODE_WALK_BOXING,     "Walk Boxing"),
    "left_jab":        (MODE_LEFT_JAB,        "Left Jab"),
    "right_jab":       (MODE_RIGHT_JAB,       "Right Jab"),
    "random_punches":  (MODE_RANDOM_PUNCHES,  "Random Punches"),
    "left_hook":       (MODE_LEFT_HOOK,       "Left Hook"),
    "right_hook":      (MODE_RIGHT_HOOK,      "Right Hook"),
    # Styled Walking
    "careful":         (MODE_CAREFUL,         "Careful"),
    "object_carrying": (MODE_OBJECT_CARRYING, "Object Carrying"),
    "crouch":          (MODE_CROUCH,          "Crouch"),
    "happy_dance":     (MODE_HAPPY_DANCE,     "Happy Dance"),
    "zombie":          (MODE_ZOMBIE,          "Zombie"),
    "point":           (MODE_POINT,           "Point"),
    "scared":          (MODE_SCARED,          "Scared"),
}


@dataclass
class SegmentSpec:
    idx:       int
    mode:      str          # "run" | "walk" | "slow_walk" | ...
    movement:  str          # "forward" | "backward" | "strafe_left" | "strafe_right"
    duration:  float        # seconds
    turn_dir:  Optional[str] = None   # "left" | "right" | None
    turn_deg:  Optional[float] = None
    speed:     float = -1.0           # m/s; -1.0 means use mode default


def parse_recipe(path: str) -> tuple[dict, list[SegmentSpec]]:
    with open(path, encoding="utf-8") as f:
        recipe = yaml.safe_load(f)

    segments_raw = recipe.get("segments", [])
    specs: list[SegmentSpec] = []
    for i, raw in enumerate(segments_raw):
        mode = raw.get("mode", "walk").lower().replace(" ", "_")
        if mode not in _MODE_MAP:
            raise ValueError(f"Segment {i}: unknown mode '{mode}'. Choose from {list(_MODE_MAP)}")

        turn = raw.get("turn")
        direction = raw.get("direction", "forward")

        if turn and direction and direction != "forward":
            raise ValueError(
                f"Segment {i}: cannot specify both 'turn' and non-forward 'direction'. "
                "During a turn the robot moves forward while rotating."
            )

        specs.append(SegmentSpec(
            idx=i,
            mode=mode,
            movement="forward" if turn else direction,
            duration=float(raw["duration"]),
            turn_dir=turn,
            turn_deg=float(raw.get("angle", 90)) if turn else None,
            speed=float(raw.get("speed", -1.0)),
        ))

    return recipe, specs


# ---------------------------------------------------------------------------
# Facing angle helpers
# ---------------------------------------------------------------------------

def _facing_vector(angle_rad: float) -> list[float]:
    return [math.cos(angle_rad), math.sin(angle_rad), 0.0]


def _movement_vector(movement_type: str, facing_angle: float) -> list[float]:
    a = facing_angle
    vectors = {
        "forward":      [ math.cos(a),  math.sin(a), 0.0],
        "backward":     [-math.cos(a), -math.sin(a), 0.0],
        "strafe_left":  [-math.sin(a),  math.cos(a), 0.0],
        "strafe_right": [ math.sin(a), -math.cos(a), 0.0],
    }
    return vectors.get(movement_type, [1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Command log writer
# ---------------------------------------------------------------------------

class CommandLogger:
    def __init__(self, output_dir: Path):
        self._path = output_dir / "command_log.jsonl"
        self._f = self._path.open("w", encoding="utf-8")
        self._t0 = time.time()

    def _write(self, obj: dict):
        obj["t"] = round(time.time() - self._t0, 4)
        self._f.write(json.dumps(obj, separators=(",", ":")) + "\n")
        self._f.flush()

    def segment_start(self, spec: SegmentSpec, facing_angle_deg: float):
        ev = {
            "event":            "segment_start",
            "segment_idx":      spec.idx,
            "mode":             spec.mode,            # internal key used by templates.py
            "mode_name":        _MODE_MAP[spec.mode][1],
            "movement":         spec.movement,
            "facing_angle_deg": round(facing_angle_deg, 2),
            "duration":         spec.duration,
            "speed":            spec.speed,
        }
        if spec.turn_dir:
            ev["turn_dir"] = spec.turn_dir
            ev["turn_deg"] = spec.turn_deg
        self._write(ev)

    def session_end(self, facing_angle_deg: float):
        self._write({"event": "session_end", "facing_angle_deg": round(facing_angle_deg, 2)})

    def close(self):
        self._f.close()


# ---------------------------------------------------------------------------
# Segment execution
# ---------------------------------------------------------------------------

def execute_segment(
    spec: SegmentSpec,
    cmd_socket,
    debug_recorder: Optional[DebugRecorder],
    logger: CommandLogger,
    planner_hz: float,
    facing_angle: float,
) -> float:
    """Execute one motion segment. Returns the final facing angle (radians)."""
    mode_int = _MODE_MAP[spec.mode][0]
    total_steps = int(planner_hz * spec.duration)
    dt = 1.0 / planner_hz

    facing_start = facing_angle
    if spec.turn_dir and spec.turn_deg:
        # In standard math coords: positive angle = counter-clockwise (left turn)
        sign = 1.0 if spec.turn_dir == "left" else -1.0
        facing_end = facing_start + math.radians(sign * spec.turn_deg)
    else:
        facing_end = facing_start

    facing_angle_deg = math.degrees(facing_start)
    logger.segment_start(spec, facing_angle_deg)

    print(
        f"  Segment {spec.idx}: {spec.mode} {spec.movement}"
        + (f" + turn {spec.turn_dir} {spec.turn_deg}°" if spec.turn_dir else "")
        + f"  ({spec.duration:.1f}s, {total_steps} steps)"
    )

    for step in range(total_steps):
        alpha = step / max(total_steps - 1, 1)
        current_facing = facing_start + alpha * (facing_end - facing_start)

        movement = _movement_vector(spec.movement, current_facing)
        facing   = _facing_vector(current_facing)
        msg = build_planner_message(
            mode=mode_int,
            movement=movement,
            facing=facing,
            speed=spec.speed,
            height=-1.0,
        )
        cmd_socket.send(msg)
        if debug_recorder is not None:
            debug_recorder.poll()
        time.sleep(dt)

    return facing_end


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Compose and annotate a multi-segment trajectory.")
    parser.add_argument("recipe", help="Path to YAML recipe file.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--control-port", type=int, default=5556)
    parser.add_argument("--control-host", default="*")
    parser.add_argument("--init-wait", type=float, default=None,
                        help="Override recipe init_wait (seconds).")
    parser.add_argument("--planner-hz", type=float, default=None,
                        help="Override recipe planner_hz.")
    parser.add_argument("--debug-url", default="tcp://127.0.0.1:5557")
    parser.add_argument("--debug-topic", default="g1_debug")
    parser.add_argument("--no-record-debug", action="store_true")
    parser.add_argument("--render-mp4", action="store_true")
    parser.add_argument("--data-fps", type=int, default=50)
    parser.add_argument("--video-fps", type=int, default=25)
    parser.add_argument("--video-width", type=int, default=960)
    parser.add_argument("--video-height", type=int, default=540)
    return parser.parse_args()


def main():
    args = parse_args()
    recipe, specs = parse_recipe(args.recipe)

    recipe_name   = recipe.get("name", Path(args.recipe).stem)
    planner_hz    = args.planner_hz or float(recipe.get("planner_hz", 20.0))
    init_wait     = args.init_wait  or float(recipe.get("init_wait", 8.0))
    control_port  = recipe.get("control_port", args.control_port)
    debug_url     = recipe.get("debug_url", args.debug_url)

    print(f"Recipe: {recipe_name}  ({len(specs)} segments)")
    for s in specs:
        print(f"  [{s.idx}] {s.mode} {s.movement}"
              + (f" turn={s.turn_dir} {s.turn_deg}°" if s.turn_dir else "")
              + f" dur={s.duration}s")

    # 1. Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            Path(__file__).resolve().parent / "output" / f"{timestamp}_{recipe_name}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. ZMQ setup
    context = zmq.Context()
    cmd_socket = context.socket(zmq.PUB)
    bind_addr = f"tcp://{args.control_host}:{control_port}"
    try:
        cmd_socket.bind(bind_addr)
        print(f"ZMQ PUB bound to {bind_addr}")
    except Exception as e:
        print(f"Bind failed ({e}), trying connect...")
        cmd_socket.connect(f"tcp://127.0.0.1:{control_port}")

    # 3. Debug recorder
    debug_recorder = None
    if not args.no_record_debug:
        try:
            import msgpack  # noqa: F401
            debug_recorder = DebugRecorder(
                context=context, zmq_url=debug_url, topic=args.debug_topic
            )
            print(f"Debug recording: {debug_url} topic={args.debug_topic}")
        except (ImportError, RuntimeError) as e:
            print(f"Warning: debug recording disabled ({e})")

    # 4. Wait for subscribers + send START
    print("Waiting for subscribers to connect...")
    time.sleep(2)
    if debug_recorder:
        debug_recorder.poll()

    print("Sending START + PLANNER...")
    print("NOTE: Make sure you pressed '9' in the MuJoCo window to start simulation.")
    cmd_socket.send(build_command_message(start=True, stop=False, planner=True))

    print(f"Waiting {init_wait:.1f}s for robot to initialize...")
    warmup_end = time.time() + init_wait
    while time.time() < warmup_end:
        if debug_recorder:
            debug_recorder.poll()
        time.sleep(0.01)

    # 5. Execute segments
    logger = CommandLogger(output_dir)
    facing_angle = 0.0  # radians
    motion_record_start = motion_record_end = 0
    motion_target_start = motion_target_end = 0
    motion_measured_start = motion_measured_end = 0

    # Mark motion-window start indices so CSV/video only include segment execution.
    if debug_recorder:
        motion_record_start = len(debug_recorder.records)
        motion_target_start = len(debug_recorder.target_rows)
        motion_measured_start = len(debug_recorder.measured_rows)

    for spec in specs:
        facing_angle = execute_segment(
            spec, cmd_socket, debug_recorder, logger, planner_hz, facing_angle
        )

    # Flush in-flight motion debug frames before marking session end.
    if debug_recorder:
        flush_end = time.time() + 0.5
        while time.time() < flush_end:
            debug_recorder.poll()
            time.sleep(0.01)

    facing_deg = math.degrees(facing_angle)
    logger.session_end(facing_deg)
    logger.close()

    # Mark motion-window end indices at session_end (exclude post-session_end data).
    if debug_recorder:
        motion_record_end = len(debug_recorder.records)
        motion_target_end = len(debug_recorder.target_rows)
        motion_measured_end = len(debug_recorder.measured_rows)

    # 6. Return to IDLE (send commands but don't record frames)
    print("Returning to IDLE...")
    idle_msg = build_planner_message(
        mode=MODE_IDLE,
        movement=[1.0, 0.0, 0.0],
        facing=[1.0, 0.0, 0.0],
        speed=-1.0, height=-1.0,
    )
    for _ in range(int(planner_hz)):
        cmd_socket.send(idle_msg)
        # Stop polling debug frames after session_end to exclude IDLE cooldown
        time.sleep(1.0 / planner_hz)

    # 7. Save CSV recording (only motion frames, excluding IDLE cooldown)
    if debug_recorder:
        # Slice recorder buffers to [motion_start, session_end].
        debug_recorder.records = debug_recorder.records[motion_record_start:motion_record_end]
        debug_recorder.target_rows = debug_recorder.target_rows[motion_target_start:motion_target_end]
        debug_recorder.measured_rows = debug_recorder.measured_rows[motion_measured_start:motion_measured_end]

        # Patch args to match save_recording expectations
        args.output_dir = str(output_dir)
        args.mode = specs[0].mode if specs else "walk"
        args.duration = sum(s.duration for s in specs)
        args.planner_hz = planner_hz
        args.control_port = control_port
        args.debug_url = debug_url
        args.debug_topic = args.debug_topic

        save_recording(debug_recorder, args)

        # Update metadata.json with recipe info
        meta_path = output_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            meta = {}
        meta["recipe_name"] = recipe_name
        meta["recipe_segments"] = [
            {
                "idx": s.idx, "mode": s.mode, "movement": s.movement,
                "turn_dir": s.turn_dir, "turn_deg": s.turn_deg, "duration": s.duration,
            }
            for s in specs
        ]
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        debug_recorder.close()

    # 8. Generate annotations
    # Build segment dicts for annotator
    t_cursor = 0.0
    segments = []
    for spec in specs:
        t_end = t_cursor + spec.duration
        segments.append({
            "idx":          spec.idx,
            "t_start":      round(t_cursor, 3),
            "t_end":        round(t_end, 3),
            "frame_start":  int(round(t_cursor * args.data_fps)),
            "frame_end":    int(round(t_end * args.data_fps)),
            "mode":         spec.mode,
            "movement":     spec.movement,
            "turn_deg":     spec.turn_deg,
            "turn_dir":     spec.turn_dir,
            "duration_sec": spec.duration,
            "speed":        spec.speed,
        })
        t_cursor = t_end

    annotate_recording(
        output_dir=output_dir,
        segments=segments,
        data_fps=args.data_fps,
        recipe_name=recipe_name,
        overwrite=True,
    )

    cmd_socket.close(0)
    context.term()

    print("\n" + "=" * 60)
    print(f"Trajectory recorded: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
