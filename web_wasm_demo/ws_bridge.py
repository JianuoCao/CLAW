#!/usr/bin/env python3
"""
WebSocket bridge server for the G1 WASM demo.

Bridges browser keyboard events → ZMQ planner commands (to C++ controller),
and streams robot qpos back to connected browser clients for WASM rendering.

Usage:
    1. Start C++ controller:  bash deploy.sh sim --input-type zmq_manager
    2. Press '9' in the MuJoCo window to start simulation
    3. Run this bridge:  python ws_bridge.py
    4. In another terminal:  cd web_wasm_demo && npm run dev
    5. Open http://localhost:5173 in your browser

Keyboard controls (forwarded from browser):
    Enter       — Start / Stop controller
    W / S       — Move forward / backward
    A / D       — Turn left / right (adjust facing angle)
    Q / E       — Snap heading ±30°
    , / .       — Strafe left / right
    R           — Stop movement (return to idle)

The bridge does NOT load MuJoCo Python — the browser handles all
physics and rendering via MuJoCo WASM.
"""

import argparse
import asyncio
import csv
import json
import math
import struct
import sys
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import yaml

import numpy as np
import uvicorn
import zmq
import zmq.asyncio
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Allow imports from parent gear_sonic_deploy directory
_parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_parent_dir))
# If compose_trajectory.py is not in parent (e.g. GR00T-WBC-Dev layout), search sibling repos
_ct_search_dirs = [
    _parent_dir,
    _parent_dir.parent / "GR00T-WholeBodyControl" / "gear_sonic_deploy",
]
for _d in _ct_search_dirs:
    if (_d / "compose_trajectory.py").is_file():
        if str(_d) not in sys.path:
            sys.path.insert(0, str(_d))
        break
from annotation.templates import compose_annotations

# _MODE_MAP: key → (planner_int, display_name)  (imported from compose_trajectory)
try:
    from compose_trajectory import _MODE_MAP
except ImportError:
    # Fallback minimal map if compose_trajectory is not importable
    _MODE_MAP = {}


def _mode_index_to_key(mode_index: int) -> str:
    """Reverse-lookup: planner mode index → string key for compose_annotations."""
    for k, (idx, _) in _MODE_MAP.items():
        if idx == mode_index:
            return k
    return "walk"

# Directory for storing recipe run results (CSV, annotation JSON)
_RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Default joint angles (offsets applied to body_q_target from ZMQ)
# ---------------------------------------------------------------------------
DEFAULT_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg
    0.0, 0.0, 0.0,                            # waist
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,      # left arm
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,     # right arm
], dtype=float)

# ---------------------------------------------------------------------------
# ZMQ message builders (matches virtual_joystick.py / zmq_planner_sender.py)
# ---------------------------------------------------------------------------
HEADER_SIZE = 1280


def _build_header(fields: list, version: int = 1, count: int = 1) -> bytes:
    header = {
        "v": version,
        "endian": "le",
        "count": count,
        "fields": fields,
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_json)} > {HEADER_SIZE}")
    return header_json.ljust(HEADER_SIZE, b"\x00")


def build_command_message(start: bool, stop: bool, planner: bool) -> bytes:
    fields = [
        {"name": "start",   "dtype": "u8", "shape": [1]},
        {"name": "stop",    "dtype": "u8", "shape": [1]},
        {"name": "planner", "dtype": "u8", "shape": [1]},
    ]
    payload = struct.pack("BBB",
                          1 if start else 0,
                          1 if stop else 0,
                          1 if planner else 0)
    return b"command" + _build_header(fields) + payload


def build_planner_message(mode: int, movement, facing,
                          speed: float = -1.0, height: float = -1.0) -> bytes:
    fields = [
        {"name": "mode",     "dtype": "i32", "shape": [1]},
        {"name": "movement", "dtype": "f32", "shape": [3]},
        {"name": "facing",   "dtype": "f32", "shape": [3]},
        {"name": "speed",    "dtype": "f32", "shape": [1]},
        {"name": "height",   "dtype": "f32", "shape": [1]},
    ]
    payload = b"".join((
        struct.pack("<i", int(mode)),
        struct.pack("<fff", *[float(v) for v in movement]),
        struct.pack("<fff", *[float(v) for v in facing]),
        struct.pack("<f", float(speed)),
        struct.pack("<f", float(height)),
    ))
    return b"planner" + _build_header(fields) + payload


# ---------------------------------------------------------------------------
# Motion sets (matches web_demo/server.py)
# ---------------------------------------------------------------------------
MOTION_SETS = [
    {
        "name": "Locomotion",
        "modes": [
            {"key": 1, "index": 1,  "name": "Slow Walk",  "speed_range": [0.2, 0.8],  "default_speed": 0.5},
            {"key": 2, "index": 2,  "name": "Walk",       "speed_range": None,         "default_speed": -1.0},
            {"key": 3, "index": 3,  "name": "Run",        "speed_range": [1.5, 3.0],  "default_speed": 2.0},
            {"key": 4, "index": 17, "name": "Happy",      "speed_range": None,         "default_speed": -1.0},
            {"key": 5, "index": 18, "name": "Stealth",    "speed_range": None,         "default_speed": -1.0},
            {"key": 6, "index": 19, "name": "Injured",    "speed_range": None,         "default_speed": -1.0},
        ],
    },
    {
        "name": "Squat / Ground",
        "height_control": True,
        "modes": [
            {"key": 1, "index": 4,  "name": "Squat",          "speed_range": None,        "default_speed": -1.0},
            {"key": 2, "index": 5,  "name": "Kneel (Two)",    "speed_range": None,        "default_speed": -1.0},
            {"key": 3, "index": 6,  "name": "Kneel (One)",    "speed_range": None,        "default_speed": -1.0},
            {"key": 4, "index": 8,  "name": "Hand Crawl",     "speed_range": [0.4, 1.5], "default_speed": 0.8},
            {"key": 5, "index": 14, "name": "Elbow Crawl",    "speed_range": [0.7, 1.5], "default_speed": 1.0},
        ],
    },
    {
        "name": "Boxing",
        "modes": [
            {"key": 1, "index": 9,  "name": "Idle Boxing",    "speed_range": None,        "default_speed": -1.0},
            {"key": 2, "index": 10, "name": "Walk Boxing",    "speed_range": [0.7, 1.5], "default_speed": 1.0},
            {"key": 3, "index": 11, "name": "Left Jab",       "speed_range": [0.7, 1.5], "default_speed": 1.0},
            {"key": 4, "index": 12, "name": "Right Jab",      "speed_range": [0.7, 1.5], "default_speed": 1.0},
            {"key": 5, "index": 13, "name": "Random Punches", "speed_range": [0.7, 1.5], "default_speed": 1.0},
            {"key": 6, "index": 15, "name": "Left Hook",      "speed_range": [0.7, 1.5], "default_speed": 1.0},
            {"key": 7, "index": 16, "name": "Right Hook",     "speed_range": [0.7, 1.5], "default_speed": 1.0},
        ],
    },
    {
        "name": "Styled Walking",
        "modes": [
            {"key": 1, "index": 20, "name": "Careful",          "speed_range": None, "default_speed": -1.0},
            {"key": 2, "index": 21, "name": "Object Carrying",  "speed_range": None, "default_speed": -1.0},
            {"key": 3, "index": 22, "name": "Crouch",           "speed_range": None, "default_speed": -1.0},
            {"key": 4, "index": 23, "name": "Happy Dance",      "speed_range": None, "default_speed": -1.0},
            {"key": 5, "index": 24, "name": "Zombie",           "speed_range": None, "default_speed": -1.0},
            {"key": 6, "index": 25, "name": "Point",            "speed_range": None, "default_speed": -1.0},
            {"key": 7, "index": 26, "name": "Scared",           "speed_range": None, "default_speed": -1.0},
        ],
    },
]

MODE_IDLE = 0

# ---------------------------------------------------------------------------
# Recipe recording globals
# ---------------------------------------------------------------------------
_recipe_recording: bool = False
_recipe_rec_frames: list[dict] = []
_recipe_task: asyncio.Task | None = None
_recipe_run_started_at: float | None = None
_recipe_first_segment_at: float | None = None
_recipe_first_motion_at: float | None = None
_recipe_first_linvel_seen_at: float | None = None
_recipe_first_target_change_at: float | None = None
_recipe_first_measured_change_at: float | None = None
_recipe_first_target_root_change_at: float | None = None
_recipe_first_measured_root_change_at: float | None = None
_recipe_debug_frames_seen: int = 0
_recipe_debug_frames_missing_linvel: int = 0
_recipe_last_target_q: np.ndarray | None = None
_recipe_last_measured_q: np.ndarray | None = None
_recipe_last_target_root: np.ndarray | None = None
_recipe_last_measured_root: np.ndarray | None = None
_recipe_debug_exception_count: int = 0
_recipe_debug_log_path: Path | None = None
_recipe_debug_frames_path: Path | None = None
_recipe_debug_snapshot_count: int = 0
_RECIPE_DEBUG_PRE_SEG_FRAMES = 10
_RECIPE_DEBUG_POST_SEG_FRAMES = 40
_recipe_pre_segment_buffer = deque(maxlen=_RECIPE_DEBUG_PRE_SEG_FRAMES)
_recipe_post_segment_frames_remaining: int = 0


def _recipe_since_run(ts: float | None) -> str:
    if ts is None or _recipe_run_started_at is None:
        return "n/a"
    return f"{(ts - _recipe_run_started_at):.3f}s"


def _recipe_since_segment(ts: float | None) -> str:
    if ts is None or _recipe_first_segment_at is None:
        return "n/a"
    return f"{(ts - _recipe_first_segment_at):.3f}s"


def _append_recipe_debug_line(line: str):
    print(line)
    if _recipe_debug_log_path is None:
        return
    try:
        with _recipe_debug_log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _append_recipe_debug_json(obj: dict):
    if _recipe_debug_frames_path is None:
        return
    try:
        with _recipe_debug_frames_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=True) + "\n")
    except Exception:
        pass


def _make_recipe_snapshot(data: dict, frame_mono: float, base_trans, quat_wxyz, body_q, trans_tgt, quat_tgt, body_q_tgt) -> dict:
    return {
        "type": "recipe_debug_frame",
        "frame_idx": _recipe_debug_snapshot_count,
        "t_from_run_s": round(frame_mono - _recipe_run_started_at, 6) if _recipe_run_started_at else None,
        "t_from_first_segment_s": (
            round(frame_mono - _recipe_first_segment_at, 6)
            if _recipe_first_segment_at else None
        ),
        "base_trans_measured": [round(float(v), 6) for v in base_trans],
        "base_quat_measured": [round(float(v), 6) for v in quat_wxyz],
        "body_q_measured_head": [round(float(v), 6) for v in body_q[:8]],
        "base_trans_target": [round(float(v), 6) for v in trans_tgt],
        "base_quat_target": [round(float(v), 6) for v in quat_tgt],
        "body_q_target_head": [round(float(v), 6) for v in body_q_tgt[:8]],
        "base_lin_vel_raw": data.get("base_lin_vel_measured"),
        "lin_vel_raw": data.get("lin_vel_measured"),
    }


def _recipe_joint_delta(curr: np.ndarray, prev: np.ndarray | None) -> float:
    if prev is None or curr.shape != prev.shape:
        return 0.0
    return float(np.max(np.abs(curr - prev)))


def _recipe_root_delta(trans: list[float], quat: list[float], prev: np.ndarray | None) -> float:
    curr = np.array(list(trans) + list(quat), dtype=float)
    if prev is None or curr.shape != prev.shape:
        return 0.0
    return float(np.max(np.abs(curr - prev)))


def _yaw_from_quat_wxyz(quat_wxyz: list[float]) -> float:
    """Return world yaw (radians) from quaternion in w,x,y,z order."""
    if len(quat_wxyz) != 4:
        return 0.0
    w, x, y, z = [float(v) for v in quat_wxyz]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _print_recipe_debug_summary(prefix: str = "Recipe timing summary"):
    if _recipe_run_started_at is None:
        return
    _append_recipe_debug_line(
        f"  {prefix}: debug_frames={_recipe_debug_frames_seen}, "
        f"missing_linvel={_recipe_debug_frames_missing_linvel}, "
        f"first_segment={_recipe_since_run(_recipe_first_segment_at)} from run_recipe, "
        f"first_linvel_field={_recipe_since_run(_recipe_first_linvel_seen_at)} from run_recipe, "
        f"first_target_root_change={_recipe_since_run(_recipe_first_target_root_change_at)} from run_recipe "
        f"({_recipe_since_segment(_recipe_first_target_root_change_at)} from first segment), "
        f"first_measured_root_change={_recipe_since_run(_recipe_first_measured_root_change_at)} from run_recipe "
        f"({_recipe_since_segment(_recipe_first_measured_root_change_at)} from first segment), "
        f"first_target_change={_recipe_since_run(_recipe_first_target_change_at)} from run_recipe "
        f"({_recipe_since_segment(_recipe_first_target_change_at)} from first segment), "
        f"first_measured_change={_recipe_since_run(_recipe_first_measured_change_at)} from run_recipe "
        f"({_recipe_since_segment(_recipe_first_measured_change_at)} from first segment), "
        f"first_nonzero_speed={_recipe_since_run(_recipe_first_motion_at)} from run_recipe "
        f"({_recipe_since_segment(_recipe_first_motion_at)} from first segment), "
        f"debug_exceptions={_recipe_debug_exception_count}"
    )

# ---------------------------------------------------------------------------
# Keyboard session frame recording globals (separate from recipe recording)
# ---------------------------------------------------------------------------
_kb_session_recording: bool      = False
_kb_session_frames: list[dict]   = []


# ---------------------------------------------------------------------------
# Keyboard session recorder
# ---------------------------------------------------------------------------
class KeyboardSessionRecorder:
    """Records keyboard-driven planner state changes and reconstructs annotation segments."""

    TURN_THRESHOLD_DEG = 5.0   # minimum facing delta to classify segment as a turn
    MIN_SEGMENT_SEC    = 0.3   # discard segments shorter than this

    def __init__(self):
        self._recording: bool      = False
        self._events: list[dict]   = []
        self._t0: float            = 0.0

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self):
        self._recording = True
        self._events    = []
        self._t0        = time.time()
        self._snapshot()

    def stop(self):
        if self._recording:
            self._snapshot()   # finalize last state
        self._recording = False

    def _snapshot(self):
        # Prefer measured speed (actual velocity); fall back to commanded speed
        spd = state.measured_speed if state.measured_speed > 0.05 else state.speed
        self._events.append({
            "t":             time.time(),
            "movement_type": state.movement_type,
            "mode_index":    state.planner_mode_index,
            "facing_angle":  state.facing_angle,
            "speed":         spd,
        })

    def on_state_change(self):
        if self._recording:
            self._snapshot()

    def build_segments(self) -> list[dict]:
        """Reconstruct annotation segments from recorded snapshots."""
        if len(self._events) < 2:
            return []

        t0       = self._t0
        segments = []
        seg_idx  = 0
        i        = 0

        while i < len(self._events) - 1:
            ev = self._events[i]

            # Skip idle phases
            if ev["movement_type"] == "idle":
                i += 1
                continue

            cur_mt   = ev["movement_type"]
            cur_mode = ev["mode_index"]

            # Find the end of this phase (same movement_type + mode_index)
            j = i + 1
            while (j < len(self._events) and
                   self._events[j]["movement_type"] == cur_mt and
                   self._events[j]["mode_index"]    == cur_mode):
                j += 1

            ev_end   = self._events[j - 1]
            t_start  = ev["t"]     - t0
            t_end    = ev_end["t"] - t0
            duration = t_end - t_start

            if duration < self.MIN_SEGMENT_SEC:
                i = j
                continue

            delta_deg = math.degrees(ev_end["facing_angle"] - ev["facing_angle"])
            if abs(delta_deg) >= self.TURN_THRESHOLD_DEG:
                turn_dir = "left" if delta_deg > 0 else "right"
                turn_deg = round(abs(delta_deg), 1)
            else:
                turn_dir = None
                turn_deg = None

            segments.append({
                "idx":          seg_idx,
                "mode":         _mode_index_to_key(cur_mode),
                "movement":     cur_mt,
                "turn_dir":     turn_dir,
                "turn_deg":     turn_deg,
                "duration_sec": round(duration, 3),
                "speed":        ev["speed"],
                "t_start":      round(t_start, 3),
                "t_end":        round(t_end, 3),
                "frame_start":  int(round(t_start * 50)),
                "frame_end":    int(round(t_end * 50)),
            })
            seg_idx += 1
            i = j

        return segments


_session_recorder = KeyboardSessionRecorder()


# ---------------------------------------------------------------------------
# Shared planner state
# ---------------------------------------------------------------------------
class PlannerState:
    def __init__(self):
        self.active        = False
        self.motion_set_idx = 0
        self.mode_key      = 1
        self.facing_angle  = 0.0
        self.measured_yaw: float | None = None
        self.momentum      = 0.0
        self.speed         = -1.0
        self.height        = -1.0
        self.movement_type = "idle"
        # Robot state (updated from ZMQ debug stream)
        self.raw_qpos: list | None        = None   # 36 floats: measured base(3)+quat(4)+joints(29)
        self.raw_qpos_target: list | None = None   # 36 floats: kinematic target pose
        self.measured_speed: float        = 0.0    # horizontal speed magnitude from g1_debug (m/s)
        # Recipe executor flag — pauses planner_loop while recipe runs
        self.recipe_running: bool = False

    @property
    def current_set(self):
        return MOTION_SETS[self.motion_set_idx % len(MOTION_SETS)]

    @property
    def current_mode_info(self):
        for m in self.current_set["modes"]:
            if m["key"] == self.mode_key:
                return m
        return self.current_set["modes"][0]

    @property
    def planner_mode_index(self):
        return self.current_mode_info["index"]

    def get_movement_vector(self):
        a = self.facing_angle
        if self.momentum < 0.1:
            return [0.0, 0.0, 0.0]
        vectors = {
            "forward":      [ math.cos(a),  math.sin(a), 0.0],
            "backward":     [-math.cos(a), -math.sin(a), 0.0],
            "strafe_left":  [-math.sin(a),  math.cos(a), 0.0],
            "strafe_right": [ math.sin(a), -math.cos(a), 0.0],
        }
        return vectors.get(self.movement_type, [0.0, 0.0, 0.0])

    def get_facing_vector(self):
        return [math.cos(self.facing_angle), math.sin(self.facing_angle), 0.0]

    def decay_momentum(self):
        self.momentum *= 0.999
        if self.momentum < 0.1:
            self.momentum = 0.0
            self.movement_type = "idle"

    def clamp_speed(self):
        sr = self.current_mode_info.get("speed_range")
        if sr and self.speed > 0:
            self.speed = max(sr[0], min(sr[1], self.speed))

    def to_dict(
        self,
        include_motion_sets: bool = False,
        include_qpos_target: bool = True,
    ) -> dict:
        mi = self.current_mode_info
        payload = {
            "type": "state",
            "active":        self.active,
            "motionSet":     self.current_set["name"],
            "motionSetIdx":  self.motion_set_idx,
            "modeName":      mi["name"],
            "modeKey":       self.mode_key,
            "modeIndex":     mi["index"],
            "facingAngleDeg": round(math.degrees(self.facing_angle), 1),
            "measuredYawDeg": (
                round(math.degrees(self.measured_yaw), 1)
                if self.measured_yaw is not None else None
            ),
            "momentum":      round(self.momentum, 3),
            "speed":         round(self.speed, 2),
            "height":        round(self.height, 2),
            "movementType":  self.movement_type,
            "measuredSpeed": self.measured_speed,
            "speedRange":    mi.get("speed_range"),
            "heightControl": self.current_set.get("height_control", False),
            "qpos":          self.raw_qpos,
        }
        if include_qpos_target:
            payload["qpos_target"] = self.raw_qpos_target
        if include_motion_sets:
            payload["motionSets"] = [
                {
                    "name":           s["name"],
                    "height_control": s.get("height_control", False),
                    "modes": [
                        {
                            "key":           m["key"],
                            "name":          m["name"],
                            "speed_range":   m.get("speed_range"),
                            "default_speed": m.get("default_speed", -1.0),
                        }
                        for m in s["modes"]
                    ],
                }
                for s in MOTION_SETS
            ]
        return payload


state = PlannerState()

# ---------------------------------------------------------------------------
# ZMQ (async)
# ---------------------------------------------------------------------------
zmq_ctx:  zmq.asyncio.Context      = zmq.asyncio.Context()
zmq_pub:  zmq.asyncio.Socket | None = None
zmq_sub:  zmq.asyncio.Socket | None = None

connected_ws: set[WebSocket] = set()


def _free_port(port: int) -> bool:
    """Kill any process listening on *port* (TCP). Returns True if anything was killed."""
    import subprocess
    result = subprocess.run(
        ["lsof", "-ti", f"tcp:{port}"],
        capture_output=True, text=True,
    )
    pids = [p.strip() for p in result.stdout.splitlines() if p.strip()]
    if not pids:
        return False
    for pid in pids:
        try:
            subprocess.run(["kill", "-9", pid], check=False)
            print(f"  Killed stale process PID {pid} that was holding port {port}")
        except Exception:
            pass
    return True


async def init_zmq(pub_port: int, sub_port: int):
    global zmq_pub, zmq_sub

    zmq_pub = zmq_ctx.socket(zmq.PUB)
    try:
        zmq_pub.bind(f"tcp://*:{pub_port}")
    except zmq.ZMQError as e:
        if "Address already in use" not in str(e):
            raise
        print(f"  Port {pub_port} in use — killing stale process and retrying…")
        if _free_port(pub_port):
            import asyncio as _asyncio
            await _asyncio.sleep(0.5)   # give kernel time to release the socket
            zmq_pub.bind(f"tcp://*:{pub_port}")
        else:
            raise RuntimeError(
                f"Port {pub_port} is already in use and no owning process was found.\n"
                f"Run:  lsof -i tcp:{pub_port}  to investigate."
            ) from e
    print(f"  ZMQ PUB  → tcp://*:{pub_port}  (sends planner commands to C++ controller)")

    zmq_sub = zmq_ctx.socket(zmq.SUB)
    zmq_sub.connect(f"tcp://127.0.0.1:{sub_port}")
    zmq_sub.setsockopt(zmq.SUBSCRIBE, b"g1_debug")
    # RCVTIMEO is for blocking sockets; zmq.asyncio uses the event loop — omit it.
    print(f"  ZMQ SUB  ← tcp://127.0.0.1:{sub_port}  (reads robot state from C++ controller)")


# ---------------------------------------------------------------------------
# Broadcast state to all browser clients
# ---------------------------------------------------------------------------
async def broadcast_state():
    msg = json.dumps(state.to_dict())
    dead: set[WebSocket] = set()
    for ws in connected_ws:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    connected_ws.difference_update(dead)


# ---------------------------------------------------------------------------
# Planner send loop — 20 Hz
#
# Always sends planner messages (even when user hasn't pressed Enter yet) to
# prevent the C++ controller's safety-reset timeout from triggering.
# Sends the START+PLANNER activate command once on startup.
# ---------------------------------------------------------------------------
async def planner_loop():
    await asyncio.sleep(2.0)   # let ZMQ PUB/SUB subscribers connect

    # Activate the planner on the C++ side once
    if zmq_pub:
        await zmq_pub.send(build_command_message(start=True, stop=False, planner=True))
        print("  Sent START+PLANNER command to C++ controller")

    hz = 20
    while True:
        t0 = time.monotonic()

        # Yield the planner commands to the recipe executor while it runs,
        # but still broadcast qpos so the browser renderer stays live.
        if state.recipe_running:
            await broadcast_state()
            await asyncio.sleep(1.0 / hz)
            continue

        state.decay_momentum()
        mv  = state.get_movement_vector()
        fv  = state.get_facing_vector()

        # Only send non-idle commands when the user has pressed Enter (active)
        if state.active and state.momentum >= 0.1:
            mode = state.planner_mode_index
        else:
            mode = MODE_IDLE

        msg = build_planner_message(mode, mv, fv, state.speed, state.height)
        if zmq_pub:
            await zmq_pub.send(msg)

        await broadcast_state()
        await asyncio.sleep(max(0.0, 1.0 / hz - (time.monotonic() - t0)))


# ---------------------------------------------------------------------------
# ZMQ debug subscriber — reads qpos at ~50 Hz
# ---------------------------------------------------------------------------
async def debug_sub_loop():
    try:
        import msgpack
    except ImportError:
        print("WARNING: msgpack not installed — robot state will not be received.\n"
              "         Install with:  pip install msgpack")
        return

    while True:
        if zmq_sub is None:
            await asyncio.sleep(0.5)
            continue
        try:
            # Use true async recv (no NOBLOCK) so this coroutine suspends until a
            # message arrives instead of polling.  This prevents ZMQ queue buildup:
            # with NOBLOCK + sleep(0.02) the sleep is imprecise under load and
            # messages accumulate, causing ever-growing render latency.
            message = await asyncio.wait_for(zmq_sub.recv(), timeout=0.5)
            payload = message.split(b"g1_debug", 1)[1]
            data    = msgpack.unpackb(payload)

            # ── Measured fields: actual physics state ──────────────────────────
            # body_q_measured is absolute (C++ already adds default_angles), MuJoCo order.
            # base_quat_measured is real IMU quaternion.
            # base_trans_measured is hardcoded {0,-1,0.793} in C++ (no real odometry).
            body_q     = np.array(data.get("body_q_measured",
                                           DEFAULT_ANGLES.tolist()), dtype=float)
            quat_wxyz  = list(data.get("base_quat_measured", [1.0, 0.0, 0.0, 0.0]))
            base_trans = list(data.get("base_trans_measured", [0.0, 0.0, 0.793]))

            # ── Target/kinematic fields: reference pose from motion sequence ──
            body_q_tgt  = np.array(data.get("body_q_target",
                                            DEFAULT_ANGLES.tolist()), dtype=float)
            quat_tgt    = list(data.get("base_quat_target",  [1.0, 0.0, 0.0, 0.0]))
            trans_tgt   = list(data.get("base_trans_target", [0.0, 0.0, 0.793]))

            now = time.time()
            frame = {
                "t":                     now,
                "base_trans_measured":   base_trans,
                "base_quat_measured":    quat_wxyz,
                "body_q_measured":       body_q.tolist(),
                "base_lin_vel_measured": data.get("base_lin_vel_measured", []),
                "base_trans_target":     trans_tgt,
                "base_quat_target":      quat_tgt,
                "body_q_target":         body_q_tgt.tolist(),
            }

            # Track latest measured base yaw so recipe turns can start from
            # the robot's current real orientation.
            state.measured_yaw = _yaw_from_quat_wxyz(quat_wxyz)

            # Capture frame for recipe recording
            if _recipe_recording:
                _recipe_rec_frames.append(frame)

            # Capture frame for keyboard session recording
            if _kb_session_recording:
                _kb_session_frames.append(frame)

            # qpos = [base_pos(3), base_quat_wxyz(4), joints(29)] = 36 floats
            state.raw_qpos        = [round(v, 6) for v in
                                     base_trans + quat_wxyz + body_q.tolist()]
            state.raw_qpos_target = [round(v, 6) for v in
                                     trans_tgt + quat_tgt + body_q_tgt.tolist()]

            global _recipe_first_linvel_seen_at, _recipe_first_target_change_at
            global _recipe_first_measured_change_at, _recipe_debug_frames_seen
            global _recipe_debug_frames_missing_linvel, _recipe_last_target_q
            global _recipe_last_measured_q, _recipe_first_target_root_change_at
            global _recipe_first_measured_root_change_at, _recipe_last_target_root
            global _recipe_last_measured_root, _recipe_debug_snapshot_count
            global _recipe_post_segment_frames_remaining

            if state.recipe_running:
                _recipe_debug_frames_seen += 1
                frame_mono = time.monotonic()
                snapshot = _make_recipe_snapshot(
                    data, frame_mono, base_trans, quat_wxyz, body_q, trans_tgt, quat_tgt, body_q_tgt
                )
                if _recipe_first_segment_at is None:
                    _recipe_pre_segment_buffer.append(snapshot)
                elif _recipe_post_segment_frames_remaining > 0:
                    _append_recipe_debug_json(snapshot)
                    _recipe_debug_snapshot_count += 1
                    _recipe_post_segment_frames_remaining -= 1

                lin_vel_raw = data.get("base_lin_vel_measured")
                if not lin_vel_raw:
                    lin_vel_raw = data.get("lin_vel_measured")
                if len(lin_vel_raw or []) >= 2:
                    if _recipe_first_linvel_seen_at is None:
                        _recipe_first_linvel_seen_at = time.monotonic()
                        _append_recipe_debug_line(
                            "  Recipe timing: first lin_vel field seen "
                            f"{_recipe_since_run(_recipe_first_linvel_seen_at)} after run_recipe; "
                            f"value={list(lin_vel_raw)[:3]}"
                        )
                else:
                    _recipe_debug_frames_missing_linvel += 1
                    if _recipe_debug_frames_missing_linvel in (1, 25, 100):
                        keys = sorted(list(data.keys()))[:24]
                        _append_recipe_debug_line(
                            "  Recipe timing: lin_vel missing in debug frame "
                            f"(count={_recipe_debug_frames_missing_linvel}); keys={keys}"
                        )

                target_delta = _recipe_joint_delta(body_q_tgt, _recipe_last_target_q)
                target_root_delta = _recipe_root_delta(trans_tgt, quat_tgt, _recipe_last_target_root)
                if (
                    _recipe_first_target_root_change_at is None
                    and _recipe_first_segment_at is not None
                    and target_root_delta > 1e-4
                ):
                    _recipe_first_target_root_change_at = time.monotonic()
                    _append_recipe_debug_line(
                        "  Recipe timing: first target root change "
                        f"{_recipe_since_run(_recipe_first_target_root_change_at)} after run_recipe, "
                        f"{_recipe_since_segment(_recipe_first_target_root_change_at)} after first segment "
                        f"(max_root_delta={target_root_delta:.6f})"
                    )
                if (
                    _recipe_first_target_change_at is None
                    and _recipe_first_segment_at is not None
                    and target_delta > 1e-4
                ):
                    _recipe_first_target_change_at = time.monotonic()
                    _append_recipe_debug_line(
                        "  Recipe timing: first target joint change "
                        f"{_recipe_since_run(_recipe_first_target_change_at)} after run_recipe, "
                        f"{_recipe_since_segment(_recipe_first_target_change_at)} after first segment "
                        f"(max_joint_delta={target_delta:.6f})"
                    )

                measured_delta = _recipe_joint_delta(body_q, _recipe_last_measured_q)
                measured_root_delta = _recipe_root_delta(base_trans, quat_wxyz, _recipe_last_measured_root)
                if (
                    _recipe_first_measured_root_change_at is None
                    and _recipe_first_segment_at is not None
                    and measured_root_delta > 1e-4
                ):
                    _recipe_first_measured_root_change_at = time.monotonic()
                    _append_recipe_debug_line(
                        "  Recipe timing: first measured root change "
                        f"{_recipe_since_run(_recipe_first_measured_root_change_at)} after run_recipe, "
                        f"{_recipe_since_segment(_recipe_first_measured_root_change_at)} after first segment "
                        f"(max_root_delta={measured_root_delta:.6f})"
                    )
                if (
                    _recipe_first_measured_change_at is None
                    and _recipe_first_segment_at is not None
                    and measured_delta > 1e-4
                ):
                    _recipe_first_measured_change_at = time.monotonic()
                    _append_recipe_debug_line(
                        "  Recipe timing: first measured joint change "
                        f"{_recipe_since_run(_recipe_first_measured_change_at)} after run_recipe, "
                        f"{_recipe_since_segment(_recipe_first_measured_change_at)} after first segment "
                        f"(max_joint_delta={measured_delta:.6f})"
                    )

                _recipe_last_target_q = body_q_tgt.copy()
                _recipe_last_measured_q = body_q.copy()
                _recipe_last_target_root = np.array(list(trans_tgt) + list(quat_tgt), dtype=float)
                _recipe_last_measured_root = np.array(list(base_trans) + list(quat_wxyz), dtype=float)

            # Real measured speed: horizontal magnitude of base linear velocity
            lin_vel = data.get("base_lin_vel_measured") or data.get("lin_vel_measured") or []
            if len(lin_vel) >= 2:
                state.measured_speed = round(float(np.linalg.norm(lin_vel[:2])), 3)
                global _recipe_first_motion_at
                if (
                    state.recipe_running
                    and _recipe_first_segment_at is not None
                    and _recipe_first_motion_at is None
                    and state.measured_speed > 0.05
                ):
                    _recipe_first_motion_at = time.monotonic()
                    since_run = (_recipe_first_motion_at - _recipe_run_started_at) if _recipe_run_started_at else None
                    since_segment = _recipe_first_motion_at - _recipe_first_segment_at
                    run_part = f"{since_run:.3f}s from run_recipe" if since_run is not None else "n/a from run_recipe"
                    _append_recipe_debug_line(
                        "  Recipe timing: first non-zero measured_speed "
                        f"({state.measured_speed:.3f} m/s) after {run_part}, "
                        f"{since_segment:.3f}s from first segment command"
                    )
        except (zmq.Again, asyncio.TimeoutError):
            # No message within timeout window — loop and wait again.
            pass
        except Exception as e:
            global _recipe_debug_exception_count
            if state.recipe_running:
                _recipe_debug_exception_count += 1
                if _recipe_debug_exception_count <= 3:
                    _append_recipe_debug_line(
                        f"  Recipe timing: debug_sub exception #{_recipe_debug_exception_count}: {type(e).__name__}: {e}"
                    )
            await asyncio.sleep(0.02)
        # No unconditional sleep here — with await recv() the coroutine already
        # yields to the event loop while waiting for the next message.


# ---------------------------------------------------------------------------
# Recipe execution helpers
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


def _mode_display_speed(mode: str, speed: float) -> str:
    """Return a human-readable speed string for a segment."""
    if speed >= 0:
        return f"{speed:.1f} m/s"
    # Look up the mode's default_speed from MOTION_SETS
    mode_int = _MODE_MAP[mode][0]
    for ms in MOTION_SETS:
        for m in ms["modes"]:
            if m["index"] == mode_int:
                ds = m.get("default_speed", -1.0)
                if ds and ds > 0:
                    return f"{ds:.1f} m/s (default)"
                return "default"
    return "default"


async def _execute_segment_async(
    mode: str,
    movement: str,
    turn_dir: str | None,
    turn_deg: float | None,
    duration: float,
    speed: float,
    height: float,
    planner_hz: float,
    facing_start: float,
):
    """
    Execute one recipe segment.
    Async generator — yields (step, total_steps, facing_angle) each tick
    so the caller can track intra-segment progress.
    Final value (via StopAsyncIteration) is the ending facing angle;
    use `async for` and capture facing from the last yielded tuple.
    """
    mode_int    = _MODE_MAP[mode][0]
    total_steps = max(1, int(planner_hz * duration))
    dt          = 1.0 / planner_hz

    if turn_dir and turn_deg:
        sign       = 1.0 if turn_dir == "left" else -1.0
        facing_end = facing_start + math.radians(sign * turn_deg)
    else:
        facing_end = facing_start

    current_facing = facing_start
    for step in range(total_steps):
        alpha          = step / max(total_steps - 1, 1)
        current_facing = facing_start + alpha * (facing_end - facing_start)
        mv  = _movement_vector(movement, current_facing)
        fv  = _facing_vector(current_facing)
        zmq_msg = build_planner_message(mode_int, mv, fv, speed, height)
        if zmq_pub:
            await zmq_pub.send(zmq_msg)
        yield step, total_steps, current_facing
        await asyncio.sleep(dt)


def _save_recipe_csv(frames: list[dict], output_dir: Path) -> Path | None:
    """Write trajectory_dynamic.csv from captured debug frames."""
    if not frames:
        return None
    path = output_dir / "trajectory_dynamic.csv"
    t0   = frames[0]["t"]
    cols = ["t", "base_x", "base_y", "base_z", "qw", "qx", "qy", "qz"]
    cols += [f"joint_{i}" for i in range(29)]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for frame in frames:
            trans  = frame.get("base_trans_measured", [0.0, 0.0, 0.793])
            quat   = frame.get("base_quat_measured",  [1.0, 0.0, 0.0, 0.0])
            joints = frame.get("body_q_measured",      [0.0] * 29)
            row    = [round(frame["t"] - t0, 4)]
            row   += [round(v, 6) for v in trans]
            row   += [round(v, 6) for v in quat]
            row   += [round(float(v), 6) for v in joints[:29]]
            writer.writerow(row)
    return path


def _save_target_csv(frames: list[dict], output_dir: Path) -> Path | None:
    """Write trajectory_kinematic.csv (kinematic reference) from captured debug frames."""
    if not frames:
        return None
    # Skip if no target data was recorded (C++ controller may not publish target fields)
    if not any(frame.get("body_q_target") for frame in frames):
        return None
    path = output_dir / "trajectory_kinematic.csv"
    t0    = frames[0]["t"]
    # Normalize horizontal position (x, y) relative to the first frame so that
    # the trajectory always starts at (0, 0, z) regardless of accumulated drift
    # from previous recipe runs.  z (height) is kept absolute.
    trans0 = frames[0].get("base_trans_target", [0.0, 0.0, 0.793])
    ox, oy = trans0[0], trans0[1]
    cols = ["t", "base_x", "base_y", "base_z", "qw", "qx", "qy", "qz"]
    cols += [f"joint_{i}" for i in range(29)]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for frame in frames:
            trans  = frame.get("base_trans_target", [0.0, 0.0, 0.793])
            quat   = frame.get("base_quat_target",  [1.0, 0.0, 0.0, 0.0])
            joints = frame.get("body_q_target",      [0.0] * 29)
            row    = [round(frame["t"] - t0, 4)]
            row   += [round(trans[0] - ox, 6), round(trans[1] - oy, 6), round(trans[2], 6)]
            row   += [round(v, 6) for v in quat]
            row   += [round(float(v), 6) for v in joints[:29]]
            writer.writerow(row)
    return path


async def handle_stop_keyboard_session(msg: dict, ws: WebSocket):
    """Stop the keyboard session recorder and generate annotation JSON + CSVs."""
    global _kb_session_recording, _kb_session_frames
    _session_recorder.stop()
    _kb_session_recording = False
    captured = list(_kb_session_frames)
    _kb_session_frames = []

    async def _send(obj: dict):
        try:
            await ws.send_text(json.dumps(obj))
        except Exception:
            pass

    segments = _session_recorder.build_segments()
    if not segments:
        await _send({
            "type":    "keyboard_session_error",
            "message": "No significant motion recorded. Move the robot for longer before stopping.",
        })
        return

    if not _MODE_MAP:
        await _send({
            "type":    "keyboard_session_error",
            "message": "Server could not import _MODE_MAP — annotation unavailable.",
        })
        return

    session_name = msg.get("name") or "keyboard_session"
    run_id       = uuid.uuid4().hex[:12]
    output_dir   = _RESULTS_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = compose_annotations(segments)
    total_dur   = segments[-1]["t_end"] if segments else 0.0

    ann_payload = {
        "schema_version":     "1.0",
        "session_name":       session_name,
        "session_type":       "keyboard",
        "total_duration_sec": round(total_dur, 3),
        "data_fps":           50,
        "segments":           segments,
        **annotations,
    }
    ann_path = output_dir / "annotation.json"
    ann_path.write_text(json.dumps(ann_payload, indent=2), encoding="utf-8")
    ann_url  = f"/results/{run_id}/annotation.json"

    # Save measured + target CSVs if frames were captured
    csv_url        = None
    csv_target_url = None
    if captured:
        p = _save_recipe_csv(captured, output_dir)
        if p:
            csv_url = f"/results/{run_id}/trajectory_dynamic.csv"
        p = _save_target_csv(captured, output_dir)
        if p:
            csv_target_url = f"/results/{run_id}/trajectory_kinematic.csv"

    await _send({
        "type":          "keyboard_session_result",
        "run_id":        run_id,
        "duration_sec":  round(total_dur, 3),
        "segment_count": len(segments),
        "ann_url":       ann_url,
        "csv_url":       csv_url,
        "csv_target_url": csv_target_url,
        "annotations":   annotations,
    })


async def handle_run_recipe(msg: dict, ws: WebSocket):
    """
    Validate → execute segments → annotate → optionally record CSV → return results.
    Runs as an asyncio Task. Handles CancelledError for clean abort.
    """
    global _recipe_task, _recipe_recording, _recipe_rec_frames
    global _recipe_run_started_at, _recipe_first_segment_at, _recipe_first_motion_at
    global _recipe_first_linvel_seen_at, _recipe_first_target_change_at
    global _recipe_first_measured_change_at, _recipe_debug_frames_seen
    global _recipe_debug_frames_missing_linvel, _recipe_last_target_q
    global _recipe_last_measured_q, _recipe_debug_exception_count
    global _recipe_first_target_root_change_at, _recipe_first_measured_root_change_at
    global _recipe_last_target_root, _recipe_last_measured_root
    global _recipe_debug_log_path, _recipe_debug_frames_path, _recipe_debug_snapshot_count
    global _recipe_post_segment_frames_remaining

    async def _send(obj: dict):
        try:
            await ws.send_text(json.dumps(obj))
        except Exception:
            pass

    # ── 1. Validate segments ────────────────────────────────────────────────
    raw_segments = msg.get("segments", [])
    if not raw_segments:
        await _send({"type": "recipe_error", "message": "No segments provided."})
        return

    if not _MODE_MAP:
        await _send({"type": "recipe_error",
                     "message": "Server could not import _MODE_MAP from compose_trajectory.py."})
        return

    parsed = []
    for i, s in enumerate(raw_segments):
        mode = s.get("mode", "walk")
        if mode not in _MODE_MAP:
            await _send({"type": "recipe_error",
                         "message": f"Segment {i}: unknown mode '{mode}'."})
            return
        turn      = s.get("turn")          # "left"|"right"|None
        direction = s.get("direction", "forward") if not turn else "forward"
        parsed.append({
            "idx":      i,
            "mode":     mode,
            "movement": direction,
            "turn_dir": turn,
            "turn_deg": float(s.get("angle", 90)) if turn else None,
            "duration": float(s.get("duration", 2.0)),
            "speed":    float(s.get("speed", -1.0)),
            "height":   float(s.get("height", -1.0)),
        })

    recipe_name = msg.get("name") or "web_recipe"
    init_wait   = float(msg.get("init_wait", 0.0))
    do_record   = bool(msg.get("record", True))
    planner_hz  = 20.0

    # ── 2. Setup output dir ─────────────────────────────────────────────────
    run_id     = uuid.uuid4().hex[:12]
    output_dir = _RESULTS_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    _recipe_debug_log_path = output_dir / "debug_timing.log"
    _recipe_debug_frames_path = output_dir / "debug_frames_head.jsonl"
    _recipe_debug_snapshot_count = 0
    _recipe_pre_segment_buffer.clear()
    _recipe_post_segment_frames_remaining = 0

    state.recipe_running = True
    _recipe_run_started_at = time.monotonic()
    _recipe_first_segment_at = None
    _recipe_first_motion_at = None
    _recipe_first_linvel_seen_at = None
    _recipe_first_target_change_at = None
    _recipe_first_measured_change_at = None
    _recipe_first_target_root_change_at = None
    _recipe_first_measured_root_change_at = None
    _recipe_debug_frames_seen = 0
    _recipe_debug_frames_missing_linvel = 0
    _recipe_last_target_q = None
    _recipe_last_measured_q = None
    _recipe_last_target_root = None
    _recipe_last_measured_root = None
    _recipe_debug_exception_count = 0
    _append_recipe_debug_line(
        f"  Recipe timing: run_recipe received name={recipe_name!r}, run_id={run_id}, "
        f"segments={len(parsed)}, init_wait={init_wait:.1f}s"
    )
    _append_recipe_debug_json({
        "type": "recipe_run_start",
        "run_id": run_id,
        "recipe_name": recipe_name,
        "segments": len(parsed),
        "init_wait": init_wait,
        "planner_hz": planner_hz,
        "output_dir": str(output_dir),
        "segments_spec": parsed,
    })

    try:
        # ── 3. Activate planner ─────────────────────────────────────────────
        if zmq_pub:
            await zmq_pub.send(build_command_message(start=True, stop=False, planner=True))

        # Progress allocation:  init_wait 0→15%,  execution 15→95%,  processing 95→100%
        PROG_INIT_END = 0.15
        PROG_EXEC_END = 0.95

        # Init-wait: send countdown every 0.5 s so the UI feels alive
        t_start_wait = time.monotonic()
        t_end_wait   = t_start_wait + init_wait
        while time.monotonic() < t_end_wait:
            remaining = max(0.0, t_end_wait - time.monotonic())
            frac      = 1.0 - remaining / max(init_wait, 0.001)
            await _send({
                "type":     "recipe_status",
                "phase":    "init_wait",
                "message":  f"Warming up controller… {remaining:.1f}s remaining",
                "progress": round(frac * PROG_INIT_END, 3),
            })
            await asyncio.sleep(min(0.5, remaining + 0.01))

        # ── 4. Start recording ──────────────────────────────────────────────
        if do_record:
            _recipe_rec_frames.clear()
            _recipe_recording = True

        # ── 5. Execute segments ─────────────────────────────────────────────
        # Start each recipe from the current measured yaw (fall back to the
        # last commanded facing when measured yaw is unavailable).
        facing_angle = state.measured_yaw if state.measured_yaw is not None else state.facing_angle
        state.facing_angle = facing_angle
        t_cursor     = 0.0
        total_dur    = sum(s["duration"] for s in parsed)
        # Keep progress updates lightweight so they do not compete with qpos streaming.
        _PROG_INTERVAL = max(1, int(planner_hz // 2))

        for spec in parsed:
            speed_str = _mode_display_speed(spec["mode"], spec["speed"])
            if spec["turn_dir"]:
                seg_desc = (f"{_MODE_MAP[spec['mode']][1]} + turn {spec['turn_dir']} "
                            f"{int(spec['turn_deg'] or 90)}°  {spec['duration']:.1f}s")
            else:
                seg_desc = (f"{_MODE_MAP[spec['mode']][1]} {spec['movement']} "
                            f"{spec['duration']:.1f}s  @{speed_str}")

            base_label = f"[{spec['idx']+1}/{len(parsed)}] {seg_desc}"

            # Send status at start of segment
            prog_start = PROG_INIT_END + (t_cursor / total_dur) * (PROG_EXEC_END - PROG_INIT_END)
            if _recipe_first_segment_at is None:
                _recipe_first_segment_at = time.monotonic()
                since_run = _recipe_first_segment_at - _recipe_run_started_at
                _append_recipe_debug_line(
                    "  Recipe timing: first segment command begins "
                    f"{since_run:.3f}s after run_recipe"
                )
                for snap in list(_recipe_pre_segment_buffer):
                    snap["t_from_first_segment_s"] = round((snap["t_from_run_s"] - since_run), 6) if snap["t_from_run_s"] is not None else None
                    snap["window"] = "pre_segment"
                    _append_recipe_debug_json(snap)
                    _recipe_debug_snapshot_count += 1
                _recipe_pre_segment_buffer.clear()
                _recipe_post_segment_frames_remaining = _RECIPE_DEBUG_POST_SEG_FRAMES
            await _send({
                "type":         "recipe_status",
                "phase":        "executing",
                "segment_idx":  spec["idx"],
                "total":        len(parsed),
                "message":      base_label,
                "progress":     round(prog_start, 3),
            })

            # Run segment — async generator yields (step, total_steps, facing) each tick
            async for step, total_steps, cur_facing in _execute_segment_async(
                mode=spec["mode"],
                movement=spec["movement"],
                turn_dir=spec["turn_dir"],
                turn_deg=spec["turn_deg"],
                duration=spec["duration"],
                speed=spec["speed"],
                height=spec["height"],
                planner_hz=planner_hz,
                facing_start=facing_angle,
            ):
                facing_angle = cur_facing
                state.facing_angle = cur_facing
                # Send smooth intra-segment progress every _PROG_INTERVAL ticks
                if step % _PROG_INTERVAL == 0:
                    seg_frac = step / max(total_steps - 1, 1)
                    t_elapsed = t_cursor + seg_frac * spec["duration"]
                    prog = PROG_INIT_END + (t_elapsed / total_dur) * (PROG_EXEC_END - PROG_INIT_END)
                    remaining = spec["duration"] * (1.0 - seg_frac)
                    await _send({
                        "type":     "recipe_status",
                        "phase":    "executing",
                        "message":  f"{base_label}  ({remaining:.1f}s left)",
                        "progress": round(prog, 3),
                    })

            t_cursor += spec["duration"]

        # Keep the final heading for subsequent segments/recipes and for the
        # planner loop after recipe execution.
        state.facing_angle = facing_angle

        # ── 6. Stop recording ───────────────────────────────────────────────
        _recipe_recording = False
        captured_frames   = list(_recipe_rec_frames)

        # Return to IDLE while preserving the current heading so playback
        # does not snap back to +X at the end of the recipe.
        idle_facing = _facing_vector(facing_angle)
        idle_msg = build_planner_message(
            MODE_IDLE, [0.0, 0.0, 0.0], idle_facing, -1.0, -1.0
        )
        for _ in range(int(planner_hz)):
            if zmq_pub:
                await zmq_pub.send(idle_msg)
            await asyncio.sleep(1.0 / planner_hz)

        await _send({"type": "recipe_status", "phase": "processing",
                     "message": "Processing results…", "progress": PROG_EXEC_END})

        # ── 7. Build annotation segments ────────────────────────────────────
        t_cur = 0.0
        ann_segments = []
        for spec in parsed:
            t_end_seg = t_cur + spec["duration"]
            ann_segments.append({
                "idx":          spec["idx"],
                "t_start":      round(t_cur, 3),
                "t_end":        round(t_end_seg, 3),
                "frame_start":  int(round(t_cur * 50)),
                "frame_end":    int(round(t_end_seg * 50)),
                "mode":         spec["mode"],
                "movement":     spec["movement"],
                "turn_deg":     spec["turn_deg"],
                "turn_dir":     spec["turn_dir"],
                "duration_sec": round(spec["duration"], 3),
                "speed":        spec["speed"],
            })
            t_cur = t_end_seg

        annotations = compose_annotations(ann_segments)

        # ── 8. Save files ────────────────────────────────────────────────────
        csv_url        = None
        csv_target_url = None
        if do_record:
            csv_path = _save_recipe_csv(captured_frames, output_dir)
            if csv_path:
                csv_url = f"/results/{run_id}/trajectory_dynamic.csv"
            tgt_path = _save_target_csv(captured_frames, output_dir)
            if tgt_path:
                csv_target_url = f"/results/{run_id}/trajectory_kinematic.csv"

        ann_payload = {
            "schema_version":    "1.0",
            "recipe_name":       recipe_name,
            "total_duration_sec": round(total_dur, 3),
            "data_fps":          50,
            "segments":          ann_segments,
            **annotations,
        }
        ann_path = output_dir / "annotation.json"
        ann_path.write_text(json.dumps(ann_payload, indent=2), encoding="utf-8")
        ann_url  = f"/results/{run_id}/annotation.json"

        # ── 9. Send results ──────────────────────────────────────────────────
        await _send({
            "type":          "recipe_result",
            "run_id":        run_id,
            "duration_sec":  round(total_dur, 3),
            "csv_url":       csv_url,
            "csv_target_url": csv_target_url,
            "ann_url":       ann_url,
            "annotations":   annotations,
        })
        _print_recipe_debug_summary()

    except asyncio.CancelledError:
        _recipe_recording = False
        if zmq_pub:
            idle_facing = _facing_vector(state.facing_angle)
            idle_msg = build_planner_message(
                MODE_IDLE, [0.0, 0.0, 0.0], idle_facing, -1.0, -1.0
            )
            await zmq_pub.send(idle_msg)
        await _send({"type": "recipe_error", "message": "Recipe cancelled.", "cancelled": True})
        _print_recipe_debug_summary("Recipe timing summary (cancelled)")
        raise

    except Exception as e:
        import traceback
        _recipe_recording = False
        await _send({"type": "recipe_error", "message": str(e),
                     "detail": traceback.format_exc()})
        _print_recipe_debug_summary("Recipe timing summary (error)")

    finally:
        state.recipe_running = False
        _recipe_task = None
        _recipe_run_started_at = None
        _recipe_first_segment_at = None
        _recipe_first_motion_at = None
        _recipe_first_linvel_seen_at = None
        _recipe_first_target_change_at = None
        _recipe_first_measured_change_at = None
        _recipe_first_target_root_change_at = None
        _recipe_first_measured_root_change_at = None
        _recipe_debug_frames_seen = 0
        _recipe_debug_frames_missing_linvel = 0
        _recipe_last_target_q = None
        _recipe_last_measured_q = None
        _recipe_last_target_root = None
        _recipe_last_measured_root = None
        _recipe_debug_exception_count = 0
        _recipe_debug_log_path = None
        _recipe_debug_frames_path = None
        _recipe_debug_snapshot_count = 0
        _recipe_pre_segment_buffer.clear()
        _recipe_post_segment_frames_remaining = 0


# ---------------------------------------------------------------------------
# Keyboard event handler
# ---------------------------------------------------------------------------
def handle_key(key: str, pressed: bool):
    """
    Process keyboard events forwarded from the browser.
    Enter toggles whether movement commands are sent (active/idle).
    Movement keys are only acted on when active.
    Mode, speed, and height are controlled via UI widgets.
    """
    if not pressed:
        return

    # Enter → toggle movement on/off (planner always keeps running)
    if key == "Enter":
        state.active = not state.active
        print(f"  Controller {'ACTIVE' if state.active else 'IDLE'}")
        _session_recorder.on_state_change()
        return

    if not state.active:
        return

    movement_keys = {
        "w": "forward",
        "s": "backward",
        ",": "strafe_left",
        ".": "strafe_right",
    }
    if key.lower() in movement_keys:
        state.movement_type = movement_keys[key.lower()]
        state.momentum = 1.0
        _session_recorder.on_state_change()
        return

    if key.lower() == "a":
        state.facing_angle += 0.1
        state.movement_type = "forward"
        state.momentum = 1.0
        _session_recorder.on_state_change()
        return
    if key.lower() == "d":
        state.facing_angle -= 0.1
        state.movement_type = "forward"
        state.momentum = 1.0
        _session_recorder.on_state_change()
        return

    # Snap heading ±30°
    if key.lower() == "q":
        state.facing_angle += math.pi / 6
        _session_recorder.on_state_change()
        return
    if key.lower() == "e":
        state.facing_angle -= math.pi / 6
        _session_recorder.on_state_change()
        return

    # Stop movement
    if key.lower() == "r":
        state.momentum = 0.0
        state.movement_type = "idle"
        _session_recorder.on_state_change()
        return


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application: FastAPI):
    pub_port = application.state.zmq_pub_port
    sub_port = application.state.zmq_sub_port
    await init_zmq(pub_port, sub_port)
    t1 = asyncio.create_task(planner_loop())
    t2 = asyncio.create_task(debug_sub_loop())
    yield
    t1.cancel()
    t2.cancel()
    if zmq_pub:
        zmq_pub.close()
    if zmq_sub:
        zmq_sub.close()
    zmq_ctx.term()


app = FastAPI(title="G1 WASM Demo — WebSocket Bridge", lifespan=lifespan)

# Allow the Vite dev server (any port on localhost) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global _recipe_task, _recipe_recording, _kb_session_recording, _kb_session_frames
    await ws.accept()
    connected_ws.add(ws)
    # Send current state immediately on connect
    await ws.send_text(json.dumps(
        state.to_dict(include_motion_sets=True, include_qpos_target=True)
    ))
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            t   = msg.get("type")
            if t == "key":
                handle_key(msg["key"], msg.get("pressed", True))
            elif t == "set_speed":
                state.speed = float(msg["value"])
                state.clamp_speed()
            elif t == "set_height":
                state.height = float(msg["value"])
            elif t == "set_mode":
                state.motion_set_idx = int(msg["setIdx"])
                state.mode_key       = int(msg["modeKey"])
                mi = state.current_mode_info
                state.speed  = mi["default_speed"] if mi.get("speed_range") else -1.0
                state.height = 0.8 if state.current_set.get("height_control") else -1.0
            elif t == "reset":
                # Cancel any running recipe task
                if _recipe_task and not _recipe_task.done():
                    _recipe_task.cancel()
                _recipe_recording    = False
                state.recipe_running = False
                # Stop all motion and return to idle, reset all planner state
                state.active         = False
                state.momentum       = 0.0
                state.movement_type  = "idle"
                state.facing_angle   = 0.0
                state.measured_yaw   = None
                state.motion_set_idx = 0
                state.mode_key       = 1
                state.speed          = -1.0
                state.height         = -1.0
                # Clear cached robot state so the browser doesn't receive stale
                # accumulated target/measured poses after reset.
                state.raw_qpos        = None
                state.raw_qpos_target = None
                # Signal run_sim_loop.py to call mj_resetData (resets robot to init pos)
                Path("/tmp/gear_sonic_sim_reset").touch()
                # Send STOP first so the C++ kinematic sequencer can reset its
                # internal target position, then re-activate the planner.
                if zmq_pub:
                    await zmq_pub.send(build_command_message(start=False, stop=True, planner=False))
                    await asyncio.sleep(0.05)
                    await zmq_pub.send(build_command_message(start=True, stop=False, planner=True))
                print("  Reset: controller stop+restart sent, sim reset signal sent")
                # If a keyboard session was recording without an explicit stop, discard frames
                # (the client's resetScene() already sends stop_keyboard_session first)
                if _kb_session_recording and not _session_recorder.is_recording:
                    _kb_session_recording = False
                    _kb_session_frames.clear()
            elif t == "run_recipe":
                if _recipe_task and not _recipe_task.done():
                    await ws.send_text(json.dumps({
                        "type": "recipe_error",
                        "message": "A recipe is already running. Cancel it first.",
                    }))
                else:
                    _recipe_task = asyncio.create_task(handle_run_recipe(msg, ws))
            elif t == "cancel_recipe":
                if _recipe_task and not _recipe_task.done():
                    _recipe_task.cancel()
                _recipe_recording    = False
                state.recipe_running = False
            elif t == "start_keyboard_session":
                _session_recorder.start()
                _kb_session_frames = []
                _kb_session_recording = True
                await ws.send_text(json.dumps({"type": "keyboard_session_started"}))
            elif t == "stop_keyboard_session":
                asyncio.create_task(handle_stop_keyboard_session(msg, ws))
    except WebSocketDisconnect:
        # If client disconnects during a recipe run, cancel it
        if _recipe_task and not _recipe_task.done():
            _recipe_task.cancel()
    finally:
        connected_ws.discard(ws)


# ---------------------------------------------------------------------------
# HTTP: download recipe results (CSV / annotation JSON)
# ---------------------------------------------------------------------------

@app.get("/results/{run_id}/{filename}")
async def download_result(run_id: str, filename: str):
    """Serve a generated recipe result file (CSV or JSON)."""
    # Basic path-traversal guard
    safe_name = Path(filename).name
    if safe_name != filename or ".." in run_id:
        raise HTTPException(status_code=400, detail="Invalid path.")
    path = _RESULTS_DIR / run_id / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(str(path), filename=safe_name)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
# Scene XML discovery (mirrors run_sim_loop / web_demo/server.py logic)
# ---------------------------------------------------------------------------
def _find_scene_xml() -> Path | None:
    script_dir = Path(__file__).resolve().parent.parent
    wbc_root = script_dir / "GR00T-WholeBodyControl"

    # 1. Follow the same ROBOT_SCENE key used by the WBC yaml
    wbc_yaml = (
        wbc_root
        / "gear_sonic"
        / "utils"
        / "mujoco_sim"
        / "wbc_configs"
        / "g1_29dof_sonic_model12.yaml"
    )
    print("wbc_yaml:", wbc_yaml)
    if wbc_yaml.exists():
        print("yes")
        try:
            cfg   = yaml.safe_load(wbc_yaml.read_text(encoding="utf-8"))
            value = cfg.get("ROBOT_SCENE") if isinstance(cfg, dict) else None
            if value:
                scene_path = wbc_root / str(value)
                if scene_path.exists():
                    return scene_path
        except Exception:
            pass

    # 2. Backward-compatible fallback (previous tooling)
    g1_dir = script_dir / "g1"
    for name in ("scene_43dof.xml", "scene.xml"):
        xml_path = g1_dir / name
        if xml_path.exists():
            return xml_path
    candidates = sorted(g1_dir.glob("scene*.xml"))
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="WebSocket ↔ ZMQ bridge for the G1 WASM web demo"
    )
    parser.add_argument("--port",         type=int, default=8080,
                        help="WebSocket server port (default: 8080)")
    parser.add_argument("--host",         default="0.0.0.0",
                        help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--zmq-pub-port", type=int, default=5556,
                        help="ZMQ PUB port for planner commands (default: 5556)")
    parser.add_argument("--zmq-sub-port", type=int, default=5557,
                        help="ZMQ SUB port for robot debug stream (default: 5557)")
    parser.add_argument("--scene",        type=Path, default=None,
                        help="Path to MuJoCo scene XML (auto-detected from WBC yaml if omitted)")
    args = parser.parse_args()

    # Resolve scene XML
    scene_xml = args.scene or _find_scene_xml()
    if scene_xml is None:
        print("WARNING: Could not find scene XML. Robot visualisation may fail.")
    else:
        print(f"  Scene XML: {scene_xml}")
    app.state.scene_xml    = scene_xml
    app.state.zmq_pub_port = args.zmq_pub_port
    app.state.zmq_sub_port = args.zmq_sub_port

    print(f"""
  ┌─────────────────────────────────────────────────────┐
  │        G1 WASM Demo  —  WebSocket Bridge            │
  ├─────────────────────────────────────────────────────┤
  │  WebSocket  ws://localhost:{args.port}/ws             │
  │                                                     │
  │  Quick start:                                       │
  │    1. bash deploy.sh sim --input-type zmq_manager   │
  │    2. Press '9' in MuJoCo window                    │
  │    3. npm run dev   (in web_wasm_demo/)             │
  │    4. Open http://localhost:5173                    │
  │                                                     │
  │  Keyboard: Enter=start  WASD=move  QE=turn  R=stop  │
  └─────────────────────────────────────────────────────┘
""")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
