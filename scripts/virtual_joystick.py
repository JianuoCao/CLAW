#!/usr/bin/env python3
"""
Virtual joystick script for pre-defined motion playback via zmq_manager.

In addition to sending planner commands, this script can automatically export
a simulation replay package after motion execution:

- `trajectory_target.csv`: [root_pos xyz, root_quat xyzw, 29 joint angles]
- `trajectory_measured.csv`: same schema using measured fields when available
- `debug_records.jsonl`: raw debug output from `--zmq-out-topic`
- `metadata.json`: run configuration and recording stats
- `video.mp4` (optional): rendered with MuJoCo tracked camera
"""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime
from typing import Callable

import numpy as np
import zmq
import yaml

BODY_29DOF_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

HAND_JOINT_NAMES = [
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
]

# Add path to import the ZMQ utility functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gear_sonic', 'utils', 'teleop', 'zmq'))

try:
    from zmq_planner_sender import build_command_message, build_planner_message
except ImportError:
    # Fallback: define the functions inline if import fails
    HEADER_SIZE = 1024
    
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
    
    def build_command_message(start: bool, stop: bool, planner: bool, delta_heading=None) -> bytes:
        fields = [
            {"name": "start", "dtype": "u8", "shape": [1]},
            {"name": "stop", "dtype": "u8", "shape": [1]},
            {"name": "planner", "dtype": "u8", "shape": [1]},
        ]
        payload = b"".join((
            struct.pack("B", 1 if start else 0),
            struct.pack("B", 1 if stop else 0),
            struct.pack("B", 1 if planner else 0),
        ))
        if delta_heading is not None:
            fields.append({"name": "delta_heading", "dtype": "f32", "shape": [1]})
            payload += struct.pack("<f", float(delta_heading))
        header = _build_header(fields, version=1, count=1)
        return b"command" + header + payload
    
    def build_planner_message(mode: int, movement, facing, speed: float = -1.0, height: float = -1.0) -> bytes:
        if len(movement) != 3 or len(facing) != 3:
            raise ValueError("movement and facing must have length 3")
        fields = [
            {"name": "mode", "dtype": "i32", "shape": [1]},
            {"name": "movement", "dtype": "f32", "shape": [3]},
            {"name": "facing", "dtype": "f32", "shape": [3]},
            {"name": "speed", "dtype": "f32", "shape": [1]},
            {"name": "height", "dtype": "f32", "shape": [1]},
        ]
        payload = b"".join((
            struct.pack("<i", int(mode)),
            struct.pack("<fff", float(movement[0]), float(movement[1]), float(movement[2])),
            struct.pack("<fff", float(facing[0]), float(facing[1]), float(facing[2])),
            struct.pack("<f", float(speed)),
            struct.pack("<f", float(height)),
        ))
        header = _build_header(fields, version=1, count=1)
        return b"planner" + header + payload

# Optional dependency used only for recording debug stream.
try:
    import msgpack
except ImportError:
    msgpack = None

# -- Enums matching the C++ LocomotionMode (from localmotion_kplanner.hpp) ---
MODE_IDLE = 0
# Locomotion
MODE_SLOW_WALK = 1   # 0.1m/s ~ 0.8m/s
MODE_WALK = 2        # 0.8m/s ~ 2.5m/s (Standard walking mode)
MODE_RUN = 3         # 2.5m/s ~ 7.5m/s
MODE_HAPPY = 17
MODE_STEALTH = 18
MODE_INJURED = 19
# Squat / Ground
MODE_SQUAT = 4
MODE_KNEEL_TWO = 5
MODE_KNEEL_ONE = 6
MODE_HAND_CRAWL = 8
MODE_ELBOW_CRAWL = 14
# Boxing
MODE_IDLE_BOXING = 9
MODE_WALK_BOXING = 10
MODE_LEFT_JAB = 11
MODE_RIGHT_JAB = 12
MODE_RANDOM_PUNCHES = 13
MODE_LEFT_HOOK = 15
MODE_RIGHT_HOOK = 16
# Styled Walking
MODE_CAREFUL = 20
MODE_OBJECT_CARRYING = 21
MODE_CROUCH = 22
MODE_HAPPY_DANCE = 23
MODE_ZOMBIE = 24
MODE_POINT = 25
MODE_SCARED = 26

def _find_scene_xml():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    # Match run_sim_loop default scene source (wbc yaml ROBOT_SCENE) first.
    wbc_yaml = (
        repo_root
        / "gear_sonic"
        / "utils"
        / "mujoco_sim"
        / "wbc_configs"
        / "g1_29dof_sonic_model12.yaml"
    )
    if wbc_yaml.exists():
        try:
            cfg = yaml.safe_load(wbc_yaml.read_text(encoding="utf-8"))
            value = cfg.get("ROBOT_SCENE") if isinstance(cfg, dict) else None
            if value:
                scene_path = repo_root / str(value)
                if scene_path.exists():
                    return scene_path
        except Exception:
            pass

    # Backward-compatible fallback used by previous tooling.
    g1_dir = script_dir / "g1"
    for name in ("scene_43dof.xml", "scene.xml"):
        xml_path = g1_dir / name
        if xml_path.exists():
            return xml_path
    candidates = sorted(g1_dir.glob("scene*.xml"))
    return candidates[0] if candidates else None


class DebugRecorder:
    """Collects debug frames from g1_deploy ZMQ output."""

    def __init__(self, context: zmq.Context, zmq_url: str, topic: str):
        if msgpack is None:
            raise RuntimeError("msgpack is required for recording. Install with: pip install msgpack")

        self.topic = topic.encode("utf-8")
        self.socket = context.socket(zmq.SUB)
        self.socket.connect(zmq_url)
        self.socket.setsockopt(zmq.SUBSCRIBE, self.topic)
        self.socket.setsockopt(zmq.RCVTIMEO, 1)

        self.records = []
        self.target_rows = []
        self.measured_rows = []
        self._record_start = time.time()
        self._msgpack_unpack: Callable[[bytes], dict] = msgpack.unpackb

    def _to_xyzw(self, quat_wxyz):
        if len(quat_wxyz) != 4:
            return None
        return [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]

    def _unpack(self, message: bytes):
        if not message.startswith(self.topic):
            return None
        payload = message[len(self.topic):]
        return self._msgpack_unpack(payload)

    def poll(self):
        polled = 0
        while True:
            try:
                message = self.socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

            data = self._unpack(message)
            if not isinstance(data, dict):
                continue

            ts = time.time() - self._record_start
            self.records.append({"t": ts, "data": data})
            polled += 1

            base_pos_target = data.get("base_trans_target")
            base_quat_target = self._to_xyzw(data.get("base_quat_target", []))
            body_q_target = data.get("body_q_target")
            if (
                isinstance(base_pos_target, list)
                and len(base_pos_target) == 3
                and base_quat_target is not None
                and isinstance(body_q_target, list)
                and len(body_q_target) == 29
            ):
                self.target_rows.append(base_pos_target + base_quat_target + body_q_target)

            base_pos_measured = data.get("base_trans_measured")
            base_quat_measured = self._to_xyzw(data.get("base_quat_measured", []))
            body_q_measured = data.get("body_q_measured")
            if (
                isinstance(base_pos_measured, list)
                and len(base_pos_measured) == 3
                and base_quat_measured is not None
                and isinstance(body_q_measured, list)
                and len(body_q_measured) == 29
            ):
                self.measured_rows.append(base_pos_measured + base_quat_measured + body_q_measured)

        return polled

    def close(self):
        try:
            self.socket.close(0)
        except Exception:
            pass


def _build_qpos_addr_map(model):
    qpos_addr = {}
    for name in BODY_29DOF_JOINT_NAMES + HAND_JOINT_NAMES:
        try:
            jid = model.joint(name).id
            qpos_addr[name] = model.jnt_qposadr[jid]
        except Exception:
            continue
    return qpos_addr


def _set_robot_qpos_from_row(mj_data, row, qpos_addr):
    root_pos = row[0:3]
    quat_xyzw = row[3:7]
    joints = row[7:36]
    quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

    mj_data.qpos[:3] = root_pos
    mj_data.qpos[3:7] = quat_wxyz
    mj_data.qvel[:] = 0.0
    mj_data.qacc[:] = 0.0

    for idx, joint_name in enumerate(BODY_29DOF_JOINT_NAMES):
        addr = qpos_addr.get(joint_name)
        if addr is not None:
            mj_data.qpos[addr] = joints[idx]

    # Trajectory data is body-only (29-DoF), so keep finger joints neutral.
    for joint_name in HAND_JOINT_NAMES:
        addr = qpos_addr.get(joint_name)
        if addr is not None:
            mj_data.qpos[addr] = 0.0


def render_video(csv_path, data_fps=50, video_fps=25, width=960, height=540):
    """Render qpos trajectory CSV to MP4 using MuJoCo + ffmpeg pipe."""
    def _import_mujoco_with_backend(backend: str):
        previous = os.environ.get("MUJOCO_GL")
        os.environ["MUJOCO_GL"] = backend
        try:
            import importlib
            import mujoco as mujoco_module
            return importlib.reload(mujoco_module)
        except Exception:
            if previous is None:
                os.environ.pop("MUJOCO_GL", None)
            else:
                os.environ["MUJOCO_GL"] = previous
            return None

    # Try currently configured backend first, then common headless options.
    tried = []
    mujoco = None
    backend_candidates = []
    current_backend = os.environ.get("MUJOCO_GL")
    if current_backend:
        backend_candidates.append(current_backend)
    backend_candidates.extend(["egl", "osmesa", "glfw"])

    for backend in backend_candidates:
        if backend in tried:
            continue
        tried.append(backend)
        mujoco = _import_mujoco_with_backend(backend)
        if mujoco is not None:
            break

    if mujoco is None:
        print(
            "mujoco unavailable for rendering (tried MUJOCO_GL backends: "
            f"{', '.join(tried)}), skipping MP4 render."
        )
        return None

    if not shutil.which("ffmpeg"):
        print("ffmpeg not found in PATH, skipping MP4 render.")
        return None

    data = np.loadtxt(csv_path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 36:
        print(f"Expected 36 columns in {csv_path}, got {data.shape[1]}. Skipping MP4 render.")
        return None

    scene_xml = _find_scene_xml()
    print(f"Using MuJoCo model XML for rendering: {scene_xml}")
    if scene_xml is None:
        print("Cannot find scene XML under gear_sonic_deploy/g1, skipping MP4 render.")
        return None

    step = max(1, round(float(data_fps) / float(video_fps)))
    frames = data[::step]
    actual_fps = float(data_fps) / step

    mj_model_cls = getattr(mujoco, "MjModel")
    mj_data_cls = getattr(mujoco, "MjData")
    mjv_camera_cls = getattr(mujoco, "MjvCamera")
    mjt_camera = getattr(mujoco, "mjtCamera")
    mj_forward = getattr(mujoco, "mj_forward")

    try:
        mj_model = mj_model_cls.from_xml_path(str(scene_xml))
    except Exception as e:
        print(f"Failed to load MuJoCo model for rendering: {e}")
        return None
    mj_model.vis.global_.offwidth = max(width, mj_model.vis.global_.offwidth)
    mj_model.vis.global_.offheight = max(height, mj_model.vis.global_.offheight)
    mj_data = mj_data_cls(mj_model)
    try:
        renderer = mujoco.Renderer(mj_model, height=height, width=width)
    except Exception as e:
        print(f"Failed to create MuJoCo renderer (headless OpenGL issue): {e}")
        print("MP4 render skipped. CSV/JSON outputs are still saved.")
        return None

    cam = mjv_camera_cls()
    cam.type = mjt_camera.mjCAMERA_TRACKING
    try:
        cam.trackbodyid = mj_model.body("pelvis").id
    except Exception:
        cam.trackbodyid = 1
    cam.distance = 3.0
    cam.azimuth = 120
    cam.elevation = -20

    video_path = os.path.join(os.path.dirname(csv_path), "video.mp4")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(actual_fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        video_path,
    ]

    try:
        proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"Failed to start ffmpeg for MP4 render: {e}")
        renderer.close()
        return None

    qpos_addr = _build_qpos_addr_map(mj_model)
    start_t = time.time()
    if proc.stdin is None:
        print("ffmpeg stdin unavailable, skipping MP4 render.")
        renderer.close()
        return None

    for i, row in enumerate(frames):
        _set_robot_qpos_from_row(mj_data, row, qpos_addr)

        mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data, camera=cam)
        frame = renderer.render()
        proc.stdin.write(frame.tobytes())

        if (i + 1) % 200 == 0:
            print(f"  rendering frame {i + 1}/{len(frames)}")

    proc.stdin.close()
    proc.wait()
    renderer.close()

    elapsed = time.time() - start_t
    print(f"Saved MP4: {video_path} ({elapsed:.1f}s)")
    return video_path


def parse_args():
    parser = argparse.ArgumentParser(description="Virtual joystick with optional auto-export.")
    parser.add_argument("--control-port", type=int, default=5556, help="ZMQ manager input port.")
    parser.add_argument("--control-host", default="*", help="Publisher bind host (default '*').")
    parser.add_argument("--init-wait", type=float, default=8.0, help="Seconds to wait after START.")
    parser.add_argument("--duration", type=float, default=5.0, help="Motion duration in seconds.")
    parser.add_argument("--planner-hz", type=float, default=20.0, help="Planner message send rate.")
    parser.add_argument(
        "--mode",
        choices=[
            "slow_walk", "walk", "run", "happy", "stealth", "injured",
            "squat", "kneel_two", "kneel_one", "hand_crawl", "elbow_crawl",
            "idle_boxing", "walk_boxing", "left_jab", "right_jab",
            "random_punches", "left_hook", "right_hook",
            "careful", "object_carrying", "crouch", "happy_dance",
            "zombie", "point", "scared",
        ],
        default="run",
        help="Locomotion mode used during motion.",
    )

    parser.set_defaults(record_debug=True)
    parser.add_argument(
        "--record-debug",
        dest="record_debug",
        action="store_true",
        help="Record g1_debug stream (default: enabled).",
    )
    parser.add_argument(
        "--no-record-debug",
        dest="record_debug",
        action="store_false",
        help="Disable debug stream recording.",
    )
    parser.add_argument("--debug-url", default="tcp://127.0.0.1:5557", help="Debug ZMQ output URL.")
    parser.add_argument("--debug-topic", default="g1_debug", help="Debug ZMQ topic prefix.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output dir for recording package (default: gear_sonic_deploy/output/<timestamp>).",
    )
    parser.add_argument("--render-mp4", action="store_true", help="Render MP4 from target trajectory.")
    parser.add_argument("--data-fps", type=int, default=50, help="Expected debug data FPS.")
    parser.add_argument("--video-fps", type=int, default=25, help="Output MP4 FPS.")
    parser.add_argument("--video-width", type=int, default=960, help="Output MP4 width.")
    parser.add_argument("--video-height", type=int, default=540, help="Output MP4 height.")
    return parser.parse_args()


def mode_from_name(name: str) -> int:
    return {
        "slow_walk":       MODE_SLOW_WALK,
        "walk":            MODE_WALK,
        "run":             MODE_RUN,
        "happy":           MODE_HAPPY,
        "stealth":         MODE_STEALTH,
        "injured":         MODE_INJURED,
        "squat":           MODE_SQUAT,
        "kneel_two":       MODE_KNEEL_TWO,
        "kneel_one":       MODE_KNEEL_ONE,
        "hand_crawl":      MODE_HAND_CRAWL,
        "elbow_crawl":     MODE_ELBOW_CRAWL,
        "idle_boxing":     MODE_IDLE_BOXING,
        "walk_boxing":     MODE_WALK_BOXING,
        "left_jab":        MODE_LEFT_JAB,
        "right_jab":       MODE_RIGHT_JAB,
        "random_punches":  MODE_RANDOM_PUNCHES,
        "left_hook":       MODE_LEFT_HOOK,
        "right_hook":      MODE_RIGHT_HOOK,
        "careful":         MODE_CAREFUL,
        "object_carrying": MODE_OBJECT_CARRYING,
        "crouch":          MODE_CROUCH,
        "happy_dance":     MODE_HAPPY_DANCE,
        "zombie":          MODE_ZOMBIE,
        "point":           MODE_POINT,
        "scared":          MODE_SCARED,
    }.get(name, MODE_RUN)


def save_recording(debug_recorder: DebugRecorder, args) -> Path:
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(__file__).resolve().parent / "output" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    target_csv = output_dir / "trajectory_target.csv"
    measured_csv = output_dir / "trajectory_measured.csv"
    raw_jsonl = output_dir / "debug_records.jsonl"
    metadata_path = output_dir / "metadata.json"

    if debug_recorder.target_rows:
        np.savetxt(str(target_csv), np.asarray(debug_recorder.target_rows), delimiter=",")
    if debug_recorder.measured_rows:
        np.savetxt(str(measured_csv), np.asarray(debug_recorder.measured_rows), delimiter=",")

    with raw_jsonl.open("w", encoding="utf-8") as f:
        for item in debug_recorder.records:
            f.write(json.dumps(item, separators=(",", ":")))
            f.write("\n")

    metadata = {
        "created_at": datetime.now().isoformat(),
        "control_port": args.control_port,
        "debug_url": args.debug_url,
        "debug_topic": args.debug_topic,
        "motion_duration_sec": args.duration,
        "planner_hz": args.planner_hz,
        "mode": args.mode,
        "num_debug_records": len(debug_recorder.records),
        "num_target_rows": len(debug_recorder.target_rows),
        "num_measured_rows": len(debug_recorder.measured_rows),
        "files": {
            "trajectory_target_csv": target_csv.name if target_csv.exists() else None,
            "trajectory_measured_csv": measured_csv.name if measured_csv.exists() else None,
            "debug_records_jsonl": raw_jsonl.name,
            "video_mp4": "video.mp4" if (output_dir / "video.mp4").exists() else None,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.render_mp4 and target_csv.exists():
        video_path = render_video(
            str(target_csv),
            data_fps=args.data_fps,
            video_fps=args.video_fps,
            width=args.video_width,
            height=args.video_height,
        )
        metadata["files"]["video_mp4"] = (
            Path(video_path).name if video_path is not None and Path(video_path).exists() else None
        )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved replay package: {output_dir}")
    return output_dir


def main():
    args = parse_args()

    # 1) Setup command publisher
    context = zmq.Context()
    cmd_socket = context.socket(zmq.PUB)
    bind_address = f"tcp://{args.control_host}:{args.control_port}"
    try:
        cmd_socket.bind(bind_address)
        print(f"Publisher bound to {bind_address}")
    except Exception as e:
        print(f"Bind failed ({e}), trying connect mode...")
        cmd_socket.connect(f"tcp://127.0.0.1:{args.control_port}")

    debug_recorder = None
    if args.record_debug:
        if msgpack is None:
            raise RuntimeError(
                "Missing dependency: msgpack. Install with `pip install msgpack` or disable --record-debug."
            )
        debug_recorder = DebugRecorder(context=context, zmq_url=args.debug_url, topic=args.debug_topic)
        print(f"Debug recording enabled: {args.debug_url} topic={args.debug_topic}")

    # ZMQ PUB/SUB slow joiner gap
    print("Waiting for subscribers to connect...")
    time.sleep(2)
    if debug_recorder is not None:
        debug_recorder.poll()

    # 2) Send START + planner enable
    print("Sending START and PLANNER commands...")
    print("NOTE: Make sure you pressed '9' in MuJoCo window to start simulation.")
    command_msg = build_command_message(start=True, stop=False, planner=True)
    cmd_socket.send(command_msg)

    print("Waiting for planner and robot to initialize...")
    warmup_end = time.time() + args.init_wait
    while time.time() < warmup_end:
        if debug_recorder is not None:
            debug_recorder.poll()
        time.sleep(0.01)

    # 3) Motion command loop
    planner_hz = float(args.planner_hz)
    total_steps = int(planner_hz * args.duration)
    motion_mode = mode_from_name(args.mode)

    facing_angle = 0.0
    movement = [1.0, 0.0, 0.0]
    facing_direction = [math.cos(facing_angle), math.sin(facing_angle), 0.0]
    motion_msg = build_planner_message(
        mode=motion_mode,
        movement=movement,
        facing=facing_direction,
        speed=-1.0,
        height=-1.0,
    )

    print(f"Executing pre-defined motion ({args.duration:.2f}s, {args.mode}, {planner_hz:.1f}Hz)...")
    for step in range(total_steps):
        cmd_socket.send(motion_msg)
        if debug_recorder is not None:
            debug_recorder.poll()
        if step % max(1, int(planner_hz)) == 0:
            print(f"  Step {step + 1}/{total_steps}")
        time.sleep(1.0 / planner_hz)

    # 4) Return to IDLE for stabilization
    print("Stopping motion (sending IDLE)...")
    idle_msg = build_planner_message(
        mode=MODE_IDLE,
        movement=[1.0, 0.0, 0.0],
        facing=[1.0, 0.0, 0.0],
        speed=-1.0,
        height=-1.0,
    )
    idle_steps = int(max(1, planner_hz))
    for _ in range(idle_steps):
        cmd_socket.send(idle_msg)
        if debug_recorder is not None:
            debug_recorder.poll()
        time.sleep(1.0 / planner_hz)

    # Drain remaining debug data and persist
    output_dir = None
    if debug_recorder is not None:
        drain_until = time.time() + 0.5
        while time.time() < drain_until:
            debug_recorder.poll()
            time.sleep(0.01)
        output_dir = save_recording(debug_recorder, args)

    print("\n" + "=" * 60)
    print("Motion complete.")
    print("Robot should now be in IDLE mode.")
    if output_dir is not None:
        print(f"Replay package path: {output_dir}")
    print("=" * 60)

    if debug_recorder is not None:
        debug_recorder.close()
    cmd_socket.close(0)
    context.term()


if __name__ == "__main__":
    main()