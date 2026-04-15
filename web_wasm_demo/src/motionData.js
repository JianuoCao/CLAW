/**
 * Reference motion loading and playback.
 * Parses CSV files exported by the motion pipeline and provides
 * frame-by-frame joint positions + base pose to drive MuJoCo replay.
 */

function parseCSV(text) {
  const lines = text.trim().split('\n');
  const header = lines[0].split(',');
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    rows.push(lines[i].split(',').map(Number));
  }
  return { header, rows };
}

export class MotionClip {
  constructor(name, jointPos, basePos, baseQuat, fps) {
    this.name = name;
    this.jointPos = jointPos;   // Float64Array[frames * 29]
    this.basePos = basePos;     // Float64Array[frames * 3]
    this.baseQuat = baseQuat;   // Float64Array[frames * 4]  (w,x,y,z)
    this.fps = fps;
    this.numFrames = jointPos.length / 29;
  }
}

export class MotionPlayer {
  constructor() {
    this.clips = {};
    this.current = null;
    this.frame = 0;
    this.loop = true;
    this.playing = true;
    this.accumulator = 0;
  }

  async loadMotion(name, dirUrl) {
    const [jpText, bpText, bqText] = await Promise.all([
      fetch(`${dirUrl}/joint_pos.csv`).then(r => r.text()),
      fetch(`${dirUrl}/body_pos.csv`).then(r => r.text()),
      fetch(`${dirUrl}/body_quat.csv`).then(r => r.text()),
    ]);

    const jp = parseCSV(jpText);
    const bp = parseCSV(bpText);
    const bq = parseCSV(bqText);

    const numFrames = jp.rows.length;

    const jointPos = new Float64Array(numFrames * 29);
    const basePos = new Float64Array(numFrames * 3);
    const baseQuat = new Float64Array(numFrames * 4);

    for (let f = 0; f < numFrames; f++) {
      for (let j = 0; j < 29; j++) jointPos[f * 29 + j] = jp.rows[f][j];
      basePos[f * 3 + 0] = bp.rows[f][0];  // body_0_x
      basePos[f * 3 + 1] = bp.rows[f][1];  // body_0_y
      basePos[f * 3 + 2] = bp.rows[f][2];  // body_0_z
      baseQuat[f * 4 + 0] = bq.rows[f][0]; // body_0_w
      baseQuat[f * 4 + 1] = bq.rows[f][1]; // body_0_x
      baseQuat[f * 4 + 2] = bq.rows[f][2]; // body_0_y
      baseQuat[f * 4 + 3] = bq.rows[f][3]; // body_0_z
    }

    const fps = 50; // default motion fps
    this.clips[name] = new MotionClip(name, jointPos, basePos, baseQuat, fps);
  }

  selectMotion(name) {
    if (!this.clips[name]) return;
    this.current = this.clips[name];
    this.frame = 0;
    this.accumulator = 0;
  }

  /** Advance one tick (called at simulation rate). */
  tick() {
    if (!this.current || !this.playing) return;
    this.frame++;
    if (this.frame >= this.current.numFrames) {
      this.frame = this.loop ? 0 : this.current.numFrames - 1;
    }
  }

  /** Write current frame into MuJoCo qpos. */
  applyFrame(model, data, mujoco) {
    const clip = this.current;
    if (!clip) return;
    const f = this.frame;
    const off = f * 29;
    const bp = f * 3;
    const bq = f * 4;
    // Free joint: qpos[0:3]=pos, qpos[3:7]=quat(w,x,y,z), qpos[7:36]=joints
    data.qpos[0] = clip.basePos[bp + 0];
    data.qpos[1] = clip.basePos[bp + 1];
    data.qpos[2] = clip.basePos[bp + 2];
    data.qpos[3] = clip.baseQuat[bq + 0];
    data.qpos[4] = clip.baseQuat[bq + 1];
    data.qpos[5] = clip.baseQuat[bq + 2];
    data.qpos[6] = clip.baseQuat[bq + 3];
    for (let j = 0; j < 29; j++) {
      data.qpos[7 + j] = clip.jointPos[off + j];
    }
  }

  get motionNames() { return Object.keys(this.clips); }
  get currentName() { return this.current ? this.current.name : '-'; }
  get totalFrames() { return this.current ? this.current.numFrames : 0; }
}
