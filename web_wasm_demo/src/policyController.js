/**
 * ONNX Runtime Web policy controller (encoder + decoder).
 *
 * This module loads the G1 encoder/decoder ONNX models and provides
 * closed-loop inference: build observation → run encoder → run decoder
 * → apply PD control torques.
 *
 * Currently a structural skeleton; the observation builder dimensions
 * will be filled in after inspecting the live ONNX input shapes.
 */

import * as ort from 'onnxruntime-web';
import { kps, kds, actionScale, defaultAngles, mujoco_to_isaaclab, NUM_JOINTS, CONTROL_DT } from './policyParams.js';

export class PolicyController {
  constructor() {
    this.encoderSession = null;
    this.decoderSession = null;
    this.ready = false;
    this.tokenState = new Float32Array(64);
    this.lastActions = new Float32Array(NUM_JOINTS);
    this.targetJointPos = new Float64Array(NUM_JOINTS);
    for (let i = 0; i < NUM_JOINTS; i++) this.targetJointPos[i] = defaultAngles[i];
  }

  async load(encoderUrl, decoderUrl, onProgress) {
    const report = (msg) => onProgress && onProgress(msg);
    try {
      report('Loading encoder ONNX (~50 MB)...');
      const encBuf = await fetch(encoderUrl).then(r => r.arrayBuffer());
      this.encoderSession = await ort.InferenceSession.create(encBuf, {
        executionProviders: ['wasm'],
      });

      report('Loading decoder ONNX (~41 MB)...');
      const decBuf = await fetch(decoderUrl).then(r => r.arrayBuffer());
      this.decoderSession = await ort.InferenceSession.create(decBuf, {
        executionProviders: ['wasm'],
      });

      this.ready = true;
      console.log('[PolicyController] Encoder inputs:', this.encoderSession.inputNames,
                  'outputs:', this.encoderSession.outputNames);
      console.log('[PolicyController] Decoder inputs:', this.decoderSession.inputNames,
                  'outputs:', this.decoderSession.outputNames);
    } catch (e) {
      console.error('[PolicyController] Failed to load ONNX models:', e);
    }
  }

  /**
   * Apply PD control torques to MuJoCo actuators based on targetJointPos.
   * This can be used with either policy output or motion-replay targets.
   */
  applyPDControl(model, data) {
    // qpos layout: [base_pos(3), base_quat(4), joints(29)]
    for (let j = 0; j < NUM_JOINTS; j++) {
      const qpos = data.qpos[7 + j];
      const qvel = data.qvel[6 + j]; // qvel has 6-DOF free joint + joint velocities
      const torque = kps[j] * (this.targetJointPos[j] - qpos) + kds[j] * (0.0 - qvel);
      const ctrlRange0 = model.actuator_ctrlrange[j * 2];
      const ctrlRange1 = model.actuator_ctrlrange[j * 2 + 1];
      data.ctrl[j] = Math.max(ctrlRange0, Math.min(ctrlRange1, torque));
    }
  }
}
