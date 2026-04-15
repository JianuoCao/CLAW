/**
 * Motor constants, PD gains, joint mappings, action scales, and default
 * standing angles for the G1 29-DOF policy.
 *
 * Ported 1:1 from policy_parameters.hpp.
 */

const NATURAL_FREQ = 10 * 2.0 * Math.PI;
const DAMPING_RATIO = 2;

const ARMATURE_5020 = 0.003609725;
const ARMATURE_7520_14 = 0.010177520;
const ARMATURE_7520_22 = 0.025101925;
const ARMATURE_4010 = 0.00425;

const S_5020 = ARMATURE_5020 * NATURAL_FREQ * NATURAL_FREQ;
const S_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ * NATURAL_FREQ;
const S_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ * NATURAL_FREQ;
const S_4010 = ARMATURE_4010 * NATURAL_FREQ * NATURAL_FREQ;

const D_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ;
const D_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ;
const D_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ;
const D_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ;

const E_5020 = 25.0;
const E_7520_14 = 88.0;
const E_7520_22 = 139.0;
const E_4010 = 5.0;

export const NUM_JOINTS = 29;

// Stiffness (Kp) per joint – MuJoCo order
export const kps = new Float64Array([
  S_7520_22, S_7520_22, S_7520_14, S_7520_22, 2*S_5020, 2*S_5020,  // L leg
  S_7520_22, S_7520_22, S_7520_14, S_7520_22, 2*S_5020, 2*S_5020,  // R leg
  S_7520_14, 2*S_5020, 2*S_5020,                                     // waist
  S_5020, S_5020, S_5020, S_5020, S_5020, S_4010, S_4010,           // L arm
  S_5020, S_5020, S_5020, S_5020, S_5020, S_4010, S_4010,           // R arm
]);

// Damping (Kd) per joint – MuJoCo order
export const kds = new Float64Array([
  D_7520_22, D_7520_22, D_7520_14, D_7520_22, 2*D_5020, 2*D_5020,
  D_7520_22, D_7520_22, D_7520_14, D_7520_22, 2*D_5020, 2*D_5020,
  D_7520_14, 2*D_5020, 2*D_5020,
  D_5020, D_5020, D_5020, D_5020, D_5020, D_4010, D_4010,
  D_5020, D_5020, D_5020, D_5020, D_5020, D_4010, D_4010,
]);

// Action scale per joint – MuJoCo order
const as = (e, s) => 0.25 * e / s;
export const actionScale = new Float64Array([
  as(E_7520_22,S_7520_22), as(E_7520_22,S_7520_22), as(E_7520_14,S_7520_14), as(E_7520_22,S_7520_22), as(E_5020,S_5020), as(E_5020,S_5020),
  as(E_7520_22,S_7520_22), as(E_7520_22,S_7520_22), as(E_7520_14,S_7520_14), as(E_7520_22,S_7520_22), as(E_5020,S_5020), as(E_5020,S_5020),
  as(E_7520_14,S_7520_14), as(E_5020,S_5020), as(E_5020,S_5020),
  as(E_5020,S_5020), as(E_5020,S_5020), as(E_5020,S_5020), as(E_5020,S_5020), as(E_5020,S_5020), as(E_4010,S_4010), as(E_4010,S_4010),
  as(E_5020,S_5020), as(E_5020,S_5020), as(E_5020,S_5020), as(E_5020,S_5020), as(E_5020,S_5020), as(E_4010,S_4010), as(E_4010,S_4010),
]);

// Default standing angles – MuJoCo order (radians)
export const defaultAngles = new Float64Array([
  -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   // L leg
  -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   // R leg
   0.0,   0.0, 0.0,                         // waist
   0.2,   0.2, 0.0, 0.6, 0.0, 0.0, 0.0,   // L arm
   0.2,  -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,   // R arm
]);

// isaaclab_to_mujoco[isaaclab_idx] = mujoco_idx
export const isaaclab_to_mujoco = [0,3,6,9,13,17,1,4,7,10,14,18,2,5,8,11,15,19,21,23,25,27,12,16,20,22,24,26,28];
// mujoco_to_isaaclab[mujoco_idx] = isaaclab_idx
export const mujoco_to_isaaclab = [0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28];

export const CONTROL_DT = 0.02; // 50 Hz
