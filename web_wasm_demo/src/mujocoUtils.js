/**
 * MuJoCo WASM scene loading and Three.js scene-graph construction.
 *
 * Coordinate convention: MuJoCo is Z-up, Three.js is Y-up.
 *   position  : (x, y, z) → (x, z, -y)
 *   quaternion: (w, x, y, z) → Three.Quaternion(x, z, -y, w)
 */

import * as THREE from 'three';

// ─── Coordinate helpers ──────────────────────────────────────────────

export function swizzlePos(mx, my, mz) {
  return [mx, mz, -my];
}

export function swizzleQuat(w, x, y, z) {
  // Returns [x, y, z, w] for Three.js Quaternion constructor
  return [x, z, -y, w];
}

// ─── Asset download ──────────────────────────────────────────────────

const MESH_FILES = [
  'pelvis.STL', 'pelvis_contour_link.STL',
  'left_hip_pitch_link.STL', 'left_hip_roll_link.STL', 'left_hip_yaw_link.STL',
  'left_knee_link.STL', 'left_ankle_pitch_link.STL', 'left_ankle_roll_link.STL',
  'right_hip_pitch_link.STL', 'right_hip_roll_link.STL', 'right_hip_yaw_link.STL',
  'right_knee_link.STL', 'right_ankle_pitch_link.STL', 'right_ankle_roll_link.STL',
  'waist_yaw_link.STL', 'waist_roll_link.STL', 'torso_link.STL',
  'logo_link.STL', 'head_link.STL', 'waist_support_link.STL',
  'left_shoulder_pitch_link.STL', 'left_shoulder_roll_link.STL',
  'left_shoulder_yaw_link.STL', 'left_elbow_link.STL',
  'left_wrist_roll_link.STL', 'left_wrist_pitch_link.STL',
  'left_wrist_yaw_link.STL',
  'left_hand_palm_link.STL',
  'left_hand_thumb_0_link.STL', 'left_hand_thumb_1_link.STL', 'left_hand_thumb_2_link.STL',
  'left_hand_index_0_link.STL', 'left_hand_index_1_link.STL',
  'left_hand_middle_0_link.STL', 'left_hand_middle_1_link.STL',
  'right_shoulder_pitch_link.STL', 'right_shoulder_roll_link.STL',
  'right_shoulder_yaw_link.STL', 'right_elbow_link.STL',
  'right_wrist_roll_link.STL', 'right_wrist_pitch_link.STL',
  'right_wrist_yaw_link.STL',
  'right_hand_palm_link.STL',
  'right_hand_thumb_0_link.STL', 'right_hand_thumb_1_link.STL', 'right_hand_thumb_2_link.STL',
  'right_hand_index_0_link.STL', 'right_hand_index_1_link.STL',
  'right_hand_middle_0_link.STL', 'right_hand_middle_1_link.STL',
];

/**
 * Download scene XMLs and mesh STLs into the Emscripten virtual FS,
 * then load the MuJoCo model. Returns {model, data}.
 */
export async function downloadAssets(mujoco, assetBase, onProgress) {
  const FS = mujoco.FS;

  try { FS.mkdir('/working'); } catch {}
  try { FS.mkdir('/working/meshes'); } catch {}

  const report = (msg) => onProgress && onProgress(msg);

  report('Downloading scene XML...');
  const [sceneXml, robotXml] = await Promise.all([
    fetch(`${assetBase}/g1/scene_29dof_with_hand.xml`).then(r => r.text()),
    fetch(`${assetBase}/g1/g1_29dof_with_hand.xml`).then(r => r.text()),
  ]);

  FS.writeFile('/working/scene_29dof_with_hand.xml', sceneXml);
  FS.writeFile('/working/g1_29dof_with_hand.xml', robotXml);

  report('Downloading meshes...');
  const total = MESH_FILES.length;
  let loaded = 0;
  const batchSize = 24;
  for (let i = 0; i < total; i += batchSize) {
    const batch = MESH_FILES.slice(i, i + batchSize);
    await Promise.all(batch.map(async (name) => {
      const buf = await fetch(`${assetBase}/g1/meshes/${name}`).then(r => r.arrayBuffer());
      FS.writeFile(`/working/meshes/${name}`, new Uint8Array(buf));
      loaded++;
      report(`Meshes ${loaded}/${total}`);
    }));
  }

  report('Loading MuJoCo model...');
  const model = mujoco.MjModel.loadFromXML('/working/scene_29dof_with_hand.xml');
  const data = new mujoco.MjData(model);
  return { model, data };
}

// ─── Three.js scene-graph construction ───────────────────────────────

// MuJoCo geom type enum
const mjGEOM_PLANE = 0;
const mjGEOM_SPHERE = 2;
const mjGEOM_CAPSULE = 3;
const mjGEOM_CYLINDER = 5;
const mjGEOM_BOX = 6;
const mjGEOM_MESH = 7;

function makeMaterial(r, g, b, a) {
  const params = { color: new THREE.Color(r, g, b), roughness: 0.6, metalness: 0.1 };
  if (a < 1.0) {
    params.transparent = true;
    params.opacity = a;
  }
  return new THREE.MeshStandardMaterial(params);
}

function makeGroundGridMaterial() {
  const tilePx = 128;
  const canvas = document.createElement('canvas');
  canvas.width = tilePx;
  canvas.height = tilePx;
  const ctx = canvas.getContext('2d');

  // MuJoCo-like checker tile with clearer contrast.
  // ctx.fillStyle = '#4f6780';
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, tilePx, tilePx);
  // ctx.fillStyle = '#2f4258';
  ctx.fillStyle = '#e1e0e0';
  ctx.fillRect(0, 0, tilePx / 2, tilePx / 2);
  ctx.fillRect(tilePx / 2, tilePx / 2, tilePx / 2, tilePx / 2);

  ctx.strokeStyle = 'rgba(145, 141, 141, 0.09)';
  ctx.lineWidth = 1;
  const major = tilePx / 2;
  const minor = tilePx / 8;
  for (let i = 0; i <= tilePx; i += minor) {
    const majorLine = (i % major) === 0;
    ctx.globalAlpha = majorLine ? 1.0 : 0.45;
    ctx.beginPath();
    ctx.moveTo(i + 0.5, 0);
    ctx.lineTo(i + 0.5, tilePx);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, i + 0.5);
    ctx.lineTo(tilePx, i + 0.5);
    ctx.stroke();
  }
  ctx.globalAlpha = 1.0;

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(25, 25);
  tex.anisotropy = 8;
  tex.colorSpace = THREE.SRGBColorSpace;

  return new THREE.MeshStandardMaterial({
    map: tex,
    color: 0xffffff,
    roughness: 0.95,
    metalness: 0.0,
  });
}

function createMeshGeometry(model, meshId) {
  const vertAdr = model.mesh_vertadr[meshId];
  const vertNum = model.mesh_vertnum[meshId];
  const faceAdr = model.mesh_faceadr[meshId];
  const faceNum = model.mesh_facenum[meshId];

  const positions = new Float32Array(faceNum * 9);
  const normals = new Float32Array(faceNum * 9);

  for (let f = 0; f < faceNum; f++) {
    for (let v = 0; v < 3; v++) {
      const vIdx = model.mesh_face[faceAdr * 3 + f * 3 + v] + vertAdr;
      const mx = model.mesh_vert[vIdx * 3 + 0];
      const my = model.mesh_vert[vIdx * 3 + 1];
      const mz = model.mesh_vert[vIdx * 3 + 2];
      const nx = model.mesh_normal[vIdx * 3 + 0];
      const ny = model.mesh_normal[vIdx * 3 + 1];
      const nz = model.mesh_normal[vIdx * 3 + 2];
      const i = f * 9 + v * 3;
      positions[i] = mx; positions[i+1] = mz; positions[i+2] = -my;
      normals[i] = nx;   normals[i+1] = nz;   normals[i+2] = -ny;
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  return geo;
}

function createGeomGeometry(type, size) {
  switch (type) {
    case mjGEOM_PLANE: {
      const geo = new THREE.PlaneGeometry(50, 50);
      geo.rotateX(-Math.PI / 2);
      return geo;
    }
    case mjGEOM_SPHERE:
      return new THREE.SphereGeometry(size[0], 20, 16);
    case mjGEOM_CAPSULE:
      return new THREE.CapsuleGeometry(size[0], size[1] * 2, 8, 16);
    case mjGEOM_CYLINDER:
      return new THREE.CylinderGeometry(size[0], size[0], size[1] * 2, 20);
    case mjGEOM_BOX:
      // MuJoCo half-extents (x,y,z) → Three.js full extents with swizzle
      return new THREE.BoxGeometry(size[0] * 2, size[2] * 2, size[1] * 2);
    default:
      return null;
  }
}

/**
 * Build the Three.js scene graph from a loaded MuJoCo model.
 * Uses a flat structure: all body groups are direct children of the scene,
 * so we can set world-space transforms directly from data.xpos / data.xquat.
 * Returns an array of THREE.Group indexed by body ID.
 */
export function buildSceneGraph(model, scene) {
  const bodyGroups = [];

  for (let b = 0; b < model.nbody; b++) {
    const group = new THREE.Group();
    group.name = `body_${b}`;
    scene.add(group);
    bodyGroups.push(group);
  }

  // Create a mesh for each geom
  for (let g = 0; g < model.ngeom; g++) {
    const geomType = model.geom_type[g];
    const bodyId = model.geom_bodyid[g];
    const dataId = model.geom_dataid[g];
    const s = [model.geom_size[g*3], model.geom_size[g*3+1], model.geom_size[g*3+2]];
    const rgba = [model.geom_rgba[g*4], model.geom_rgba[g*4+1], model.geom_rgba[g*4+2], model.geom_rgba[g*4+3]];

    // Skip tiny collision spheres (radius < 0.01, no mesh)
    if (geomType === mjGEOM_SPHERE && s[0] < 0.01) continue;

    let geometry;
    if (geomType === mjGEOM_MESH && dataId >= 0) {
      geometry = createMeshGeometry(model, dataId);
    } else {
      geometry = createGeomGeometry(geomType, s);
    }
    if (!geometry) continue;

    let mat;
    if (geomType === mjGEOM_PLANE) {
      mat = makeGroundGridMaterial();
    } else {
      mat = makeMaterial(rgba[0], rgba[1], rgba[2], rgba[3]);
    }

    const mesh = new THREE.Mesh(geometry, mat);
    mesh.castShadow = geomType !== mjGEOM_PLANE;
    mesh.receiveShadow = true;

    // Geom-local transform (in body frame), swizzled
    const lp = swizzlePos(model.geom_pos[g*3], model.geom_pos[g*3+1], model.geom_pos[g*3+2]);
    const lq = swizzleQuat(model.geom_quat[g*4], model.geom_quat[g*4+1], model.geom_quat[g*4+2], model.geom_quat[g*4+3]);
    mesh.position.set(lp[0], lp[1], lp[2]);
    mesh.quaternion.set(lq[0], lq[1], lq[2], lq[3]);

    bodyGroups[bodyId].add(mesh);
  }

  return bodyGroups;
}

// ─── Joint qpos address map ──────────────────────────────────────────

/**
 * Build an array of qpos addresses for a list of joint names (in order).
 * Returns -1 for any name not found. Used to correctly map server body-joint
 * data into the model's qpos array even when extra joints (e.g. hands) are
 * interleaved between arm joints.
 */
export function buildBodyJointQposAddrs(model, jointNames) {
  const nameToQposAdr = {};
  const names = model.names; // Uint8Array of null-terminated name strings
  for (let j = 0; j < model.njnt; j++) {
    const start = model.name_jntadr[j];
    let end = start;
    while (end < names.length && names[end] !== 0) end++;
    const name = new TextDecoder().decode(names.subarray(start, end));
    nameToQposAdr[name] = model.jnt_qposadr[j];
  }
  return jointNames.map(n => (n in nameToQposAdr ? nameToQposAdr[n] : -1));
}

// ─── Per-frame body-transform update ────────────────────────────────

export function updateBodies(model, data, bodyGroups) {
  for (let b = 1; b < model.nbody; b++) {
    const px = data.xpos[b*3], py = data.xpos[b*3+1], pz = data.xpos[b*3+2];
    const qw = data.xquat[b*4], qx = data.xquat[b*4+1], qy = data.xquat[b*4+2], qz = data.xquat[b*4+3];
    const [sx, sy, sz] = swizzlePos(px, py, pz);
    const [sqx, sqy, sqz, sqw] = swizzleQuat(qw, qx, qy, qz);
    bodyGroups[b].position.set(sx, sy, sz);
    bodyGroups[b].quaternion.set(sqx, sqy, sqz, sqw);
  }
}
