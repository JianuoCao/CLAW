import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import load_mujoco from 'mujoco-js';
import { downloadAssets, buildSceneGraph, updateBodies, swizzlePos, buildBodyJointQposAddrs } from './mujocoUtils.js';
import { defaultAngles, NUM_JOINTS } from './policyParams.js';

// Joint names in the order the server sends them (matches BODY_29DOF_JOINT_NAMES in virtual_joystick.py)
const BODY_JOINT_NAMES = [
  'left_hip_pitch_joint',  'left_hip_roll_joint',  'left_hip_yaw_joint',
  'left_knee_joint',       'left_ankle_pitch_joint', 'left_ankle_roll_joint',
  'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
  'right_knee_joint',      'right_ankle_pitch_joint', 'right_ankle_roll_joint',
  'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
  'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
  'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
  'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
  'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
];

// Use same-origin /ws so Vite's proxy routes it to ws_bridge.py (port 8080).
const WS_URL = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
const ASSET_BASE = '';
const DYNAMIC_BASE_POS = [0.0, 0.0, 0.793];
const RECIPE_INIT_WAIT_SEC = 0.0;

// ─── UI helpers ──────────────────────────────────────────────────────

function setProgress(pct, msg) {
  document.getElementById('loading-bar').style.width = `${pct}%`;
  document.getElementById('loading-status').textContent = msg.toUpperCase();
}

function showUI() {
  document.getElementById('loading-overlay').classList.add('hidden');
  ['top-bar', 'left-panel', 'right-panel', 'controls-bar'].forEach(
    id => document.getElementById(id).style.display = ''
  );
}

// ─── App ─────────────────────────────────────────────────────────────

class G1WasmDemo {
  constructor() {
    this.mujoco = null;
    this.model  = null;
    this.data   = null;
    this.bodyGroups = [];

    this.bodyJointQposAddrs = null; // qpos address per body joint, built after model load

    this.ws          = null;
    this.wsConnected = false;
    this.serverState = {};
    this.pendingState = null;

    // Qpos double-buffer for interpolation
    this.qposPrev    = null;   // previous frame's qpos (Float64Array)
    this.qposCurr    = null;   // latest received qpos  (Float64Array)
    this.qposTargetPrev = null;
    this.qposTargetCurr = null;
    this.qposAlpha   = 1.0;    // interpolation progress 0→1
    this.qposLastMs  = 0;      // timestamp of last qpos arrival
    this.SERVER_DT   = 0.05;   // server broadcasts at 20 Hz → 50 ms

    // Frozen-to-live transition: lerp from the visual reset pose to the first server qpos
    this._unfreezeFromQpos = null; // qpos snapshot at the moment of unfreeze
    this._unfreezeStartMs  = 0;
    this._UNFREEZE_DUR_MS  = 500;  // ms to blend from reset pose to server pose

    // Kinematic root origin: x,y offset of the first received qpos_target after unfreeze.
    // Subtracted from all kinematic root positions so the rendered robot starts at
    // (0, 0, z) regardless of accumulated drift from previous recipe runs.
    this._targetRootOrigin = null; // [ox, oy] or null

    this.paused      = false;
    this.frameCount  = 0;
    this.lastFpsTime = 0;
    this.fps         = 0;

    this.cameraFollow = false;
    this._camTarget   = new THREE.Vector3(); // pre-alloc, reused every frame
    this._frozen      = false;  // true after reset: ignore server qpos until Enter

    // ── Cached DOM refs (populated in cacheDom) ───────────────────
    this._dom = {};
    this._prevActive      = null;
    this._prevMovement    = null;
    this._prevMode        = null;
    this._prevHeading     = null;
    this._prevSpeedRange      = undefined;
    this._prevHeightCtrl      = undefined;
    this._prevStrafeDisabled  = undefined;
  }

  // ── Cache DOM elements once at startup ──────────────────────────
  cacheDom() {
    const ids = [
      'hud-active', 'hud-movement', 'hud-mode', 'hud-heading', 'hud-fps',
      'btn-activate', 'speed-row', 'height-row',
      'connection-dot', 'conn-label',
      'set-tabs', 'mode-buttons', 'strafe-ctrl',
    ];
    for (const id of ids) this._dom[id] = document.getElementById(id);
    this._dom['btn-act-label'] = this._dom['btn-activate'].querySelector('.btn-act-label');
  }

  async init() {
    this.cacheDom();
    this.setupRenderer();

    setProgress(5, 'Loading MuJoCo WASM...');
    this.mujoco = await load_mujoco();

    const { model, data } = await downloadAssets(this.mujoco, ASSET_BASE, msg => {
      setProgress(30, msg);
    });
    this.model = model;
    this.data  = data;
    this.bodyJointQposAddrs = buildBodyJointQposAddrs(model, BODY_JOINT_NAMES);

    setProgress(60, 'Building 3D scene...');
    this.bodyGroups = buildSceneGraph(model, this.scene);

    this.resetScene();   // apply standing default pose

    setProgress(80, 'Connecting to server...');
    this.connectWebSocket();
    this.setupKeyboard();
    this.setupModeUI();
    this.setupRecipeUI();

    setProgress(100, 'Ready');
    showUI();
    this.renderer.setAnimationLoop(this.render.bind(this));
  }

  // ── Three.js setup ──────────────────────────────────────────────

  setupRenderer() {
    this.scene = new THREE.Scene();
    // this.scene.background = new THREE.Color(0xaecfff);
    this.scene.background = new THREE.Color(0xffffff);
    // this.scene.fog = new THREE.Fog(0xaecfff, 14, 42);
    // this.scene.fog= new THREE.Fog(0xffffff, 14, 42);
    this.camera = new THREE.PerspectiveCamera(
      45, window.innerWidth / window.innerHeight, 0.01, 100
    );
    this.camera.position.set(3, 2.5, 3);

    this.renderer = new THREE.WebGLRenderer({
      canvas: document.getElementById('canvas'),
      antialias: true,
    });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.32;

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.8, 0);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI * 0.48;
    this.controls.update();

    const ambient = new THREE.AmbientLight(0xffffff, 0.68);
    this.scene.add(ambient);

    const sun = new THREE.DirectionalLight(0xfff8ee, 1.45);
    sun.position.set(5, 10, 4);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);   // 1024 is enough, 2048 costs ~4× GPU time
    sun.shadow.camera.near = 0.1;
    sun.shadow.camera.far  = 40;
    const d = 8;
    sun.shadow.camera.left   = -d;
    sun.shadow.camera.right  =  d;
    sun.shadow.camera.top    =  d;
    sun.shadow.camera.bottom = -d;
    this.scene.add(sun);
    this.sunLight = sun;

    const fill = new THREE.DirectionalLight(0xb8ccff, 0.4);
    fill.position.set(-4, 5, -3);
    this.scene.add(fill);

    window.addEventListener('resize', () => {
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(window.innerWidth, window.innerHeight);
    });
  }

  // ── WebSocket ───────────────────────────────────────────────────

  connectWebSocket() {
    const dot   = this._dom['connection-dot'];
    const label = this._dom['conn-label'];

    const tryConnect = () => {
      this.ws = new WebSocket(WS_URL);

      this.ws.onopen = () => {
        this.wsConnected = true;
        dot.classList.add('on');
        label.textContent = 'ONLINE';
      };

      this.ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);
          if (msg.type === 'state') {
            if (!msg.motionSets) {
              msg.motionSets = this.pendingState?.motionSets || this.serverState.motionSets;
            }
            this.pendingState = msg;
          } else if (msg.type === 'recipe_status') {
            this._recipeUI?.onStatus(msg);
          } else if (msg.type === 'recipe_result') {
            this._recipeUI?.onResult(msg);
          } else if (msg.type === 'recipe_error') {
            this._recipeUI?.onError(msg);
          }
        } catch {}
      };

      this.ws.onclose = () => {
        this.wsConnected = false;
        dot.classList.remove('on');
        label.textContent = 'OFFLINE';
        setTimeout(tryConnect, 2000);
      };

      this.ws.onerror = () => { this.ws.close(); };
    };

    tryConnect();
  }

  sendKey(key, pressed) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'key', key, pressed }));
    }
  }

  sendSetSpeed(value) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'set_speed', value }));
    }
  }

  sendSetHeight(value) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'set_height', value }));
    }
  }

  sendSetMode(setIdx, modeKey) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'set_mode', setIdx, modeKey }));
    }
  }

  // ── UI update (called from onmessage, ~20 Hz) ───────────────────

  _updateUI(s) {
    const dom = this._dom;

    // Build mode buttons once (first state message that has motionSets)
    if (s.motionSets && !this._modeButtonsBuilt) {
      this.buildModeButtons(s.motionSets);
    }

    // Slider visibility
    if (s.speedRange !== this._prevSpeedRange) {
      dom['speed-row'].classList.toggle('visible', !!s.speedRange);
      this._prevSpeedRange = s.speedRange;
    }
    if (s.heightControl !== this._prevHeightCtrl) {
      dom['height-row'].classList.toggle('visible', !!s.heightControl);
      this._prevHeightCtrl = s.heightControl;
    }

    // Initialize sliders with default values when mode changes
    const modeKey = `${s.motionSetIdx}:${s.modeKey}`;
    if (modeKey !== this._prevSliderModeKey) {
      this._prevSliderModeKey = modeKey;
      if (s.speedRange) {
        const sl = document.getElementById('speed-slider');
        sl.min   = s.speedRange[0];
        sl.max   = s.speedRange[1];
        sl.value = s.speed > 0 ? s.speed : s.speedRange[0];
        sl.dispatchEvent(new Event('input'));
      }
      if (s.heightControl) {
        const sl = document.getElementById('height-slider');
        sl.value = s.height > 0 ? s.height : 0.8;
        sl.dispatchEvent(new Event('input'));
      }
    }

    // Strafe availability: disabled for Squat/Ground (setIdx=1) and Idle Boxing (setIdx=2, modeKey=1)
    const strafeDisabled = s.motionSetIdx === 1 ||
                           (s.motionSetIdx === 2 && s.modeKey === 1);
    if (strafeDisabled !== this._prevStrafeDisabled) {
      const sc = dom['strafe-ctrl'];
      if (sc) {
        sc.querySelectorAll('.hint-key').forEach(el =>
          el.classList.toggle('key-unavailable', strafeDisabled));
        const desc = sc.querySelector('.ctrl-desc');
        if (desc) desc.classList.toggle('key-unavailable', strafeDisabled);
      }
      this._prevStrafeDisabled = strafeDisabled;
    }

    // Activate button
    if (s.active !== this._prevActive) {
      dom['btn-activate'].classList.toggle('active', s.active);
      dom['btn-act-label'].textContent = s.active ? 'Deactivate Controller' : 'Activate Controller';
      this._prevActive = s.active;
    }

    // HUD cells — only write DOM when value changed
    const activeStr = s.active ? 'ACTIVE' : 'IDLE';
    if (activeStr !== this._prevActive_str) {
      dom['hud-active'].textContent = activeStr;
      dom['hud-active'].className   = 'hud-cell-value ' + (s.active ? 'green' : 'dim');
      this._prevActive_str = activeStr;
    }

    const movStr = (s.movementType || 'idle').toUpperCase();
    if (movStr !== this._prevMovement) {
      dom['hud-movement'].textContent = movStr;
      dom['hud-movement'].className   = 'hud-cell-value ' +
        (s.movementType && s.movementType !== 'idle' ? 'green' : 'dim');
      this._prevMovement = movStr;
    }

    const modeStr = s.modeName || '—';
    if (modeStr !== this._prevMode) {
      dom['hud-mode'].textContent = modeStr;
      this._prevMode = modeStr;
    }

    const headStr = (s.facingAngleDeg !== undefined ? s.facingAngleDeg.toFixed(1) : '0.0') + '°';
    if (headStr !== this._prevHeading) {
      dom['hud-heading'].textContent = headStr;
      this._prevHeading = headStr;
    }

    // Mode highlight
    if (this._modeButtonsBuilt) this.updateModeHighlight();

  }

  _applyPendingState(timeMS) {
    const msg = this.pendingState;
    if (!msg) return;
    this.pendingState = null;
    if (!msg.motionSets && this.serverState.motionSets) {
      msg.motionSets = this.serverState.motionSets;
    }
    this.serverState = msg;

    if (!this._frozen && msg.qpos && msg.qpos.length === 36) {
      if (!this.qposCurr) {
        this.qposPrev = new Float64Array(msg.qpos);
        this.qposCurr = new Float64Array(msg.qpos);
      } else {
        if (!this.qposPrev) this.qposPrev = new Float64Array(36);
        this.qposPrev.set(this.qposCurr);
        this.qposCurr.set(msg.qpos);
      }
      this.qposAlpha  = 0;
      this.qposLastMs = timeMS;
      if (this._recipeUI?._isRunning) {
        window._pbQposCount  = (window._pbQposCount  || 0) + 1;
        window._pbQposLastMs = this.qposLastMs;
      }
    }

    if (!this._frozen && msg.qpos_target && msg.qpos_target.length === 36) {
      // On the first target frame after unfreeze, record the x,y root position as
      // the origin so that accumulated drift from previous recipe runs is zeroed out.
      if (!this._targetRootOrigin) {
        this._targetRootOrigin = [msg.qpos_target[0], msg.qpos_target[1]];
      }
      const [ox, oy] = this._targetRootOrigin;
      if (!this.qposTargetCurr) {
        // First frame: initialise both buffers from raw data, then offset both.
        this.qposTargetPrev = new Float64Array(msg.qpos_target);
        this.qposTargetCurr = new Float64Array(msg.qpos_target);
        this.qposTargetPrev[0] -= ox; this.qposTargetPrev[1] -= oy;
        this.qposTargetCurr[0] -= ox; this.qposTargetCurr[1] -= oy;
      } else {
        if (!this.qposTargetPrev) this.qposTargetPrev = new Float64Array(36);
        // qposTargetCurr already has the offset applied — copy it to prev as-is.
        this.qposTargetPrev.set(this.qposTargetCurr);
        // Set curr from raw message and apply offset once.
        this.qposTargetCurr.set(msg.qpos_target);
        this.qposTargetCurr[0] -= ox;
        this.qposTargetCurr[1] -= oy;
      }
    }

    this._updateUI(msg);
  }

  // ── Keyboard ────────────────────────────────────────────────────

  setupKeyboard() {
    const forwardKeys = ['w', 'a', 's', 'd', 'q', 'e', 'r', ',', '.', 'Enter'];

    // Map e.key → data-key attribute value used in HTML
    const toDataKey = (key) => {
      if (key === 'Enter') return 'enter';
      if (key === ' ')     return 'space';
      return key.toLowerCase();
    };

    // Returns true when focus is inside a form element (recipe modal inputs etc.)
    const inFormField = () => {
      const tag = document.activeElement?.tagName;
      return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
    };

    document.addEventListener('keydown', (e) => {
      if (e.repeat) return;

      // Never steal keys while the user is typing in an input / select
      if (inFormField()) return;

      // Green highlight: any hint-key with matching data-key
      const highlightKey = e.code === 'Comma' ? ',' : e.code === 'Period' ? '.' : e.key;
      document.querySelectorAll(`.hint-key[data-key="${toDataKey(highlightKey)}"]`)
              .forEach(el => el.classList.add('key-pressed'));

      if (e.code === 'Space') {
        e.preventDefault();
        this.paused = !this.paused;
        return;
      }
      if (e.code === 'Enter') {
        e.preventDefault();
        // Snapshot current WASM qpos so we can lerp from it to the server pose
        this._unfreezeFromQpos = new Float64Array(this.data.qpos);
        this._unfreezeStartMs  = performance.now();
        this._frozen = false;   // unfreeze: resume tracking server qpos
        this.sendKey('Enter', true);
        return;
      }

      // Normalize comma/period by physical key code to work regardless of IME or layout
      const key = e.code === 'Comma' ? ',' : e.code === 'Period' ? '.' : e.key;
      if (forwardKeys.includes(key.toLowerCase())) {
        // Block strafe keys when unavailable in current mode
        if ((key === ',' || key === '.') && this._prevStrafeDisabled) return;
        this.sendKey(key, true);
      }
    });

    document.addEventListener('keyup', (e) => {
      if (e.repeat) return;
      if (inFormField()) return;

      // Remove green highlight
      const unhighlightKey = e.code === 'Comma' ? ',' : e.code === 'Period' ? '.' : e.key;
      document.querySelectorAll(`.hint-key[data-key="${toDataKey(unhighlightKey)}"]`)
              .forEach(el => el.classList.remove('key-pressed'));

      // Normalize comma/period by physical key code to work regardless of IME or layout
      const key = e.key === 'Enter' ? 'Enter' : e.code === 'Comma' ? ',' : e.code === 'Period' ? '.' : e.key;
      if (forwardKeys.includes(key.toLowerCase()) || key === 'Enter') {
        this.sendKey(key, false);
      }
    });

    document.getElementById('btn-reload').addEventListener('click', () => {
      this.resetScene();
    });

    document.getElementById('btn-activate').addEventListener('click', () => {
      this._unfreezeFromQpos = new Float64Array(this.data.qpos);
      this._unfreezeStartMs  = performance.now();
      this._frozen = false;
      this.sendKey('Enter', true);
    });
  }

  resetScene() {
    // 1. Stop accepting server qpos until user re-activates with Enter
    this._frozen        = true;
    this._unfreezeFromQpos = null;
    this.pendingState = null;
    // Reset recipe UI if it was running (clears is-playing, hides playhead, back to build)
    if (this._recipeUI?._isRunning) {
      this._recipeUI._showBuild();
    }
    this.qposPrev   = null;
    this.qposCurr   = null;
    this.qposTargetPrev = null;
    this.qposTargetCurr = null;
    this.qposAlpha  = 1.0;
    this._targetRootOrigin = null;  // clear drift offset on reset

    // 2. Snap visualization to origin standing pose
    this.mujoco.mj_resetData(this.model, this.data);
    this.data.qpos[0] = DYNAMIC_BASE_POS[0];    // global x
    this.data.qpos[1] = DYNAMIC_BASE_POS[1];    // global y
    this.data.qpos[2] = DYNAMIC_BASE_POS[2];    // global z (standing height)
    this.data.qpos[3] = 1.0;    // quat w
    this.data.qpos[4] = 0.0;
    this.data.qpos[5] = 0.0;
    this.data.qpos[6] = 0.0;
    for (let j = 0; j < NUM_JOINTS; j++) {
      const addr = this.bodyJointQposAddrs[j];
      if (addr >= 0) this.data.qpos[addr] = defaultAngles[j];
    }
    // hand joints remain at 0 (from mj_resetData) — matches virtual_joystick.py
    this.mujoco.mj_forward(this.model, this.data);
    updateBodies(this.model, this.data, this.bodyGroups);

    // 3. Reset camera to default view
    this.controls.target.set(0, 0.8, 0);
    this.camera.position.set(3, 2.5, 3);
    this.controls.update();

    // 4. Tell bridge: stop movement, deactivate, clear heading
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'reset' }));
    }
  }

  // ── Mode UI ─────────────────────────────────────────────────────

  setupModeUI() {
    document.getElementById('speed-slider').addEventListener('input', (e) => {
      this.sendSetSpeed(parseFloat(e.target.value));
    });
    document.getElementById('height-slider').addEventListener('input', (e) => {
      this.sendSetHeight(parseFloat(e.target.value));
    });
  }

  buildModeButtons(motionSets) {
    const tabs      = this._dom['set-tabs'];
    const container = this._dom['mode-buttons'];
    tabs.innerHTML      = '';
    container.innerHTML = '';

    // Expose motionSets globally so the inline editor can look up speed ranges
    window._motionSetsData = motionSets;

    motionSets.forEach((set, setIdx) => {
      const tab = document.createElement('button');
      tab.className = 'set-tab';
      tab.textContent = set.name;
      tab.dataset.setIdx = setIdx;
      tab.onclick = () => {
        this.sendSetMode(setIdx, (this.serverState.motionSets?.[setIdx]?.modes?.[0]?.key) ?? 1);
      };
      tabs.appendChild(tab);

      set.modes.forEach((mode) => {
        const item = document.createElement('div');
        item.className = 'mode-item';
        item.dataset.setIdx    = setIdx;
        item.dataset.modeKey   = mode.key;
        const strKey = _modeNameToKey(mode.name);
        item.dataset.modeStrKey = strKey;

        // Store speed range for inline editor
        if (mode.speed_range) {
          item.dataset.speedMin     = mode.speed_range[0];
          item.dataset.speedMax     = mode.speed_range[1];
          item.dataset.speedDefault = mode.default_speed ?? 1.0;
        }
        // Height control: derive from the set's height_control flag
        item.dataset.hasHeight = set.height_control ? '1' : '0';

        const speedBadge = mode.speed_range ? '<span class="mode-tag speed-tag">SPD</span>' : '';

        item.innerHTML = `
          <span class="mode-name">${mode.name}</span>
          <div class="mode-badges">${speedBadge}</div>`;

        item.onclick = () => { this.sendSetMode(setIdx, mode.key); };
        container.appendChild(item);
      });
    });

    this._updateSetVisibility(0);
    this._modeButtonsBuilt = true;
  }

  _updateSetVisibility(activeSetIdx) {
    if (this._prevSetIdx === activeSetIdx) return;
    this._prevSetIdx = activeSetIdx;
    this._dom['mode-buttons'].querySelectorAll('.mode-item').forEach(item => {
      item.style.display = parseInt(item.dataset.setIdx) === activeSetIdx ? 'flex' : 'none';
    });
    this._dom['set-tabs'].querySelectorAll('.set-tab').forEach((tab, i) => {
      tab.classList.toggle('active', i === activeSetIdx);
    });
  }

  updateModeHighlight() {
    const s = this.serverState;
    if (s.motionSetIdx === undefined) return;

    if (document.body.classList.contains('mode-edit') && window._segSelectionActive) {
      return;
    }

    this._updateSetVisibility(s.motionSetIdx);

    // Only update active item if selection changed
    const key = `${s.motionSetIdx}:${s.modeKey}`;
    if (key === this._prevModeKey) return;
    this._prevModeKey = key;

    this._dom['mode-buttons'].querySelectorAll('.mode-item').forEach(item => {
      const match = parseInt(item.dataset.setIdx) === s.motionSetIdx &&
                    parseInt(item.dataset.modeKey) === s.modeKey;
      item.classList.toggle('active', match);
    });
  }

  // ── Recipe Composer UI ──────────────────────────────────────────

  setupRecipeUI() {
    this._recipeUI = new RecipeUI(
      (msg) => {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify(msg));
        }
      },
      () => this.serverState?.motionSets || [],
    );
  }

  // ── Render loop — pure GPU work only ────────────────────────────

  render(timeMS) {
    this._applyPendingState(timeMS);

    // FPS counter (1 DOM write per second)
    this.frameCount++;
    if (timeMS - this.lastFpsTime > 1000) {
      this._dom['hud-fps'].textContent = this.frameCount;
      this.frameCount  = 0;
      this.lastFpsTime = timeMS;
    }

    const renderMode = window.getRenderMode?.() || 'kinematic';
    const posePrev = renderMode === 'kinematic'
      ? (this.qposTargetPrev || this.qposPrev)
      : this.qposPrev;
    const poseCurr = renderMode === 'kinematic'
      ? (this.qposTargetCurr || this.qposCurr)
      : this.qposCurr;

    if (!this.paused && poseCurr) {
      // ── Interpolation + dead-reckoning between server qpos frames ──
      // alpha 0→1  = normal lerp between prev and curr frames.
      // alpha 1→1.5 = predictive extrapolation along the prev→curr velocity,
      //               so the display keeps advancing if the next packet is late
      //               rather than freezing at the last received pose.
      // Cap at 1.5 to limit drift on genuine packet loss.
      const elapsed   = (timeMS - this.qposLastMs) / 1000;
      const alpha     = Math.min(elapsed / this.SERVER_DT, 1.5);
      const prev      = posePrev;
      const curr      = poseCurr;
      const qpos      = this.data.qpos;

      // Unfreeze blend: lerp from the frozen visual pose to the live server pose
      // over _UNFREEZE_DUR_MS so the robot doesn't snap to a distant position.
      let unfreezeBlend = 1.0;
      if (this._unfreezeFromQpos) {
        const dt = timeMS - this._unfreezeStartMs;
        unfreezeBlend = Math.min(dt / this._UNFREEZE_DUR_MS, 1.0);
        if (unfreezeBlend >= 1.0) this._unfreezeFromQpos = null; // blend done
      }
      const fromQ = this._unfreezeFromQpos;

      // Root pose (pos + quat): always contiguous at qpos[0..6]
      // Dynamic mode keeps root translation fixed for a stable in-place view.
      // Kinematic mode follows the target root translation and orientation.
      for (let i = 0; i < 7; i++) {
        let base = prev ? prev[i] + alpha * (curr[i] - prev[i]) : curr[i];
        if (renderMode === 'dynamic' && i < 3) {
          base = DYNAMIC_BASE_POS[i];
        }
        qpos[i] = fromQ ? (fromQ[i] + unfreezeBlend * (base - fromQ[i])) : base;
      }
      // Body joints: server data is at indices [7..35] in BODY_JOINT_NAMES order,
      // but the with_hand model interleaves hand joints so we must use name-mapped addresses.
      for (let j = 0; j < NUM_JOINTS; j++) {
        const addr = this.bodyJointQposAddrs[j];
        if (addr < 0) continue;
        const base = prev ? prev[7 + j] + alpha * (curr[7 + j] - prev[7 + j]) : curr[7 + j];
        qpos[addr] = fromQ ? (fromQ[addr] + unfreezeBlend * (base - fromQ[addr])) : base;
      }
      // hand joints are not driven by server — they stay at their reset default (0)

      this.mujoco.mj_forward(this.model, this.data);
      updateBodies(this.model, this.data, this.bodyGroups);


      if (this.cameraFollow) {
        const [tx, ty, tz] = swizzlePos(qpos[0], qpos[1], qpos[2]);
        this._camTarget.set(tx, ty + 0.5, tz);
        // In kinematic mode the robot walks through the world, so we want the
        // camera to keep it centred at all times while still allowing the user
        // to orbit/zoom freely.  OrbitControls stores its view as a spherical
        // offset from `controls.target`; when `target` changes the camera
        // position moves by the same delta, preserving the orbit shape.
        // lerp = 1.0  → robot always at the exact screen centre (pan disabled)
        // lerp = 0.05 → slow follow (original behaviour for dynamic/fixed mode)
        const followLerp = renderMode === 'kinematic' ? 1.0 : 0.05;
        this.controls.target.lerp(this._camTarget, followLerp);
          this.sunLight.position.set(tx + 5, 10, tz + 4);
          this.sunLight.target.position.set(tx, 0, tz);
          this.sunLight.target.updateMatrixWorld();
      }
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

// ─── Recipe Composer ─────────────────────────────────────────────────────

/**
 * Annotation style labels matching templates.py STYLES order.
 * Index 0-5 are canonical sentences; 6-13 are per-style; 14+ are mixed.
 */
// Order matches compose_annotations() output:
// [0-2] Timed canonicals (Robot / Person / verb-first)
// [3-5] Action-only canonicals (Robot / Person / verb-first)
// [6-13] 8 styles (Instruction … Concise no-time)
// [14-16] Mixed diversity
const ANN_STYLE_LABELS = [
  'Canonical (Robot, Timed)',    // 0: "The robot walks forward for 3.0 seconds."
  'Canonical (Person, Timed)',   // 1: "The person walks forward for 3.0 seconds."
  'Canonical (Timed)',           // 2: "Walks forward for 3.0 seconds."
  'Canonical (Robot, Action)',   // 3: "The robot walks forward."
  'Canonical (Person, Action)',  // 4: "The person walks forward."
  'Canonical (Action)',          // 5: "Walks forward."
  'Instruction',                 // 6
  'Natural',                     // 7
  'Narrative',                   // 8
  'Concise',                     // 9
  'Instruction (no time)',       // 10
  'Natural (no time)',           // 11
  'Narrative (no time)',         // 12
  'Concise (no time)',           // 13
  'Mixed A',                     // 14
  'Mixed B',                     // 15
  'Mixed C',                     // 16
];

function _escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

class RecipeUI {
  /**
   * @param {(msg: object) => void} send  - function to send a WS message
   * @param {() => object[]}        getMotionSets - returns motionSets from serverState
   */
  constructor(send, getMotionSets) {
    this._send         = send;
    this._getMotionSets = getMotionSets;
    this._segments     = [];   // [{mode, direction, turn_dir, turn_deg, duration, speed}]
    this._result       = null;
    this._activeTab    = 'full';
    this._domCache     = {};
    this._isRunning        = false;   // true while a recipe is executing
    this._minimized        = false;   // true when HUD is shown instead of full modal
    this._playheadStarted  = false;   // true once playhead has been kicked off for this run
    this._runLog           = null;    // client-side timing markers for the current run
    this._initDom();
  }

  // ── DOM wiring ────────────────────────────────────────────────

  _q(id) {
    if (!this._domCache[id]) this._domCache[id] = document.getElementById(id);
    return this._domCache[id];
  }

  _initDom() {
    // Open / close
    document.getElementById('btn-recipe')?.addEventListener('click', () => this.open());
    document.getElementById('recipe-close').addEventListener('click', () => this.close());

    // Build view
    document.getElementById('recipe-add-seg').addEventListener('click', () => this._addSegment());
    document.getElementById('recipe-run').addEventListener('click', () => this._runRecipe());

    // Segment reorder from visual track drag
    document.addEventListener('recipe-reorder', e => {
      const { from, to } = e.detail;
      if (from < 0 || from >= this._segments.length) return;
      const seg = this._segments.splice(from, 1)[0];
      this._segments.splice(to, 0, seg);
    });

    // Running view
    document.getElementById('recipe-cancel').addEventListener('click', () => {
      this._send({ type: 'cancel_recipe' });
    });

    // Minimize / expand
    document.getElementById('recipe-minimize').addEventListener('click', () => this._minimize());
    document.getElementById('recipe-hud-expand').addEventListener('click', () => this._expand());
    document.getElementById('recipe-hud-cancel').addEventListener('click', () => {
      this._send({ type: 'cancel_recipe' });
    });

    // Results view — tabs
    document.getElementById('recipe-results').addEventListener('click', (e) => {
      const tab = e.target.closest('.res-tab');
      if (!tab) return;
      this._activeTab = tab.dataset.tab;
      document.querySelectorAll('.res-tab').forEach(t =>
        t.classList.toggle('active', t.dataset.tab === this._activeTab));
      document.getElementById('res-full').classList.toggle('hidden', this._activeTab !== 'full');
      document.getElementById('res-seg').classList.toggle('hidden', this._activeTab !== 'seg');
    });

    // Copy all
    document.getElementById('res-copy-all').addEventListener('click', () => {
      if (!this._result) return;
      const text = this._result.annotations.full_trajectory?.[0] || '';
      navigator.clipboard.writeText(text).then(() => {
        const btn = document.getElementById('res-copy-all');
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = 'Copy All'; }, 1500);
      });
    });

    // New recipe
    document.getElementById('recipe-new').addEventListener('click', () => this._showBuild());
  }

  // ── Segment management ────────────────────────────────────────

  _addSegment() {
    // Look up default_speed for 'walk' from motionSets
    // 'walk' is in the Locomotion set (no height_control), so height = -1.0
    let defaultSpeed = -1.0;
    for (const ms of (this._getMotionSets() || [])) {
      for (const m of ms.modes) {
        if (_modeNameToKey(m.name) === 'walk' && m.default_speed > 0) {
          defaultSpeed = m.default_speed;
        }
      }
    }
    this._segments.push({
      mode: 'walk', direction: 'forward',
      turn_dir: null, turn_deg: 90,
      duration: 2.0, speed: defaultSpeed, height: -1.0,
    });
    this._rebuildSegmentList();
    // Scroll to new row
    const list = document.getElementById('recipe-segments');
    list.lastElementChild?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  _removeSegment(idx) {
    this._segments.splice(idx, 1);
    this._rebuildSegmentList();
  }

  _rebuildSegmentList() {
    const container = document.getElementById('recipe-segments');
    container.innerHTML = '';

    if (this._segments.length === 0) {
      container.innerHTML = '<div class="seg-empty">No segments yet — click "+ Add Segment" below.</div>';
      return;
    }

    const motionSets = this._getMotionSets();

    this._segments.forEach((seg, idx) => {
      const row = document.createElement('div');
      row.className = 'recipe-segment';
      row.dataset.idx = idx;

      // Mode select (grouped by set)
      const modeSelect = document.createElement('select');
      modeSelect.className = 'seg-mode';
      if (motionSets.length > 0) {
        motionSets.forEach(set => {
          const grp = document.createElement('optgroup');
          grp.label = set.name;
          set.modes.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.name.toLowerCase().replace(/\s+/g, '_').replace(/[()]/g, '');
            // Map display name to internal key (match ws_bridge MODE_NAME_TO_KEY)
            opt.value = _modeNameToKey(m.name);
            opt.textContent = m.name;
            if (opt.value === seg.mode) opt.selected = true;
            grp.appendChild(opt);
          });
          modeSelect.appendChild(grp);
        });
      } else {
        // Fallback if motionSets not yet received
        [['walk','Walk'],['run','Run'],['slow_walk','Slow Walk']].forEach(([v,l]) => {
          const opt = document.createElement('option');
          opt.value = v; opt.textContent = l;
          if (v === seg.mode) opt.selected = true;
          modeSelect.appendChild(opt);
        });
      }
      // ── Speed input ────────────────────────────────────────────
      const speedWrap = document.createElement('div');
      speedWrap.className = 'seg-speed-wrap';
      const speedInput = document.createElement('input');
      speedInput.type = 'number'; speedInput.className = 'seg-speed';
      speedInput.min = 0.1; speedInput.max = 5.0; speedInput.step = 0.1;
      const speedUnit = document.createElement('span');
      speedUnit.className = 'seg-dur-unit'; speedUnit.textContent = 'm/s';
      speedWrap.appendChild(speedInput); speedWrap.appendChild(speedUnit);

      // ── Height hidden input (tracks per-segment height) ────────
      const heightInput = document.createElement('input');
      heightInput.type = 'hidden'; heightInput.className = 'seg-height';
      heightInput.value = seg.height ?? -1.0;

      // Helper: get mode info from motionSets by internal key
      const getModeInfo = (modeKey) => {
        for (const ms of (this._getMotionSets() || [])) {
          for (const m of ms.modes) {
            if (_modeNameToKey(m.name) === modeKey) return m;
          }
        }
        return null;
      };

      // Sync speed input visibility + value to a given mode key
      const syncSpeed = (modeKey, forceDefault = false) => {
        const info = getModeInfo(modeKey);
        if (info?.speed_range) {
          speedWrap.style.display = 'flex';
          if (forceDefault || this._segments[idx].speed < 0) {
            const ds = (info.default_speed > 0) ? info.default_speed : info.speed_range[0];
            speedInput.value = ds;
            this._segments[idx].speed = ds;
          } else {
            speedInput.value = this._segments[idx].speed;
          }
          speedInput.min  = info.speed_range[0];
          speedInput.max  = info.speed_range[1];
        } else {
          speedWrap.style.display = 'none';
          this._segments[idx].speed = -1.0;
        }
      };

      // Sync height hidden input to a given mode key
      const syncHeight = (modeKey, forceDefault = false) => {
        const hasHeightCtrl = this._getMotionSets()?.find(ms =>
          ms.modes?.some(m => _modeNameToKey(m.name) === modeKey)
        )?.height_control ?? false;
        if (hasHeightCtrl) {
          if (forceDefault || this._segments[idx].height < 0) {
            heightInput.value = 0.8;
            this._segments[idx].height = 0.8;
          } else {
            heightInput.value = this._segments[idx].height;
          }
        } else {
          heightInput.value = -1.0;
          this._segments[idx].height = -1.0;
        }
      };

      syncSpeed(seg.mode);
      syncHeight(seg.mode);

      modeSelect.addEventListener('change', () => {
        this._segments[idx].mode = modeSelect.value;
        syncSpeed(modeSelect.value, true);  // reset to new mode's default
        syncHeight(modeSelect.value, true); // reset to new mode's default
      });
      speedInput.addEventListener('change', () => {
        this._segments[idx].speed = parseFloat(speedInput.value) || -1.0;
      });
      heightInput.addEventListener('change', () => {
        this._segments[idx].height = parseFloat(heightInput.value);
      });

      // Direction select (includes turn options)
      const dirSelect = document.createElement('select');
      dirSelect.className = 'seg-dir';
      const dirOptions = [
        { value: 'forward',      label: '→ Forward' },
        { value: 'backward',     label: '← Backward' },
        { value: 'strafe_left',  label: '↑ Strafe Left' },
        { value: 'strafe_right', label: '↓ Strafe Right' },
        { value: 'idle',         label: '• Idle' },
        { value: 'turn_left',    label: '↺ Turn Left' },
        { value: 'turn_right',   label: '↻ Turn Right' },
      ];
      dirOptions.forEach(({ value, label }) => {
        const opt = document.createElement('option');
        opt.value = value; opt.textContent = label;
        const isTurn = seg.turn_dir !== null;
        if (isTurn && value === `turn_${seg.turn_dir}`) opt.selected = true;
        else if (!isTurn && value === seg.direction) opt.selected = true;
        dirSelect.appendChild(opt);
      });

      // Angle input (shown only when turn selected)
      const angleWrap = document.createElement('div');
      angleWrap.className = 'seg-angle-wrap';
      const angleInput = document.createElement('input');
      angleInput.type = 'number'; angleInput.className = 'seg-angle';
      angleInput.min = 10; angleInput.max = 720; angleInput.step = 5;
      angleInput.value = seg.turn_deg ?? 90;
      const angleUnit = document.createElement('span');
      angleUnit.className = 'seg-angle-unit'; angleUnit.textContent = '°';
      angleWrap.appendChild(angleInput); angleWrap.appendChild(angleUnit);

      const isTurnSel = seg.turn_dir !== null;
      angleWrap.style.display = isTurnSel ? 'flex' : 'none';

      dirSelect.addEventListener('change', () => {
        const val = dirSelect.value;
        if (val === 'turn_left' || val === 'turn_right') {
          this._segments[idx].turn_dir = val === 'turn_left' ? 'left' : 'right';
          this._segments[idx].direction = 'forward';
          angleWrap.style.display = 'flex';
        } else {
          this._segments[idx].turn_dir = null;
          this._segments[idx].direction = val;
          angleWrap.style.display = 'none';
        }
      });
      angleInput.addEventListener('change', () => {
        this._segments[idx].turn_deg = parseFloat(angleInput.value) || 90;
      });

      // Duration input
      const durWrap = document.createElement('div');
      durWrap.className = 'seg-dur-wrap';
      const durInput = document.createElement('input');
      durInput.type = 'number'; durInput.className = 'seg-dur';
      durInput.min = 0.5; durInput.max = 60; durInput.step = 0.5;
      durInput.value = seg.duration;
      const durUnit = document.createElement('span');
      durUnit.className = 'seg-dur-unit'; durUnit.textContent = 's';
      durWrap.appendChild(durInput); durWrap.appendChild(durUnit);
      durInput.addEventListener('change', () => {
        this._segments[idx].duration = parseFloat(durInput.value) || 2.0;
      });

      // Delete button
      const delBtn = document.createElement('button');
      delBtn.className = 'seg-del'; delBtn.textContent = '×';
      delBtn.title = 'Remove segment';
      delBtn.addEventListener('click', () => this._removeSegment(idx));

      // Index label
      const numSpan = document.createElement('span');
      numSpan.className = 'seg-num'; numSpan.textContent = idx + 1;

      row.appendChild(numSpan);
      row.appendChild(modeSelect);
      row.appendChild(dirSelect);
      row.appendChild(angleWrap);
      row.appendChild(speedWrap);
      row.appendChild(heightInput);
      row.appendChild(durWrap);
      row.appendChild(delBtn);
      container.appendChild(row);
    });
  }

  // ── Run recipe ────────────────────────────────────────────────

  _runRecipe() {
    const errEl = document.getElementById('recipe-build-error');
    errEl.style.display = 'none';

    if (this._segments.length === 0) {
      errEl.textContent = 'Add at least one segment before running.';
      errEl.style.display = 'block';
      return;
    }

    const name     = document.getElementById('recipe-name').value.trim() || 'web_recipe';
    const doRecord = true;
    const initWait = RECIPE_INIT_WAIT_SEC;

    const segments = this._segments.map(s => {
      const seg = { mode: s.mode, duration: s.duration };
      if (s.turn_dir) {
        seg.turn  = s.turn_dir;
        seg.angle = s.turn_deg;
      } else {
        seg.direction = s.direction;
      }
      if (s.speed  > 0) seg.speed  = s.speed;
      if (s.height > 0) seg.height = s.height;
      return seg;
    });

    const renderMode = window.getRenderMode?.() || 'dynamic';
    this._runLog = {
      clientPlayClickedAt: performance.now(),
      clientPlayClickedIso: new Date().toISOString(),
      runRecipeSentAt: null,
      executingSeenAt: null,
    };
    console.info(
      `[recipe] PLAY clicked @ ${this._runLog.clientPlayClickedIso} ` +
      `(init_wait=${initWait.toFixed(1)}s, segments=${segments.length}, render=${renderMode})`
    );
    this._runLog.runRecipeSentAt = performance.now();
    this._send({ type: 'run_recipe', name, segments, record: doRecord, init_wait: initWait, render_mode: renderMode });
    this._showRunning();
    console.info(
      `[recipe] run_recipe sent +${(this._runLog.runRecipeSentAt - this._runLog.clientPlayClickedAt).toFixed(1)}ms`
    );
    this._setProgress(0, `Warming up controller… (${initWait.toFixed(1)}s)`);
  }

  // ── View transitions ──────────────────────────────────────────

  open() {
    this._minimized = false;
    document.getElementById('recipe-hud').classList.add('hidden');
    document.getElementById('recipe-modal').classList.remove('hidden');
    if (this._segments.length > 0) this._rebuildSegmentList();
  }

  close() {
    // While a recipe is running, minimize to HUD instead of hiding everything
    if (this._isRunning) {
      this._minimize();
    } else {
      document.getElementById('recipe-modal').classList.add('hidden');
    }
  }

  _minimize() {
    this._minimized = true;
    document.getElementById('recipe-modal').classList.add('hidden');
    document.getElementById('recipe-hud').classList.remove('hidden');
  }

  _expand() {
    this._minimized = false;
    document.getElementById('recipe-hud').classList.add('hidden');
    document.getElementById('recipe-modal').classList.remove('hidden');
    // Ensure running view is visible when expanding back
    if (this._isRunning) {
      document.getElementById('recipe-build').classList.add('hidden');
      document.getElementById('recipe-running').classList.remove('hidden');
      document.getElementById('recipe-results').classList.add('hidden');
    }
  }

  _showBuild() {
    this._isRunning = false;
    this._setSeqRunState(false);
    document.body.classList.remove('is-playing');
    window.hidePlayhead?.();
    // Hide HUD if it was showing
    document.getElementById('recipe-hud').classList.add('hidden');
    document.getElementById('recipe-minimize').classList.add('hidden');
    this._minimized = false;
    // Ensure modal is visible
    document.getElementById('recipe-modal').classList.remove('hidden');
    document.getElementById('recipe-build').classList.remove('hidden');
    document.getElementById('recipe-running').classList.add('hidden');
    document.getElementById('recipe-results').classList.add('hidden');
    document.getElementById('recipe-build-error').style.display = 'none';
    this._rebuildSegmentList();
  }

  _showRunning() {
    this._isRunning       = true;
    this._playheadStarted = false;
    window.hidePlayhead?.();   // reset from any previous run
    document.getElementById('recipe-results').classList.add('hidden');
    // In edit mode: keep recipe-modal + track visible; show HUD for progress only
    if (document.body.classList.contains('mode-edit')) {
      document.getElementById('recipe-minimize').classList.add('hidden');
      this._setSeqRunState(true);
      document.body.classList.add('is-playing');
      const bar = document.getElementById('seq-run-bar');
      const label = document.getElementById('seq-run-label');
      if (bar) bar.style.width = '0%';
      if (label) label.textContent = `Initializing… (${RECIPE_INIT_WAIT_SEC.toFixed(1)}s)`;
      // Never show floating HUD popup in edit mode.
      document.getElementById('recipe-hud').classList.add('hidden');
      return;
    }
    document.getElementById('recipe-minimize').classList.remove('hidden');
    document.getElementById('recipe-build').classList.add('hidden');
    document.getElementById('recipe-running').classList.remove('hidden');
    // Reset label + bar for fresh run
    document.getElementById('recipe-prog-label').textContent = `Warming up controller… (${RECIPE_INIT_WAIT_SEC.toFixed(1)}s)`;
    document.getElementById('recipe-prog-bar').style.width = '0%';
  }

  _showResults() {
    this._isRunning = false;
    this._setSeqRunState(false);
    if (document.body.classList.contains('mode-edit')) {
      // Edit mode: track stays visible; results surface via file panel only
      document.body.classList.remove('is-playing');
      document.getElementById('recipe-hud').classList.add('hidden');
      document.getElementById('recipe-minimize').classList.add('hidden');
      // Update export button state and sync download links
      window._setExportState?.('ready');
      window._syncFilePanel?.();
      // Animate file icons flying to export button 1s after play ends, then open panel
      window.hidePlayhead?.();
      window.setFilePanelPreview?.(this._result?.annotations?.full_trajectory);
      setTimeout(() => {
        window.triggerFileFly?.();
        setTimeout(() => window.openFilePanel?.(), 500);
      }, 1000);
      return;
    }
    // Auto-expand from HUD when results are ready
    if (this._minimized) this._expand();
    document.getElementById('recipe-hud').classList.add('hidden');
    document.getElementById('recipe-minimize').classList.add('hidden');
    document.getElementById('recipe-build').classList.add('hidden');
    document.getElementById('recipe-running').classList.add('hidden');
    document.getElementById('recipe-results').classList.remove('hidden');
  }

  _setProgress(pct, label) {
    const w = `${Math.round(pct * 100)}%`;
    const cleaned = this._cleanStatusLabel(label);
    document.getElementById('recipe-prog-bar').style.width = w;
    document.getElementById('recipe-prog-label').textContent = cleaned;
    const seqBar = document.getElementById('seq-run-bar');
    const seqLabel = document.getElementById('seq-run-label');
    if (seqBar) seqBar.style.width = w;
    if (seqLabel) seqLabel.textContent = cleaned;
    // Mirror to HUD when minimized
    document.getElementById('recipe-hud-bar').style.width = w;
    // Truncate long labels for the chip
    const short = cleaned.length > 48 ? cleaned.slice(0, 46) + '…' : cleaned;
    document.getElementById('recipe-hud-label').textContent = short;
  }

  _setSeqRunState(active) {
    const el = document.getElementById('seq-runstat');
    if (!el) return;
    el.classList.toggle('on', !!active);
  }

  _cleanStatusLabel(label) {
    let text = String(label || '').trim();
    // Remove leading progress counters like [1/1].
    text = text.replace(/^\s*\[\d+\/\d+\]\s*/i, '');
    // Remove viewer wording from server status text.
    text = text.replace(/\s*\(?(in\s+)?viewer\)?/ig, '');
    text = text.replace(/\s{2,}/g, ' ').trim();
    return text || 'Running…';
  }

  // ── WS callbacks ──────────────────────────────────────────────

  onStatus(msg) {
    const pct = msg.progress ?? 0;
    this._setProgress(pct, msg.message || '');
    if (msg.phase === 'init_wait') {
      console.info(`[recipe] bridge status: ${msg.message || 'warming up controller'}`);
    }
    // Start playhead exactly when the robot begins executing its first segment
    if (!this._playheadStarted && msg.phase === 'executing') {
      if (this._runLog && this._runLog.executingSeenAt == null) {
        this._runLog.executingSeenAt = performance.now();
        const sinceClick = this._runLog.executingSeenAt - this._runLog.clientPlayClickedAt;
        const sinceSend = this._runLog.executingSeenAt - this._runLog.runRecipeSentAt;
        console.info(
          `[recipe] first executing status after +${sinceClick.toFixed(1)}ms from PLAY ` +
          `(+${sinceSend.toFixed(1)}ms from run_recipe send)`
        );
      }
      this._playheadStarted    = true;
      window._pbQposCount      = 0;
      window._pbQposLastMs     = performance.now();
      window.startPlayhead?.();
    }
  }

  onResult(result) {
    if (this._runLog) {
      const now = performance.now();
      console.info(
        `[recipe] result received after +${(now - this._runLog.clientPlayClickedAt).toFixed(1)}ms from PLAY`
      );
    }
    this._result = result;
    this._renderResults(result);
    this._showResults();
  }

  onError(msg) {
    if (this._runLog) {
      const now = performance.now();
      console.warn(
        `[recipe] error received after +${(now - this._runLog.clientPlayClickedAt).toFixed(1)}ms from PLAY: ${msg.message}`
      );
    }
    this._isRunning = false;
    this._setSeqRunState(false);
    document.body.classList.remove('is-playing');
    window.hidePlayhead?.();
    document.getElementById('recipe-hud').classList.add('hidden');
    document.getElementById('recipe-minimize').classList.add('hidden');
    if (msg.cancelled) {
      // User explicitly cancelled — just go back to build
      if (this._minimized) this._expand();
      this._minimized = false;
      this._showBuild();
      return;
    }
    // Auto-expand from HUD so the error is visible
    if (this._minimized) this._expand();
    this._minimized = false;
    // Show error inline in the running view with a Back button
    const label = document.getElementById('recipe-prog-label');
    label.innerHTML =
      `<span style="color:var(--red)">Error: ${_escHtml(msg.message)}</span>` +
      `<br><button id="recipe-err-back" style="margin-top:10px;padding:6px 18px;` +
      `background:transparent;border:1px solid var(--border-hi);` +
      `border-radius:var(--radius);color:var(--text2);` +
      `font-family:var(--mono);font-size:10px;cursor:pointer">← Back to Edit</button>`;
    document.getElementById('recipe-prog-bar').style.width = '0%';
    document.getElementById('recipe-err-back')?.addEventListener('click', () => {
      this._showBuild();
    });
  }

  // ── Render results ─────────────────────────────────────────────

  _renderResults(result) {
    const ann = result.annotations || {};

    // ── Full trajectory tab ──────────────────────────────────────
    const fullPane = document.getElementById('res-full');
    fullPane.innerHTML = '';
    const fullList = ann.full_trajectory || [];
    fullList.forEach((text, i) => {
      const label  = ANN_STYLE_LABELS[i] || `Style ${i}`;
      const isCanon = i < 6;
      const row    = document.createElement('div');
      row.className = 'ann-row';

      const pill = document.createElement('span');
      pill.className = 'ann-style-pill' + (isCanon ? ' canonical' : '');
      pill.textContent = label;

      const textEl = document.createElement('span');
      textEl.className = 'ann-text';
      textEl.textContent = text;

      const copyBtn = document.createElement('button');
      copyBtn.className = 'ann-copy'; copyBtn.textContent = 'Copy';
      copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(text).then(() => {
          copyBtn.textContent = 'Copied';
          setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1200);
        });
      });

      row.appendChild(pill); row.appendChild(textEl); row.appendChild(copyBtn);
      fullPane.appendChild(row);
    });

    // ── Per-segment tab ──────────────────────────────────────────
    const segPane = document.getElementById('res-seg');
    segPane.innerHTML = '';
    const STYLES_ORDER = [
      'Instruction','Natural','Narrative','Concise',
      'Instruction (no time)','Natural (no time)','Narrative (no time)','Concise (no time)',
    ];
    (ann.per_segment || []).forEach(segAnn => {
      const acc = document.createElement('div');
      acc.className = 'seg-accordion';

      const header = document.createElement('div');
      header.className = 'seg-acc-header';
      header.innerHTML = `<span>Segment ${segAnn.segment_idx + 1}</span><span class="seg-acc-arrow">▶</span>`;
      header.addEventListener('click', () => {
        const body  = acc.querySelector('.seg-acc-body');
        const arrow = acc.querySelector('.seg-acc-arrow');
        const open  = body.classList.toggle('open');
        arrow.classList.toggle('open', open);
      });

      const body = document.createElement('div');
      body.className = 'seg-acc-body';
      (segAnn.annotations || []).forEach((text, i) => {
        const row  = document.createElement('div');
        row.className = 'ann-row';

        const pill = document.createElement('span');
        pill.className = 'ann-style-pill';
        pill.textContent = STYLES_ORDER[i] || `Style ${i}`;

        const textEl = document.createElement('span');
        textEl.className = 'ann-text'; textEl.textContent = text;

        const copyBtn = document.createElement('button');
        copyBtn.className = 'ann-copy'; copyBtn.textContent = 'Copy';
        copyBtn.addEventListener('click', () => {
          navigator.clipboard.writeText(text).then(() => {
            copyBtn.textContent = 'Copied';
            setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1200);
          });
        });

        row.appendChild(pill); row.appendChild(textEl); row.appendChild(copyBtn);
        body.appendChild(row);
      });

      acc.appendChild(header); acc.appendChild(body);
      segPane.appendChild(acc);
    });

    // ── Downloads ─────────────────────────────────────────────────
    const csvLink       = document.getElementById('res-dl-csv');
    const csvTargetLink = document.getElementById('res-dl-csv-target');
    const jsonLink      = document.getElementById('res-dl-json');

    if (result.csv_url) {
      csvLink.href = result.csv_url;
      csvLink.classList.remove('hidden');
    } else {
      csvLink.classList.add('hidden');
    }

    if (csvTargetLink) {
      if (result.csv_target_url) {
        csvTargetLink.href = result.csv_target_url;
        csvTargetLink.classList.remove('hidden');
      } else {
        csvTargetLink.classList.add('hidden');
      }
    }

    // Annotation JSON: server-side URL for the full annotation.json
    if (result.ann_url) {
      jsonLink.href = result.ann_url;
    }

    // Also switch to full-trajectory tab by default
    this._activeTab = 'full';
    document.querySelectorAll('.res-tab').forEach(t =>
      t.classList.toggle('active', t.dataset.tab === 'full'));
    document.getElementById('res-full').classList.remove('hidden');
    document.getElementById('res-seg').classList.add('hidden');
  }
}

// Map ws_bridge display name → internal mode key (mirrors annotation.js MODE_NAME_TO_KEY)
function _modeNameToKey(displayName) {
  const MAP = {
    'Slow Walk': 'slow_walk', 'Walk': 'walk', 'Run': 'run',
    'Happy': 'happy', 'Stealth': 'stealth', 'Injured': 'injured',
    'Squat': 'squat', 'Kneel (Two)': 'kneel_two', 'Kneel (One)': 'kneel_one',
    'Hand Crawl': 'hand_crawl', 'Elbow Crawl': 'elbow_crawl',
    'Idle Boxing': 'idle_boxing', 'Walk Boxing': 'walk_boxing',
    'Left Jab': 'left_jab', 'Right Jab': 'right_jab',
    'Random Punches': 'random_punches', 'Left Hook': 'left_hook', 'Right Hook': 'right_hook',
    'Careful': 'careful', 'Object Carrying': 'object_carrying', 'Crouch': 'crouch',
    'Happy Dance': 'happy_dance', 'Zombie': 'zombie', 'Point': 'point', 'Scared': 'scared',
  };
  return MAP[displayName] ?? displayName.toLowerCase().replace(/\s+/g, '_').replace(/[()]/g, '');
}

// ─── Bootstrap ───────────────────────────────────────────────────────

const demo = new G1WasmDemo();
demo.init().catch((err) => {
  console.error('Fatal:', err);
  document.getElementById('loading-status').textContent = `Error: ${err.message}`;
});
