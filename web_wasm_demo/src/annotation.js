/**
 * Client-side motion annotation tracker.
 *
 * Builds a persistent, growing narrative sentence from successive motion
 * segments detected in the ws_bridge state stream. Narrative persists until
 * reset() is called (hooked to the Reset/Backspace button in main.js).
 *
 * State fields consumed:
 *   s.movementType    "forward" | "backward" | "strafe_left" | "strafe_right" | "idle"
 *   s.modeName        "Slow Walk" | "Walk" | "Run" | ...
 *   s.facingAngleDeg  float
 *   s.speed           float (m/s, -1 if unset)
 *   s.active          bool
 */

// ── modeName (from ws_bridge) → internal mode key ─────────────────────────

const MODE_NAME_TO_KEY = {
  'slow walk':       'slow_walk',
  'walk':            'walk',
  'run':             'run',
  'happy':           'happy',
  'stealth':         'stealth',
  'injured':         'injured',
  'squat':           'squat',
  'kneel (two)':     'kneel_two',
  'kneel (one)':     'kneel_one',
  'hand crawl':      'hand_crawl',
  'elbow crawl':     'elbow_crawl',
  'idle boxing':     'idle_boxing',
  'walk boxing':     'walk_boxing',
  'left jab':        'left_jab',
  'right jab':       'right_jab',
  'random punches':  'random_punches',
  'left hook':       'left_hook',
  'right hook':      'right_hook',
  'careful':         'careful',
  'object carrying': 'object_carrying',
  'crouch':          'crouch',
  'happy dance':     'happy_dance',
  'zombie':          'zombie',
  'point':           'point',
  'scared':          'scared',
};

// ── Canonical verb table (matches templates.py canonical_verb) ────────────

const CANONICAL_VERB = {
  // Locomotion
  run:             'runs',
  walk:            'walks',
  slow_walk:       'walks slowly',
  happy:           'walks happily',
  stealth:         'sneaks',
  injured:         'limps',
  // Squat / Ground
  squat:           'squat-walks',
  kneel_two:       'kneels on both knees',
  kneel_one:       'kneels on one knee',
  hand_crawl:      'crawls on hands and knees',
  elbow_crawl:     'army crawls',
  // Boxing
  idle_boxing:     'shadow boxes',
  walk_boxing:     'walks while boxing',
  left_jab:        'throws a left jab',
  right_jab:       'throws a right jab',
  random_punches:  'throws random punches',
  left_hook:       'throws a left hook',
  right_hook:      'throws a right hook',
  // Styled Walking
  careful:         'walks carefully',
  object_carrying: 'walks while carrying an object',
  crouch:          'crouch-walks',
  happy_dance:     'dances happily',
  zombie:          'shambles',
  point:           'walks while pointing',
  scared:          'scurries',
};

const CANONICAL_DIR = {
  forward:      'forward',
  backward:     'backward',
  strafe_left:  'left',
  strafe_right: 'right',
  idle:         '',
};

const TURN_PHRASES = {
  left:  'turns left',
  right: 'turns right',
};

const MODE_ING = {
  // Locomotion
  run:             'running',
  walk:            'walking',
  slow_walk:       'walking slowly',
  happy:           'walking happily',
  stealth:         'sneaking',
  injured:         'limping',
  // Squat / Ground
  squat:           'squat-walking',
  kneel_two:       'kneeling on both knees',
  kneel_one:       'kneeling on one knee',
  hand_crawl:      'crawling on hands and knees',
  elbow_crawl:     'army crawling',
  // Boxing
  idle_boxing:     'shadow boxing',
  walk_boxing:     'walking while boxing',
  left_jab:        'throwing a left jab',
  right_jab:       'throwing a right jab',
  random_punches:  'throwing random punches',
  left_hook:       'throwing a left hook',
  right_hook:      'throwing a right hook',
  // Styled Walking
  careful:         'walking carefully',
  object_carrying: 'walking while carrying an object',
  crouch:          'crouch-walking',
  happy_dance:     'dancing happily',
  zombie:          'shambling',
  point:           'walking while pointing',
  scared:          'scurrying',
};

// Heading delta (degrees) to confirm a turn is in progress
const TURN_THRESHOLD_DEG = 6.0;

// ---------------------------------------------------------------------------

function _modeKey(modeName) {
  if (!modeName) return 'walk';
  const key = MODE_NAME_TO_KEY[modeName.toLowerCase()];
  return key || 'walk';
}

function _speedLabel(speed) {
  if (!speed || speed < 0.05) return '';
  return ` at ${speed.toFixed(1)} m/s`;
}

function _escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ---------------------------------------------------------------------------

export class AnnotationTracker {
  constructor() {
    // Completed action segments — stored as verb phrases (without subject)
    // Each entry: { phrase: string, modeKey: string }
    this.completedPhrases = [];

    // Current live phrase being built
    this.currentText = '';

    // Previous state for transition detection
    this._prevMovement = null;
    this._prevModeKey  = null;
    this._prevActive   = null;
    this._prevFacing   = null;
    this._wasTurning   = false;

    // Accumulated heading delta to confirm a turn
    this._facingAccum  = 0.0;

    // Turn direction detected from accumulation
    this._turnDir = null;
  }

  /**
   * Call once per server state update (~20 Hz).
   * @param {object} s  ws_bridge state dict.
   */
  onStateUpdate(s) {
    const modeKey  = _modeKey(s.modeName);
    const movement = s.movementType || 'idle';
    const facing   = s.facingAngleDeg ?? 0;
    // Use measuredSpeed (actual robot velocity from g1_debug) when available,
    // fall back to commanded speed for display only.
    const speed    = (s.measuredSpeed != null && s.measuredSpeed > 0.05)
                       ? s.measuredSpeed
                       : (s.speed ?? -1);
    const active   = s.active ?? false;

    // Accumulate heading delta
    if (this._prevFacing !== null) {
      let delta = facing - this._prevFacing;
      if (delta >  180) delta -= 360;
      if (delta < -180) delta += 360;
      this._facingAccum += delta;
    }
    this._prevFacing = facing;

    const turning = Math.abs(this._facingAccum) > TURN_THRESHOLD_DEG;
    if (turning) {
      this._turnDir = this._facingAccum < 0 ? 'left' : 'right';
    }

    // Detect meaningful state transition
    const modeChanged     = modeKey  !== this._prevModeKey;
    const movementChanged = movement !== this._prevMovement;
    const activeChanged   = active   !== this._prevActive;
    const turnStarted     = turning  && !this._wasTurning;
    const turnEnded       = !turning && this._wasTurning;

    const isTransition = modeChanged || movementChanged || activeChanged || turnStarted || turnEnded;

    if (isTransition) {
      // Commit current phrase to completed list
      if (this.currentText && this._prevActive) {
        this.completedPhrases.push({ phrase: this.currentText, modeKey: this._prevModeKey });
      }
      this._facingAccum = 0.0;
      this._turnDir = null;
    }

    this._wasTurning   = turning;
    this._prevModeKey  = modeKey;
    this._prevMovement = movement;
    this._prevActive   = active;

    this.currentText = this._buildPhrase(modeKey, movement, active, turning, speed);
  }

  /**
   * Build a verb phrase for the current live state.
   * Does NOT include "The robot" — that is prepended by the narrative builder.
   */
  _buildPhrase(modeKey, movement, active, turning, speed) {
    if (!active)                          return 'Idle';
    if (movement === 'idle' && !turning)  return 'standing still';

    const speedStr = _speedLabel(speed);

    if (turning) {
      const dir = this._turnDir || (this._facingAccum < 0 ? 'left' : 'right');
      return `${TURN_PHRASES[dir]} while ${MODE_ING[modeKey] || 'walking'}${speedStr}`;
    }

    const verb = CANONICAL_VERB[modeKey] || 'moves';
    const dir  = CANONICAL_DIR[movement] || '';
    return dir ? `${verb} ${dir}${speedStr}` : `${verb}${speedStr}`;
  }

  /**
   * Build the full growing narrative from all completed phrases + current action.
   * Format: "The robot {phrase0}, then {phrase1}, then {phrase2}, and {current} ▌"
   */
  getNarrative(includeCurrent = true) {
    const all = [...this.completedPhrases.map(p => p.phrase)];
    if (includeCurrent && this.currentText && this.currentText !== 'Idle') {
      all.push(this.currentText);
    }

    if (all.length === 0) return '';
    if (all.length === 1) {
      return includeCurrent
        ? `The robot ${all[0]}`
        : `The robot ${all[0]}.`;
    }

    // Join with ", then" for middle segments, ", and" for last transition
    const parts = [`The robot ${all[0]}`];
    for (let i = 1; i < all.length - 1; i++) {
      parts.push(`then ${all[i]}`);
    }
    const lastConn = includeCurrent ? 'and' : 'and finally';
    parts.push(`${lastConn} ${all[all.length - 1]}`);

    return parts.join(', ') + (includeCurrent ? '' : '.');
  }

  /**
   * Returns HTML for the transcript display.
   * Past phrases are dim; the current action is highlighted via CSS class.
   */
  getHistoryHTML() {
    // Show all completed phrases dimly as a growing sentence
    return this.completedPhrases
      .map((p, i) => {
        const prefix = i === 0 ? 'The robot ' : 'then ';
        return `<span class="transcript-item">${_escapeHtml(prefix + p.phrase)},</span>`;
      })
      .join(' ');
  }

  /**
   * Reset the full transcript (call on scene reset).
   */
  reset() {
    this.completedPhrases = [];
    this.currentText      = '';
    this._prevMovement    = null;
    this._prevModeKey     = null;
    this._prevActive      = null;
    this._prevFacing      = null;
    this._wasTurning      = false;
    this._facingAccum     = 0.0;
    this._turnDir         = null;
  }
}
