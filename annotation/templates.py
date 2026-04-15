"""
Deterministic diverse text annotation generator for robot motion trajectories.

Diversity comes from Cartesian products of synonym banks across 8 annotation
styles (4 timed + 4 no-time). No LLM required — ~200+ distinct phrasings
per trajectory.

Segment dict schema (from compose_trajectory.py / annotate.py):
  {
    "idx":          int,
    "t_start":      float,
    "t_end":        float,
    "frame_start":  int,
    "frame_end":    int,
    "mode":         "run" | "walk" | "slow_walk" | "happy" | "stealth" |
                    "injured" | "squat" | "kneel_two" | "kneel_one" |
                    "hand_crawl" | "elbow_crawl" | "idle_boxing" |
                    "walk_boxing" | "left_jab" | "right_jab" |
                    "random_punches" | "left_hook" | "right_hook" |
                    "careful" | "object_carrying" | "crouch" |
                    "happy_dance" | "zombie" | "point" | "scared",
    "movement":     "forward" | "backward" | "strafe_left" | "strafe_right",
    "turn_deg":     float | None,   # degrees rotated (positive = left)
    "turn_dir":     "left" | "right" | None,
    "duration_sec": float,
    "speed":        float,          # m/s; -1.0 means use mode default
  }
"""

from __future__ import annotations
from typing import Any

# ---------------------------------------------------------------------------
# Synonym banks
# ---------------------------------------------------------------------------

VERB_BANKS: dict[str, list[str]] = {
    # Locomotion
    "run":             ["run", "sprint", "dash", "jog quickly", "move at full speed"],
    "walk":            ["walk", "stride", "march", "move briskly"],
    "slow_walk":       ["walk slowly", "creep", "move cautiously", "step carefully", "shuffle"],
    "happy":           ["walk happily", "strut", "move cheerfully", "bounce along"],
    "stealth":         ["sneak", "move stealthily", "creep silently", "move covertly"],
    "injured":         ["limp", "hobble", "move with difficulty", "stagger"],
    # Squat / Ground
    "squat":           ["squat-walk", "move in a low squat", "crouch-walk", "duck-walk"],
    "kneel_two":       ["kneel on both knees", "move while kneeling", "two-knee walk"],
    "kneel_one":       ["kneel on one knee", "move on one knee", "single-knee crawl"],
    "hand_crawl":      ["crawl on hands and knees", "crawl forward", "hand-and-knee crawl"],
    "elbow_crawl":     ["army crawl", "elbow-crawl", "low crawl", "crawl on elbows"],
    # Boxing
    "idle_boxing":     ["shadow box", "box in place", "hold a boxing stance"],
    "walk_boxing":     ["advance while boxing", "walk in a boxing stance", "move aggressively"],
    "left_jab":        ["throw a left jab", "jab with the left hand", "execute a left jab"],
    "right_jab":       ["throw a right jab", "jab with the right hand", "execute a right jab"],
    "random_punches":  ["throw punches", "unleash a flurry of punches", "punch rapidly"],
    "left_hook":       ["throw a left hook", "hook with the left", "execute a left hook"],
    "right_hook":      ["throw a right hook", "hook with the right", "execute a right hook"],
    # Styled Walking
    "careful":         ["walk carefully", "tread carefully", "move with caution", "step deliberately"],
    "object_carrying": ["carry an object", "walk while carrying", "move with a load"],
    "crouch":          ["crouch-walk", "move while crouching", "walk in a low crouch"],
    "happy_dance":     ["dance happily", "perform a happy dance", "dance joyfully"],
    "zombie":          ["shamble", "zombie-walk", "shuffle zombie-like", "lurch forward"],
    "point":           ["walk while pointing", "move and point", "stride while gesturing"],
    "scared":          ["scurry away", "move fearfully", "flee", "move in a panic"],
}

VERB_BANKS_3RD: dict[str, list[str]] = {
    # Locomotion
    "run":             ["runs", "sprints", "dashes", "jogs quickly", "moves at full speed"],
    "walk":            ["walks", "strides", "marches", "moves briskly"],
    "slow_walk":       ["walks slowly", "creeps", "moves cautiously", "steps carefully"],
    "happy":           ["walks happily", "struts", "moves cheerfully", "bounces along"],
    "stealth":         ["sneaks", "moves stealthily", "creeps silently", "moves covertly"],
    "injured":         ["limps", "hobbles", "moves with difficulty", "staggers"],
    # Squat / Ground
    "squat":           ["squat-walks", "moves in a low squat", "crouch-walks", "duck-walks"],
    "kneel_two":       ["kneels on both knees", "moves while kneeling", "two-knee walks"],
    "kneel_one":       ["kneels on one knee", "moves on one knee", "single-knee crawls"],
    "hand_crawl":      ["crawls on hands and knees", "crawls forward", "hand-crawls"],
    "elbow_crawl":     ["army crawls", "elbow-crawls", "low crawls", "crawls on elbows"],
    # Boxing
    "idle_boxing":     ["shadow boxes", "boxes in place", "holds a boxing stance"],
    "walk_boxing":     ["advances while boxing", "walks in a boxing stance", "moves aggressively"],
    "left_jab":        ["throws a left jab", "jabs with the left", "executes a left jab"],
    "right_jab":       ["throws a right jab", "jabs with the right", "executes a right jab"],
    "random_punches":  ["throws punches", "unleashes a flurry of punches", "punches rapidly"],
    "left_hook":       ["throws a left hook", "hooks with the left", "executes a left hook"],
    "right_hook":      ["throws a right hook", "hooks with the right", "executes a right hook"],
    # Styled Walking
    "careful":         ["walks carefully", "treads carefully", "moves with caution", "steps deliberately"],
    "object_carrying": ["carries an object", "walks while carrying", "moves with a load"],
    "crouch":          ["crouch-walks", "moves while crouching", "walks in a low crouch"],
    "happy_dance":     ["dances happily", "performs a happy dance", "dances joyfully"],
    "zombie":          ["shambles", "zombie-walks", "shuffles zombie-like", "lurches forward"],
    "point":           ["walks while pointing", "moves and points", "strides while gesturing"],
    "scared":          ["scurries away", "moves fearfully", "flees", "moves in a panic"],
}

DIRECTION_BANKS: dict[str, list[str]] = {
    "forward":      ["forward", "ahead", "straight", "in the forward direction"],
    "backward":     ["backward", "back", "in reverse", "rearward"],
    "strafe_left":  ["left", "to the left", "sideways to the left"],
    "strafe_right": ["right", "to the right", "sideways to the right"],
}

TURN_VERB_BANKS: dict[str, list[str]] = {
    "left":  ["turn left", "rotate left", "veer left", "pivot left", "swing left"],
    "right": ["turn right", "rotate right", "veer right", "pivot right", "swing right"],
}

TURN_VERB_BANKS_PRESENT: dict[str, list[str]] = {
    "left":  ["turns left", "rotates left", "veers left", "pivots left"],
    "right": ["turns right", "rotates right", "veers right", "pivots right"],
}

CONNECTIVES: list[str] = [
    "then", "followed by", "after which", "next", "and then", "afterwards",
    "before", "subsequently",
]

# TEMPO_ADVERBS — for speed-controllable modes entries are ordered slow → fast
# so that _speed_adverb() can map the actual speed value to the right descriptor.
# For non-speed-controllable modes the order is arbitrary (seed picks for variety).
TEMPO_ADVERBS: dict[str, list[str]] = {
    # Locomotion  (speed-controllable, ordered slow → fast)
    "run":             ["at a jog", "quickly", "rapidly", "at full speed"],
    "walk":            ["at walking pace", "steadily", "briskly"],
    "slow_walk":       ["very slowly", "slowly", "carefully", "at a measured pace"],
    # Locomotion  (non-speed-controllable)
    "happy":           ["cheerfully", "with a bounce", "in high spirits"],
    "stealth":         ["without a sound", "covertly", "silently"],
    "injured":         ["with great difficulty", "painfully", "unsteadily"],
    # Squat / Ground  (hand_crawl / elbow_crawl are speed-controllable)
    "squat":           ["in a low crouch", "close to the ground"],
    "kneel_two":       ["on both knees", "in a kneeling position"],
    "kneel_one":       ["on one knee", "in a half-kneel"],
    "hand_crawl":      ["slowly on all fours", "on all fours", "quickly on all fours"],
    "elbow_crawl":     ["flat to the ground", "low to the ground", "rapidly"],
    # Boxing  (walk_boxing and punch modes are speed-controllable)
    "idle_boxing":     ["with guard up", "defensively", "in stance"],
    "walk_boxing":     ["cautiously", "steadily", "aggressively"],
    "left_jab":        ["with control", "with precision", "with force"],
    "right_jab":       ["with control", "with precision", "with force"],
    "random_punches":  ["lightly", "with intensity", "in a rapid flurry"],
    "left_hook":       ["with control", "powerfully", "with full force"],
    "right_hook":      ["with control", "powerfully", "with full force"],
    # Styled Walking  (non-speed-controllable)
    "careful":         ["with deliberate steps", "carefully", "attentively"],
    "object_carrying": ["balancing carefully", "steadily", "with both hands"],
    "crouch":          ["in a crouched posture", "low to the ground"],
    "happy_dance":     ["joyfully", "with enthusiasm", "energetically"],
    "zombie":          ["mindlessly", "lifelessly", "with arms outstretched"],
    "point":           ["while gesturing", "with a directing hand"],
    "scared":          ["in a panic", "frantically", "with urgency"],
}

# Speed range (m/s) for modes that accept a speed parameter.
# Entries are ordered to match TEMPO_ADVERBS (slow → fast).
_MODE_SPEED_RANGE: dict[str, tuple[float, float]] = {
    "slow_walk":      (0.2, 0.8),
    "walk":           (0.8, 2.5),
    "run":            (1.5, 3.0),
    "hand_crawl":     (0.4, 1.5),
    "elbow_crawl":    (0.7, 1.5),
    "walk_boxing":    (0.7, 1.5),
    "left_jab":       (0.7, 1.5),
    "right_jab":      (0.7, 1.5),
    "random_punches": (0.7, 1.5),
    "left_hook":      (0.7, 1.5),
    "right_hook":     (0.7, 1.5),
}

# ---------------------------------------------------------------------------
# Speed → adverb helper
# ---------------------------------------------------------------------------

def _speed_adverb(mode: str, speed: float, seed: int = 0) -> str:
    """Return a tempo adverb for the segment.

    If *speed* >= 0 and the mode has a known speed range, the adverb is chosen
    by mapping the speed linearly into the TEMPO_ADVERBS bank (ordered
    slow → fast).  Otherwise the bank is sampled with *seed* for variety.
    """
    bank = TEMPO_ADVERBS.get(mode, ["steadily"])
    if speed >= 0 and mode in _MODE_SPEED_RANGE:
        lo, hi = _MODE_SPEED_RANGE[mode]
        t = (speed - lo) / max(hi - lo, 1e-6)
        t = max(0.0, min(1.0, t))
        idx = round(t * (len(bank) - 1))
        return bank[idx]
    return _pick(bank, seed)


# ---------------------------------------------------------------------------
# Turn precision helpers
# ---------------------------------------------------------------------------

def _approx_turn_desc(deg: float, direction: str) -> str:
    """Return a human-friendly turn description for a given angle in degrees."""
    abs_deg = abs(deg)
    if abs_deg < 15:
        return f"a slight {direction} turn"
    elif abs_deg < 60:
        return f"a partial {direction} turn (~{round(abs_deg)}°)"
    elif abs_deg < 120:
        return f"a quarter {direction} turn (~90°)"
    elif abs_deg < 240:
        return f"a half {direction} turn (~180°)"
    elif abs_deg < 315:
        return f"a three-quarter {direction} turn (~270°)"
    else:
        return f"a nearly full {direction} rotation (~360°)"


def _exact_turn_desc(deg: float) -> str:
    return f"{int(round(abs(deg)))} degrees"


# ---------------------------------------------------------------------------
# Single-segment renderers per style
# ---------------------------------------------------------------------------

def _pick(bank: list[str], seed: int) -> str:
    return bank[seed % len(bank)]


def _ing(verb: str) -> str:
    """Return the -ing form of a simple verb phrase (handles trailing 'e')."""
    # Only apply to the first word (the actual verb)
    parts = verb.split(" ", 1)
    v = parts[0]
    if v.endswith("e") and not v.endswith("ee"):
        v = v[:-1] + "ing"
    else:
        v = v + "ing"
    return (v + " " + parts[1]) if len(parts) > 1 else v


def _render_instruction(seg: dict[str, Any], seed: int) -> str:
    """Imperative sentence. E.g.: 'Run forward for 3 seconds.'"""
    dur = seg["duration_sec"]
    if seg["turn_dir"]:
        turn_dir = seg["turn_dir"]
        verb = _pick(TURN_VERB_BANKS[turn_dir], seed)
        angle = _exact_turn_desc(seg["turn_deg"]) if seg["turn_deg"] is not None else ""
        mode = seg["mode"]
        while_clause = f" while {_ing(_pick(VERB_BANKS[mode], seed + 1))}"
        angle_part = f" {angle}" if angle else ""
        return f"{verb.capitalize()}{angle_part}{while_clause} over {dur:.1f}s."
    else:
        verb = _pick(VERB_BANKS[seg["mode"]], seed)
        direction = _pick(DIRECTION_BANKS[seg["movement"]], seed + 1)
        return f"{verb.capitalize()} {direction} for {dur:.1f} seconds."


def _render_natural(seg: dict[str, Any], seed: int) -> str:
    """Natural language with adverbs. E.g.: 'Sprint ahead at full speed for about 3 seconds.'"""
    dur = seg["duration_sec"]
    mode = seg["mode"]
    if seg["turn_dir"]:
        turn_dir = seg["turn_dir"]
        verb = _pick(TURN_VERB_BANKS[turn_dir], seed + 2)
        angle_desc = _approx_turn_desc(seg["turn_deg"] or 90.0, turn_dir)
        while_mode = _ing(_pick(VERB_BANKS[mode], seed))
        return f"{verb.capitalize()} ({angle_desc}) while {while_mode} over {dur:.1f}s."
    else:
        verb = _pick(VERB_BANKS[mode], seed + 2)
        direction = _pick(DIRECTION_BANKS[seg["movement"]], seed)
        # Skip tempo adverb if verb phrase already contains an adverb qualifier
        _adverb_words = {"quickly", "slowly", "cautiously", "carefully", "briskly"}
        if not _adverb_words.intersection(verb.split()):
            adverb = _speed_adverb(mode, seg.get("speed", -1.0), seed + 1)
            return f"{verb.capitalize()} {direction} {adverb} for about {dur:.1f} seconds."
        return f"{verb.capitalize()} {direction} for about {dur:.1f} seconds."


def _render_narrative(seg: dict[str, Any], seed: int) -> str:
    """Third-person present. E.g.: 'The robot dashes forward for 3 seconds.'"""
    dur = seg["duration_sec"]
    mode = seg["mode"]
    if seg["turn_dir"]:
        turn_dir = seg["turn_dir"]
        verb = _pick(TURN_VERB_BANKS_PRESENT[turn_dir], seed + 1)
        angle = _approx_turn_desc(seg["turn_deg"] or 90.0, turn_dir)
        while_mode = _ing(_pick(VERB_BANKS[mode], seed))
        return f"The robot {verb} ({angle}) while {while_mode} for {dur:.1f}s."
    else:
        verb = _pick(VERB_BANKS_3RD[mode], seed + 3)
        direction = _pick(DIRECTION_BANKS[seg["movement"]], seed + 2)
        _adverb_words = {"quickly", "slowly", "cautiously", "carefully", "briskly"}
        if not _adverb_words.intersection(verb.split()):
            adverb = _speed_adverb(mode, seg.get("speed", -1.0), seed)
            return f"The robot {verb} {direction} {adverb} for {dur:.1f} seconds."
        return f"The robot {verb} {direction} for {dur:.1f} seconds."


def _render_concise(seg: dict[str, Any], seed: int) -> str:
    """Short keyword form. E.g.: 'run forward 3s'"""
    dur = seg["duration_sec"]
    if seg["turn_dir"]:
        deg_str = f"{int(round(abs(seg['turn_deg'])))}°" if seg["turn_deg"] else ""
        return f"{seg['mode'].replace('_', ' ')} + turn {seg['turn_dir']} {deg_str} {dur:.1f}s".strip()
    else:
        return f"{seg['mode'].replace('_', ' ')} {seg['movement'].replace('_', ' ')} {dur:.1f}s"


def _render_instruction_no_time(seg: dict[str, Any], seed: int) -> str:
    """Imperative sentence without duration. E.g.: 'Run forward.'"""
    if seg["turn_dir"]:
        turn_dir = seg["turn_dir"]
        verb = _pick(TURN_VERB_BANKS[turn_dir], seed)
        angle = _exact_turn_desc(seg["turn_deg"]) if seg["turn_deg"] is not None else ""
        mode = seg["mode"]
        while_clause = f" while {_ing(_pick(VERB_BANKS[mode], seed + 1))}"
        angle_part = f" {angle}" if angle else ""
        return f"{verb.capitalize()}{angle_part}{while_clause}."
    else:
        verb = _pick(VERB_BANKS[seg["mode"]], seed)
        direction = _pick(DIRECTION_BANKS[seg["movement"]], seed + 1)
        return f"{verb.capitalize()} {direction}."


def _render_natural_no_time(seg: dict[str, Any], seed: int) -> str:
    """Natural language without duration. E.g.: 'Sprint ahead at full speed.'"""
    mode = seg["mode"]
    if seg["turn_dir"]:
        turn_dir = seg["turn_dir"]
        verb = _pick(TURN_VERB_BANKS[turn_dir], seed + 2)
        angle_desc = _approx_turn_desc(seg["turn_deg"] or 90.0, turn_dir)
        while_mode = _ing(_pick(VERB_BANKS[mode], seed))
        return f"{verb.capitalize()} ({angle_desc}) while {while_mode}."
    else:
        verb = _pick(VERB_BANKS[mode], seed + 2)
        direction = _pick(DIRECTION_BANKS[seg["movement"]], seed)
        _adverb_words = {"quickly", "slowly", "cautiously", "carefully", "briskly"}
        if not _adverb_words.intersection(verb.split()):
            adverb = _speed_adverb(mode, seg.get("speed", -1.0), seed + 1)
            return f"{verb.capitalize()} {direction} {adverb}."
        return f"{verb.capitalize()} {direction}."


def _render_narrative_no_time(seg: dict[str, Any], seed: int) -> str:
    """Third-person present without duration. E.g.: 'The robot dashes forward.'"""
    mode = seg["mode"]
    if seg["turn_dir"]:
        turn_dir = seg["turn_dir"]
        verb = _pick(TURN_VERB_BANKS_PRESENT[turn_dir], seed + 1)
        angle = _approx_turn_desc(seg["turn_deg"] or 90.0, turn_dir)
        while_mode = _ing(_pick(VERB_BANKS[mode], seed))
        return f"The robot {verb} ({angle}) while {while_mode}."
    else:
        verb = _pick(VERB_BANKS_3RD[mode], seed + 3)
        direction = _pick(DIRECTION_BANKS[seg["movement"]], seed + 2)
        _adverb_words = {"quickly", "slowly", "cautiously", "carefully", "briskly"}
        if not _adverb_words.intersection(verb.split()):
            adverb = _speed_adverb(mode, seg.get("speed", -1.0), seed)
            return f"The robot {verb} {direction} {adverb}."
        return f"The robot {verb} {direction}."


def _render_concise_no_time(seg: dict[str, Any], seed: int) -> str:
    """Short keyword form without duration. E.g.: 'run forward'"""
    if seg["turn_dir"]:
        deg_str = f"{int(round(abs(seg['turn_deg'])))}°" if seg["turn_deg"] else ""
        return f"{seg['mode'].replace('_', ' ')} + turn {seg['turn_dir']} {deg_str}".strip()
    else:
        return f"{seg['mode'].replace('_', ' ')} {seg['movement'].replace('_', ' ')}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

STYLES = [
    "instruction",
    "natural",
    "narrative",
    "concise",
    "instruction_no_time",
    "natural_no_time",
    "narrative_no_time",
    "concise_no_time",
]

_STYLE_FNS = {
    "instruction": _render_instruction,
    "natural":     _render_natural,
    "narrative":   _render_narrative,
    "concise":     _render_concise,
    "instruction_no_time": _render_instruction_no_time,
    "natural_no_time": _render_natural_no_time,
    "narrative_no_time": _render_narrative_no_time,
    "concise_no_time": _render_concise_no_time,
}


def render_segment(seg: dict[str, Any], style: str = "natural", seed: int = 0) -> str:
    """Render a single segment as a text phrase in the given style."""
    fn = _STYLE_FNS.get(style)
    if fn is None:
        raise ValueError(f"Unknown style '{style}'. Choose from {STYLES}")
    return fn(seg, seed)


# ---------------------------------------------------------------------------
# Canonical sentence builders (timed + action-only, with subject variants)
# ---------------------------------------------------------------------------

_CANONICAL_VERB = {
    # Locomotion
    "run":             "runs",
    "walk":            "walks",
    "slow_walk":       "walks slowly",
    "happy":           "walks happily",
    "stealth":         "sneaks",
    "injured":         "limps",
    # Squat / Ground
    "squat":           "squat-walks",
    "kneel_two":       "kneels on both knees",
    "kneel_one":       "kneels on one knee",
    "hand_crawl":      "crawls on hands and knees",
    "elbow_crawl":     "army crawls",
    # Boxing
    "idle_boxing":     "shadow boxes",
    "walk_boxing":     "walks while boxing",
    "left_jab":        "throws a left jab",
    "right_jab":       "throws a right jab",
    "random_punches":  "throws random punches",
    "left_hook":       "throws a left hook",
    "right_hook":      "throws a right hook",
    # Styled Walking
    "careful":         "walks carefully",
    "object_carrying": "walks while carrying an object",
    "crouch":          "crouch-walks",
    "happy_dance":     "dances happily",
    "zombie":          "shambles",
    "point":           "walks while pointing",
    "scared":          "scurries",
}
_CANONICAL_VERB_ING = {
    # Locomotion
    "run":             "running",
    "walk":            "walking",
    "slow_walk":       "walking slowly",
    "happy":           "walking happily",
    "stealth":         "sneaking",
    "injured":         "limping",
    # Squat / Ground
    "squat":           "squat-walking",
    "kneel_two":       "kneeling on both knees",
    "kneel_one":       "kneeling on one knee",
    "hand_crawl":      "crawling on hands and knees",
    "elbow_crawl":     "army crawling",
    # Boxing
    "idle_boxing":     "shadow boxing",
    "walk_boxing":     "walking while boxing",
    "left_jab":        "throwing a left jab",
    "right_jab":       "throwing a right jab",
    "random_punches":  "throwing random punches",
    "left_hook":       "throwing a left hook",
    "right_hook":      "throwing a right hook",
    # Styled Walking
    "careful":         "walking carefully",
    "object_carrying": "walking while carrying an object",
    "crouch":          "crouch-walking",
    "happy_dance":     "dancing happily",
    "zombie":          "shambling",
    "point":           "walking while pointing",
    "scared":          "scurrying",
}
_SUBJECTS = ["The robot", "The person", ""]   # "" = verb-first (no subject)


def _segment_verb_phrase(seg: dict[str, Any], include_time: bool) -> str:
    """
    Canonical verb phrase for a segment (no subject).
    include_time=True  → "runs forward for 3.0 seconds"
    include_time=False → "runs forward"
    """
    dur  = seg["duration_sec"]
    mode = seg["mode"]
    verb = _CANONICAL_VERB[mode]

    if seg["turn_dir"]:
        turn_dir = seg["turn_dir"]
        angle    = _exact_turn_desc(seg["turn_deg"]) if seg["turn_deg"] is not None else ""
        angle_part   = f" {angle}" if angle else ""
        while_clause = f" while {_CANONICAL_VERB_ING[mode]}"
        phrase = f"turns {turn_dir}{angle_part}{while_clause}"
    else:
        direction = DIRECTION_BANKS[seg["movement"]][0]
        phrase = f"{verb} {direction}"

    return f"{phrase} for {dur:.1f} seconds" if include_time else phrase


def _build_sentence(phrases: list[str], subject: str) -> str:
    """
    Conjoin verb phrases with ', then ' (middle) and ', and finally ' (last).
    Prepend subject to the first phrase; capitalise first letter if no subject.
    """
    if not phrases:
        return ""

    if subject:
        head = f"{subject} {phrases[0]}"
    else:
        head = phrases[0].capitalize()

    if len(phrases) == 1:
        return head + "."

    parts = [head]
    for p in phrases[1:-1]:
        parts.append(f"then {p}")
    parts.append(f"and finally {phrases[-1]}")
    return ", ".join(parts) + "."


def build_canonical_sentences(segments: list[dict[str, Any]]) -> list[str]:
    """
    Build all 6 canonical sentences:
      3 subjects × 2 time modes (timed + action-only)
    Order: timed first (robot / person / no-subject), then action-only.
    """
    results = []
    for include_time in (True, False):
        phrases = [_segment_verb_phrase(s, include_time) for s in segments]
        for subj in _SUBJECTS:
            results.append(_build_sentence(phrases, subj))
    return results


def compose_annotations(segments: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Generate diverse annotations for a full trajectory of segments.

    Returns:
        {
            "full_trajectory": [str, str, str, str],   # one per style
            "per_segment": [
                {"segment_idx": int, "annotations": [str, str, str, str]},
                ...
            ]
        }
    """
    per_segment = []
    for seg in segments:
        annots = [render_segment(seg, style=s, seed=seg["idx"]) for s in STYLES]
        per_segment.append({"segment_idx": seg["idx"], "annotations": annots})

    # Full-trajectory: join each style's segment phrases with varied connectives
    full_trajectory = []
    for style_idx, style in enumerate(STYLES):
        phrases = [render_segment(seg, style=style, seed=style_idx) for seg in segments]
        if style in {"concise", "concise_no_time"}:
            joined = "; ".join(p.rstrip(".") for p in phrases)
        else:
            conn = CONNECTIVES[style_idx % len(CONNECTIVES)]
            # Strip trailing punctuation from all but the last phrase before joining
            parts = []
            for i, p in enumerate(phrases):
                if i < len(phrases) - 1:
                    parts.append(p.rstrip("."))
                else:
                    parts.append(p)
            joined = f", {conn} ".join(parts)
        full_trajectory.append(joined)

    # Add extra diversity: mix styles across segments using different seeds
    extra_phrases = []
    for seed_offset in range(1, 4):
        parts = []
        for i, seg in enumerate(segments):
            style = STYLES[(i + seed_offset) % len(STYLES)]
            phrase = render_segment(seg, style=style, seed=seed_offset * 10 + i)
            parts.append(phrase.rstrip("."))
        conn = CONNECTIVES[(seed_offset + 2) % len(CONNECTIVES)]
        extra_phrases.append(f", {conn} ".join(parts) + ".")
    full_trajectory.extend(extra_phrases)

    # Prepend the 6 canonical sentences (3 subjects × timed/action-only)
    canonical = build_canonical_sentences(segments)
    full_trajectory = canonical + full_trajectory

    return {
        "full_trajectory": full_trajectory,
        "per_segment":     per_segment,
    }
