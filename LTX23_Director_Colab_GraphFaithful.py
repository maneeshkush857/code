#!/usr/bin/env python3
"""LTX-2.3 Director 2.0 MV - Graph-Faithful Google Colab re-execution.

This program faithfully re-executes the ORIGINAL ComfyUI graph stored in
``LTX-2.3_Director_2.0-MV-Workflow-30s.json`` (the sole source of truth). The
JSON graph is loaded, parsed into an internal representation, validated, and
executed node-by-node using the original ComfyUI nodes. It is designed to run
on Google Colab free tier (T4 GPU) with aggressive memory management borrowed
as INFRASTRUCTURE ONLY from ``LTX23_Director_Master_V2.py``.

CRITICAL DESIGN CONTRACT
------------------------
* The ComfyUI JSON graph is authoritative. Values are PARSED from it, never
  hardcoded (validation expectations are the only place literals appear, and
  they are compared against parsed values).
* ``LTXDirector`` (node id=131) remains the Master Timeline Controller. It is
  NOT replaced by a Python 5-segment loop or ``linear_blend_overlap``. The
  timeline (31.5 s / 756 frames / 24 fps), the 5 reference-image segments, the
  4-LoRA stack (0.4/0.6/0.7/0.9), guide_data + motion_guide_data, both sampling
  stages, the audio latent path, both CropGuides, all VAEs, and the final
  ``VHS_VideoCombine`` are preserved exactly as authored.
* No 5-scene / linear-blend architecture is copied from V2. Only the memory
  engine is borrowed.
* No silent exception swallowing in core execution. No placeholder reference
  images (a missing image is a hard error at runtime on Colab).

IMPORT SAFETY
-------------
Only the standard library (plus ``json`` and ``ctypes``, both stdlib) is
imported at module top level. Heavy dependencies (``torch``, ``psutil``,
``comfy``, ``comfy.model_management``, ``nodes``) are imported lazily inside
helper functions, so that JSON parse / graph build / timeline extraction /
validation (the ``--selftest`` path) runs on a machine with NO GPU/torch.

CELL LAYOUT (Colab notebook cells emulated as banner-delimited blocks)
----------------------------------------------------------------------
* CELL 1  - Environment + memory protection (import-safe)
* CELL 3-7 - Colab install / ComfyUI bootstrap (authored in FEAT-002)
* CELL 8  - Load workflow JSON
* CELL 9  - Parse graph into internal representation
* CELL 10 - Validate parsed graph (part 1)
* CELL 12 - Deep memory cleanup + memory guard
* CELL 27 - Validate parsed graph (final report driver)
* CELL 29 - LTXDirector timeline parser
* Remaining execution cells (node registry / executor / run) - FEAT-002/003.
"""

from __future__ import annotations

# ════════════════════════════════════════════════════════════════════════════
# CELL 1: ENVIRONMENT + MEMORY PROTECTION (import-safe, stdlib only)
# ════════════════════════════════════════════════════════════════════════════
import argparse
import ctypes
import gc
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

# Mirror the CUDA allocator / malloc tuning from V2 (lines 40-42). Setting these
# in os.environ is harmless when torch/CUDA are absent, and must be done before
# torch is ever imported (which only happens lazily, later).
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.8",
)
os.environ.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")
os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "65536")

# Default path to the source-of-truth workflow JSON. Overridable for Colab via
# the WORKFLOW_JSON_PATH env var or the CLI. Defaults to the file that ships
# next to this script in the repo.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_JSON_PATH = os.environ.get(
    "WORKFLOW_JSON_PATH",
    os.path.join(_SCRIPT_DIR, "LTX-2.3_Director_2.0-MV-Workflow-30s.json"),
)

# Memory-guard thresholds (borrowed as infrastructure from V2's memory engine).
MIN_FREE_RAM_GB = 2.0
VRAM_SAFETY_BUFFER_GB = 1.2

# The LTXDirector node id in the source graph (Master Timeline Controller).
LTXDIRECTOR_NODE_ID = 131


# ════════════════════════════════════════════════════════════════════════════
# CELL 1 (cont.) - LAZY / GUARDED HEAVY IMPORTS
# ----------------------------------------------------------------------------
# These helpers import torch / psutil / comfy only when actually needed so the
# parse+validate path stays import-safe on a GPU-less box.
# ════════════════════════════════════════════════════════════════════════════
def _try_import(module_name: str) -> Any | None:
    """Import ``module_name`` lazily, returning None if unavailable.

    This never raises for a missing dependency; it returns None so callers can
    degrade gracefully. It does NOT hide genuine errors from an installed but
    broken module - those propagate.
    """
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def _import_torch() -> Any | None:
    """Return the ``torch`` module if importable, else None."""
    return _try_import("torch")


def _import_psutil() -> Any | None:
    """Return the ``psutil`` module if importable, else None."""
    return _try_import("psutil")


def _import_comfy_mm() -> Any | None:
    """Return ``comfy.model_management`` if importable, else None."""
    return _try_import("comfy.model_management")


# ════════════════════════════════════════════════════════════════════════════
# CELL 12: DEEP MEMORY CLEANUP + MEMORY GUARD
# (infrastructure only, borrowed from V2 CELL 7 memory engine; import-safe)
# ════════════════════════════════════════════════════════════════════════════
def malloc_trim() -> bool:
    """Release free heap back to the OS via glibc ``malloc_trim(0)``.

    Returns True on success. On non-Linux platforms (no ``libc.so.6``) this
    returns False rather than crashing.
    """
    if not sys.platform.startswith("linux"):
        return False
    try:
        libc = ctypes.CDLL("libc.so.6")
    except OSError:
        return False
    try:
        libc.malloc_trim(0)
        return True
    except (AttributeError, OSError):
        return False


def get_free_ram_gb() -> float | None:
    """Return available system RAM in GB.

    Prefers ``psutil.virtual_memory().available``; falls back to parsing
    ``/proc/meminfo`` (MemAvailable). Returns None if neither is available.
    """
    psutil = _import_psutil()
    if psutil is not None:
        try:
            return psutil.virtual_memory().available / 1e9
        except Exception as exc:  # noqa: BLE001 - report, don't hide
            print(f"[memory] psutil.virtual_memory failed: {exc!r}")

    # Fallback: /proc/meminfo (Linux only).
    meminfo = "/proc/meminfo"
    if os.path.exists(meminfo):
        try:
            with open(meminfo, "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        kb = float(line.split()[1])
                        return kb * 1024 / 1e9
        except (OSError, ValueError, IndexError) as exc:
            print(f"[memory] /proc/meminfo parse failed: {exc!r}")
    return None


def get_free_vram_gb() -> float | None:
    """Return free VRAM in GB via ``torch.cuda.mem_get_info()``.

    Returns None (never crashes) if torch/CUDA is unavailable.
    """
    torch = _import_torch()
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        free_bytes, _total = torch.cuda.mem_get_info()
        return free_bytes / 1e9
    except Exception as exc:  # noqa: BLE001 - report, don't hide
        print(f"[memory] torch.cuda.mem_get_info failed: {exc!r}")
        return None


def _drop_page_cache() -> None:
    """Best-effort Linux page-cache advisory drop for the workflow file.

    Uses ``os.posix_fadvise(..., POSIX_FADV_DONTNEED)`` where available. This is
    advisory and safe; it silently no-ops on platforms without posix_fadvise.
    """
    if not hasattr(os, "posix_fadvise"):
        return
    try:
        if os.path.exists(WORKFLOW_JSON_PATH):
            fd = os.open(WORKFLOW_JSON_PATH, os.O_RDONLY)
            try:
                size = os.fstat(fd).st_size
                os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
    except OSError as exc:
        print(f"[memory] posix_fadvise notice: {exc!r}")


def deep_memory_cleanup(tag: str = "") -> None:
    """Aggressively free CPU + GPU memory. Import-safe.

    Calls (only when available): ComfyUI model unload/cleanup/soft-empty and
    clears ``current_loaded_models``; ``torch.cuda.empty_cache`` /
    ``ipc_collect``; ``gc.collect``; glibc ``malloc_trim``; and an advisory
    page-cache drop. When torch/comfy are missing it still runs gc + malloc_trim
    so the function is meaningful on a plain host.
    """
    mm = _import_comfy_mm()
    if mm is not None:
        for fn_name in ("unload_all_models", "cleanup_models", "soft_empty_cache"):
            fn = getattr(mm, fn_name, None)
            if callable(fn):
                try:
                    fn()
                except Exception as exc:  # noqa: BLE001 - report, don't hide
                    print(f"[memory] comfy.{fn_name} failed: {exc!r}")
        loaded = getattr(mm, "current_loaded_models", None)
        if isinstance(loaded, list):
            loaded.clear()

    gc.collect()

    torch = _import_torch()
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception as exc:  # noqa: BLE001 - report, don't hide
            print(f"[memory] torch cuda cleanup failed: {exc!r}")

    gc.collect()
    _drop_page_cache()
    malloc_trim()
    if tag:
        print(f"[memory] deep_memory_cleanup done ({tag})")


def memory_guard(
    min_free_ram_gb: float = MIN_FREE_RAM_GB,
    vram_buffer_gb: float = VRAM_SAFETY_BUFFER_GB,
    tag: str = "",
    raise_on_violation: bool = False,
) -> bool:
    """Enforce free-RAM / VRAM-buffer thresholds.

    If free RAM drops below ``min_free_ram_gb`` (or free VRAM is known and below
    ``vram_buffer_gb``), run :func:`deep_memory_cleanup` and, if the threshold is
    still violated, either raise ``MemoryError`` (when ``raise_on_violation``) or
    return False. Returns True when thresholds are satisfied. Never swallows the
    violation silently.
    """
    def _check() -> tuple[bool, float | None, float | None]:
        ram = get_free_ram_gb()
        vram = get_free_vram_gb()
        ram_ok = ram is None or ram >= min_free_ram_gb
        vram_ok = vram is None or vram >= vram_buffer_gb
        return (ram_ok and vram_ok), ram, vram

    ok, ram, vram = _check()
    if ok:
        return True

    print(
        f"⚠️ [memory_guard{':' + tag if tag else ''}] threshold violated "
        f"(free RAM={ram}, free VRAM={vram}); running deep cleanup"
    )
    deep_memory_cleanup(f"memory_guard:{tag}")

    ok, ram, vram = _check()
    if ok:
        return True

    msg = (
        f"memory_guard{':' + tag if tag else ''}: insufficient memory after "
        f"cleanup (free RAM={ram} GB < {min_free_ram_gb} GB or "
        f"free VRAM={vram} GB < {vram_buffer_gb} GB)"
    )
    if raise_on_violation:
        raise MemoryError(msg)
    print(f"⚠️ {msg}")
    return False


# ════════════════════════════════════════════════════════════════════════════
# CELL 3-7: COLAB BOOTSTRAP - ComfyUI install, custom-node fetch, model download
# (authored in FEAT-002)
# ════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════
# CELL 8: LOAD WORKFLOW JSON
# ════════════════════════════════════════════════════════════════════════════
def load_workflow_json(path: str | None = None) -> dict[str, Any]:
    """Load and lightly validate the source-of-truth workflow JSON.

    ``path`` defaults to :data:`WORKFLOW_JSON_PATH`. Verifies that the top-level
    structure has ``nodes``, ``links``, ``last_node_id`` and ``last_link_id``.
    Raises a clear error on missing file, invalid JSON, or missing keys.
    """
    resolved = path or WORKFLOW_JSON_PATH
    if not os.path.exists(resolved):
        raise FileNotFoundError(
            f"Workflow JSON not found at '{resolved}'. Set WORKFLOW_JSON_PATH "
            f"or pass --json <path>."
        )
    try:
        with open(resolved, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Workflow JSON at '{resolved}' is not valid JSON: {exc}") from exc

    # ValueError (not TypeError) is intentional: these validate the CONTENT of
    # an external data file, not a Python type contract with the caller.
    if not isinstance(data, dict):
        raise ValueError(f"Workflow JSON at '{resolved}' must be a JSON object.")  # noqa: TRY004

    for key in ("nodes", "links", "last_node_id", "last_link_id"):
        if key not in data:
            raise ValueError(
                f"Workflow JSON at '{resolved}' is missing required top-level key '{key}'."
            )
    if not isinstance(data["nodes"], list):
        raise ValueError("Workflow JSON 'nodes' must be a list.")  # noqa: TRY004
    if not isinstance(data["links"], list):
        raise ValueError("Workflow JSON 'links' must be a list.")  # noqa: TRY004
    return data


# ════════════════════════════════════════════════════════════════════════════
# CELL 9: PARSE GRAPH INTO INTERNAL REPRESENTATION
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class ParsedNode:
    """A single ComfyUI node parsed from the workflow JSON."""

    id: int
    type: str
    cnr_id: str | None = None
    ver: str | None = None
    inputs: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    widgets_values: Any = None
    mode: int = 0
    order: int = 0
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def widget_version(self) -> str | None:
        """Best-effort node version: 'ver' if present, else properties ver."""
        if self.ver:
            return self.ver
        return self.properties.get("ver") or self.properties.get("cnr_id")


@dataclass
class ParsedGraph:
    """Internal representation of the parsed ComfyUI workflow graph."""

    raw: dict[str, Any]
    nodes_by_id: dict[int, ParsedNode]
    links: list[list[Any]]
    groups: list[dict[str, Any]]
    node_type_counts: dict[str, int]
    last_node_id: int
    last_link_id: int

    # --- accessors ---------------------------------------------------------
    def get_node(self, node_id: int) -> ParsedNode | None:
        """Return the node with ``node_id`` or None."""
        return self.nodes_by_id.get(node_id)

    def require_node(self, node_id: int) -> ParsedNode:
        """Return the node with ``node_id`` or raise KeyError."""
        node = self.nodes_by_id.get(node_id)
        if node is None:
            raise KeyError(f"No node with id={node_id} in parsed graph.")
        return node

    def nodes_of_type(self, node_type: str) -> list[ParsedNode]:
        """Return all nodes whose ``type`` equals ``node_type`` (order-stable)."""
        return [n for n in self.nodes_by_id.values() if n.type == node_type]

    def incoming_links(self, node_id: int) -> list[list[Any]]:
        """Return links whose destination node is ``node_id``.

        Link format: [link_id, from_node, from_slot, to_node, to_slot, type].
        """
        return [lk for lk in self.links if len(lk) >= 4 and lk[3] == node_id]

    def outgoing_links(self, node_id: int) -> list[list[Any]]:
        """Return links whose source node is ``node_id``."""
        return [lk for lk in self.links if len(lk) >= 2 and lk[1] == node_id]

    @property
    def node_count(self) -> int:
        return len(self.nodes_by_id)

    @property
    def link_count(self) -> int:
        return len(self.links)


def parse_graph(data: dict[str, Any]) -> ParsedGraph:
    """Build a :class:`ParsedGraph` from the loaded workflow dict.

    Everything is derived from the parsed JSON; nothing is reconstructed from
    memory. Link entries are normalized to plain lists of the canonical form
    ``[link_id, from_node, from_slot, to_node, to_slot, type]``.
    """
    nodes_by_id: dict[int, ParsedNode] = {}
    node_type_counts: dict[str, int] = {}

    for raw_node in data["nodes"]:
        node = ParsedNode(
            id=raw_node["id"],
            type=raw_node["type"],
            cnr_id=raw_node.get("cnr_id"),
            ver=raw_node.get("ver"),
            inputs=raw_node.get("inputs") or [],
            outputs=raw_node.get("outputs") or [],
            widgets_values=raw_node.get("widgets_values"),
            mode=raw_node.get("mode", 0),
            order=raw_node.get("order", 0),
            properties=raw_node.get("properties") or {},
        )
        nodes_by_id[node.id] = node
        node_type_counts[node.type] = node_type_counts.get(node.type, 0) + 1

    links: list[list[Any]] = [list(lk) for lk in data["links"]]
    groups: list[dict[str, Any]] = data.get("groups") or []

    return ParsedGraph(
        raw=data,
        nodes_by_id=nodes_by_id,
        links=links,
        groups=groups,
        node_type_counts=node_type_counts,
        last_node_id=data["last_node_id"],
        last_link_id=data["last_link_id"],
    )


# ════════════════════════════════════════════════════════════════════════════
# CELL 29: LTXDIRECTOR TIMELINE PARSER
# (LTXDirector id=131 is the Master Timeline Controller - parse, never replace)
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class DirectorTimeline:
    """Parsed LTXDirector timeline + derived facts.

    All fields are parsed from the LTXDirector node's serialized timeline JSON
    widget and its numeric widgets. The original serialized string is retained.
    """

    node_id: int
    raw_timeline_string: str
    timeline_config: dict[str, Any]
    timeline_segments: list[dict[str, Any]]
    audio_segments: list[dict[str, Any]]
    motion_segments: list[dict[str, Any]]
    global_prompt: str
    guide_strength: str
    duration_seconds: float
    frames: int
    fps: int


def _find_timeline_widget(widgets_values: Any) -> tuple[int, dict[str, Any], str]:
    """Locate the serialized timeline JSON widget in ``widgets_values``.

    Detection: attempt ``json.loads`` on each string widget and pick the one
    whose parsed object contains timeline keys ('segments' / 'global_prompt').
    Returns (widget_index, parsed_dict, raw_string). Raises if not found.
    """
    if not isinstance(widgets_values, (list, tuple)):
        # ValueError (not TypeError): malformed workflow content, not a caller bug.
        raise ValueError(  # noqa: TRY004
            "LTXDirector widgets_values is not a list; cannot find timeline."
        )

    for idx, widget in enumerate(widgets_values):
        if not isinstance(widget, str):
            continue
        stripped = widget.strip()
        if not stripped.startswith("{"):
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and (
            "segments" in parsed or "global_prompt" in parsed
        ):
            return idx, parsed, widget

    raise ValueError(
        "Could not locate the LTXDirector serialized timeline JSON widget "
        "(no string widget parses to an object with 'segments'/'global_prompt')."
    )


def _first_string_matching(widgets_values: Any, predicate) -> str | None:
    """Return the first string widget satisfying ``predicate``, else None."""
    if not isinstance(widgets_values, (list, tuple)):
        return None
    for widget in widgets_values:
        if isinstance(widget, str) and predicate(widget):
            return widget
    return None


def _cross_check_fps(graph: ParsedGraph, derived_fps: int) -> int:
    """Cross-check the derived fps against LTXVConditioning and VHS nodes.

    Emits a warning if the authoritative nodes disagree, but returns the
    timeline-derived value (frames/duration) as the source of truth.
    """
    checks: list[tuple[str, int | None]] = []

    cond_nodes = graph.nodes_of_type("LTXVConditioning")
    if cond_nodes:
        wv = cond_nodes[0].widgets_values
        if isinstance(wv, (list, tuple)) and wv and isinstance(wv[0], (int, float)):
            checks.append(("LTXVConditioning", int(wv[0])))

    vhs_nodes = graph.nodes_of_type("VHS_VideoCombine")
    if vhs_nodes:
        wv = vhs_nodes[0].widgets_values
        if isinstance(wv, dict) and "frame_rate" in wv:
            checks.append(("VHS_VideoCombine", int(wv["frame_rate"])))

    for name, value in checks:
        if value is not None and value != derived_fps:
            print(
                f"⚠️ [timeline] fps mismatch: derived={derived_fps} but "
                f"{name} widget reports {value}"
            )
    return derived_fps


def parse_ltxdirector_timeline(graph: ParsedGraph) -> DirectorTimeline:
    """Parse the LTXDirector (id=131) timeline into a :class:`DirectorTimeline`.

    Finds the LTXDirector node, extracts and parses its serialized timeline JSON
    widget, and exposes convenient objects. ``motion_segments`` may legitimately
    be empty in this graph - that is handled gracefully; motion is never
    invented. Duration comes from LTXDirector numeric widgets (31.5) and frames
    (756); fps = round(frames / duration) and is cross-checked against
    LTXVConditioning and VHS_VideoCombine.
    """
    director_nodes = graph.nodes_of_type("LTXDirector")
    if not director_nodes:
        raise ValueError("No LTXDirector node found in graph (Master Timeline Controller missing).")
    director = graph.get_node(LTXDIRECTOR_NODE_ID) or director_nodes[0]

    wv = director.widgets_values
    _idx, timeline, raw_string = _find_timeline_widget(wv)

    segments = timeline.get("segments") or []
    audio_segments = timeline.get("audioSegments") or []
    motion_segments = timeline.get("motionSegments") or []  # may be empty

    global_prompt = timeline.get("global_prompt", "")

    # guide_strength: LTXDirector widget that is a comma-separated float string
    # like '1.00,1.00,1.00,1.00,1.00' (widget index 10 in this graph).
    def _is_guide_strength(text: str) -> bool:
        stripped = text.strip()
        if not stripped or "," not in stripped:
            return False
        parts = stripped.split(",")
        try:
            for part in parts:
                float(part)
        except ValueError:
            return False
        return True

    guide_strength = _first_string_matching(wv, _is_guide_strength) or ""

    # Derive duration + frames from the LTXDirector numeric widgets. The graph
    # stores duration (31.5) and frame count (756) among the leading numeric
    # widgets; prefer the timeline's own normalDurationFrames when present.
    frames = int(timeline.get("normalDurationFrames") or 0)
    duration_seconds = 0.0
    if isinstance(wv, (list, tuple)):
        floats = [w for w in wv if isinstance(w, float)]
        ints = [w for w in wv if isinstance(w, bool) is False and isinstance(w, int)]
        if floats:
            # The duration in seconds is the max float among the leading widgets
            # (31.5 here), which represents the timeline length.
            duration_seconds = float(max(floats))
        if frames <= 0 and ints:
            frames = int(max(ints))
    if frames <= 0:
        frames = int(timeline.get("normalDurationFrames") or 0)

    if duration_seconds <= 0 or frames <= 0:
        raise ValueError(
            f"Could not derive timeline duration/frames from LTXDirector widgets "
            f"(duration={duration_seconds}, frames={frames})."
        )

    fps = round(frames / duration_seconds)
    fps = _cross_check_fps(graph, fps)

    config_keys = (
        "mainTrackEnabled",
        "audioTrackEnabled",
        "motionTrackEnabled",
        "overrideAudio",
        "inpaint_audio",
        "normalStartFrame",
        "normalDurationFrames",
    )
    timeline_config = {k: timeline.get(k) for k in config_keys}
    if "frame_rate" in timeline:
        timeline_config["frame_rate"] = timeline["frame_rate"]

    return DirectorTimeline(
        node_id=director.id,
        raw_timeline_string=raw_string,
        timeline_config=timeline_config,
        timeline_segments=segments,
        audio_segments=audio_segments,
        motion_segments=motion_segments,
        global_prompt=global_prompt,
        guide_strength=guide_strength,
        duration_seconds=duration_seconds,
        frames=frames,
        fps=fps,
    )


# ════════════════════════════════════════════════════════════════════════════
# CELL 10 + CELL 27: ORIGINAL WORKFLOW VALIDATOR
# ════════════════════════════════════════════════════════════════════════════
# Expected node-type multiset. These are compared against PARSED counts; a
# mismatch is reported precisely (expected vs found).
_EXPECTED_NODE_TYPES: dict[str, int] = {
    "VAELoader": 2,
    "SamplerCustomAdvanced": 2,
    "LTXVSeparateAVLatent": 2,
    "LTXVConcatAVLatent": 2,
    "LTXDirectorGuide": 2,
    "LTXDirectorCropGuides": 2,
    "LatentUpscaleModelLoader": 1,
    "LTXVLatentUpsampler": 1,
    "KSamplerSelect": 2,
    "CFGGuider": 2,
    "BasicScheduler": 2,
    "VHS_VideoCombine": 1,
    "VAELoaderKJ": 1,
    "VAEDecode": 1,
    "UnetLoaderGGUF": 1,
    "RandomNoise": 1,
    "Power Lora Loader (rgthree)": 1,
    "ModelPreviewOverrideKJ": 1,
    "LTXVConditioning": 1,
    "LTXVAudioVAEDecode": 1,
    "LTXDirector": 1,
    "DualCLIPLoader": 1,
    "ConditioningZeroOut": 1,
}

_EXPECTED_NODE_COUNT = 32
_EXPECTED_LINK_COUNT = 65

_EXPECTED_LORAS: list[tuple[str, float]] = [
    ("ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors", 0.4),
    ("LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors", 0.6),
    ("ltx2.3-transition.safetensors", 0.7),
    ("LTX2.3-MVCamera-drclips.safetensors", 0.9),
]

# Node ids for model/stage/guide checks (structural identity of the graph).
_LORA_NODE_ID = 138
_UNET_NODE_ID = 135
_DUALCLIP_NODE_ID = 12
_AUDIO_VAE_NODE_ID = 8
_VIDEO_VAE_NODE_ID = 36
_TINY_VAE_NODE_ID = 6
_SPATIAL_UPSCALER_NODE_ID = 13
_STAGE1_SCHED_NODE_ID = 33
_STAGE2_SCHED_NODE_ID = 21
_STAGE1_KSAMPLER_NODE_ID = 20
_STAGE2_KSAMPLER_NODE_ID = 32
_STAGE1_GUIDE_NODE_ID = 132
_STAGE2_GUIDE_NODE_ID = 133
_VHS_NODE_ID = 139

_EXPECTED_MODELS = {
    "unet_gguf": "ltx-2-3-22b-dev-Q4_K_M.gguf",
    "dualclip": ["gemma_3_12B_it_fp4_mixed.safetensors", "ltx-2.3_text_projection_bf16.safetensors"],
    "audio_vae": "LTX23_audio_vae_bf16.safetensors",
    "video_vae": "LTX23_video_vae_bf16.safetensors",
    "tiny_vae": "taeltx2_3.safetensors",
    "spatial_upscaler": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
}

_EXPECTED_AUDIO_FILE = "Late night trap.mp3"
_EXPECTED_AUDIO_TRIMSTART = 446.9222739141953
_EXPECTED_AUDIO_DURATION_FRAMES = 2880
_EXPECTED_TIMELINE_FRAMES = 756
_EXPECTED_TIMELINE_FPS = 24
_EXPECTED_TIMELINE_DURATION = 31.5
_EXPECTED_STAGE1 = {"sampler": "euler", "scheduler": "linear_quadratic", "steps": 8, "denoise": 1.0, "guide": 1.0}
_EXPECTED_STAGE2 = {"sampler": "euler", "scheduler": "linear_quadratic", "steps": 4, "denoise": 0.42, "guide": 0.5}
_EXPECTED_VHS = {
    "format": "video/h264-mp4",
    "pix_fmt": "yuv420p",
    "crf": 8,
    "frame_rate": 24,
    "filename_prefix": "LTX2.3/Video",
}


class ValidationError(Exception):
    """Raised when the parsed graph does not match the original workflow."""


def _extract_loras(node: ParsedNode) -> list[tuple[str, float, bool]]:
    """Extract (lora_name, strength, on) tuples from a rgthree Power Lora node."""
    result: list[tuple[str, float, bool]] = []
    wv = node.widgets_values
    if not isinstance(wv, (list, tuple)):
        return result
    for entry in wv:
        if isinstance(entry, dict) and "lora" in entry and "strength" in entry:
            result.append((entry["lora"], float(entry["strength"]), bool(entry.get("on"))))
    return result


def _fmt(ok: bool, label: str, detail: str) -> tuple[bool, str]:
    status = "PASS" if ok else "FAIL"
    return ok, f"  [{status}] {label:<13} {detail}"


def validate_workflow(graph: ParsedGraph, timeline: DirectorTimeline) -> bool:
    """Validate the parsed graph and print the ORIGINAL WORKFLOW VALIDATION report.

    Compares parsed values against expectations derived from the source graph.
    Prints one line per category (Nodes / Connections / Models / LoRAs /
    Timeline / Audio / Stage 1 / Stage 2 / Final Video) as PASS or a precise
    mismatch. Returns True only if every check passes; otherwise raises
    :class:`ValidationError` with the exact expected-vs-found values.
    """
    lines: list[str] = []
    failures: list[str] = []

    def record(ok: bool, label: str, detail: str) -> None:
        ok2, line = _fmt(ok, label, detail)
        lines.append(line)
        if not ok2:
            failures.append(f"{label}: {detail}")

    # --- Nodes -------------------------------------------------------------
    count_ok = graph.node_count == _EXPECTED_NODE_COUNT
    missing_types: list[str] = []
    for ntype, expected in _EXPECTED_NODE_TYPES.items():
        found = graph.node_type_counts.get(ntype, 0)
        if found != expected:
            missing_types.append(f"{ntype} expected={expected} found={found}")
    types_ok = not missing_types
    if count_ok and types_ok:
        record(True, "Nodes", f"{graph.node_count} nodes, all {len(_EXPECTED_NODE_TYPES)} types present")
    else:
        detail = f"count expected={_EXPECTED_NODE_COUNT} found={graph.node_count}"
        if missing_types:
            detail += " | type mismatches: " + "; ".join(missing_types)
        record(False, "Nodes", detail)

    # --- Connections -------------------------------------------------------
    conn_ok = graph.link_count == _EXPECTED_LINK_COUNT
    record(
        conn_ok,
        "Connections",
        f"{graph.link_count} links"
        + ("" if conn_ok else f" (expected {_EXPECTED_LINK_COUNT})"),
    )

    # --- Models ------------------------------------------------------------
    model_problems: list[str] = []

    def _widget_list(node_id: int) -> list[Any]:
        node = graph.get_node(node_id)
        if node is None:
            model_problems.append(f"node id={node_id} missing")
            return []
        wv = node.widgets_values
        return list(wv) if isinstance(wv, (list, tuple)) else []

    unet_wv = _widget_list(_UNET_NODE_ID)
    if not unet_wv or unet_wv[0] != _EXPECTED_MODELS["unet_gguf"]:
        model_problems.append(
            f"unet expected={_EXPECTED_MODELS['unet_gguf']} found={unet_wv[:1]}"
        )
    dualclip_wv = _widget_list(_DUALCLIP_NODE_ID)
    if dualclip_wv[:2] != _EXPECTED_MODELS["dualclip"]:
        model_problems.append(
            f"dualclip expected={_EXPECTED_MODELS['dualclip']} found={dualclip_wv[:2]}"
        )
    for node_id, key in (
        (_AUDIO_VAE_NODE_ID, "audio_vae"),
        (_VIDEO_VAE_NODE_ID, "video_vae"),
        (_TINY_VAE_NODE_ID, "tiny_vae"),
        (_SPATIAL_UPSCALER_NODE_ID, "spatial_upscaler"),
    ):
        wv = _widget_list(node_id)
        if not wv or wv[0] != _EXPECTED_MODELS[key]:
            model_problems.append(f"{key} expected={_EXPECTED_MODELS[key]} found={wv[:1]}")
    models_ok = not model_problems
    record(
        models_ok,
        "Models",
        "gguf + dualclip pair + audio/video/tiny VAE + spatial upscaler"
        if models_ok
        else "; ".join(model_problems),
    )

    # --- LoRAs -------------------------------------------------------------
    lora_node = graph.get_node(_LORA_NODE_ID)
    lora_problems: list[str] = []
    parsed_loras: list[tuple[str, float, bool]] = []
    if lora_node is None:
        lora_problems.append(f"Power Lora Loader node id={_LORA_NODE_ID} missing")
    else:
        parsed_loras = _extract_loras(lora_node)
        if len(parsed_loras) != len(_EXPECTED_LORAS):
            lora_problems.append(
                f"count expected={len(_EXPECTED_LORAS)} found={len(parsed_loras)}"
            )
        else:
            for (name, strength, on), (exp_name, exp_strength) in zip(
                parsed_loras, _EXPECTED_LORAS
            ):
                if name != exp_name:
                    lora_problems.append(f"name expected={exp_name} found={name}")
                if abs(strength - exp_strength) > 1e-9:
                    lora_problems.append(
                        f"{name} strength expected={exp_strength} found={strength}"
                    )
                if not on:
                    lora_problems.append(f"{name} is not enabled (on=False)")
    loras_ok = not lora_problems
    if loras_ok:
        strengths = "/".join(str(s) for _n, s, _o in parsed_loras)
        record(True, "LoRAs", f"{len(parsed_loras)} LoRAs on, strengths {strengths}")
    else:
        record(False, "LoRAs", "; ".join(lora_problems))

    # --- Timeline ----------------------------------------------------------
    tl_problems: list[str] = []
    if timeline.frames != _EXPECTED_TIMELINE_FRAMES:
        tl_problems.append(f"frames expected={_EXPECTED_TIMELINE_FRAMES} found={timeline.frames}")
    if timeline.fps != _EXPECTED_TIMELINE_FPS:
        tl_problems.append(f"fps expected={_EXPECTED_TIMELINE_FPS} found={timeline.fps}")
    if abs(timeline.duration_seconds - _EXPECTED_TIMELINE_DURATION) > 1e-6:
        tl_problems.append(
            f"duration expected={_EXPECTED_TIMELINE_DURATION} found={timeline.duration_seconds}"
        )
    if len(timeline.timeline_segments) != 5:
        tl_problems.append(f"image segments expected=5 found={len(timeline.timeline_segments)}")
    tl_ok = not tl_problems
    record(
        tl_ok,
        "Timeline",
        f"{timeline.duration_seconds}s / {timeline.frames} frames / {timeline.fps} fps, "
        f"{len(timeline.timeline_segments)} image segments"
        if tl_ok
        else "; ".join(tl_problems),
    )

    # --- Audio -------------------------------------------------------------
    audio_problems: list[str] = []
    if not timeline.audio_segments:
        audio_problems.append("no audio segments found")
    else:
        aseg = timeline.audio_segments[0]
        fname = aseg.get("fileName") or aseg.get("audioFile", "")
        if _EXPECTED_AUDIO_FILE not in str(fname):
            audio_problems.append(f"audioFile expected~'{_EXPECTED_AUDIO_FILE}' found={fname}")
        trim = aseg.get("trimStart")
        if trim is None or abs(float(trim) - _EXPECTED_AUDIO_TRIMSTART) > 1e-3:
            audio_problems.append(
                f"trimStart expected≈{_EXPECTED_AUDIO_TRIMSTART} found={trim}"
            )
        adur = aseg.get("audioDurationFrames")
        if adur != _EXPECTED_AUDIO_DURATION_FRAMES:
            audio_problems.append(
                f"audioDurationFrames expected={_EXPECTED_AUDIO_DURATION_FRAMES} found={adur}"
            )
    audio_ok = not audio_problems
    record(
        audio_ok,
        "Audio",
        f"'{_EXPECTED_AUDIO_FILE}' trimStart≈{_EXPECTED_AUDIO_TRIMSTART}, "
        f"{_EXPECTED_AUDIO_DURATION_FRAMES} frames"
        if audio_ok
        else "; ".join(audio_problems),
    )

    # --- Stage 1 / Stage 2 -------------------------------------------------
    def _check_stage(
        label: str,
        ksampler_id: int,
        sched_id: int,
        guide_id: int,
        expected: dict[str, Any],
    ) -> None:
        problems: list[str] = []
        ks = _widget_list(ksampler_id)
        if not ks or ks[0] != expected["sampler"]:
            problems.append(f"sampler expected={expected['sampler']} found={ks[:1]}")
        sched = _widget_list(sched_id)
        # BasicScheduler widgets: [scheduler_name, steps, denoise]
        if not sched or sched[0] != expected["scheduler"]:
            problems.append(f"scheduler expected={expected['scheduler']} found={sched[:1]}")
        if len(sched) < 3:
            problems.append(f"scheduler widgets incomplete: {sched}")
        else:
            if int(sched[1]) != expected["steps"]:
                problems.append(f"steps expected={expected['steps']} found={sched[1]}")
            if abs(float(sched[2]) - expected["denoise"]) > 1e-9:
                problems.append(f"denoise expected={expected['denoise']} found={sched[2]}")
        guide = _widget_list(guide_id)
        # LTXDirectorGuide widgets: guide strength is at index 2.
        if len(guide) < 3:
            problems.append(f"guide widgets incomplete for node {guide_id}: {guide}")
        elif abs(float(guide[2]) - expected["guide"]) > 1e-9:
            problems.append(f"guide strength expected={expected['guide']} found={guide[2]}")
        ok = not problems
        record(
            ok,
            label,
            f"{expected['sampler']}/{expected['scheduler']} steps={expected['steps']} "
            f"denoise={expected['denoise']} guide={expected['guide']}"
            if ok
            else "; ".join(problems),
        )

    _check_stage("Stage 1", _STAGE1_KSAMPLER_NODE_ID, _STAGE1_SCHED_NODE_ID, _STAGE1_GUIDE_NODE_ID, _EXPECTED_STAGE1)
    _check_stage("Stage 2", _STAGE2_KSAMPLER_NODE_ID, _STAGE2_SCHED_NODE_ID, _STAGE2_GUIDE_NODE_ID, _EXPECTED_STAGE2)

    # --- Final Video -------------------------------------------------------
    vhs_node = graph.get_node(_VHS_NODE_ID)
    vhs_problems: list[str] = []
    if vhs_node is None or not isinstance(vhs_node.widgets_values, dict):
        vhs_problems.append(f"VHS_VideoCombine node id={_VHS_NODE_ID} missing or malformed")
    else:
        wv = vhs_node.widgets_values
        for key, exp in _EXPECTED_VHS.items():
            found = wv.get(key)
            if found != exp:
                vhs_problems.append(f"{key} expected={exp} found={found}")
    video_ok = not vhs_problems
    record(
        video_ok,
        "Final Video",
        f"VHS_VideoCombine {_EXPECTED_VHS['format']} {_EXPECTED_VHS['pix_fmt']} "
        f"crf={_EXPECTED_VHS['crf']} fps={_EXPECTED_VHS['frame_rate']} "
        f"prefix='{_EXPECTED_VHS['filename_prefix']}'"
        if video_ok
        else "; ".join(vhs_problems),
    )

    # --- Report ------------------------------------------------------------
    print("=" * 70)
    print("ORIGINAL WORKFLOW VALIDATION")
    print("=" * 70)
    for line in lines:
        print(line)
    print("=" * 70)

    if failures:
        raise ValidationError(
            "Workflow validation failed:\n  - " + "\n  - ".join(failures)
        )
    return True


# ════════════════════════════════════════════════════════════════════════════
# NODE REGISTRY / MEMORY-AWARE EXECUTOR / FULL PIPELINE  (authored in FEAT-002)
# ════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════
# CLI ENTRYPOINT
# ════════════════════════════════════════════════════════════════════════════
def run_selftest(json_path: str | None = None) -> int:
    """Load -> parse -> timeline -> validate against the REAL repo JSON.

    Does NOT import torch/comfy. Returns 0 on success, non-zero on failure.
    Prints measured facts (node/link counts, timeline, LoRA names+strengths).
    """
    try:
        data = load_workflow_json(json_path)
        graph = parse_graph(data)
        timeline = parse_ltxdirector_timeline(graph)
        validate_workflow(graph, timeline)
    except (FileNotFoundError, ValueError, KeyError, ValidationError) as exc:
        print(f"❌ SELFTEST FAILED: {exc}")
        return 1

    lora_node = graph.get_node(_LORA_NODE_ID)
    loras = _extract_loras(lora_node) if lora_node else []

    print()
    print("MEASURED FACTS")
    print("-" * 70)
    print(f"  Node count : {graph.node_count}")
    print(f"  Link count : {graph.link_count}")
    print(
        f"  Timeline   : {timeline.duration_seconds}s / "
        f"{timeline.frames} frames / {timeline.fps} fps"
    )
    print(f"  Image segs : {len(timeline.timeline_segments)}")
    print(f"  Audio segs : {len(timeline.audio_segments)}")
    print(f"  Motion segs: {len(timeline.motion_segments)}")
    print("  LoRA stack :")
    for name, strength, on in loras:
        print(f"    - {name}  strength={strength}  on={on}")
    print("-" * 70)
    print("✅ SELFTEST PASSED")
    return 0


def run_pipeline(json_path: str | None = None) -> int:
    """Full Colab pipeline entrypoint (implemented in FEAT-002).

    This stub validates the graph first (import-safe), then hands off to the
    memory-aware executor. The executor and Colab bootstrap are authored in
    FEAT-002; running --run in this GPU-less sandbox is not supported.
    """
    print(
        "--run: the full graph-faithful Colab pipeline (ComfyUI bootstrap, "
        "model download, memory-aware graph executor) is authored in later "
        "features and requires a Colab GPU runtime. Use --selftest here."
    )
    # Validate up front so misconfiguration is caught before any heavy work.
    data = load_workflow_json(json_path)
    graph = parse_graph(data)
    timeline = parse_ltxdirector_timeline(graph)
    validate_workflow(graph, timeline)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LTX-2.3 Director 2.0 MV - graph-faithful re-execution.",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Load+parse+validate the workflow JSON without importing torch/comfy.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the full Colab pipeline (implemented in FEAT-002; needs a GPU).",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help=f"Path to the workflow JSON (default: {WORKFLOW_JSON_PATH}).",
    )
    args = parser.parse_args(argv)

    if args.run:
        return run_pipeline(args.json_path)
    # Default action in the sandbox is the selftest.
    return run_selftest(args.json_path)


if __name__ == "__main__":
    sys.exit(main())
