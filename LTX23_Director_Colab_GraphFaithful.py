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
# CELL 2: SYSTEM INFO
# (import-safe; warns instead of crashing when GPU/psutil are absent)
# ════════════════════════════════════════════════════════════════════════════
# Colab free-tier T4 target. These are advisory thresholds only.
TARGET_MIN_FREE_RAM_GB = 12.2
TARGET_GPU_SUBSTRINGS = ("T4", "L4", "A100", "V100", "P100")


def cell2_system_info() -> dict[str, Any]:
    """Print and return a system-info snapshot (GPU, RAM, Python, disk).

    Never crashes when torch/psutil/CUDA are unavailable; it warns instead. The
    returned dict is convenient for checkpointing and the final audit report.
    """
    info: dict[str, Any] = {}
    print("=" * 70)
    print("CELL 2: SYSTEM INFO")
    print("=" * 70)

    info["python_version"] = sys.version.split()[0]
    print(f"  Python           : {info['python_version']}")

    # GPU / VRAM (torch.cuda).
    torch = _import_torch()
    gpu_name: str | None = None
    total_vram_gb: float | None = None
    if torch is not None:
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                total_vram_gb = props.total_memory / 1e9
        except Exception as exc:  # noqa: BLE001 - report, don't hide
            print(f"  [warn] torch.cuda query failed: {exc!r}")
    info["gpu_name"] = gpu_name
    info["total_vram_gb"] = total_vram_gb
    if gpu_name:
        print(f"  GPU              : {gpu_name} ({total_vram_gb:.1f} GB VRAM)")
        if not any(sub in gpu_name for sub in TARGET_GPU_SUBSTRINGS):
            print(
                f"  ⚠️ GPU '{gpu_name}' is not a recognized T4-class Colab GPU; "
                f"proceed with caution."
            )
    else:
        print("  GPU              : none detected (CPU-only / no CUDA)")
        print("  ⚠️ No CUDA GPU detected. Full generation requires a Colab GPU runtime.")

    # CPU RAM (psutil).
    total_ram_gb: float | None = None
    free_ram_gb = get_free_ram_gb()
    psutil = _import_psutil()
    if psutil is not None:
        try:
            total_ram_gb = psutil.virtual_memory().total / 1e9
        except Exception as exc:  # noqa: BLE001 - report, don't hide
            print(f"  [warn] psutil.virtual_memory failed: {exc!r}")
    info["total_ram_gb"] = total_ram_gb
    info["free_ram_gb"] = free_ram_gb
    ram_desc = f"{free_ram_gb:.1f} GB free" if free_ram_gb is not None else "unknown"
    total_desc = f" / {total_ram_gb:.1f} GB total" if total_ram_gb is not None else ""
    print(f"  RAM              : {ram_desc}{total_desc}")
    if free_ram_gb is not None and free_ram_gb < TARGET_MIN_FREE_RAM_GB:
        print(
            f"  ⚠️ Free RAM {free_ram_gb:.1f} GB is below the ~{TARGET_MIN_FREE_RAM_GB} "
            f"GB target for this workflow. Memory guards will engage aggressively."
        )

    # Disk free.
    try:
        usage = os.statvfs(_SCRIPT_DIR)
        disk_free_gb = usage.f_bavail * usage.f_frsize / 1e9
        info["disk_free_gb"] = disk_free_gb
        print(f"  Disk free        : {disk_free_gb:.1f} GB")
    except (OSError, AttributeError) as exc:
        info["disk_free_gb"] = None
        print(f"  Disk free        : unknown ({exc!r})")

    print("=" * 70)
    return info


# ════════════════════════════════════════════════════════════════════════════
# CELL 3-7: COLAB BOOTSTRAP - ComfyUI install, custom-node fetch, model download
# ----------------------------------------------------------------------------
# These cells run ON COLAB. They shell out to git/pip and verify files on disk.
# All values (model / LoRA / audio / image names) are READ FROM the parsed graph
# and timeline, never hardcoded twice. Every step reports its result and raises a
# clear error on failure - no silent swallowing.
# ════════════════════════════════════════════════════════════════════════════

# ComfyUI upstream. comfy-core node versions observed in the source JSON range
# roughly 0.7.0 .. 0.24.0 (see context). We install the latest ComfyUI because
# its bundled core nodes are backward compatible with these node classes; the
# exact per-node vers are reported (not pinned) so drift is visible.
COMFYUI_REPO_URL = "https://github.com/comfyanonymous/ComfyUI.git"

# Default Colab install root. Overridable via COMFYUI_ROOT env var.
COMFYUI_ROOT = os.environ.get("COMFYUI_ROOT", "/content/ComfyUI")

# Exact pinned custom nodes. Where the source JSON's properties.ver gives a
# commit hash we pin the commit; otherwise we pin the version tag. Each entry:
#   folder            -> subdir under custom_nodes
#   url               -> git repo
#   ref               -> commit hash or tag to check out (None = default branch)
#   ref_kind          -> 'commit' | 'tag' | 'branch'
#   provides          -> node classes this repo supplies (for the audit)
#   required_version  -> the version string exactly as recorded in the JSON
PINNED_CUSTOM_NODES: list[dict[str, Any]] = [
    {
        "folder": "WhatDreamsCost-ComfyUI",
        "url": "https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI.git",
        "ref": "2.0.0",
        "ref_kind": "tag",
        "provides": ["LTXDirector", "LTXDirectorGuide", "LTXDirectorCropGuides"],
        "required_version": "whatdreamscost-comfyui 2.0.0 (CropGuides 1.3.9)",
    },
    {
        "folder": "ComfyUI-KJNodes",
        "url": "https://github.com/kijai/ComfyUI-KJNodes.git",
        "ref": "996b010ae4613ae0743121ace5975830dcf8e6af",
        "ref_kind": "commit",
        "provides": ["ModelPreviewOverrideKJ", "VAELoaderKJ"],
        "required_version": "comfyui-kjnodes @ 996b010ae4613ae0743121ace5975830dcf8e6af",
    },
    {
        "folder": "ComfyUI-GGUF",
        "url": "https://github.com/city96/ComfyUI-GGUF.git",
        "ref": "6ea2651e7df66d7585f6ffee804b20e92fb38b8a",
        "ref_kind": "commit",
        "provides": ["UnetLoaderGGUF"],
        "required_version": "ComfyUI-GGUF @ 6ea2651e7df66d7585f6ffee804b20e92fb38b8a",
    },
    {
        "folder": "rgthree-comfy",
        "url": "https://github.com/rgthree/rgthree-comfy.git",
        "ref": "dbc5fa5e89b6a8b6a1a1dda787505b690f18026c",
        "ref_kind": "commit",
        "provides": ["Power Lora Loader (rgthree)"],
        "required_version": "rgthree-comfy @ dbc5fa5e89b6a8b6a1a1dda787505b690f18026c",
    },
    {
        "folder": "ComfyUI-LTXVideo",
        "url": "https://github.com/Lightricks/ComfyUI-LTXVideo.git",
        "ref": None,
        "ref_kind": "branch",
        "provides": [
            "LTXVConcatAVLatent",
            "LTXVSeparateAVLatent",
            "LTXVLatentUpsampler",
            "LTXVAudioVAEDecode",
            "LTXVConditioning",
        ],
        "required_version": "ComfyUI-LTXVideo (comfy-core LTXV nodes 0.7.0-0.16.0)",
    },
    {
        "folder": "ComfyUI-VideoHelperSuite",
        "url": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
        "ref": "v1.7.9",
        "ref_kind": "tag",
        "provides": ["VHS_VideoCombine"],
        "required_version": "comfyui-videohelpersuite 1.7.9",
    },
]

# Node classes that MUST exist after install (CELL 4b) mapped to the repo that
# supplies them, so a missing class produces a precise, named error.
REQUIRED_NODE_CLASSES: dict[str, str] = {
    "LTXDirector": "WhatDreamsCost-ComfyUI 2.0.0",
    "LTXDirectorGuide": "WhatDreamsCost-ComfyUI 2.0.0",
    "LTXDirectorCropGuides": "WhatDreamsCost-ComfyUI 1.3.9",
    "Power Lora Loader (rgthree)": "rgthree-comfy @ dbc5fa5e",
    "ModelPreviewOverrideKJ": "ComfyUI-KJNodes @ 996b010a",
    "VAELoaderKJ": "ComfyUI-KJNodes 1.2.5",
    "UnetLoaderGGUF": "ComfyUI-GGUF @ 6ea2651e",
    "LTXVLatentUpsampler": "ComfyUI-LTXVideo",
    "LTXVConcatAVLatent": "ComfyUI-LTXVideo",
    "LTXVSeparateAVLatent": "ComfyUI-LTXVideo",
    "LTXVConditioning": "ComfyUI-LTXVideo",
    "LTXVAudioVAEDecode": "ComfyUI-LTXVideo",
    "DualCLIPLoader": "comfy-core (bundled with ComfyUI)",
    "VHS_VideoCombine": "ComfyUI-VideoHelperSuite 1.7.9",
}


def run_cmd(cmd: list[str], cwd: str | None = None, check: bool = True) -> int:
    """Run a shell command with real error reporting (no silent failure).

    Streams the command, prints its exit status, and raises RuntimeError with
    the captured tail of output when ``check`` and the command fails.
    """
    import subprocess  # local import: not needed for the selftest path

    printable = " ".join(cmd)
    print(f"  $ {printable}" + (f"   (cwd={cwd})" if cwd else ""))
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout.rstrip()[-2000:])
    if proc.returncode != 0:
        tail = (proc.stderr or "").rstrip()[-2000:]
        print(tail)
        if check:
            raise RuntimeError(
                f"Command failed (exit {proc.returncode}): {printable}\n{tail}"
            )
    return proc.returncode


def _pip_install(args: list[str]) -> None:
    """pip install into the current interpreter with error reporting."""
    run_cmd([sys.executable, "-m", "pip", "install", "--no-input", *args])


def cell3_install_comfyui(comfyui_root: str = COMFYUI_ROOT) -> str:
    """CELL 3: clone ComfyUI and install its requirements.

    Installing the latest ComfyUI is acceptable for the core node classes used
    here; comfy-core node vers seen in the source JSON range ~0.7.0-0.24.0 and
    are backward compatible. Returns the ComfyUI root path.
    """
    print("=" * 70)
    print("CELL 3: INSTALL COMFYUI")
    print("=" * 70)
    if os.path.isdir(os.path.join(comfyui_root, ".git")):
        print(f"  ComfyUI already present at {comfyui_root}; skipping clone.")
    else:
        parent = os.path.dirname(comfyui_root) or "."
        os.makedirs(parent, exist_ok=True)
        run_cmd(["git", "clone", "--depth", "1", COMFYUI_REPO_URL, comfyui_root])
    req = os.path.join(comfyui_root, "requirements.txt")
    if os.path.exists(req):
        _pip_install(["-r", req])
    else:
        raise RuntimeError(f"ComfyUI requirements.txt not found at {req}")
    print(f"  ComfyUI ready at {comfyui_root}")
    return comfyui_root


def _git_checkout_ref(folder_path: str, node: dict[str, Any]) -> None:
    """Check out the pinned ref (commit/tag/branch) for a custom node repo."""
    ref = node.get("ref")
    if not ref:
        return
    ref_kind = node.get("ref_kind", "commit")
    # Fetch the specific ref (tags need an explicit fetch after a shallow clone).
    run_cmd(["git", "fetch", "--depth", "1", "origin", ref], cwd=folder_path, check=False)
    rc = run_cmd(["git", "checkout", ref], cwd=folder_path, check=False)
    if rc != 0:
        # Fall back to a full fetch then checkout (covers tags/commits missing
        # from the shallow clone). Failure here is a hard error.
        run_cmd(["git", "fetch", "--unshallow"], cwd=folder_path, check=False)
        run_cmd(["git", "fetch", "--all", "--tags"], cwd=folder_path, check=False)
        run_cmd(["git", "checkout", ref], cwd=folder_path)
    print(f"  {node['folder']} pinned to {ref_kind} {ref}")


def cell4_install_custom_nodes(comfyui_root: str = COMFYUI_ROOT) -> None:
    """CELL 4: install the exact pinned custom nodes and their requirements."""
    print("=" * 70)
    print("CELL 4: INSTALL PINNED CUSTOM NODES")
    print("=" * 70)
    custom_nodes_dir = os.path.join(comfyui_root, "custom_nodes")
    os.makedirs(custom_nodes_dir, exist_ok=True)
    for node in PINNED_CUSTOM_NODES:
        folder_path = os.path.join(custom_nodes_dir, node["folder"])
        print(f"- {node['folder']}  ({node['required_version']})")
        if os.path.isdir(os.path.join(folder_path, ".git")):
            print(f"  already present at {folder_path}; fetching pinned ref.")
        else:
            run_cmd(["git", "clone", node["url"], folder_path])
        _git_checkout_ref(folder_path, node)
        req = os.path.join(folder_path, "requirements.txt")
        if os.path.exists(req):
            _pip_install(["-r", req])
        else:
            print(f"  (no requirements.txt for {node['folder']})")
        print(f"  ✅ installed {node['folder']}")
    print("=" * 70)


def cell4b_verify_node_classes(node_mappings: dict[str, Any]) -> None:
    """CELL 4b: assert every required node class exists after install.

    ``node_mappings`` is ComfyUI's ``NODE_CLASS_MAPPINGS``. A missing class
    raises a CLEAR error naming the node, its supplying repo, and the required
    version. NO silent fallback.
    """
    print("=" * 70)
    print("CELL 4b: VERIFY REQUIRED NODE CLASSES")
    print("=" * 70)
    missing: list[str] = []
    for class_name, source in REQUIRED_NODE_CLASSES.items():
        if class_name in node_mappings:
            print(f"  [PASS] {class_name}")
        else:
            print(f"  [FAIL] {class_name}  <- provided by {source}")
            missing.append(f"'{class_name}' (install {source})")
    if missing:
        raise RuntimeError(
            "Required ComfyUI node classes are missing after install:\n  - "
            + "\n  - ".join(missing)
            + "\nInstall the exact custom nodes/versions listed above; do NOT "
            "fall back to a substitute node."
        )
    print("=" * 70)


def _models_root(comfyui_root: str) -> str:
    return os.path.join(comfyui_root, "models")


# Map each parsed model to the ComfyUI models subdirectory it belongs in and,
# when a public download URL is configurable via env, use it. The env var names
# let a Colab user supply URLs without editing the source.
def _model_specs_from_graph(graph: ParsedGraph) -> list[dict[str, Any]]:
    """Build the model verification specs by READING names from the graph."""

    def w0(node_id: int) -> str:
        node = graph.require_node(node_id)
        wv = node.widgets_values
        if not isinstance(wv, (list, tuple)) or not wv:
            raise ValueError(f"node id={node_id} has no widget file name")
        return str(wv[0])

    dualclip = graph.require_node(_DUALCLIP_NODE_ID).widgets_values
    return [
        {"name": w0(_UNET_NODE_ID), "subdir": "unet", "env": "URL_UNET_GGUF"},
        {"name": str(dualclip[0]), "subdir": "text_encoders", "env": "URL_CLIP_GEMMA"},
        {"name": str(dualclip[1]), "subdir": "text_encoders", "env": "URL_CLIP_PROJ"},
        {"name": w0(_VIDEO_VAE_NODE_ID), "subdir": "vae", "env": "URL_VIDEO_VAE"},
        {"name": w0(_AUDIO_VAE_NODE_ID), "subdir": "vae", "env": "URL_AUDIO_VAE"},
        {"name": w0(_TINY_VAE_NODE_ID), "subdir": "vae_approx", "env": "URL_TINY_VAE"},
        {
            "name": w0(_SPATIAL_UPSCALER_NODE_ID),
            "subdir": "upscale_models",
            "env": "URL_SPATIAL_UPSCALER",
        },
    ]


def _download_url(url: str, dest_path: str) -> None:
    """Download ``url`` to ``dest_path`` with a progress-less streaming copy."""
    import urllib.request  # local import: not needed for the selftest path

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp = dest_path + ".part"
    print(f"  downloading -> {dest_path}")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    os.replace(tmp, dest_path)


def _verify_or_download(name: str, directory: str, url: str | None) -> str:
    """Return the verified path of ``name`` in ``directory``.

    Downloads from ``url`` when provided and the file is missing. If the file is
    still missing, raises a hard error naming the exact filename, directory, and
    (when known) the URL. No placeholder is ever created.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"  [OK] {name}")
        return path
    if url:
        _download_url(url, path)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"  [OK] {name} (downloaded)")
            return path
    raise FileNotFoundError(
        f"Required asset missing: '{name}'\n"
        f"  expected in: {directory}\n"
        f"  download URL: {url or '(none configured - set the matching URL_* env var)'}\n"
        f"  Place the exact file there; no placeholder will be substituted."
    )


def cell5_download_verify_models(graph: ParsedGraph, comfyui_root: str = COMFYUI_ROOT) -> dict[str, str]:
    """CELL 5: verify/download all model files into the correct subdirs."""
    print("=" * 70)
    print("CELL 5: DOWNLOAD / VERIFY MODELS")
    print("=" * 70)
    models_root = _models_root(comfyui_root)
    resolved: dict[str, str] = {}
    for spec in _model_specs_from_graph(graph):
        directory = os.path.join(models_root, spec["subdir"])
        url = os.environ.get(spec["env"])
        resolved[spec["name"]] = _verify_or_download(spec["name"], directory, url)
    print("=" * 70)
    return resolved


def cell6_download_verify_loras(graph: ParsedGraph, comfyui_root: str = COMFYUI_ROOT) -> dict[str, str]:
    """CELL 6: verify/download the 4 LoRAs with exact strengths from node 138."""
    print("=" * 70)
    print("CELL 6: DOWNLOAD / VERIFY LORAS")
    print("=" * 70)
    lora_node = graph.get_node(_LORA_NODE_ID)
    if lora_node is None:
        raise ValueError(f"Power Lora Loader node id={_LORA_NODE_ID} missing from graph.")
    loras = _extract_loras(lora_node)
    if not loras:
        raise ValueError("No LoRAs parsed from the Power Lora Loader node.")
    loras_dir = os.path.join(_models_root(comfyui_root), "loras")
    resolved: dict[str, str] = {}
    for name, strength, on in loras:
        env_key = "URL_LORA_" + "".join(c if c.isalnum() else "_" for c in name).upper()
        url = os.environ.get(env_key)
        path = _verify_or_download(name, loras_dir, url)
        resolved[name] = path
        print(f"       strength={strength}  on={on}")
    print("=" * 70)
    return resolved


def cell7_download_verify_audio_images(
    graph: ParsedGraph, timeline: DirectorTimeline, comfyui_root: str = COMFYUI_ROOT
) -> dict[str, str]:
    """CELL 7: verify audio + the 5 reference images in input/whatdreamscost.

    A missing reference image is a HARD error: placeholders would silently
    destroy character consistency across the 5 timeline segments. The imageB64
    '/api/view?...' hints in the timeline describe the expected input subfolder
    layout (input/whatdreamscost/<file>).
    """
    print("=" * 70)
    print("CELL 7: DOWNLOAD / VERIFY AUDIO + REFERENCE IMAGES")
    print("=" * 70)
    input_root = os.path.join(comfyui_root, "input")
    resolved: dict[str, str] = {}

    # Audio (from the parsed audio segment; filename read, never hardcoded).
    if not timeline.audio_segments:
        raise ValueError("Timeline has no audio segment; cannot verify audio file.")
    aseg = timeline.audio_segments[0]
    audio_rel = str(aseg.get("audioFile") or aseg.get("fileName") or "")
    if not audio_rel:
        raise ValueError("Audio segment has no audioFile/fileName.")
    audio_dir = os.path.join(input_root, os.path.dirname(audio_rel) or "whatdreamscost")
    audio_name = os.path.basename(audio_rel)
    resolved[audio_rel] = _verify_or_download(
        audio_name, audio_dir, os.environ.get("URL_AUDIO_MP3")
    )

    # Reference images (read from the timeline segments; NO placeholders).
    for seg in timeline.timeline_segments:
        img_rel = str(seg.get("imageFile") or "")
        if not img_rel:
            raise ValueError(
                "A timeline image segment is missing 'imageFile'; refusing to "
                "substitute a placeholder (would break character consistency)."
            )
        img_dir = os.path.join(input_root, os.path.dirname(img_rel) or "whatdreamscost")
        img_name = os.path.basename(img_rel)
        env_key = "URL_IMG_" + "".join(c if c.isalnum() else "_" for c in img_name).upper()
        resolved[img_rel] = _verify_or_download(
            img_name, img_dir, os.environ.get(env_key)
        )
    print(f"  verified 1 audio + {len(timeline.timeline_segments)} reference images")
    print("=" * 70)
    return resolved


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
# CELL 11: NODE REGISTRY
# ----------------------------------------------------------------------------
# After ComfyUI + custom nodes are importable, map each JSON node 'type' string
# to the actual ComfyUI class in NODE_CLASS_MAPPINGS. Handles display-name vs
# class-name differences (e.g. 'Power Lora Loader (rgthree)'). This is where the
# guarded heavy imports (nodes / comfy) actually happen. Any unmapped type is a
# clear, named error.
# ════════════════════════════════════════════════════════════════════════════
# A few source 'type' strings differ from their NODE_CLASS_MAPPINGS key. This
# table records known aliases; lookups still fall back to display_name matching.
_NODE_TYPE_ALIASES: dict[str, str] = {
    # display name -> possible class-mapping key (checked in addition to itself)
    "Power Lora Loader (rgthree)": "Power Lora Loader (rgthree)",
}


def _load_comfy_node_mappings(comfyui_root: str = COMFYUI_ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    """Import ComfyUI and return (NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS).

    Adds ComfyUI to sys.path, initializes custom nodes, and imports ``nodes``.
    Import happens here (guarded) so the selftest path never triggers it.
    """
    if comfyui_root not in sys.path:
        sys.path.insert(0, comfyui_root)

    nodes_mod = _try_import("nodes")
    if nodes_mod is None:
        raise RuntimeError(
            f"Could not import ComfyUI 'nodes' module. Is ComfyUI installed at "
            f"{comfyui_root}? Run CELL 3 (cell3_install_comfyui) first."
        )
    # Initialize custom nodes so their NODE_CLASS_MAPPINGS get merged in.
    init_custom = getattr(nodes_mod, "init_extra_nodes", None) or getattr(
        nodes_mod, "init_custom_nodes", None
    )
    if callable(init_custom):
        try:
            init_custom()
        except Exception as exc:  # noqa: BLE001 - report; custom node import errors matter
            print(f"  [warn] init_extra_nodes raised: {exc!r}")
    class_mappings = dict(getattr(nodes_mod, "NODE_CLASS_MAPPINGS", {}))
    display_mappings = dict(getattr(nodes_mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))
    return class_mappings, display_mappings


def build_node_registry(
    graph: ParsedGraph, comfyui_root: str = COMFYUI_ROOT
) -> dict[str, Any]:
    """CELL 11: map every JSON node 'type' to its ComfyUI class.

    Returns a dict {type_string: class}. Raises a clear error listing every
    unmapped type. Also runs CELL 4b verification against the loaded mappings.
    """
    print("=" * 70)
    print("CELL 11: BUILD NODE REGISTRY")
    print("=" * 70)
    class_mappings, display_mappings = _load_comfy_node_mappings(comfyui_root)

    # CELL 4b: verify required classes exist BEFORE we try to map the full graph.
    cell4b_verify_node_classes(class_mappings)

    # Invert display-name mapping so we can resolve display names to class keys.
    display_to_key: dict[str, str] = {v: k for k, v in display_mappings.items()}

    registry: dict[str, Any] = {}
    unmapped: list[str] = []
    for node_type in sorted(graph.node_type_counts):
        cls = class_mappings.get(node_type)
        if cls is None:
            alias = _NODE_TYPE_ALIASES.get(node_type)
            if alias and alias in class_mappings:
                cls = class_mappings[alias]
            elif node_type in display_to_key:
                cls = class_mappings.get(display_to_key[node_type])
        if cls is None:
            unmapped.append(node_type)
        else:
            registry[node_type] = cls
    if unmapped:
        raise RuntimeError(
            "Unmapped node types (no ComfyUI class found in NODE_CLASS_MAPPINGS):\n  - "
            + "\n  - ".join(unmapped)
            + "\nEnsure the exact custom nodes from CELL 4 are installed."
        )
    print(f"  mapped {len(registry)} node types to ComfyUI classes")
    print("=" * 70)
    return registry


# ════════════════════════════════════════════════════════════════════════════
# CHECKPOINT / RESUME SYSTEM  (CELL 34 requirement)
# ----------------------------------------------------------------------------
# Persists execution state to workflow_state.json: current node, phase,
# completed nodes, output paths, timeline state, error state. On restart the
# executor resumes from the last safe checkpoint and skips completed expensive
# stages. This never changes graph semantics - it only records what already ran.
# ════════════════════════════════════════════════════════════════════════════
WORKFLOW_STATE_PATH = os.environ.get(
    "WORKFLOW_STATE_PATH", os.path.join(_SCRIPT_DIR, "workflow_state.json")
)


class Checkpoint:
    """Read/write the resumable workflow_state.json checkpoint."""

    def __init__(self, path: str = WORKFLOW_STATE_PATH) -> None:
        self.path = path
        self.state: dict[str, Any] = {
            "phase": "init",
            "current_node": None,
            "completed_nodes": [],
            "output_paths": {},
            "timeline": {},
            "error": None,
        }

    def load(self) -> bool:
        """Load an existing checkpoint. Returns True if one was found."""
        if not os.path.exists(self.path):
            return False
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                self.state = json.load(handle)
            print(f"[checkpoint] resumed from {self.path} (phase={self.state.get('phase')})")
            return True
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[checkpoint] could not read {self.path}: {exc!r}; starting fresh")
            return False

    def save(self) -> None:
        """Atomically persist the current state."""
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(self.state, handle, indent=2)
            os.replace(tmp, self.path)
        except OSError as exc:
            print(f"[checkpoint] save failed ({exc!r}); continuing without checkpoint")

    def is_done(self, node_id: int) -> bool:
        return node_id in self.state.get("completed_nodes", [])

    def mark_phase(self, phase: str) -> None:
        self.state["phase"] = phase
        self.save()

    def mark_current(self, node_id: int) -> None:
        self.state["current_node"] = node_id
        self.save()

    def mark_done(self, node_id: int) -> None:
        done = self.state.setdefault("completed_nodes", [])
        if node_id not in done:
            done.append(node_id)
        self.state["current_node"] = None
        self.save()

    def record_output(self, key: str, path: str) -> None:
        self.state.setdefault("output_paths", {})[key] = path
        self.save()

    def record_error(self, detail: str) -> None:
        self.state["error"] = detail
        self.save()

    def clear_error(self) -> None:
        self.state["error"] = None
        self.save()


# ════════════════════════════════════════════════════════════════════════════
# CELL 35: STRUCTURED ERROR REPORTING
# ----------------------------------------------------------------------------
# Wraps core node execution so every failure prints a banner with node id, type,
# FUNCTION, input types, model-loaded flag, free RAM/VRAM, current stage, and the
# full traceback. Forbidden anywhere in core execution: 'except Exception: pass'.
# ════════════════════════════════════════════════════════════════════════════
class NodeExecutionError(RuntimeError):
    """A node failed to execute; carries structured diagnostic context."""


def _report_node_failure(
    node: ParsedNode,
    function_name: str,
    input_types: dict[str, str],
    stage: str,
    exc: BaseException,
) -> str:
    """Build and print the structured failure banner. Returns the banner text."""
    import traceback

    ram = get_free_ram_gb()
    vram = get_free_vram_gb()
    banner = [
        "!" * 70,
        f"NODE EXECUTION FAILED  [Node {node.id}] {node.type}",
        "!" * 70,
        f"  Node id       : {node.id}",
        f"  Type          : {node.type}",
        f"  FUNCTION      : {function_name}",
        f"  Input types   : {input_types}",
        f"  Stage         : {stage}",
        f"  Free RAM (GB) : {ram}",
        f"  Free VRAM(GB) : {vram}",
        f"  Error         : {type(exc).__name__}: {exc}",
        "-" * 70,
        traceback.format_exc().rstrip(),
        "!" * 70,
    ]
    text = "\n".join(banner)
    print(text)
    return text


# ════════════════════════════════════════════════════════════════════════════
# CELL 15: MEMORY-AWARE EXECUTOR
# ----------------------------------------------------------------------------
# execute_node()/call_comfy_node() instantiate the EXACT original node class,
# inspect its FUNCTION attribute, call the correct method, and map JSON link
# inputs to kwargs BY INPUT NAME. ComfyUI return conventions (tuple/list/dict)
# are preserved. Per-node RAM/VRAM is logged before and after in the exact
# '[Node N] Type / RAM before / VRAM before / Executing... / RAM after / VRAM
# after' format. A topological runner walks the PARSED graph in dependency
# order, wiring outputs->inputs exactly as the JSON links specify.
# ════════════════════════════════════════════════════════════════════════════
# Stages whose loads/decodes are expensive enough to warrant memory guarding.
_EXPENSIVE_NODE_TYPES = frozenset(
    {
        "UnetLoaderGGUF",
        "DualCLIPLoader",
        "Power Lora Loader (rgthree)",
        "LTXDirector",
        "LTXVLatentUpsampler",
        "SamplerCustomAdvanced",
        "VAEDecode",
        "LTXVAudioVAEDecode",
        "VHS_VideoCombine",
    }
)


def _link_by_id(graph: ParsedGraph, link_id: int) -> list[Any] | None:
    """Return the link tuple with ``link_id`` or None.

    Link format: [link_id, from_node, from_slot, to_node, to_slot, type].
    """
    for lk in graph.links:
        if lk and lk[0] == link_id:
            return lk
    return None


class GraphExecutor:
    """Memory-aware topological executor over the PARSED ComfyUI graph."""

    def __init__(
        self,
        graph: ParsedGraph,
        timeline: DirectorTimeline,
        registry: dict[str, Any],
        checkpoint: Checkpoint | None = None,
    ) -> None:
        self.graph = graph
        self.timeline = timeline
        self.registry = registry
        self.checkpoint = checkpoint or Checkpoint()
        # results[node_id] -> the node's output tuple/list (indexable by slot).
        self.results: dict[int, Any] = {}
        self._instances: dict[int, Any] = {}
        self.stage = "init"
        # Metrics for the final audit (CELL 18).
        self.ram_peak_used_gb: float = 0.0
        self.vram_peak_used_gb: float = 0.0

    # -- topological ordering ------------------------------------------------
    def _topological_order(self) -> list[int]:
        """Kahn topological sort using link dependencies.

        Falls back to the node 'order' field to break ties so execution follows
        the authored ordering when the DAG allows multiple valid orders.
        """
        nodes = self.graph.nodes_by_id
        indeg: dict[int, int] = {nid: 0 for nid in nodes}
        deps: dict[int, set[int]] = {nid: set() for nid in nodes}
        for lk in self.graph.links:
            if len(lk) < 4:
                continue
            src, dst = lk[1], lk[3]
            if src in nodes and dst in nodes and src not in deps[dst]:
                deps[dst].add(src)
                indeg[dst] += 1
        # Ready set ordered by authored 'order' then id for determinism.
        ready = sorted(
            (nid for nid, d in indeg.items() if d == 0),
            key=lambda n: (nodes[n].order, n),
        )
        order: list[int] = []
        while ready:
            nid = ready.pop(0)
            order.append(nid)
            for other in nodes:
                if nid in deps[other]:
                    deps[other].discard(nid)
                    indeg[other] -= 1
                    if indeg[other] == 0:
                        ready.append(other)
            ready.sort(key=lambda n: (nodes[n].order, n))
        if len(order) != len(nodes):
            unresolved = [nid for nid in nodes if nid not in order]
            raise RuntimeError(
                f"Graph is not a DAG or has dangling deps; could not order nodes: {unresolved}"
            )
        return order

    # -- input wiring --------------------------------------------------------
    def _resolve_inputs(self, node: ParsedNode) -> tuple[dict[str, Any], dict[str, str]]:
        """Resolve a node's linked inputs to kwargs keyed by input NAME.

        Returns (kwargs, input_types) where input_types maps name->declared link
        type for the failure banner. Only connected inputs are resolved here;
        widget parameters are added later by :meth:`_widget_kwargs`.
        """
        kwargs: dict[str, Any] = {}
        input_types: dict[str, str] = {}
        for inp in node.inputs:
            link_id = inp.get("link")
            name = inp.get("name")
            if link_id is None or name is None:
                continue
            lk = _link_by_id(self.graph, link_id)
            if lk is None:
                raise NodeExecutionError(
                    f"Node {node.id} ({node.type}) input '{name}' references missing "
                    f"link id={link_id}."
                )
            from_node, from_slot = lk[1], lk[2]
            input_types[name] = str(lk[5]) if len(lk) > 5 else "?"
            if from_node not in self.results:
                raise NodeExecutionError(
                    f"Node {node.id} ({node.type}) input '{name}' needs output of "
                    f"node {from_node} which has not executed yet (topology error)."
                )
            src_out = self.results[from_node]
            kwargs[name] = self._index_output(src_out, from_slot, from_node)
        return kwargs, input_types

    @staticmethod
    def _index_output(output: Any, slot: int, from_node: int) -> Any:
        """Index a node's output by slot, honoring ComfyUI tuple/list returns."""
        if isinstance(output, (tuple, list)):
            if slot < len(output):
                return output[slot]
            raise NodeExecutionError(
                f"Output slot {slot} out of range for node {from_node} "
                f"(returned {len(output)} outputs)."
            )
        # Single non-sequence return maps to slot 0.
        if slot == 0:
            return output
        raise NodeExecutionError(
            f"Node {from_node} returned a scalar but slot {slot} was requested."
        )

    # -- widget -> parameter mapping ----------------------------------------
    @staticmethod
    def _class_input_order(node_cls: Any) -> tuple[list[str], set[str]]:
        """Return (ordered_input_names, connected_capable_names) from INPUT_TYPES.

        ComfyUI declares inputs via INPUT_TYPES(); widget values fill the
        non-linked inputs in declared order. We collect required+optional names
        in declaration order so widget positions can be matched by position.
        """
        names: list[str] = []
        get = getattr(node_cls, "INPUT_TYPES", None)
        if not callable(get):
            return names, set()
        try:
            spec = get()
        except Exception:  # noqa: BLE001 - some nodes need args; skip widget-by-name
            return names, set()
        for section in ("required", "optional"):
            names.extend(spec.get(section) or {})
        return names, set(names)

    def _widget_kwargs(
        self, node: ParsedNode, cls: Any, linked_names: set[str]
    ) -> dict[str, Any]:
        """Map the node's widgets_values to parameter kwargs by declared order.

        Widget values fill INPUT_TYPES inputs that are NOT satisfied by a link,
        in declaration order (ComfyUI's convention). Dict widgets_values (e.g.
        VHS_VideoCombine) are mapped by key directly.
        """
        wv = node.widgets_values
        if wv is None:
            return {}
        input_names, _ = self._class_input_order(cls)

        if isinstance(wv, dict):
            # Only pass keys the class actually declares as inputs.
            return {k: v for k, v in wv.items() if k in set(input_names)}

        if not isinstance(wv, (list, tuple)):
            return {}

        # Positional fill: walk declared inputs, skipping linked ones, and assign
        # successive widget values. Widget-only extras (seed control combos etc.)
        # that have no declared input are ignored.
        result: dict[str, Any] = {}
        widget_iter = iter(wv)
        for iname in input_names:
            if iname in linked_names:
                continue
            try:
                value = next(widget_iter)
            except StopIteration:
                break
            result[iname] = value
        return result

    # -- single node execution ----------------------------------------------
    def execute_node(self, node: ParsedNode) -> Any:
        """Instantiate and run one node; log RAM/VRAM before and after.

        Preserves ComfyUI return conventions. Never substitutes a different
        node. On failure prints the structured banner and raises
        NodeExecutionError.
        """
        cls = self.registry.get(node.type)
        if cls is None:
            raise NodeExecutionError(
                f"No registered class for node {node.id} type '{node.type}'."
            )

        function_name = getattr(cls, "FUNCTION", None)
        if not function_name:
            raise NodeExecutionError(
                f"Node {node.id} ({node.type}) class has no FUNCTION attribute; "
                f"cannot execute (refusing to guess)."
            )

        kwargs, input_types = self._resolve_inputs(node)
        kwargs.update(self._widget_kwargs(node, cls, set(kwargs)))

        expensive = node.type in _EXPENSIVE_NODE_TYPES
        if expensive:
            memory_guard(tag=f"pre:{node.type}")

        ram_before = get_free_ram_gb()
        vram_before = get_free_vram_gb()
        print(f"[Node {node.id}] {node.type}")
        print(f"    RAM before  : {_gb(ram_before)}")
        print(f"    VRAM before : {_gb(vram_before)}")
        print("    Executing...")

        try:
            instance = self._instances.get(node.id)
            if instance is None:
                instance = cls()
                self._instances[node.id] = instance
            method = getattr(instance, function_name)
            output = method(**kwargs)
        except Exception as exc:
            self._report_and_raise(node, function_name, input_types, exc)
            raise  # unreachable (helper raises), satisfies type-checkers

        ram_after = get_free_ram_gb()
        vram_after = get_free_vram_gb()
        print(f"    RAM after   : {_gb(ram_after)}")
        print(f"    VRAM after  : {_gb(vram_after)}")
        self._update_peaks(ram_before, ram_after, vram_before, vram_after)

        if expensive:
            deep_memory_cleanup(f"post:{node.type}")
        return output

    def _report_and_raise(
        self,
        node: ParsedNode,
        function_name: str,
        input_types: dict[str, str],
        exc: Exception,
    ) -> None:
        detail = _report_node_failure(node, function_name, input_types, self.stage, exc)
        self.checkpoint.record_error(detail.splitlines()[1])
        raise NodeExecutionError(
            f"[Node {node.id}] {node.type}.{function_name} failed during stage "
            f"'{self.stage}': {type(exc).__name__}: {exc}"
        ) from exc

    def _update_peaks(
        self,
        ram_before: float | None,
        ram_after: float | None,
        vram_before: float | None,
        vram_after: float | None,
    ) -> None:
        if ram_before is not None and ram_after is not None:
            self.ram_peak_used_gb = max(self.ram_peak_used_gb, max(0.0, ram_before - ram_after))
        if vram_before is not None and vram_after is not None:
            self.vram_peak_used_gb = max(
                self.vram_peak_used_gb, max(0.0, vram_before - vram_after)
            )

    # -- alias requested by the spec ----------------------------------------
    def call_comfy_node(self, node: ParsedNode) -> Any:
        """Compatibility alias for :meth:`execute_node` (spec naming)."""
        return self.execute_node(node)

    # -- full topological run ------------------------------------------------
    def run(self) -> dict[int, Any]:
        """Execute every node in topological order, honoring the checkpoint.

        Stage labels track where we are (Stage 1 / Stage 2 / Decode / Combine)
        for the failure banner. Completed nodes recorded in the checkpoint are
        skipped so a resumed run does not repeat expensive stages.
        """
        order = self._topological_order()
        print("=" * 70)
        print(f"EXECUTING GRAPH: {len(order)} nodes in topological order")
        print("=" * 70)
        for node_id in order:
            node = self.graph.require_node(node_id)
            if node.mode == 4:  # bypassed/muted node in ComfyUI
                print(f"[Node {node_id}] {node.type}  (muted; skipped)")
                continue
            if self.checkpoint.is_done(node_id) and node_id in self.results:
                print(f"[Node {node_id}] {node.type}  (checkpoint: already done)")
                continue
            self.stage = self._stage_for(node)
            self.checkpoint.mark_current(node_id)
            self.results[node_id] = self.execute_node(node)
            self.checkpoint.mark_done(node_id)
        print("=" * 70)
        print("GRAPH EXECUTION COMPLETE")
        print("=" * 70)
        return self.results

    def _stage_for(self, node: ParsedNode) -> str:
        """Human-readable stage label for diagnostics/checkpointing."""
        if node.id in (_STAGE1_GUIDE_NODE_ID, _STAGE1_KSAMPLER_NODE_ID, _STAGE1_SCHED_NODE_ID, 19):
            return "Stage 1 (euler/8/denoise1.0/linear_quadratic)"
        if node.id in (
            _SPATIAL_UPSCALER_NODE_ID,
            14,
            _STAGE2_GUIDE_NODE_ID,
            _STAGE2_KSAMPLER_NODE_ID,
            _STAGE2_SCHED_NODE_ID,
            31,
        ):
            return "Stage 2 (upsampler + euler/4/denoise0.42/linear_quadratic)"
        if node.type in ("VAEDecode", "LTXVAudioVAEDecode", "LTXVSeparateAVLatent"):
            return "Decode"
        if node.type == "VHS_VideoCombine":
            return "Combine"
        if node.type == "LTXDirector":
            return "Director master timeline"
        return "Load / wire"


def _gb(value: float | None) -> str:
    """Format an optional GB value for logging."""
    return f"{value:.2f} GB" if isinstance(value, (int, float)) else "n/a"


# ════════════════════════════════════════════════════════════════════════════
# CELLs 13, 14, 16, 17, 18, 19  (orchestrated through the executor's topo run)
# ----------------------------------------------------------------------------
# CELL 13: LOAD ORIGINAL GRAPH COMPONENTS
#   UnetLoaderGGUF -> Power Lora Loader -> ModelPreviewOverrideKJ -> LTXDirector
#   DualCLIPLoader, VAELoader x2, VAELoaderKJ, LatentUpscaleModelLoader. Each of
#   these is a node in the parsed graph with its widgets_values read from the
#   JSON. The executor instantiates each loader class, calls its FUNCTION with
#   widget kwargs, wraps it in memory_guard/deep_memory_cleanup, and stores the
#   output for downstream linking. They execute in topological order (the graph's
#   own dependency structure), not a manually-ordered script. Lazy-load: the
#   executor only instantiates a node when it becomes ready (all deps satisfied).
#   Reuse: immutable results are cached in self.results and never re-loaded.
#
# CELL 14: LTXDirector MASTER TIMELINE (node id=131).
#   See cell14_verify_director_outputs below.
#
# CELL 16: DECODE (VAEDecode / LTXVAudioVAEDecode / LTXVSeparateAVLatent).
#   VAEDecode uses the video VAE (LTX23_video_vae_bf16) to decode video latents
#   to IMAGE. LTXVAudioVAEDecode uses the audio VAE (LTX23_audio_vae_bf16) to
#   decode audio latents to AUDIO. Both SeparateAVLatent nodes split the
#   composite AV latent from the sampler. All memory-guarded by the executor.
#
# CELL 17: VHS_VideoCombine (node id=139).
#   Wires VAEDecode IMAGE, LTXVAudioVAEDecode AUDIO, LTXDirector frame_rate (24)
#   into VHS with settings from its dict widgets_values (h264-mp4, yuv420p, crf 8
#   frame_rate 24, filename_prefix 'LTX2.3/Video', save_output true). This is the
#   final output combiner; it does NOT rely on a separate FFmpeg mux for primary
#   audio sync since the Director/audio-VAE output feeds VHS directly.
# ════════════════════════════════════════════════════════════════════════════
def cell14_verify_director_outputs(executor: GraphExecutor) -> None:
    """CELL 14: confirm LTXDirector produced all 8 outputs and stays master.

    The Director is executed as part of the topological run (it feeds every
    downstream stage). Here we assert it produced its 8-tuple so no downstream
    node silently received a substitute.
    """
    director_out = executor.results.get(LTXDIRECTOR_NODE_ID)
    if director_out is None:
        raise NodeExecutionError(
            f"LTXDirector (id={LTXDIRECTOR_NODE_ID}) did not execute; it is the "
            f"master timeline controller and must run."
        )
    if not isinstance(director_out, (tuple, list)) or len(director_out) < 8:
        raise NodeExecutionError(
            f"LTXDirector produced {len(director_out) if isinstance(director_out, (tuple, list)) else 'non-sequence'} "
            f"outputs; expected 8 (model, positive, video_latent, audio_latent, "
            f"guide_data, motion_guide_data, frame_rate, combined_audio)."
        )
    print(
        "[CELL 14] LTXDirector master timeline produced 8 outputs "
        "(model/positive/video_latent/audio_latent/guide_data/motion_guide_data/"
        "frame_rate/combined_audio)."
    )


def cell18_validate_final_output(
    executor: GraphExecutor, exec_seconds: float
) -> dict[str, Any]:
    """CELL 18: verify the produced MP4 and report measured facts.

    Reports the measured output frame count/duration (when derivable), RAM peak,
    VRAM peak, and execution time. Compares to the expected 756 frames / 31.5 s /
    24 fps but does NOT claim a guaranteed crash-free run.
    """
    print("=" * 70)
    print("CELL 18: VALIDATE FINAL OUTPUT")
    print("=" * 70)
    report: dict[str, Any] = {}

    vhs_out = executor.results.get(_VHS_NODE_ID)
    mp4_path = _extract_mp4_path(vhs_out)
    report["mp4_path"] = mp4_path
    if mp4_path and os.path.exists(mp4_path):
        size = os.path.getsize(mp4_path)
        print(f"  Output MP4    : {mp4_path} ({size / 1e6:.2f} MB)")
        report["mp4_exists"] = True
    else:
        print(f"  ⚠️ Output MP4 not found on disk (VHS returned: {vhs_out!r})")
        report["mp4_exists"] = False

    measured_frames, measured_duration = _measure_output(executor)
    report["measured_frames"] = measured_frames
    report["measured_duration"] = measured_duration
    print(
        f"  Frames        : measured={measured_frames} "
        f"expected={_EXPECTED_TIMELINE_FRAMES}"
    )
    print(
        f"  Duration      : measured={measured_duration} "
        f"expected={_EXPECTED_TIMELINE_DURATION}s @ {_EXPECTED_TIMELINE_FPS}fps"
    )
    if measured_frames is not None and measured_frames != _EXPECTED_TIMELINE_FRAMES:
        print(
            f"  ⚠️ Frame count differs from expected {_EXPECTED_TIMELINE_FRAMES}; "
            f"inspect the timeline/sampler output."
        )

    report["ram_peak_used_gb"] = round(executor.ram_peak_used_gb, 3)
    report["vram_peak_used_gb"] = round(executor.vram_peak_used_gb, 3)
    report["exec_seconds"] = round(exec_seconds, 2)
    print(f"  RAM peak used : {report['ram_peak_used_gb']} GB (approx)")
    print(f"  VRAM peak used: {report['vram_peak_used_gb']} GB (approx)")
    print(f"  Exec time     : {report['exec_seconds']} s")
    print(
        "  NOTE: metrics are measured, not guaranteed. This run reflects the "
        "actual graph execution; it does not promise a crash-free run on all "
        "hardware."
    )
    print("=" * 70)
    return report


def _extract_mp4_path(vhs_output: Any) -> str | None:
    """Best-effort extraction of the saved MP4 path from VHS_VideoCombine output.

    VHS returns a dict-like ui/result payload; the filenames live under
    result[0][1] (a list of written files). We scan for a .mp4 path.
    """
    candidates: list[str] = []

    def _scan(obj: Any) -> None:
        if isinstance(obj, str):
            if obj.lower().endswith(".mp4"):
                candidates.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _scan(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _scan(v)

    _scan(vhs_output)
    # Prefer an existing file; else the last candidate.
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[-1] if candidates else None


def _measure_output(executor: GraphExecutor) -> tuple[int | None, float | None]:
    """Measure frame count + duration from the decoded IMAGE tensor if present."""
    frames: int | None = None
    decode_out = executor.results.get(1)  # VAEDecode node id=1 -> IMAGE
    image = None
    if isinstance(decode_out, (tuple, list)) and decode_out:
        image = decode_out[0]
    elif decode_out is not None:
        image = decode_out
    try:
        if image is not None and hasattr(image, "shape"):
            # ComfyUI IMAGE tensors are [batch(frames), H, W, C].
            frames = int(image.shape[0])
    except Exception as exc:  # noqa: BLE001 - measurement is best-effort, report it
        print(f"  [warn] could not measure frame count: {exc!r}")
    duration = frames / executor.timeline.fps if frames else None
    return frames, duration


def cell19_display_download(mp4_path: str | None) -> None:
    """CELL 19: display the MP4 in Colab and offer a download. Prints the path."""
    print("=" * 70)
    print("CELL 19: DISPLAY / DOWNLOAD")
    print("=" * 70)
    if not mp4_path or not os.path.exists(mp4_path):
        print("  No MP4 available to display/download.")
        return
    print(f"  Final video: {mp4_path}")

    # Colab display (guarded: no-op outside Colab / IPython).
    ipy = _try_import("IPython.display")
    if ipy is not None:
        try:
            with open(mp4_path, "rb") as handle:
                data = handle.read()
            b64 = _b64(data)
            ipy.display(
                ipy.HTML(
                    f'<video width="512" controls>'
                    f'<source src="data:video/mp4;base64,{b64}" type="video/mp4"></video>'
                )
            )
        except Exception as exc:  # noqa: BLE001 - display is optional, report it
            print(f"  [warn] inline display failed: {exc!r}")

    colab_files = _try_import("google.colab.files")
    if colab_files is not None:
        try:
            colab_files.download(mp4_path)
        except Exception as exc:  # noqa: BLE001 - download is optional, report it
            print(f"  [warn] colab download prompt failed: {exc!r}")
    print("=" * 70)


def _b64(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode("ascii")


# ════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE DRIVER  (ties CELLs 2-19 together for --run on Colab)
# ════════════════════════════════════════════════════════════════════════════
def run_full_pipeline(json_path: str | None = None, comfyui_root: str = COMFYUI_ROOT) -> int:
    """Execute the entire graph-faithful pipeline on a Colab GPU runtime.

    This is the real --run path. It is import-safe to DEFINE (heavy imports are
    all lazy), but calling it requires ComfyUI + a GPU. Steps: system info ->
    load+parse+validate the JSON (fail fast) -> install ComfyUI + pinned custom
    nodes -> verify assets -> build node registry (+CELL 4b) -> topological
    execute (Director master timeline, Stage 1, Stage 2, decode, combine) ->
    validate output -> display/download.
    """
    import time

    checkpoint = Checkpoint()
    checkpoint.load()

    try:
        cell2_system_info()

        # Parse + validate up front so misconfiguration fails before heavy work.
        data = load_workflow_json(json_path)
        graph = parse_graph(data)
        timeline = parse_ltxdirector_timeline(graph)
        validate_workflow(graph, timeline)
        checkpoint.state["timeline"] = {
            "frames": timeline.frames,
            "fps": timeline.fps,
            "duration_seconds": timeline.duration_seconds,
        }
        checkpoint.mark_phase("assets")

        # Bootstrap (idempotent; skips work already present).
        cell3_install_comfyui(comfyui_root)
        cell4_install_custom_nodes(comfyui_root)
        cell5_download_verify_models(graph, comfyui_root)
        cell6_download_verify_loras(graph, comfyui_root)
        cell7_download_verify_audio_images(graph, timeline, comfyui_root)

        checkpoint.mark_phase("registry")
        registry = build_node_registry(graph, comfyui_root)

        checkpoint.mark_phase("execute")
        executor = GraphExecutor(graph, timeline, registry, checkpoint)
        start = time.time()
        executor.run()
        exec_seconds = time.time() - start

        cell14_verify_director_outputs(executor)

        checkpoint.mark_phase("validate")
        report = cell18_validate_final_output(executor, exec_seconds)
        if report.get("mp4_path"):
            checkpoint.record_output("mp4", report["mp4_path"])

        checkpoint.mark_phase("display")
        cell19_display_download(report.get("mp4_path"))

        checkpoint.mark_phase("done")
        checkpoint.clear_error()
        print("✅ PIPELINE COMPLETE")
        return 0
    except (NodeExecutionError, ValidationError) as exc:
        checkpoint.record_error(str(exc))
        print(f"❌ PIPELINE FAILED: {exc}")
        return 1
    except (FileNotFoundError, RuntimeError, ValueError, KeyError, MemoryError) as exc:
        checkpoint.record_error(str(exc))
        print(f"❌ PIPELINE FAILED: {type(exc).__name__}: {exc}")
        return 1


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
    """Full Colab pipeline entrypoint.

    Delegates to :func:`run_full_pipeline`, which runs CELLs 2-19: system info,
    ComfyUI + pinned custom-node install, asset verification, node registry,
    the memory-aware topological graph executor (LTXDirector master timeline,
    Stage 1, Stage 2, decode, VHS combine), final-output validation, and the
    Colab display/download. Requires a Colab GPU runtime; heavy imports are
    lazy so importing this module stays GPU-free.
    """
    return run_full_pipeline(json_path)


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
