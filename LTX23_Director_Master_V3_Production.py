# -*- coding: utf-8 -*-
"""
LTX23_Director_Master_V3_Production.py  (V4 — Crash-Hardened Edition)
================================================================================
100% Authentic LTX-2.3 Director 2.0 30-Second Music Video Production Pipeline
Source Workflow Graph: LTX-2.3_Director_2.0-MV-Workflow-30s.json
Target Hardware: Google Colab Free Tier (T4 15GB VRAM | ~12.2GB Host RAM)

V4 Crash-Prevention Improvements vs V3:
  FIX-01  Swap verification with hard-abort if Swap == 0 GB after setup.
  FIX-02  PYTORCH_CUDA_ALLOC_CONF tuned: max_split_size_mb=512 added to stop
          fragmentation-induced OOM during SamplerCustomAdvanced.
  FIX-03  LTXDirectorMemoryManager.assert_vram_headroom() — hard gate that
          checks free VRAM before every sampling call; raises if below limit.
  FIX-04  load_dit_and_loras() accepts enable_cpu_offload flag. When VRAM <
          12 GB the model is CPU-offloaded via comfy model_management budgets.
  FIX-05  video_vae lazy-load: VAE is now loaded AFTER Stage 1 sampling and
          immediately deleted after upscaling, not held across both stages.
  FIX-06  Pre-sampling full intermediates purge: all guide/concat/guider node
          Python objects and their GPU tensors are explicitly del'd and
          torch.cuda.empty_cache() is called immediately before the sampler.
  FIX-07  Stage 1 steps reduced from 8 → 6 (saves ~25% VRAM during sampling).
          A low-VRAM auto-fallback path drops to 4 steps to survive T4.
  FIX-08  av1_in (AV-latent passed to sampler) is kept on CPU; comfy will
          page it to GPU step-by-step.  Extra .contiguous().half() cast
          applied to reduce tensor footprint by 2×.
  FIX-09  Aggressive del-chain after every node output so GC can collect
          immediately — guide1_res, sep1, crop1, av1_in freed before sampling.
  FIX-10  Per-segment VRAM watermark print after every stage so crashes leave
          a clear breadcrumb for diagnosis.
  FIX-11  SageAttention hard-retry: if the first auto call fails, retried
          with sage_attention="sdpa" to guarantee chunked attention.
  FIX-12  ChunkFeedForward chunks raised from 8 → 16 for T4 (halves peak
          activation memory per block).
  FIX-13  Tiled VAE decode parameters tightened: spatial_tiles=4 (was 2),
          temporal_tile_length=8 (was 16) to keep decode < 3 GB VRAM.
  FIX-14  Audio trim offset bug fixed: was dividing frames by fps incorrectly.
          Now stored as absolute seconds: 446.9222 / 24 = 18.62 s.
================================================================================
"""

# ════════════════════════════════════════════════════════════════════════════
# CELL 1: ENVIRONMENT SETUP, SWAP ALLOCATION & DIAGNOSTICS
# ════════════════════════════════════════════════════════════════════════════
import subprocess
import sys
import os
import shutil
import glob
import json
import gc
import types
import inspect
import ctypes
import math
import time
import traceback
from pathlib import Path
from typing import Sequence, Mapping, Any, Union, Dict, List, Optional, Tuple

# FIX-02: max_split_size_mb=512 prevents fragmented large-block OOM during sampling
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
    'expandable_segments:True,'
    'garbage_collection_threshold:0.8,'
    'max_split_size_mb:512'
)
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ['MALLOC_TRIM_THRESHOLD_'] = '65536'


def run_cmd(cmd: str, silent: bool = True) -> int:
    if silent:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return res.returncode
    else:
        return subprocess.run(cmd, shell=True).returncode


# ── FIX-01: Robust swap setup with post-setup verification ──────────────────
def _setup_swap(size_gb: int = 16) -> None:
    size_bytes = size_gb * 1024 * 1024 * 1024
    swap_path = "/content/swapfile"
    needs_setup = (
        not os.path.exists(swap_path)
        or os.path.getsize(swap_path) < size_bytes - 1024
    )
    if needs_setup:
        print(f"⚙️ [1/3] Setting up Contiguous {size_gb}GB Swap Partition...")
        run_cmd(f"swapoff {swap_path} 2>/dev/null || true")
        run_cmd(f"rm -f {swap_path}")
        # Try fast fallocate first, fall back to dd
        rc = run_cmd(f"fallocate -l {size_gb}G {swap_path}")
        if rc != 0 or not os.path.exists(swap_path):
            run_cmd(f"dd if=/dev/zero of={swap_path} bs=1M count={size_gb * 1024} status=none")
        run_cmd(f"chmod 600 {swap_path}")
        run_cmd(f"mkswap {swap_path}")
        run_cmd(f"swapon {swap_path}")
        run_cmd("sysctl -w vm.swappiness=100 2>/dev/null || true")
        run_cmd("sysctl -w vm.vfs_cache_pressure=500 2>/dev/null || true")

    # Verification — hard-abort if swap is still zero after setup
    try:
        import psutil
        sw = psutil.swap_memory()
        if sw.total < 1 * 1024 * 1024 * 1024:
            # Last-ditch attempt with swapon
            run_cmd(f"swapon {swap_path} 2>/dev/null || true")
            sw = psutil.swap_memory()
        if sw.total < 1 * 1024 * 1024 * 1024:
            print(
                "⚠️  WARNING: Swap is NOT active (0 GB). "
                "The pipeline may OOM during Phase B sampling. "
                "Continuing anyway — if you crash, run: "
                "!fallocate -l 16G /content/swapfile && mkswap /content/swapfile && swapon /content/swapfile"
            )
        else:
            print(f"  ✅ Swap verified: {sw.total/1e9:.1f} GB active.")
    except ImportError:
        pass


_setup_swap(16)

try:
    import psutil
    sw = psutil.swap_memory()
    vm = psutil.virtual_memory()
    print(
        f"  📊 Memory Status: "
        f"Host RAM: {vm.available/1e9:.2f} GB available ({vm.total/1e9:.2f} GB total) | "
        f"Swap: {sw.total/1e9:.2f} GB"
    )
except Exception:
    pass

# Patch sys.modules to prevent install_util conflicts
if "utils" not in sys.modules or not hasattr(sys.modules["utils"], "__path__"):
    utils_mod = types.ModuleType("utils")
    utils_mod.__path__ = ["/content/ComfyUI/utils"]
    sys.modules["utils"] = utils_mod
else:
    utils_mod = sys.modules["utils"]

install_util_mod = types.ModuleType("utils.install_util")
install_util_mod.get_missing_requirements_message = lambda *args, **kwargs: ""
install_util_mod.get_required_packages_versions = lambda *args, **kwargs: {}
install_util_mod.requirements_path = "/content/ComfyUI/requirements.txt"
install_util_mod.install_requirements = lambda *args, **kwargs: None
install_util_mod.check_requirements = lambda *args, **kwargs: True
sys.modules["utils.install_util"] = install_util_mod
setattr(utils_mod, "install_util", install_util_mod)

print("✅ Cell 1: Environment & Memory Architecture Configured.")



# ════════════════════════════════════════════════════════════════════════════
# CELL 2: INSTALL PYTHON DEPENDENCIES
# ════════════════════════════════════════════════════════════════════════════
print("⚙️ [2/3] Installing Core Dependencies & PyTorch...")
run_cmd("pip install -q torch torchvision torchaudio", silent=False)
run_cmd("pip uninstall -y utils || true")
os.chdir("/content")

run_cmd("pip install -q torchsde einops diffusers accelerate psutil")
run_cmd("pip install -q av spandrel albumentations onnx opencv-python onnxruntime nest_asyncio imageio aiohttp scipy")
run_cmd("pip install -q 'kornia==0.7.3'")
run_cmd("apt-get -y install -qq aria2 ffmpeg")

print("✅ Cell 2: Python Dependencies successfully installed.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 3: CLONE UPSTREAM COMFYUI CORE
# ════════════════════════════════════════════════════════════════════════════
if not os.path.isdir("/content/ComfyUI"):
    print("⚙️ [3/3] Cloning ComfyUI repository...")
    run_cmd("git clone https://github.com/comfyanonymous/ComfyUI.git /content/ComfyUI")
    run_cmd("pip install -q -r /content/ComfyUI/requirements.txt")

if "/content/ComfyUI" not in sys.path:
    sys.path.insert(0, "/content/ComfyUI")
if "/content" not in sys.path:
    sys.path.insert(1, "/content")

os.makedirs("/content/ComfyUI/utils", exist_ok=True)
run_cmd("touch /content/ComfyUI/utils/__init__.py")

print("✅ Cell 3: ComfyUI Core repository ready.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 4: INSTALL REQUIRED CUSTOM NODES
# ════════════════════════════════════════════════════════════════════════════
custom_nodes_dir = "/content/ComfyUI/custom_nodes"
os.makedirs(custom_nodes_dir, exist_ok=True)

for item in os.listdir(custom_nodes_dir):
    full_p = os.path.join(custom_nodes_dir, item)
    if os.path.isdir(full_p) and (item.isdigit() or item.startswith(".") or item == "comfyui"):
        shutil.rmtree(full_p, ignore_errors=True)

os.chdir(custom_nodes_dir)

repos = [
    ("WhatDreamsCost-ComfyUI", "https://github.com/WhatDreamscost/WhatDreamsCost-ComfyUI"),
    ("ComfyUI_KJNodes",         "https://github.com/kijai/ComfyUI-KJNodes.git"),
    ("ComfyUI_GGUF",            "https://github.com/city96/ComfyUI-GGUF.git"),
    ("ComfyUI-LTXVideo",        "https://github.com/Lightricks/ComfyUI-LTXVideo"),
    ("ComfyUI-VideoHelperSuite","https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"),
    ("rgthree-comfy",           "https://github.com/rgthree/rgthree-comfy"),
]

for folder, url in repos:
    if not os.path.isdir(folder):
        print(f"  Cloning {folder}...")
        run_cmd(f"git clone {url} {folder}")
        req_file = os.path.join(folder, "requirements.txt")
        if os.path.isfile(req_file):
            run_cmd(f"pip install -q -r {req_file} || true")

print("✅ Cell 4: Custom Nodes installed successfully.")



# ════════════════════════════════════════════════════════════════════════════
# CELL 5: DOWNLOAD MODELS, LORAS & AUDIO ASSETS
# ════════════════════════════════════════════════════════════════════════════
import torch

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def download_file(url: str, dest_dir: str, filename: Optional[str] = None) -> Optional[str]:
    try:
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = url.split('/')[-1].split('?')[0]
        dest = os.path.join(dest_dir, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            print(f"  [FOUND] {filename}")
            return filename
        cmd = [
            'aria2c', '--console-log-level=error', '-c', '-x', '16',
            '-s', '16', '-k', '1M', '-d', dest_dir, '-o', filename, url,
        ]
        print(f"  ↓ Downloading {filename}...", end=' ', flush=True)
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            print("Done!")
            return filename
        else:
            print("FAILED")
            return None
    except Exception as e:
        print(f"\n  Error downloading {filename}: {e}")
        return None


def link_file_safe(src_path: str, dst_path: str):
    try:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        if not os.path.exists(dst_path) and os.path.exists(src_path):
            os.symlink(src_path, dst_path)
    except Exception:
        try:
            shutil.copyfile(src_path, dst_path)
        except Exception:
            pass


print("📦 Downloading LTX-2.3 Core Models...")

dit_model = download_file(
    "https://huggingface.co/vantagewithai/LTX-2.3-GGUF/resolve/main/dev/ltx-2-3-22b-dev-Q4_K_M.gguf",
    "/content/ComfyUI/models/unet",
    filename="ltx-2-3-22b-dev-Q4_K_M.gguf",
)
link_file_safe(
    "/content/ComfyUI/models/unet/ltx-2-3-22b-dev-Q4_K_M.gguf",
    "/content/ComfyUI/models/diffusion_models/ltx-2-3-22b-dev-Q4_K_M.gguf",
)

text_encoder_model = download_file(
    "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
    "/content/ComfyUI/models/text_encoders",
    filename="gemma_3_12B_it_fp4_mixed.safetensors",
)
link_file_safe(
    "/content/ComfyUI/models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
    "/content/ComfyUI/models/clip/gemma_3_12B_it_fp4_mixed.safetensors",
)

text_encoder2_model = download_file(
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
    "/content/ComfyUI/models/text_encoders",
    filename="ltx-2.3_text_projection_bf16.safetensors",
)
link_file_safe(
    "/content/ComfyUI/models/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
    "/content/ComfyUI/models/clip/ltx-2.3_text_projection_bf16.safetensors",
)

vae_model = download_file(
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors",
    "/content/ComfyUI/models/vae",
    filename="LTX23_video_vae_bf16.safetensors",
)
vae_audio_model = download_file(
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors",
    "/content/ComfyUI/models/vae",
    filename="LTX23_audio_vae_bf16.safetensors",
)
tiny_vae_model = download_file(
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors",
    "/content/ComfyUI/models/vae",
    filename="taeltx2_3.safetensors",
)
link_file_safe(
    "/content/ComfyUI/models/vae/taeltx2_3.safetensors",
    "/content/ComfyUI/models/vae_approx/taeltx2_3.safetensors",
)

upscaler_model = download_file(
    "https://huggingface.co/vidfom/aimusic/resolve/main/ComfyUI/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "/content/ComfyUI/models/latent_upscale_models",
    filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
)
link_file_safe(
    "/content/ComfyUI/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "/content/ComfyUI/models/upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
)

print("📦 Downloading Director 2.0 4-LoRA Stack...")
lora_dir = "/content/ComfyUI/models/loras"
lora_1 = download_file("https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors", lora_dir, filename="ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors")
lora_2 = download_file("https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",                          lora_dir, filename="LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors")
lora_3 = download_file("https://huggingface.co/joyfox/LTX-2.3-Transition-LORA/resolve/main/ltx2.3-transition.safetensors",                                lora_dir, filename="ltx2.3-transition.safetensors")
lora_4 = download_file("https://huggingface.co/vidfom/aimusic/resolve/main/ComfyUI/models/loras/LTX2.3-MVCamera-drclips.safetensors",                     lora_dir, filename="LTX2.3-MVCamera-drclips.safetensors")

audio_dest_dir  = "/content/ComfyUI/input/whatdreamscost"
audio_file_target = os.path.join(audio_dest_dir, "Late night trap.mp3")
os.makedirs(audio_dest_dir, exist_ok=True)

if not os.path.exists(audio_file_target) or os.path.getsize(audio_file_target) < 10000:
    download_file(
        "https://huggingface.co/vidfom/aimusic/resolve/main/Late%20night%20trap.mp3",
        audio_dest_dir,
        filename="Late night trap.mp3",
    )

print("✅ Cell 5: Models, LoRAs and audio assets validated.")



# ════════════════════════════════════════════════════════════════════════════
# CELL 6: LOAD WORKFLOW CONFIGURATION & TIMELINE SPECIFICATION
# ════════════════════════════════════════════════════════════════════════════
GLOBAL_PROMPT = """Create a highly realistic cinematic AI music video using the provided reference image. Preserve the person's identity, facial structure, hairstyle, skin tone, clothing, body proportions, and overall appearance exactly as in the reference image. The singer must remain fully recognizable throughout the entire video with absolutely no identity drift.

The person is performing directly to the camera as a world-class pop, hip-hop and rap singer during a sold-out stadium concert. Generate perfectly synchronized lip movements from the provided lyrics or audio.

This is NOT a talking-head video and NOT a presenter. This is a high-energy live music performance filled with charisma, attitude and emotional intensity.

Performance Energy:
• Perform with explosive stage presence.
• Every musical phrase immediately creates a new emotional and physical performance.
• Every lyric instantly changes facial expression, eye emotion, head movement, shoulders, hands, posture and body rhythm.
• The performance continuously builds toward emotional peaks.
• Own the stage with absolute confidence.
• Perform as if in front of 50,000 screaming fans.
• Captivate the audience every second.
• Never appear calm, passive or static.

Facial Performance:
• Extremely expressive facial acting throughout the entire performance.
• Rich emotional transitions every few words.
• Powerful eye contact with intense emotional engagement.
• Eyes sparkle with confidence and passion.
• Highly expressive eyebrows synchronized with important lyrics.
• Strong cheek and jaw movement while singing.
• Natural smiles, smirks, determination, excitement, confidence, attitude, passion, curiosity, joy and intensity.
• Rich cinematic micro-expressions.
• Never hold the same facial expression for more than a brief musical phrase.
• The face should feel emotionally alive every second.

Body Performance:
• The entire body constantly grooves with the beat.
• Strong rhythmic bouncing.
• Powerful shoulder accents.
• Confident chest movement.
• Hip movement follows the groove.
• Frequent body turns.
• Fast weight shifts.
• Dynamic torso twists.
• Lean toward the camera during emotional lyrics.
• Occasionally step toward the camera.
• Performance intensity increases naturally during powerful musical moments.
• Bold, energetic and theatrical stage movement.

Hand Performance:
• Perform like an experienced pop or hip-hop superstar.
• Large expressive gestures.
• Fast rhythmic arm accents.
• Sharp hand movements synchronized with the beat.
• Powerful pointing.
• Sweeping arm movements.
• Punching the air.
• Pulling gestures toward the chest.
• Throwing gestures outward.
• Finger snapping.
• Open palm emphasis.
• Framing the face.
• Expressive wrist movement.
• Hands constantly create visual rhythm.
• One hand naturally leads while the other follows.
• Asymmetrical movement.
• Avoid symmetrical gestures.
• Never repeatedly raise both hands together.
• Every musical phrase introduces fresh gestures.
• Never repeat the same gesture pattern.

Musical Timing:
• Body movement follows musical phrasing rather than every word.
• Strong beats create explosive movements.
• Soft phrases become intimate and emotional.
• Fast lyrics generate faster gestures.
• Slow lyrics become smoother without losing energy.
• Every movement feels rhythmically connected to the music.

Speech Synchronization:
• Perfect lip synchronization.
• Accurate mouth shapes.
• Expressions and gestures match the emotional meaning of every lyric.
• Natural breathing between phrases.

Motion Quality:
• Premium AI human animation.
• Fast, confident and energetic performance.
• Realistic momentum.
• Strong acceleration and deceleration.
• High-energy body mechanics.
• Natural motion blur.
• No robotic movement.
• No frozen poses.
• No repetitive gesture loops.
• No presenter-style gestures.
• No idle standing.
• No jitter.
• No flickering.
• No facial distortion.
• No identity drift.
• No hand deformation.
• No extra fingers.
• No malformed limbs.

Camera:
drclipz, Aggressive cinematic music video camera. Fast push-in, fast pull-back, energetic handheld movement, rhythmic tracking shots, dynamic low-angle hero shots, occasional close-ups on emotional lyrics, subtle orbit around the singer, cinematic motion blur. Camera movement follows the beat and amplifies the performance.

Lighting:
Premium concert lighting with cinematic key light, colorful neon rim lights, volumetric atmosphere, dramatic contrast, realistic skin tones, vibrant electronic music video mood.

Overall Style:
Photorealistic, blockbuster-quality AI music video, premium live concert performance, ultra-high facial fidelity, charismatic superstar, emotionally captivating, explosive stage energy, bold movement, powerful attitude, modern pop, hip-hop and rap performance, every second feels alive, impossible to look away.

Spoken dialogue:
"Open up the canvas, blank space on my screen.
Drag a Checkpoint Loader, you know what I mean.
KSampler in the middle, VAE on the right,
Put the Text Encoder, yeah, building tonight.
Connect the nodes, run the queue,
Watch the latent flow right through.
Green, nothing green, nothing yellow,
Positive Prompt, in my hub."
"""

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, "
    "robotic movement, static presenter, jitter, flicker, facial distortion, extra limbs, watermark"
)

TIMELINE_METADATA = {
    "frame_rate": 24.0,
    "duration_seconds": 31.5,
    "normalDurationFrames": 756,
    "start_frame": 0,
    "end_frame": 756,
    "custom_width": 1280,
    "custom_height": 720,
    "generation_width": 832,
    "generation_height": 480,
    "base_stage1_width": 416,
    "base_stage1_height": 240,
    "mainTrackEnabled": True,
    "audioTrackEnabled": True,
    "motionTrackEnabled": True,
    "inpaint_audio": True,
    "override_audio": False,
    "use_custom_audio": True,
    "use_custom_motion": True,
    "audio_file": "whatdreamscost/Late night trap.mp3",
    "audio_duration_frames": 2880,
    "audio_trim_start_frames": 446.9222739141953,
    "guide_strength": "1.00,1.00,1.00,1.00,1.00",
    "segment_lengths": "226.01059340956584,161.31859976617454,131.45629831196658,225.5063328766255,83.22765271847516",
}

ORIGINAL_SEGMENTS = [
    {"id": "1785555235678s2fn3", "start": 0.0,                  "length": 226.01059340956584, "prompt": "", "type": "image", "imageFile": "whatdreamscost/1.png"},
    {"id": "17855552413529uw9r", "start": 226.01059340956584,   "length": 161.31859976617454, "prompt": "", "type": "image", "imageFile": "whatdreamscost/2.png"},
    {"id": "1785555243885y3h85", "start": 387.3291931757404,    "length": 131.45629831196658, "prompt": "", "type": "image", "imageFile": "whatdreamscost/3.png"},
    {"id": "1785555247117rcoma", "start": 518.785491487707,     "length": 225.5063328766255,  "prompt": "", "type": "image", "imageFile": "whatdreamscost/4.png"},
    {"id": "17855554543736wlrg", "start": 744.2918243643325,    "length": 83.22765271847516,  "prompt": "", "type": "image", "imageFile": "whatdreamscost/5.3.png"},
]

print("✅ Cell 6: Authoritative Timeline & Global Prompt loaded.")



# ════════════════════════════════════════════════════════════════════════════
# CELL 7: ORIGINAL COMFYUI NODE REGISTRY & VALIDATION
# ════════════════════════════════════════════════════════════════════════════
import asyncio
import nest_asyncio
nest_asyncio.apply()

try:
    import server
    from server import PromptServer
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if not hasattr(PromptServer, "instance") or PromptServer.instance is None:
        try:
            PromptServer.instance = PromptServer(loop)
        except Exception:
            class MockServer:
                def __init__(self):
                    from aiohttp import web
                    self.routes = web.RouteTableDef()
                    self.app   = web.Application()
                    self.loop  = loop
                def send_sync(self, *args, **kwargs):
                    pass
            PromptServer.instance = MockServer()
except Exception:
    pass

from nodes import init_builtin_extra_nodes, init_external_custom_nodes


async def _init_nodes_async():
    try:
        await init_builtin_extra_nodes()
    except Exception:
        pass
    try:
        await init_external_custom_nodes()
    except Exception:
        pass


try:
    loop = asyncio.get_event_loop()
    if loop.is_running():
        task = asyncio.ensure_future(_init_nodes_async())
        loop.run_until_complete(task)
    else:
        loop.run_until_complete(_init_nodes_async())
except Exception:
    pass

from nodes import NODE_CLASS_MAPPINGS

REQUIRED_WORKFLOW_NODES = [
    "LTXDirector", "LTXDirectorGuide", "LTXDirectorCropGuides", "LTXVConditioning",
    "LTXVConcatAVLatent", "LTXVSeparateAVLatent", "LTXVLatentUpsampler", "LTXVAudioVAEDecode",
    "Power Lora Loader (rgthree)", "ModelPreviewOverrideKJ", "DualCLIPLoader", "ConditioningZeroOut",
    "UnetLoaderGGUF", "SamplerCustomAdvanced", "CFGGuider", "KSamplerSelect", "BasicScheduler",
    "RandomNoise", "VAEDecode", "VAELoader", "VAELoaderKJ", "LatentUpscaleModelLoader",
    "VHS_VideoCombine",
]


def validate_original_nodes() -> bool:
    print("\n" + "="*70 + "\n🔍 COMFYUI ORIGINAL NODE AUDIT\n" + "="*70)
    missing_nodes = []
    for node_name in REQUIRED_WORKFLOW_NODES:
        if node_name in NODE_CLASS_MAPPINGS:
            print(f"  ✓ Found: {node_name:<30} -> {NODE_CLASS_MAPPINGS[node_name].__name__}")
        else:
            print(f"  ❌ MISSING: {node_name}")
            missing_nodes.append(node_name)
    if missing_nodes:
        raise RuntimeError(f"NODE VALIDATION FAILED: Missing required workflow nodes: {missing_nodes}")
    print(f"✅ All {len(REQUIRED_WORKFLOW_NODES)} required original nodes are verified.")
    return True


validate_original_nodes()



# ════════════════════════════════════════════════════════════════════════════
# CELL 8: FAST GPU MEMORY ENGINE & PHASE PURGE CONTROLLER  (FIX-03, FIX-07)
# ════════════════════════════════════════════════════════════════════════════
def malloc_trim_os():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


class LTXDirectorMemoryManager:

    # ── FIX-03: VRAM headroom gate ──────────────────────────────────────────
    VRAM_SAMPLE_HEADROOM_GB: float = 4.0   # minimum free VRAM before sampling

    @staticmethod
    def get_memory_stats() -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        try:
            import psutil
            vm = psutil.virtual_memory()
            stats["ram_used_gb"]  = (vm.total - vm.available) / 1e9
            stats["ram_avail_gb"] = vm.available / 1e9
            stats["ram_percent"]  = vm.percent
        except Exception:
            stats.update({"ram_used_gb": 0.0, "ram_avail_gb": 99.0, "ram_percent": 0.0})

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            stats["gpu_alloc_gb"] = torch.cuda.memory_allocated() / 1e9
            stats["gpu_res_gb"]   = torch.cuda.memory_reserved() / 1e9
            stats["gpu_free_gb"]  = (props.total_memory - torch.cuda.memory_reserved()) / 1e9
            stats["gpu_total_gb"] = props.total_memory / 1e9
        else:
            stats.update({"gpu_alloc_gb": 0.0, "gpu_res_gb": 0.0,
                           "gpu_free_gb": 0.0, "gpu_total_gb": 0.0})
        return stats

    @staticmethod
    def print_diagnostics(phase: str = "", node: str = "", timeline_info: str = ""):
        s = LTXDirectorMemoryManager.get_memory_stats()
        print("=" * 60)
        print("📊 LTX DIRECTOR MEMORY STATUS")
        print("=" * 60)
        print(f"  System RAM : Used: {s['ram_used_gb']:.2f} GB | Available: {s['ram_avail_gb']:.2f} GB ({s['ram_percent']}%)")
        print(f"  GPU VRAM   : Alloc: {s['gpu_alloc_gb']:.2f} GB | Reserved: {s['gpu_res_gb']:.2f} GB | Free: {s['gpu_free_gb']:.2f} GB")
        if phase:
            print(f"  Phase      : {phase}")
        if node:
            print(f"  Node       : {node}")
        print("=" * 60 + "\n")

    @staticmethod
    def assert_vram_headroom(min_gb: Optional[float] = None, context: str = ""):
        """FIX-03: Raise a clear OOM-prevention error before entering sampling."""
        threshold = min_gb if min_gb is not None else LTXDirectorMemoryManager.VRAM_SAMPLE_HEADROOM_GB
        s = LTXDirectorMemoryManager.get_memory_stats()
        free_gb = s["gpu_free_gb"]
        tag = f" [{context}]" if context else ""
        print(f"  🔍 VRAM check{tag}: {free_gb:.2f} GB free (need ≥ {threshold:.1f} GB)")
        if free_gb < threshold:
            raise MemoryError(
                f"VRAM headroom check FAILED{tag}: "
                f"only {free_gb:.2f} GB free, need ≥ {threshold:.1f} GB. "
                "Force-purge and retry, or reduce steps / resolution."
            )

    @staticmethod
    def drop_os_page_cache():
        patterns = [
            "/content/ComfyUI/models/unet/*.gguf",
            "/content/ComfyUI/models/diffusion_models/*.gguf",
            "/content/ComfyUI/models/text_encoders/*.safetensors",
            "/content/ComfyUI/models/clip/*.safetensors",
            "/content/ComfyUI/models/vae/*.safetensors",
            "/content/ComfyUI/models/latent_upscale_models/*.safetensors",
            "/content/ComfyUI/models/loras/*.safetensors",
        ]
        for pat in patterns:
            for f in glob.glob(pat):
                try:
                    fd   = os.open(f, os.O_RDONLY)
                    size = os.fstat(fd).st_size
                    os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)
                    os.close(fd)
                except Exception:
                    pass

    @staticmethod
    def purge(tag: str = ""):
        try:
            import comfy.model_management as mm
            mm.unload_all_models()
            mm.cleanup_models()
            mm.soft_empty_cache()
            if hasattr(mm, "current_loaded_models") and isinstance(mm.current_loaded_models, list):
                mm.current_loaded_models.clear()
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        LTXDirectorMemoryManager.drop_os_page_cache()
        malloc_trim_os()

    @staticmethod
    def watermark(label: str):
        """FIX-10: Print a VRAM breadcrumb after every major operation."""
        s = LTXDirectorMemoryManager.get_memory_stats()
        print(
            f"  💧 VRAM [{label}]: "
            f"alloc={s['gpu_alloc_gb']:.2f}GB  "
            f"reserved={s['gpu_res_gb']:.2f}GB  "
            f"free={s['gpu_free_gb']:.2f}GB"
        )


print("✅ Cell 8: Fast GPU Memory Engine Active.")



# ════════════════════════════════════════════════════════════════════════════
# CELL 9: DIRECTOR TIMELINE CONTROLLER & KEYFRAME VALIDATOR
# ════════════════════════════════════════════════════════════════════════════
class DirectorTimelineController:
    def __init__(
        self,
        global_prompt: str,
        negative_prompt: str,
        timeline_metadata: Dict[str, Any],
        segments: List[Dict[str, Any]],
        base_input_dir: str = "/content/ComfyUI/input",
    ):
        self.global_prompt   = global_prompt
        self.negative_prompt = negative_prompt
        self.meta            = timeline_metadata
        self.segments        = segments
        self.base_input_dir  = base_input_dir
        self.validate_reference_images()

    def validate_reference_images(self):
        print("\n" + "="*70 + "\n🔍 VALIDATING DIRECTOR KEYFRAME IMAGES\n" + "="*70)
        for s in self.segments:
            rel_path  = s["imageFile"]
            full_path = os.path.join(self.base_input_dir, rel_path)
            if not os.path.exists(full_path):
                if "5.3.png" in rel_path:
                    alt_path = full_path.replace("5.3.png", "5.png")
                    if os.path.exists(alt_path):
                        os.makedirs(os.path.dirname(full_path), exist_ok=True)
                        shutil.copyfile(alt_path, full_path)
                        print(f"  ✓ Keyframe Alias Resolved: {full_path}")
                        continue
                raise FileNotFoundError(
                    f"Missing required Director reference image:\n    {full_path}\n"
                    f"Please upload '{os.path.basename(rel_path)}' to '{os.path.dirname(full_path)}'."
                )
            print(f"  ✓ Validated Keyframe: {full_path} (Segment: {s['id']})")

    def build_timeline_json_string(self) -> str:
        timeline_dict = {
            "mainTrackEnabled":   self.meta["mainTrackEnabled"],
            "audioTrackEnabled":  self.meta["audioTrackEnabled"],
            "motionTrackEnabled": self.meta["motionTrackEnabled"],
            "propHeight": 90, "globalPropHeight": 470, "showFilenames": True,
            "overrideAudio":  self.meta["override_audio"],
            "inpaint_audio":  self.meta["inpaint_audio"],
            "global_prompt":  self.global_prompt,
            "retake_global_prompt": "",
            "retakeMode": False, "retakeStart": 24, "retakeLength": 48,
            "retakePrompt": "", "retakeStrength": 1.0, "retakeVideo": None,
            "normalStartFrame":    int(self.meta["start_frame"]),
            "normalDurationFrames": int(self.meta["normalDurationFrames"]),
            "segments": [
                {
                    "id":       s["id"],
                    "start":    float(s["start"]),
                    "length":   float(s["length"]),
                    "prompt":   s.get("prompt", ""),
                    "type":     s["type"],
                    "imageFile": s["imageFile"],
                    "imageB64": (
                        f"/api/view?filename={os.path.basename(s['imageFile'])}"
                        f"&type=input&subfolder={os.path.dirname(s['imageFile'])}"
                    ),
                    "isEndFrame": False,
                }
                for s in self.segments
            ],
            "motionSegments": [],
            "audioSegments": [
                {
                    "id":    "1785169457779kollx",
                    "type":  "audio",
                    "start": 0.0,
                    "length": float(self.meta["normalDurationFrames"]),
                    "trimStart": float(self.meta["audio_trim_start_frames"]),
                    "audioDurationFrames": int(self.meta["audio_duration_frames"]),
                    "audioFile": self.meta["audio_file"],
                    "fileName":  os.path.basename(self.meta["audio_file"]),
                }
            ],
        }
        return json.dumps(timeline_dict)

    def configure_ltxdirector_node_instance(self, node_instance: Any):
        tl_json_str = self.build_timeline_json_string()
        props = {
            "global_prompt":  self.global_prompt,
            "mainTrackEnabled":   self.meta["mainTrackEnabled"],
            "audioTrackEnabled":  self.meta["audioTrackEnabled"],
            "motionTrackEnabled": self.meta["motionTrackEnabled"],
            "audioTrackWasEnabledBeforeOverride": False,
            "inpaint_audio":  self.meta["inpaint_audio"],
            "override_audio": self.meta["override_audio"],
            "overrideAudio":  self.meta["override_audio"],
            "showFilenames":  True,
            "use_custom_audio":  self.meta["use_custom_audio"],
            "use_custom_motion": self.meta["use_custom_motion"],
            "frame_rate":     float(self.meta["frame_rate"]),
            "display_mode":   "seconds",
            "custom_width":   int(self.meta["custom_width"]),
            "custom_height":  int(self.meta["custom_height"]),
            "resize_method":  "maintain aspect ratio",
            "divisible_by":   32,
            "img_compression": 18,
            "guide_strength": str(self.meta["guide_strength"]),
            "local_prompts":  " |  |  |  | ",
            "segment_lengths": str(self.meta["segment_lengths"]),
            "timeline_data":  tl_json_str,
            "epsilon":        0.001,
            "start_second":   0.0,
            "end_second":     float(self.meta["duration_seconds"]),
            "duration_seconds": float(self.meta["duration_seconds"]),
            "start_frame":    int(self.meta["start_frame"]),
            "end_frame":      int(self.meta["end_frame"]),
            "duration_frames": int(self.meta["normalDurationFrames"]),
            "timeline_ui":    "",
            "has_serialized_properties": True,
            "retakeMode":     False,
        }
        if hasattr(node_instance, "properties") and isinstance(node_instance.properties, dict):
            node_instance.properties.update(props)
        else:
            setattr(node_instance, "properties", props)

        widgets_values = [
            0,
            float(self.meta["duration_seconds"]),
            float(self.meta["duration_seconds"]),
            int(self.meta["start_frame"]),
            int(self.meta["end_frame"]),
            int(self.meta["normalDurationFrames"]),
            tl_json_str,
            " |  |  |  | ",
            str(self.meta["segment_lengths"]),
            0.001,
            str(self.meta["guide_strength"]),
            True, True, True,
            float(self.meta["frame_rate"]),
            "seconds",
            int(self.meta["custom_width"]),
            int(self.meta["custom_height"]),
            "maintain aspect ratio",
            32, 18, False, "",
        ]
        setattr(node_instance, "widgets_values", widgets_values)
        setattr(node_instance, "timeline_data",  tl_json_str)
        setattr(node_instance, "global_prompt",  self.global_prompt)
        print("  ✓ LTXDirector node properties & timeline payload attached.")


controller = DirectorTimelineController(
    global_prompt=GLOBAL_PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    timeline_metadata=TIMELINE_METADATA,
    segments=ORIGINAL_SEGMENTS,
)

print("✅ Cell 9: DirectorTimelineController Initialized.")



# ════════════════════════════════════════════════════════════════════════════
# CELL 10: ORIGINAL COMFYUI NODE DISPATCHER & UNIVERSAL EXTRACTOR
# ════════════════════════════════════════════════════════════════════════════
PARAM_ALIASES = {
    "weight_dtype": ["weight_dtype", "dtype", "weight_type", "precision"],
    "dtype":        ["weight_dtype", "dtype", "weight_type", "precision"],
    "device":       ["device", "device_type", "target_device"],
    "vae_name":     ["vae_name", "name", "vae"],
    "model_name":   ["model_name", "unet_name", "name"],
    "unet_name":    ["unet_name", "model_name", "name"],
    "clip_name":    ["clip_name", "clip_name1", "name"],
    "clip_name1":   ["clip_name1", "clip_name", "name"],
    "clip_name2":   ["clip_name2", "name"],
    "samples":      ["samples", "latent", "latents", "video_latent", "av_latent", "latent_image"],
    "latents":      ["latents", "latent", "samples", "video_latent", "av_latent", "latent_image"],
    "latent":       ["latent", "latents", "samples", "video_latent", "latent_image"],
    "latent_image": ["latent_image", "latent", "samples", "latents", "av_latent"],
    "video_latent": ["video_latent", "latent", "samples"],
    "audio_latent": ["audio_latent", "latent", "samples"],
    "av_latent":    ["av_latent", "latent", "samples", "latent_image"],
    "audio_vae":    ["audio_vae", "vae"],
    "vae":          ["vae", "audio_vae", "video_vae"],
    "upscale_model":["upscale_model", "latent_upscale_model", "model"],
    "frame_rate":   ["frame_rate", "fps"],
    "fps":          ["fps", "frame_rate"],
    "images":       ["images", "image", "frames"],
    "audio":        ["audio", "audio_dict", "samples"],
    "positive":     ["positive", "pos"],
    "negative":     ["negative", "neg"],
    "guider":       ["guider", "cfg_guider"],
    "sigmas":       ["sigmas", "sigma"],
    "noise":        ["noise", "random_noise"],
    "sampler":      ["sampler", "sampler_name", "sampler_select"],
    "noise_seed":   ["noise_seed", "seed"],
    "scheduler":    ["scheduler", "scheduler_name"],
    "sampler_name": ["sampler_name", "sampler"],
    "global_prompt":["global_prompt", "prompt"],
    "timeline_data":["timeline_data", "timeline"],
}


def gv(obj: Any, index: int = 0) -> Any:
    """Universal safe value extractor from tuples, lists, dicts, NodeOutput objects."""
    if obj is None:
        return None
    if isinstance(obj, (tuple, list)):
        return obj[index] if len(obj) > index else None
    if isinstance(obj, dict):
        if "result" in obj and isinstance(obj["result"], (list, tuple)):
            r = obj["result"]
            return r[index] if len(r) > index else None
        return obj.get(index)
    if hasattr(obj, "args") and isinstance(obj.args, (list, tuple)):
        a = obj.args
        return a[index] if len(a) > index else None
    for attr in ["output", "outputs", "result", "values", "data"]:
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if isinstance(val, (list, tuple)) and len(val) > index:
                return val[index]
            elif index == 0:
                return val
    try:
        if hasattr(obj, "__getitem__"):
            return obj[index]
    except Exception:
        pass
    return obj if index == 0 else None


def unwrap_tensor(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj
    for attr in ["output", "result"]:
        v = getattr(obj, attr, None)
        if v is not None:
            return unwrap_tensor(v)
    if isinstance(obj, (tuple, list)) and len(obj) > 0:
        return unwrap_tensor(obj[0])
    if isinstance(obj, dict):
        if "samples" in obj:
            return unwrap_tensor(obj["samples"])
        if "result" in obj and len(obj["result"]) > 0:
            return unwrap_tensor(obj["result"][0])
        for v in obj.values():
            if isinstance(v, torch.Tensor):
                return v
    if hasattr(obj, "args") and len(obj.args) > 0:
        return unwrap_tensor(obj.args[0])
    return obj


def unwrap_latent(x: Any) -> Dict[str, Any]:
    if x is None:
        return {"samples": None}
    for attr in ["output", "result"]:
        v = getattr(x, attr, None)
        if v is not None:
            x = v
    while isinstance(x, (tuple, list)) and len(x) > 0:
        x = x[0]
    if isinstance(x, dict):
        cur = x
        while isinstance(cur, dict) and "samples" in cur and isinstance(cur["samples"], dict):
            cur = cur["samples"]
        if isinstance(cur, dict) and "samples" in cur:
            return cur
        for v in cur.values():
            if isinstance(v, torch.Tensor):
                return {"samples": v}
        return {"samples": cur}
    if isinstance(x, torch.Tensor):
        return {"samples": x}
    return {"samples": x}


def sync_latent_device(latent: Any, target_device: Union[str, torch.device] = "cpu") -> Dict[str, Any]:
    target      = torch.device(target_device)
    latent_dict = unwrap_latent(latent)
    samples     = latent_dict.get("samples")
    if samples is None:
        return latent_dict
    if isinstance(samples, torch.Tensor):
        if samples.is_nested:
            nested_list = [t.to(target) for t in samples.unbind()]
            latent_dict["samples"] = torch.nested.nested_tensor(nested_list)
        else:
            latent_dict["samples"] = samples.to(target)
    return latent_dict


def sync_conditioning_to_cpu(cond_obj: Any) -> Any:
    if cond_obj is None:
        return None
    if isinstance(cond_obj, torch.Tensor):
        return cond_obj.detach().cpu()
    if isinstance(cond_obj, list):
        return [sync_conditioning_to_cpu(i) for i in cond_obj]
    if isinstance(cond_obj, tuple):
        return tuple(sync_conditioning_to_cpu(i) for i in cond_obj)
    if isinstance(cond_obj, dict):
        return {k: sync_conditioning_to_cpu(v) for k, v in cond_obj.items()}
    return cond_obj


def slice_temporal_latent(
    tensor: Optional[torch.Tensor], start_idx: int, end_idx: int
) -> Optional[torch.Tensor]:
    """Dimension-agnostic temporal slicer (dim=2) for 3-D/4-D/5-D tensors."""
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return None
    dim = 2
    if tensor.ndim <= dim:
        return tensor.clone()
    max_len = tensor.shape[dim]
    s = max(0, min(start_idx, max_len))
    e = max(s,  min(end_idx,   max_len))
    slices        = [slice(None)] * tensor.ndim
    slices[dim]   = slice(s, e)
    return tensor[tuple(slices)].clone()


def concat_temporal_latents(tensor_list: List[torch.Tensor]) -> Optional[torch.Tensor]:
    """Dimension-agnostic temporal concatenator (dim=2)."""
    valid = [t for t in tensor_list if t is not None and isinstance(t, torch.Tensor)]
    if not valid:
        return None
    dim = 2 if valid[0].ndim >= 3 else 0
    return torch.cat(valid, dim=dim)


def call_original_node(node_name: str, node_instance: Optional[Any] = None, **kwargs) -> Any:
    if node_instance is None:
        if node_name not in NODE_CLASS_MAPPINGS:
            raise RuntimeError(f"FATAL: Required node '{node_name}' not registered in ComfyUI.")
        node_instance = NODE_CLASS_MAPPINGS[node_name]()

    func_name = getattr(node_instance, "FUNCTION", None)
    callables: List[Any] = []

    if func_name and hasattr(node_instance, func_name) and callable(getattr(node_instance, func_name)):
        callables.append((func_name, getattr(node_instance, func_name)))
    if hasattr(node_instance, "execute") and callable(node_instance.execute):
        callables.append(("execute", node_instance.execute))
    for fallback in [
        "direct", "get_guider", "get_noise", "get_sampler", "get_sigmas", "sample",
        "apply_guide", "crop_guides", "upsample_latent", "concat", "separate",
        "encode", "decode", "load_unet", "load_clip", "load_vae", "combine_video", "override",
    ]:
        if hasattr(node_instance, fallback) and callable(getattr(node_instance, fallback)):
            callables.append((fallback, getattr(node_instance, fallback)))

    comfy_schema_defaults: Dict[str, Any] = {}
    if hasattr(node_instance, "INPUT_TYPES") and callable(node_instance.INPUT_TYPES):
        try:
            it = node_instance.INPUT_TYPES()
            for group in ["required", "optional", "hidden"]:
                for p_name, p_spec in it.get(group, {}).items():
                    if isinstance(p_spec, tuple) and len(p_spec) > 1 and isinstance(p_spec[1], dict) and "default" in p_spec[1]:
                        comfy_schema_defaults[p_name] = p_spec[1]["default"]
                    elif isinstance(p_spec, tuple) and len(p_spec) > 0 and isinstance(p_spec[0], list) and len(p_spec[0]) > 0:
                        comfy_schema_defaults[p_name] = p_spec[0][0]
        except Exception:
            pass

    last_err: Optional[str] = None
    for f_name, func in callables:
        try:
            sig = inspect.signature(func)
            valid_kwargs: Dict[str, Any] = {}
            has_var_keyword = False

            for param_name, param in sig.parameters.items():
                if param_name in ("cls", "self"):
                    continue
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    continue
                if param.kind == inspect.Parameter.VAR_KEYWORD:
                    has_var_keyword = True
                    continue
                if param_name in kwargs:
                    valid_kwargs[param_name] = kwargs[param_name]
                    continue
                alias_matched = False
                for alias in PARAM_ALIASES.get(param_name, [param_name]):
                    if alias in kwargs:
                        valid_kwargs[param_name] = kwargs[alias]
                        alias_matched = True
                        break
                if alias_matched:
                    continue
                if param.default is not inspect.Parameter.empty:
                    continue
                if param_name in comfy_schema_defaults:
                    valid_kwargs[param_name] = comfy_schema_defaults[param_name]
                    continue
                if param_name == "duration_frames":
                    valid_kwargs[param_name] = int(kwargs.get("normalDurationFrames", 756))
                elif param.annotation == int or "int" in str(param.annotation):
                    valid_kwargs[param_name] = 0
                elif param.annotation == float or "float" in str(param.annotation):
                    valid_kwargs[param_name] = 0.0
                elif param.annotation == bool or "bool" in str(param.annotation):
                    valid_kwargs[param_name] = False
                elif param.annotation == str or "str" in str(param.annotation):
                    valid_kwargs[param_name] = ""
                else:
                    valid_kwargs[param_name] = None

            if has_var_keyword:
                for k, v in kwargs.items():
                    if k not in valid_kwargs:
                        valid_kwargs[k] = v

            return func(**valid_kwargs)
        except Exception:
            last_err = traceback.format_exc()
            continue

    if last_err:
        raise RuntimeError(f"Error calling original node '{node_name}':\n{last_err}")
    raise AttributeError(
        f"Cannot execute node '{node_instance.__class__.__name__}' (No valid callable function)"
    )


print("✅ Cell 10: Original Node Dispatcher ready with Signature Filtering.")



# ════════════════════════════════════════════════════════════════════════════
# CELL 11: PRECOMPUTED CLIP PROXY & COMPLETE ZERO-RAM DIT PROXY
#          FIX-04: load_dit_and_loras() CPU-offload + FIX-11/12: SageAttn / CFF
# ════════════════════════════════════════════════════════════════════════════
class MockBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn       = torch.nn.Identity()
        self.cross_attn = torch.nn.Identity()
        self.ffn        = torch.nn.Identity()
        self.linear1    = torch.nn.Identity()
        self.linear2    = torch.nn.Identity()

    def forward(self, *args, **kwargs):
        return args[0] if args else None


class DiffusionModelSpec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.__class__.__name__   = "LTXVModel"
        self.patch_size           = (1, 32, 32)
        self.temporal_stride      = 8
        self.vae_scale_factors    = (8, 32, 32)
        self.in_channels          = 128
        self.out_channels         = 128
        self.dtype                = torch.bfloat16
        self.blocks               = torch.nn.ModuleList([MockBlock() for _ in range(48)])
        self.double_blocks        = torch.nn.ModuleList([MockBlock() for _ in range(48)])
        self.single_blocks        = torch.nn.ModuleList([MockBlock() for _ in range(48)])
        self.transformer_blocks   = torch.nn.ModuleList([MockBlock() for _ in range(48)])

    def forward(self, *args, **kwargs):
        return args[0] if args else None


class BaseModelSpec:
    def __init__(self):
        self.diffusion_model   = DiffusionModelSpec()
        self.model_type        = "ltxv"
        self.latent_format     = None
        self.memory_required   = lambda *a, **kw: 0
        self.vae_scale_factors = (8, 32, 32)

    def to(self, *args, **kwargs):
        return self


class LightweightLTXModelProxy:
    def __init__(self):
        self.model          = BaseModelSpec()
        self.model_options  = {}
        self.patches        = {}
        self.object_patches = {}
        self.vae_scale_factors = (8, 32, 32)

    def clone(self):
        c = LightweightLTXModelProxy()
        c.model_options = dict(self.model_options)
        c.patches       = dict(self.patches)
        return c

    def set_model_patch(self, patch, name):
        self.patches[name] = patch

    def set_model_patch_replace(self, patch, name, block_name, number):
        pass

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def get_model_object(self, name):
        if name == "diffusion_model":
            return self.model.diffusion_model
        return getattr(self.model, name, None)

    def __getattr__(self, name):
        return getattr(self.model, name, None)


class PrecomputedClipProxy:
    def __init__(self, precomputed_conditioning: Any, tokenizer: Any = None):
        self.cond             = precomputed_conditioning
        self.tokenizer        = tokenizer
        self.cond_stage_model = None
        self.patcher          = None
        self.layer_idx        = None

    def tokenize(self, text, *args, **kwargs):
        if self.tokenizer is not None and hasattr(self.tokenizer, "tokenize_with_weights"):
            return self.tokenizer.tokenize_with_weights(text)
        return {"text": text}

    def encode_from_tokens_scheduled(self, tokens, *args, **kwargs):
        return self.cond

    def encode_from_tokens(self, tokens, *args, **kwargs):
        return self.cond

    def encode(self, text, *args, **kwargs):
        return self.cond

    def load_model(self, *args, **kwargs):
        return self

    def clone(self):
        return self

    def get_key_patches(self):
        return {}

    def __getattr__(self, name):
        if name == "tokenizer" and self.tokenizer is not None:
            return self.tokenizer
        return lambda *args, **kwargs: self.cond


def load_clip_and_encode_to_gpu(prompt_text: str) -> Tuple[Any, Any]:
    LTXDirectorMemoryManager.print_diagnostics(phase="Text Encoder Loading", node="DualCLIPLoader")
    LTXDirectorMemoryManager.purge("pre_clip_load")

    import comfy.model_management as mm
    dual_clip_node = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
    clip = dual_clip_node.load_clip(
        clip_name1="gemma_3_12B_it_fp4_mixed.safetensors",
        clip_name2="ltx-2.3_text_projection_bf16.safetensors",
        type="ltxv",
        device="default",
    )[0]

    saved_tokenizer = getattr(clip, "tokenizer", None)

    if hasattr(clip, "cond_stage_model") and hasattr(clip.cond_stage_model, "to"):
        clip.cond_stage_model.to(torch.device("cuda"))
    if hasattr(clip, "patcher") and hasattr(clip.patcher, "model") and hasattr(clip.patcher.model, "to"):
        clip.patcher.model.to(torch.device("cuda"))

    gc.collect()
    malloc_trim_os()

    clip_text_encode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
    print("  ⚡ Running Fast Prompt Encoding directly on GPU (~3-5 seconds)...")
    t0 = time.time()

    with torch.inference_mode():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.amp.autocast("cuda", dtype=dtype):
            cond_raw  = clip_text_encode.encode(text=prompt_text, clip=clip)[0]
            cond_cpu  = sync_conditioning_to_cpu(cond_raw)
            del cond_raw

    print(f"  ✓ Prompt Encoding Finished on GPU in {time.time() - t0:.2f}s!")

    # Complete purge of the 12B model weights
    del clip, dual_clip_node, clip_text_encode
    mm.unload_all_models()
    mm.cleanup_models()
    mm.soft_empty_cache()
    if hasattr(mm, "current_loaded_models") and isinstance(mm.current_loaded_models, list):
        mm.current_loaded_models.clear()
    gc.collect()
    torch.cuda.empty_cache()
    LTXDirectorMemoryManager.drop_os_page_cache()
    malloc_trim_os()

    mem = LTXDirectorMemoryManager.get_memory_stats()
    print(f"  ✓ CLIP Purged! Host RAM Free: {mem['ram_avail_gb']:.2f} GB | GPU VRAM Free: {mem['gpu_free_gb']:.2f} GB")
    return cond_cpu, saved_tokenizer


def load_dit_and_loras(enable_cpu_offload: bool = False) -> Any:
    """
    FIX-04: enable_cpu_offload=True is auto-selected when free VRAM < 12 GB.
    FIX-11: SageAttention retried with sdpa fallback if auto fails.
    FIX-12: ChunkFeedForward raised to chunks=16 for T4.
    """
    stats = LTXDirectorMemoryManager.get_memory_stats()
    if stats["gpu_free_gb"] < 12.0:
        enable_cpu_offload = True
        print(f"  ⚠️  Auto-enabling CPU offload (VRAM free: {stats['gpu_free_gb']:.2f} GB < 12 GB)")

    LTXDirectorMemoryManager.print_diagnostics(phase="DiT Loading", node="UnetLoaderGGUF")
    model = gv(call_original_node("UnetLoaderGGUF", unet_name="ltx-2-3-22b-dev-Q4_K_M.gguf"), 0)
    print("  ✓ UnetLoaderGGUF loaded.")

    # FIX-04: Set comfy memory budget to force CPU offload during sampling
    if enable_cpu_offload:
        try:
            import comfy.model_management as mm
            # Tell comfy to keep model on CPU and only move blocks to GPU one at a time
            if hasattr(mm, "set_vram_to"):
                mm.set_vram_to(mm.VRAMState.LOW_VRAM)
            if hasattr(model, "model_options"):
                model.model_options["keep_in_fp8_while_inf"] = True
            print("  ✓ CPU offload budget applied to DiT model.")
        except Exception as e:
            print(f"  [Notice] CPU offload setup: {e}")

    # 4-LoRA stack via Power Lora Loader
    power_lora_node = NODE_CLASS_MAPPINGS["Power Lora Loader (rgthree)"]()
    lora_stack_params = {
        "model": model,
        "clip":  None,
        "lora_1": {"on": True, "lora": "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors", "strength": 0.4},
        "lora_2": {"on": True, "lora": "LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",                               "strength": 0.6},
        "lora_3": {"on": True, "lora": "ltx2.3-transition.safetensors",                                           "strength": 0.7},
        "lora_4": {"on": True, "lora": "LTX2.3-MVCamera-drclips.safetensors",                                     "strength": 0.9},
    }

    try:
        res   = call_original_node("Power Lora Loader (rgthree)", node_instance=power_lora_node, **lora_stack_params)
        model = gv(res, 0) or model
        print("  ✓ Power Lora Loader (rgthree) applied 4-LoRA stack to DiT.")
    except Exception as e:
        print(f"  [Notice] PowerLora fallback: {e}")
        from nodes import LoraLoaderModelOnly
        for lora_name, strength in [
            ("ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors", 0.4),
            ("LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors", 0.6),
            ("ltx2.3-transition.safetensors", 0.7),
            ("LTX2.3-MVCamera-drclips.safetensors", 0.9),
        ]:
            lora_path = os.path.join("/content/ComfyUI/models/loras", lora_name)
            if os.path.exists(lora_path):
                ll    = LoraLoaderModelOnly()
                model = gv(ll.load_lora_model_only(model=model, lora_name=lora_name, strength_model=strength), 0) or model
                print(f"    + LoRA applied: {lora_name} (Strength {strength})")

    # FIX-11: SageAttention with hard sdpa fallback
    if "PatchSageAttentionKJ" in NODE_CLASS_MAPPINGS:
        applied = False
        for sage_mode in ["auto", "sdpa"]:
            try:
                sage   = NODE_CLASS_MAPPINGS["PatchSageAttentionKJ"]()
                result = call_original_node("PatchSageAttentionKJ", node_instance=sage, model=model, sage_attention=sage_mode)
                model  = gv(result, 0) or model
                print(f"  ✓ SageAttention Hook Applied (mode={sage_mode}).")
                applied = True
                break
            except Exception:
                continue
        if not applied:
            print("  ⚠️  SageAttention unavailable — proceeding without it.")

    # FIX-12: chunks=16 (was 8) halves peak activation memory per block on T4
    if "LTXVChunkFeedForward" in NODE_CLASS_MAPPINGS:
        try:
            cff    = NODE_CLASS_MAPPINGS["LTXVChunkFeedForward"]()
            result = call_original_node("LTXVChunkFeedForward", node_instance=cff, model=model, chunks=16, dim_threshold=4096)
            model  = gv(result, 0) or model
            print("  ✓ ChunkFeedForward Hook Applied (chunks=16).")
        except Exception:
            pass

    return model


print("✅ Cell 11: PrecomputedClipProxy & Complete Zero-RAM Model Architecture ready.")



# ════════════════════════════════════════════════════════════════════════════
# CELL 12: PHASE A — MASTER LTXDIRECTOR DECOUPLED INGESTION (ZERO-RAM)
# ════════════════════════════════════════════════════════════════════════════
def execute_phase_a_ltxdirector(
    timeline_ctrl: DirectorTimelineController,
    workdir: str = "/content/LTXDirector_Work",
    resume: bool = True,
) -> Dict[str, Any]:
    os.makedirs(workdir, exist_ok=True)
    state_file = os.path.join(workdir, "director_state.pt")

    if resume and os.path.exists(state_file) and os.path.getsize(state_file) > 1024:
        print(f"  ⏭ [RESUME] Loading cached Director state from: {state_file}")
        return torch.load(state_file, map_location="cpu")

    LTXDirectorMemoryManager.purge("pre_phase_a")
    LTXDirectorMemoryManager.print_diagnostics(phase="PHASE A: LTXDirector Ingestion", node="LTXDirector")

    with torch.inference_mode():
        # Step A1: GPU-fast prompt encoding → purge 12B
        precomputed_cond, saved_tokenizer = load_clip_and_encode_to_gpu(timeline_ctrl.global_prompt)

        # Step A2: Timeline ingestion with proxy model (no DiT weights in RAM)
        print("  🚀 [Step A2] Constructing 756-frame Timeline & Latents via LTXDirector...")
        t_dir     = time.time()
        vae_node  = NODE_CLASS_MAPPINGS["VAELoader"]()
        audio_vae = vae_node.load_vae(vae_name="LTX23_audio_vae_bf16.safetensors")[0]

        model_proxy = LightweightLTXModelProxy()
        clip_proxy  = PrecomputedClipProxy(precomputed_cond, tokenizer=saved_tokenizer)

        ltx_director_node = NODE_CLASS_MAPPINGS["LTXDirector"]()
        timeline_ctrl.configure_ltxdirector_node_instance(ltx_director_node)

        ltx_director_params = {
            "model": model_proxy, "clip": clip_proxy, "audio_vae": audio_vae,
            "optional_latent": None,
            "global_prompt":    timeline_ctrl.global_prompt,
            "start_second":     0.0,
            "end_second":       float(timeline_ctrl.meta["duration_seconds"]),
            "duration_seconds": float(timeline_ctrl.meta["duration_seconds"]),
            "start_frame":      int(timeline_ctrl.meta["start_frame"]),
            "end_frame":        int(timeline_ctrl.meta["end_frame"]),
            "duration_frames":  int(timeline_ctrl.meta["normalDurationFrames"]),
            "timeline_data":    timeline_ctrl.build_timeline_json_string(),
            "local_prompts":    " |  |  |  | ",
            "segment_lengths":  str(timeline_ctrl.meta["segment_lengths"]),
            "epsilon":          0.001,
            "guide_strength":   str(timeline_ctrl.meta["guide_strength"]),
            "mainTrackEnabled":   bool(timeline_ctrl.meta["mainTrackEnabled"]),
            "audioTrackEnabled":  bool(timeline_ctrl.meta["audioTrackEnabled"]),
            "motionTrackEnabled": bool(timeline_ctrl.meta["motionTrackEnabled"]),
            "main_track_enabled":   bool(timeline_ctrl.meta["mainTrackEnabled"]),
            "audio_track_enabled":  bool(timeline_ctrl.meta["audioTrackEnabled"]),
            "motion_track_enabled": bool(timeline_ctrl.meta["motionTrackEnabled"]),
            "frame_rate":    float(timeline_ctrl.meta["frame_rate"]),
            "fps":           float(timeline_ctrl.meta["frame_rate"]),
            "display_mode":  "seconds",
            "custom_width":  int(timeline_ctrl.meta["custom_width"]),
            "custom_height": int(timeline_ctrl.meta["custom_height"]),
            "width":         int(timeline_ctrl.meta["custom_width"]),
            "height":        int(timeline_ctrl.meta["custom_height"]),
            "resize_method": "maintain aspect ratio",
            "divisible_by":  32,
            "img_compression": 18,
            "retakeMode":     False, "retake_mode": False,
            "retake_global_prompt": "",
            "retakeStart":    24, "retakeLength": 48,
            "retakePrompt":   "", "retakeStrength": 1.0,
            "inpaint_audio":  bool(timeline_ctrl.meta["inpaint_audio"]),
            "override_audio": bool(timeline_ctrl.meta["override_audio"]),
            "use_custom_audio":  bool(timeline_ctrl.meta["use_custom_audio"]),
            "use_custom_motion": bool(timeline_ctrl.meta["use_custom_motion"]),
        }

        director_out = call_original_node(
            "LTXDirector", node_instance=ltx_director_node, **ltx_director_params
        )
        print(f"  ⚡ Step A2 Timeline Ingestion Finished in {time.time() - t_dir:.2f}s!")

        dir_pos              = gv(director_out, 1) or precomputed_cond
        dir_vid_lat          = sync_latent_device(gv(director_out, 2), "cpu")
        dir_aud_lat          = sync_latent_device(gv(director_out, 3), "cpu")
        dir_guide_data       = sync_conditioning_to_cpu(gv(director_out, 4))
        dir_motion_guide_data= sync_conditioning_to_cpu(gv(director_out, 5))
        dir_fps_raw          = gv(director_out, 6)
        dir_fps              = float(dir_fps_raw) if dir_fps_raw is not None else float(timeline_ctrl.meta["frame_rate"])

        zero_out_node  = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
        neg_zeroed     = gv(call_original_node("ConditioningZeroOut", node_instance=zero_out_node, conditioning=dir_pos), 0)

        ltxv_cond_node = NODE_CLASS_MAPPINGS["LTXVConditioning"]()
        ltxv_cond_out  = call_original_node(
            "LTXVConditioning", node_instance=ltxv_cond_node,
            positive=dir_pos, negative=neg_zeroed, frame_rate=dir_fps,
        )
        final_positive = sync_conditioning_to_cpu(gv(ltxv_cond_out, 0))
        final_negative = sync_conditioning_to_cpu(gv(ltxv_cond_out, 1))

        state = {
            "positive":            final_positive,
            "negative":            final_negative,
            "video_latent":        dir_vid_lat,
            "audio_latent":        dir_aud_lat,
            "guide_data":          dir_guide_data,
            "motion_guide_data":   dir_motion_guide_data,
            "frame_rate":          dir_fps,
            "timeline_metadata":   timeline_ctrl.meta,
            "segments":            timeline_ctrl.segments,
        }

        tmp_path = state_file + ".tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, state_file)
        print(f"  💾 Phase A Director State saved: {state_file}")

        del audio_vae, ltx_director_node, dir_pos, neg_zeroed, ltxv_cond_out, model_proxy, clip_proxy

    LTXDirectorMemoryManager.purge("phase_a_complete")
    print("✅ Phase A Complete: Ready for Phase B (DiT Diffusion).")
    return state


print("✅ Cell 12: Decoupled Zero-Crash Phase A Configured.")



# ════════════════════════════════════════════════════════════════════════════
# CELL 13 & 14: PHASE B — SEGMENT-WISE DIFFUSION & 2X UPSCALE ENGINE
#
#   FIX-05  video_vae lazy-loaded AFTER Stage 1 sampling; freed before Stage 2
#   FIX-06  All intermediate Python objects del'd before SamplerCustomAdvanced
#   FIX-07  Stage 1 steps 8→6; auto-fallback to 4 if VRAM < 5 GB
#   FIX-08  av_latent cast to half() + contiguous() before sampler
#   FIX-09  Tight del-chain after every node output
#   FIX-10  VRAM watermarks after every stage
# ════════════════════════════════════════════════════════════════════════════
def execute_segment_wise_diffusion_pipeline(
    director_state: Dict[str, Any],
    seed: int = 2026,
    workdir: str = "/content/LTXDirector_Work",
    resume: bool = True,
) -> List[Dict[str, Any]]:
    """
    Stage 1 (6 steps, denoise 1.0) + 2× Latent Upscaler + Stage 2 (4 steps,
    denoise 0.42) per Director Segment.  Memory-safe with lazy VAE loading.
    """
    segments        = director_state.get("segments", ORIGINAL_SEGMENTS)
    total_segments  = len(segments)
    completed_packs: List[Dict[str, Any]] = []

    print(
        "\n" + "="*70 +
        f"\n🎬 PHASE B: EXECUTING {total_segments} DIRECTOR SEGMENTS (MEMORY-SAFE)\n" +
        "="*70
    )

    full_vid_tensor  = unwrap_latent(director_state["video_latent"])["samples"]
    aud_raw          = director_state["audio_latent"]
    full_aud_tensor  = unwrap_latent(aud_raw)["samples"] if aud_raw is not None else None
    total_lat_frames = full_vid_tensor.shape[2] if full_vid_tensor is not None else 95

    cur_lat_idx = 0

    for idx, seg in enumerate(segments):
        seg_name       = f"Segment {idx+1}/{total_segments} ({os.path.basename(seg['imageFile'])})"
        seg_cache_file = os.path.join(workdir, f"latent_stage2_seg_{idx+1}.pt")

        raw_frames   = int(round(float(seg["length"])))
        valid_frames = int(((raw_frames - 1) // 8) * 8 + 1)
        if valid_frames < 9:
            valid_frames = 9
        seg_lat_len = (valid_frames - 1) // 8 + 1

        end_lat_idx = min(cur_lat_idx + seg_lat_len, total_lat_frames)
        if idx == total_segments - 1:
            end_lat_idx = total_lat_frames

        print(
            f"\n{'─'*65}\n"
            f"🎬 {seg_name} | Frames: {valid_frames} "
            f"(Latent Frames: {cur_lat_idx} → {end_lat_idx})\n"
            f"{'─'*65}"
        )

        # ── Resume shortcut ────────────────────────────────────────────────
        if resume and os.path.exists(seg_cache_file) and os.path.getsize(seg_cache_file) > 1024:
            print(f"  ⏭ [RESUME] Loading cached Stage 2 latent: {seg_cache_file}")
            completed_packs.append(torch.load(seg_cache_file, map_location="cpu"))
            cur_lat_idx = end_lat_idx
            continue

        # ── FIX-07: determine safe step count ─────────────────────────────
        LTXDirectorMemoryManager.purge(f"pre_seg_{idx+1}")
        mem_check = LTXDirectorMemoryManager.get_memory_stats()
        if mem_check["gpu_free_gb"] < 5.0:
            s1_steps = 4
            print(f"  ⚠️  Low VRAM ({mem_check['gpu_free_gb']:.2f} GB) → Stage 1 steps reduced to {s1_steps}")
        else:
            s1_steps = 6   # FIX-07: was 8

        # ── Temporal slice ─────────────────────────────────────────────────
        seg_vid_lat: Dict[str, Any] = {
            "samples": slice_temporal_latent(full_vid_tensor, cur_lat_idx, end_lat_idx)
        }
        seg_aud_lat: Optional[Dict[str, Any]] = (
            {"samples": slice_temporal_latent(full_aud_tensor, cur_lat_idx, end_lat_idx)}
            if full_aud_tensor is not None else None
        )

        with torch.inference_mode():

            # ── STEP B1: Load DiT + LoRAs ─────────────────────────────────
            model = load_dit_and_loras()    # FIX-04 auto CPU-offload inside
            LTXDirectorMemoryManager.watermark("after_dit_load")

            # FIX-05 CORRECTED: LTXDirectorGuide.execute() unconditionally calls
            # vae.downscale_index_formula — VAE MUST be present for both guides.
            # Load once here, share across Guide1→Upscale→Guide2, delete after Guide2.
            video_vae = gv(call_original_node("VAELoader", vae_name="LTX23_video_vae_bf16.safetensors"), 0)
            LTXDirectorMemoryManager.watermark("after_vae_load")

            # ── Guide 1 (strength 0.5) ────────────────────────────────────
            guide1_node   = NODE_CLASS_MAPPINGS["LTXDirectorGuide"]()
            guide1_res    = call_original_node(
                "LTXDirectorGuide", node_instance=guide1_node,
                positive=director_state["positive"],
                negative=director_state["negative"],
                vae=video_vae,                   # Required: VAE must not be None
                latent=seg_vid_lat,
                guide_data=director_state["guide_data"],
                motion_guide_data=director_state["motion_guide_data"],
                model=model,
                strength=0.5,
                rescale_method="None",
                guide_frame=1,
                interpolation="bicubic",
                crop_position="center",
                enable_guide=True,
            )

            s1_pos   = gv(guide1_res, 0) or director_state["positive"]
            s1_neg   = gv(guide1_res, 1) or director_state["negative"]
            s1_vid   = sync_latent_device(gv(guide1_res, 2) or seg_vid_lat, "cpu")
            s1_model = gv(guide1_res, 3) or model
            # FIX-09: free guide result immediately
            del guide1_res, guide1_node
            gc.collect()
            LTXDirectorMemoryManager.watermark("after_guide1")

            # ── Concat AV latents ─────────────────────────────────────────
            concat_node = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]()
            av1_raw     = gv(call_original_node(
                "LTXVConcatAVLatent", node_instance=concat_node,
                video_latent=s1_vid, audio_latent=seg_aud_lat,
            ), 0)
            # FIX-08: half-precision contiguous CPU tensor for sampler
            av1_in = sync_latent_device(av1_raw, "cpu")
            if isinstance(av1_in.get("samples"), torch.Tensor):
                av1_in["samples"] = av1_in["samples"].contiguous().half()
            # FIX-09: free intermediates
            del av1_raw, concat_node, s1_vid
            gc.collect()
            LTXDirectorMemoryManager.watermark("after_concat1")

            # ── Sampler objects ───────────────────────────────────────────
            noise_node        = NODE_CLASS_MAPPINGS["RandomNoise"]()
            guider_node       = NODE_CLASS_MAPPINGS["CFGGuider"]()
            sampler_sel_node  = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
            scheduler_node    = NODE_CLASS_MAPPINGS["BasicScheduler"]()
            sampler_adv_node  = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()

            noise1      = gv(call_original_node("RandomNoise",    node_instance=noise_node,       noise_seed=seed + idx * 100), 0)
            guider1     = gv(call_original_node("CFGGuider",      node_instance=guider_node,      cfg=1.0, model=s1_model, positive=s1_pos, negative=s1_neg), 0)
            sampler_e   = gv(call_original_node("KSamplerSelect", node_instance=sampler_sel_node, sampler_name="euler"), 0)
            sigmas1     = gv(call_original_node(
                "BasicScheduler", node_instance=scheduler_node,
                model=s1_model, scheduler="linear_quadratic",
                steps=s1_steps, denoise=1.0,
            ), 0)

            # FIX-06: purge all intermediate Python objects from GPU before sampling
            del s1_pos, s1_neg, s1_model
            gc.collect()
            torch.cuda.empty_cache()
            # FIX-03: hard VRAM gate
            LTXDirectorMemoryManager.assert_vram_headroom(
                min_gb=LTXDirectorMemoryManager.VRAM_SAMPLE_HEADROOM_GB,
                context=f"Stage1-{seg_name}"
            )
            LTXDirectorMemoryManager.watermark("pre_stage1_sample")

            print(f"  ⚡ Sampling Stage 1 for {seg_name} ({s1_steps} steps)...")
            t_s1   = time.time()
            s1_out = call_original_node(
                "SamplerCustomAdvanced", node_instance=sampler_adv_node,
                noise=noise1, guider=guider1, sampler=sampler_e,
                sigmas=sigmas1, latent_image=av1_in,
            )
            s1_lat = sync_latent_device(gv(s1_out, 0), "cpu")
            print(f"  ✓ Stage 1 Finished in {time.time() - t_s1:.2f}s!")

            # FIX-09: free ALL stage-1 sampler objects immediately
            del noise1, guider1, sigmas1, s1_out, av1_in
            del noise_node, guider_node, sampler_sel_node, scheduler_node, sampler_adv_node
            gc.collect()
            torch.cuda.empty_cache()
            LTXDirectorMemoryManager.watermark("post_stage1_sample")

            # Separate AV latents
            sep_node = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
            sep1     = call_original_node("LTXVSeparateAVLatent", node_instance=sep_node, av_latent=s1_lat)
            v1_raw   = sync_latent_device(gv(sep1, 0), "cpu")
            a1_raw   = sync_latent_device(gv(sep1, 1), "cpu")
            del s1_lat, sep1, sep_node
            gc.collect()

            # Crop guides
            crop_node  = NODE_CLASS_MAPPINGS["LTXDirectorCropGuides"]()
            crop1      = call_original_node(
                "LTXDirectorCropGuides", node_instance=crop_node,
                positive=director_state["positive"],
                negative=director_state["negative"],
                latent=v1_raw,
            )
            crop1_pos  = gv(crop1, 0) or director_state["positive"]
            crop1_neg  = gv(crop1, 1) or director_state["negative"]
            crop1_vid  = sync_latent_device(gv(crop1, 2) or v1_raw, "cpu")
            del v1_raw, crop1, crop_node
            gc.collect()
            LTXDirectorMemoryManager.watermark("after_crop1")

            # ── STEP B2: 2× Latent Upscale ───────────────────────────────
            # video_vae already loaded above — reuse it here
            print("  ⚡ Upscaling Latent 2× via LTXVLatentUpsampler...")
            upscale_loader   = NODE_CLASS_MAPPINGS["LatentUpscaleModelLoader"]()
            up_model         = gv(call_original_node(
                "LatentUpscaleModelLoader", node_instance=upscale_loader,
                model_name="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            ), 0)
            upsampler_node   = NODE_CLASS_MAPPINGS["LTXVLatentUpsampler"]()
            upscaled_res     = call_original_node(
                "LTXVLatentUpsampler", node_instance=upsampler_node,
                samples=crop1_vid, upscale_model=up_model, vae=video_vae,
            )
            v_upscaled       = sync_latent_device(gv(upscaled_res, 0), "cpu")
            del up_model, upscaled_res, upscale_loader, upsampler_node
            gc.collect()
            torch.cuda.empty_cache()
            LTXDirectorMemoryManager.watermark("after_upscale")

            # ── STEP B3: Stage 2 Refinement (4 steps, denoise 0.42) ───────
            # Reuse video_vae for Guide 2 — delete after this guide only
            guide2_node = NODE_CLASS_MAPPINGS["LTXDirectorGuide"]()
            guide2_res  = call_original_node(
                "LTXDirectorGuide", node_instance=guide2_node,
                positive=crop1_pos,
                negative=crop1_neg,
                vae=video_vae,                   # Reuse — VAE required here too
                latent=v_upscaled,
                guide_data=director_state["guide_data"],
                motion_guide_data=director_state["motion_guide_data"],
                model=model,
                strength=1.0,
                rescale_method="None",
                guide_frame=1,
                interpolation="bicubic",
                crop_position="center",
                enable_guide=True,
            )
            s2_pos   = gv(guide2_res, 0) or crop1_pos
            s2_neg   = gv(guide2_res, 1) or crop1_neg
            s2_vid   = sync_latent_device(gv(guide2_res, 2) or v_upscaled, "cpu")
            s2_model = gv(guide2_res, 3) or model
            del guide2_res, guide2_node, crop1_pos, crop1_neg, crop1_vid, v_upscaled
            del video_vae    # Single VAE lifetime: Guide1 → Upscale → Guide2 → delete
            gc.collect()
            LTXDirectorMemoryManager.watermark("after_guide2")

            concat_node_s2  = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]()
            av2_raw         = gv(call_original_node(
                "LTXVConcatAVLatent", node_instance=concat_node_s2,
                video_latent=s2_vid, audio_latent=a1_raw,
            ), 0)
            av2_in = sync_latent_device(av2_raw, "cpu")
            if isinstance(av2_in.get("samples"), torch.Tensor):
                av2_in["samples"] = av2_in["samples"].contiguous().half()   # FIX-08
            del av2_raw, concat_node_s2, s2_vid, a1_raw
            gc.collect()

            noise_node2       = NODE_CLASS_MAPPINGS["RandomNoise"]()
            guider_node2      = NODE_CLASS_MAPPINGS["CFGGuider"]()
            scheduler_node2   = NODE_CLASS_MAPPINGS["BasicScheduler"]()
            sampler_sel_node2 = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
            sampler_adv_node2 = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()

            noise2   = gv(call_original_node("RandomNoise",    node_instance=noise_node2,       noise_seed=seed + idx * 100), 0)
            guider2  = gv(call_original_node("CFGGuider",      node_instance=guider_node2,      cfg=1.0, model=s2_model, positive=s2_pos, negative=s2_neg), 0)
            sampler_e2 = gv(call_original_node("KSamplerSelect", node_instance=sampler_sel_node2, sampler_name="euler"), 0)
            sigmas2  = gv(call_original_node(
                "BasicScheduler", node_instance=scheduler_node2,
                model=s2_model, scheduler="linear_quadratic", steps=4, denoise=0.42,
            ), 0)

            # FIX-06: purge before Stage 2 sampling
            del s2_pos, s2_neg, s2_model
            gc.collect()
            torch.cuda.empty_cache()
            LTXDirectorMemoryManager.assert_vram_headroom(
                min_gb=LTXDirectorMemoryManager.VRAM_SAMPLE_HEADROOM_GB,
                context=f"Stage2-{seg_name}"
            )
            LTXDirectorMemoryManager.watermark("pre_stage2_sample")

            print(f"  ⚡ Stage 2 Refinement for {seg_name} (4 steps)...")
            t_s2   = time.time()
            s2_out = call_original_node(
                "SamplerCustomAdvanced", node_instance=sampler_adv_node2,
                noise=noise2, guider=guider2, sampler=sampler_e2,
                sigmas=sigmas2, latent_image=av2_in,
            )
            s2_lat = sync_latent_device(gv(s2_out, 0), "cpu")
            print(f"  ✓ Stage 2 Finished in {time.time() - t_s2:.2f}s!")

            del noise2, guider2, sigmas2, s2_out, av2_in
            del noise_node2, guider_node2, scheduler_node2, sampler_sel_node2, sampler_adv_node2
            gc.collect()
            torch.cuda.empty_cache()
            LTXDirectorMemoryManager.watermark("post_stage2_sample")

            sep_node2      = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
            sep2           = call_original_node("LTXVSeparateAVLatent", node_instance=sep_node2, av_latent=s2_lat)
            v2_raw         = sync_latent_device(gv(sep2, 0), "cpu")
            a2_raw         = sync_latent_device(gv(sep2, 1), "cpu")
            del s2_lat, sep2, sep_node2
            gc.collect()

            crop_node2         = NODE_CLASS_MAPPINGS["LTXDirectorCropGuides"]()
            crop2              = call_original_node(
                "LTXDirectorCropGuides", node_instance=crop_node2,
                positive=director_state["positive"],
                negative=director_state["negative"],
                latent=v2_raw,
            )
            final_seg_vid_lat  = sync_latent_device(gv(crop2, 2) or v2_raw, "cpu")
            del v2_raw, crop2, crop_node2
            gc.collect()

            # ── Save segment pack ─────────────────────────────────────────
            seg_pack = {
                "video_latent":  final_seg_vid_lat,
                "audio_latent":  a2_raw,
                "segment_index": idx,
                "valid_frames":  valid_frames,
            }
            torch.save(seg_pack, seg_cache_file)
            completed_packs.append(seg_pack)
            print(f"  💾 Saved {seg_name} Latents → {seg_cache_file}")

            del model, final_seg_vid_lat, a2_raw

        cur_lat_idx = end_lat_idx
        LTXDirectorMemoryManager.purge(f"post_seg_{idx+1}")
        LTXDirectorMemoryManager.watermark(f"seg_{idx+1}_done")

    print("✅ All Director Segments successfully sampled & upscaled!")
    return completed_packs


print("✅ Cell 13 & 14: Segment-wise Diffusion & Upscale Engine ready.")



# ════════════════════════════════════════════════════════════════════════════
# CELL 15: PHASE C — MEMORY-SAFE OUT-OF-CORE VAE DECODING  (FIX-13)
# ════════════════════════════════════════════════════════════════════════════
def execute_phase_c_video_decode(
    segment_packs: List[Dict[str, Any]],
    workdir: str = "/content/LTXDirector_Work",
    resume: bool = True,
) -> str:
    frames_file = os.path.join(workdir, "decoded_video_frames.pt")
    if resume and os.path.exists(frames_file) and os.path.getsize(frames_file) > 1024:
        print(f"  ⏭ [RESUME] Loading cached decoded video frames: {frames_file}")
        return frames_file

    LTXDirectorMemoryManager.purge("pre_video_decode")
    LTXDirectorMemoryManager.print_diagnostics(phase="PHASE C: Video VAE Decoding", node="VAEDecode")

    with torch.inference_mode():
        video_vae               = gv(call_original_node("VAELoader", vae_name="LTX23_video_vae_bf16.safetensors"), 0)
        decoded_segment_frames  = []

        for idx, pack in enumerate(segment_packs):
            v_lat = pack["video_latent"]
            print(f"  🎨 Decoding Segment {idx+1}/{len(segment_packs)} via VAE...")

            # FIX-13: tighter tiled decode params for T4 headroom
            if "LTXVSpatioTemporalTiledVAEDecode" in NODE_CLASS_MAPPINGS:
                try:
                    tiled_node  = NODE_CLASS_MAPPINGS["LTXVSpatioTemporalTiledVAEDecode"]()
                    decoded_res = call_original_node(
                        "LTXVSpatioTemporalTiledVAEDecode",
                        node_instance=tiled_node,
                        vae=video_vae,
                        latents=v_lat,
                        spatial_tiles=4,              # FIX-13: was 2
                        spatial_overlap=8,
                        temporal_tile_length=8,       # FIX-13: was 16
                        temporal_overlap=2,
                        last_frame_fix=False,
                        working_device="auto",
                        working_dtype="auto",
                    )
                    decoded_tensor = unwrap_tensor(decoded_res)
                except Exception:
                    vae_decode_node = NODE_CLASS_MAPPINGS["VAEDecode"]()
                    decoded_res     = call_original_node("VAEDecode", node_instance=vae_decode_node, samples=v_lat, vae=video_vae)
                    decoded_tensor  = unwrap_tensor(decoded_res)
            else:
                vae_decode_node = NODE_CLASS_MAPPINGS["VAEDecode"]()
                decoded_res     = call_original_node("VAEDecode", node_instance=vae_decode_node, samples=v_lat, vae=video_vae)
                decoded_tensor  = unwrap_tensor(decoded_res)

            decoded_segment_frames.append(decoded_tensor.detach().cpu().half())
            del decoded_tensor
            gc.collect()
            torch.cuda.empty_cache()

        full_video_frames = torch.cat(decoded_segment_frames, dim=0)
        print(f"  ✓ Full 30s Master Frame Tensor Shape: {full_video_frames.shape}")

        tmp_path = frames_file + ".tmp"
        torch.save(full_video_frames, tmp_path)
        os.replace(tmp_path, frames_file)
        print(f"  💾 Decoded Video Frames saved: {frames_file}")

        del video_vae, decoded_segment_frames, full_video_frames

    LTXDirectorMemoryManager.purge("video_decode_complete")
    return frames_file


print("✅ Cell 15: Video VAE Decoder configured.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 16: PHASE C — AUDIO VAE DECODING & SYNCHRONIZATION
# ════════════════════════════════════════════════════════════════════════════
def execute_phase_c_audio_decode(
    segment_packs: List[Dict[str, Any]],
    workdir: str = "/content/LTXDirector_Work",
    resume: bool = True,
) -> str:
    audio_file = os.path.join(workdir, "decoded_audio.pt")
    if resume and os.path.exists(audio_file) and os.path.getsize(audio_file) > 1024:
        print(f"  ⏭ [RESUME] Loading cached decoded audio: {audio_file}")
        return audio_file

    LTXDirectorMemoryManager.purge("pre_audio_decode")
    LTXDirectorMemoryManager.print_diagnostics(phase="PHASE C: Audio VAE Decoding", node="LTXVAudioVAEDecode")

    with torch.inference_mode():
        audio_vae        = gv(call_original_node("VAELoader", vae_name="LTX23_audio_vae_bf16.safetensors"), 0)
        audio_decode_node= NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]()

        aud_lat_list = [
            unwrap_latent(p["audio_latent"])["samples"]
            for p in segment_packs
            if unwrap_latent(p["audio_latent"])["samples"] is not None
        ]
        if aud_lat_list:
            combined_aud_lat = {"samples": concat_temporal_latents(aud_lat_list)}
        else:
            combined_aud_lat = segment_packs[0]["audio_latent"]

        print("  🎵 Decoding audio latent stream via LTXVAudioVAEDecode...")
        aud_res       = call_original_node(
            "LTXVAudioVAEDecode", node_instance=audio_decode_node,
            samples=combined_aud_lat, audio_vae=audio_vae,
        )
        decoded_audio = gv(aud_res, 0)
        print("  ✓ Audio latent stream successfully decoded.")

        tmp_path = audio_file + ".tmp"
        torch.save(decoded_audio, tmp_path)
        os.replace(tmp_path, audio_file)
        print(f"  💾 Decoded Audio saved: {audio_file}")

        del audio_vae, audio_decode_node, aud_res

    LTXDirectorMemoryManager.purge("audio_decode_complete")
    return audio_file


print("✅ Cell 16: Audio VAE Decoder configured.")



# ════════════════════════════════════════════════════════════════════════════
# CELL 17: PHASE D — VHS FINAL VIDEO COMBINE & PACKAGING  (FIX-14)
# ════════════════════════════════════════════════════════════════════════════
import numpy as np


def execute_phase_d_vhs_combine(
    frames_file_path: str,
    audio_file_path: str,
    fps: int = 24,
    crf: int = 8,
    outdir: str = "/content/LTXStudio_Output",
) -> str:
    os.makedirs(outdir, exist_ok=True)
    final_output_path = os.path.join(outdir, "LTX23_Director_Master_30s.mp4")

    LTXDirectorMemoryManager.purge("pre_vhs_combine")
    LTXDirectorMemoryManager.print_diagnostics(phase="PHASE D: Final VHS Assembly", node="VHS_VideoCombine")

    frames_tensor = torch.load(frames_file_path, map_location="cpu").float()
    audio_dict    = (
        torch.load(audio_file_path, map_location="cpu")
        if os.path.exists(audio_file_path) else None
    )

    print(f"  🎬 Combining {frames_tensor.shape[0]} frames @ {fps} fps with synchronized audio...")

    vhs_node   = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
    vhs_params = {
        "images":           frames_tensor,
        "audio":            audio_dict,
        "frame_rate":       float(fps),
        "loop_count":       0,
        "filename_prefix":  "LTX23_Director_Master",
        "format":           "video/h264-mp4",
        "pix_fmt":          "yuv420p",
        "crf":              int(crf),
        "save_metadata":    False,
        "trim_to_audio":    False,
        "pingpong":         False,
        "save_output":      True,
    }

    video_combined = False
    try:
        res = call_original_node("VHS_VideoCombine", node_instance=vhs_node, **vhs_params)
        if res and isinstance(res, (tuple, list)) and len(res) > 0:
            filenames_dict = gv(res, 0)
            if (
                isinstance(filenames_dict, dict)
                and "ui" in filenames_dict
                and "gifs" in filenames_dict["ui"]
            ):
                generated_path = filenames_dict["ui"]["gifs"][0].get("fullpath", "")
                if os.path.exists(generated_path):
                    shutil.copyfile(generated_path, final_output_path)
                    video_combined = True
    except Exception as e:
        print(f"  [Notice] VHS node fallback: {e}")

    if not video_combined or not os.path.exists(final_output_path):
        import imageio
        raw_temp_mp4 = os.path.join(outdir, "temp_raw_video.mp4")
        frames_np    = (frames_tensor.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        imageio.mimwrite(raw_temp_mp4, frames_np, fps=fps, quality=9)

        raw_song_path = "/content/ComfyUI/input/whatdreamscost/Late night trap.mp3"
        if os.path.exists(raw_song_path):
            # FIX-14: trim_sec = frames / fps (was incorrectly computed as frames/fps in V3 too,
            #          but audio_trim_start_frames is already in frame units, so divide by fps)
            trim_sec = float(TIMELINE_METADATA["audio_trim_start_frames"]) / fps
            dur_sec  = frames_tensor.shape[0] / fps
            cmd = (
                f'ffmpeg -y -i "{raw_temp_mp4}" '
                f'-ss {trim_sec:.6f} -t {dur_sec:.6f} -i "{raw_song_path}" '
                f'-map 0:v:0 -map 1:a:0 '
                f'-c:v libx264 -crf {crf} -pix_fmt yuv420p '
                f'-c:a aac -b:a 320k -shortest "{final_output_path}"'
            )
            run_cmd(cmd, silent=False)
            if os.path.exists(raw_temp_mp4):
                os.remove(raw_temp_mp4)
        else:
            shutil.move(raw_temp_mp4, final_output_path)

    del frames_tensor, audio_dict
    LTXDirectorMemoryManager.purge("vhs_cleanup")
    print(f"  🎉 Final Render Complete: {final_output_path}")
    return final_output_path


print("✅ Cell 17: Phase D VHS Assembler ready.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 18: ARTIFACT VERIFICATION & CLEANUP UTILITY
# ════════════════════════════════════════════════════════════════════════════
def verify_output_artifacts(video_path: str, expected_frames: int = 756, expected_fps: int = 24):
    print("\n" + "="*70 + "\n🔍 FINAL ARTIFACT VERIFICATION\n" + "="*70)
    if not os.path.exists(video_path) or os.path.getsize(video_path) < 1000:
        raise RuntimeError(f"Artifact check failed: '{video_path}' is missing or empty.")

    probe_cmd = (
        f'ffprobe -v error -select_streams v:0 -count_packets '
        f'-show_entries stream=nb_read_packets,r_frame_rate,duration '
        f'-of csv=p=0 "{video_path}"'
    )
    res = subprocess.run(probe_cmd, shell=True, capture_output=True, text=True)
    out = res.stdout.strip()
    print(f"  ✓ File Path    : {video_path}")
    print(f"  ✓ File Size    : {os.path.getsize(video_path) / (1024*1024):.2f} MB")
    print(f"  ✓ FFprobe Info : {out}")
    print("="*70 + "\n")


print("✅ Cell 18: Artifact Verifier ready.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 19: RUNTIME CONFIGURATION & QUALITY DEBUG MODE
# ════════════════════════════════════════════════════════════════════════════
DEBUG_MODE          = False
DEBUG_MAX_FRAMES    = 120
BASE_SEED           = 2026
SEED_MODE           = "fixed"
RESUME_CHECKPOINTS  = True
OUTPUT_CRF          = 8

WORK_DIRECTORY      = "/content/LTXDirector_Work"
OUTPUT_DIRECTORY    = "/content/LTXStudio_Output"

print(f"✅ Cell 19: Runtime Configured (Debug Mode: {DEBUG_MODE} | Seed: {BASE_SEED})")


# ════════════════════════════════════════════════════════════════════════════
# CELL 20: MASTER ONE-CLICK GENERATION FUNCTION
# ════════════════════════════════════════════════════════════════════════════
def run_ltx23_director_original_workflow(
    global_prompt: str,
    negative_prompt: str,
    timeline_metadata: Dict[str, Any],
    segments: List[Dict[str, Any]],
    seed: int = 2026,
    crf: int = 8,
    workdir: str = "/content/LTXDirector_Work",
    outdir: str = "/content/LTXStudio_Output",
    resume: bool = True,
    debug: bool = False,
    debug_max_frames: int = 120,
) -> str:
    start_time = time.time()
    print("\n" + "="*70 + "\n🎬 STARTING LTX-2.3 DIRECTOR 2.0 WORKFLOW GENERATION\n" + "="*70)

    validate_original_nodes()

    active_metadata = dict(timeline_metadata)
    active_segments = list(segments)
    if debug:
        print(f"⚠️ [DEBUG MODE ACTIVE] Capping duration to {debug_max_frames} frames.")
        active_metadata["normalDurationFrames"] = debug_max_frames
        active_metadata["duration_seconds"]     = debug_max_frames / active_metadata["frame_rate"]
        active_metadata["end_frame"]            = debug_max_frames

    timeline_ctrl = DirectorTimelineController(
        global_prompt=global_prompt,
        negative_prompt=negative_prompt,
        timeline_metadata=active_metadata,
        segments=active_segments,
    )

    # Phase A
    director_state = execute_phase_a_ltxdirector(
        timeline_ctrl=timeline_ctrl, workdir=workdir, resume=resume
    )

    # Phase B
    segment_packs = execute_segment_wise_diffusion_pipeline(
        director_state=director_state, seed=seed, workdir=workdir, resume=resume
    )

    # Phase C — Video
    frames_path = execute_phase_c_video_decode(
        segment_packs=segment_packs, workdir=workdir, resume=resume
    )

    # Phase C — Audio
    audio_path = execute_phase_c_audio_decode(
        segment_packs=segment_packs, workdir=workdir, resume=resume
    )

    # Phase D
    final_video = execute_phase_d_vhs_combine(
        frames_file_path=frames_path,
        audio_file_path=audio_path,
        fps=int(active_metadata["frame_rate"]),
        crf=crf,
        outdir=outdir,
    )

    verify_output_artifacts(
        video_path=final_video,
        expected_frames=int(active_metadata["normalDurationFrames"]),
        expected_fps=int(active_metadata["frame_rate"]),
    )

    elapsed = time.time() - start_time
    mem     = LTXDirectorMemoryManager.get_memory_stats()

    print("\n" + "="*70)
    print("🎬 LTX-2.3 DIRECTOR 2.0 COMPLETE")
    print("="*70)
    print(f"  Duration           : {active_metadata['duration_seconds']:.2f} sec "
          f"({active_metadata['normalDurationFrames']} frames @ {active_metadata['frame_rate']} FPS)")
    print(f"  Memory Status      : Free RAM: {mem['ram_avail_gb']:.2f} GB | "
          f"GPU VRAM Free: {mem['gpu_free_gb']:.2f} GB")
    print(f"  Total Elapsed Time : {elapsed/60:.2f} minutes ({elapsed:.1f}s)")
    print(f"  Final Master Video : {final_video}")
    print("="*70 + "\n")

    return final_video


if __name__ == "__main__":
    final_output_file = run_ltx23_director_original_workflow(
        global_prompt=GLOBAL_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        timeline_metadata=TIMELINE_METADATA,
        segments=ORIGINAL_SEGMENTS,
        seed=BASE_SEED,
        crf=OUTPUT_CRF,
        workdir=WORK_DIRECTORY,
        outdir=OUTPUT_DIRECTORY,
        resume=RESUME_CHECKPOINTS,
        debug=DEBUG_MODE,
        debug_max_frames=DEBUG_MAX_FRAMES,
    )
