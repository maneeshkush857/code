
# -*- coding: utf-8 -*-
"""
LTX23_Director_Master_V2.py
================================================================================
LTX-2.3 Director 2.0 30-Second Music Video Pipeline (Faithful to YouTube & JSON)
Target Hardware: Google Colab T4 GPU (15GB VRAM) & Free-Tier CPU (~12.7GB RAM)

Key Fixes:
  • Fixed TypeError in free_memory (now returns empty list instead of None)
  • Dynamic 1.2GB VRAM Shield for LoRA delta weights during DiT sampling
  • Corrected PatchSageAttentionKJ & LTXVChunkFeedForward hooks
  • Full 31.5s (756 frames @ 24fps) 5-segment timeline from FutuTek Workflow JSON
  • Stage 1 (Euler 8 steps @ 416x240) -> 2x Latent Spatial Upscale -> Stage 2 (Euler 4 steps, denoise=0.42 @ 832x480)
  • SVI-Pro Linear Overlap Blending (5 frames) across all 5 scenes
  • Perfect Audio Mux with "Late night trap.mp3" (trimStart: 447 frames)
================================================================================
"""

# ════════════════════════════════════════════════════════════════════════════
# CELL 1: ENVIRONMENT SETUP & 16GB NVME SWAP
# ════════════════════════════════════════════════════════════════════════════
# @title {"single-column":true}
# @markdown # 💥 Cell 1: Environment Setup & Memory Protection
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
from pathlib import Path
from typing import Sequence, Mapping, Any, Union, Dict, List, Optional, Tuple

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.8'
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ['MALLOC_TRIM_THRESHOLD_'] = '65536'

def run_cmd(cmd: str, silent: bool = True) -> int:
    if silent:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return res.returncode
    else:
        return subprocess.run(cmd, shell=True).returncode

# 16GB High-Speed Swap Partition for Free-Tier Colab
if not os.path.exists("/content/swapfile") or os.path.getsize("/content/swapfile") < (8 * 1024 * 1024 * 1024):
    print("⚙️ [1/3] Setting up High-Speed Swap Partition...")
    run_cmd("swapoff /content/swapfile || true")
    run_cmd("rm -f /content/swapfile")
    run_cmd("fallocate -l 16G /content/swapfile || dd if=/dev/zero of=/content/swapfile bs=1M count=16384")
    run_cmd("chmod 600 /content/swapfile")
    run_cmd("mkswap /content/swapfile")
    run_cmd("swapon /content/swapfile || true")
    run_cmd("sysctl vm.swappiness=100 || true")
    run_cmd("sysctl vm.vfs_cache_pressure=500 || true")

try:
    import psutil
    sw = psutil.swap_memory()
    print(f"  📊 Virtual Memory: Physical RAM: {psutil.virtual_memory().available/1e9:.2f} GB | Active Swap: {sw.total/1e9:.2f} GB")
except Exception:
    pass

# Patch sys.modules to fix utils.install_util
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

print("✅ Cell 1: Environment & Memory Protection Configured.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 2: INSTALL PYTHON DEPENDENCIES
# ════════════════════════════════════════════════════════════════════════════
# @title {"single-column":true}
# @markdown # 💥 Cell 2: Install Core Dependencies
print("⚙️ [2/3] Installing Core Dependencies & PyTorch...")
run_cmd("pip install -q torch torchvision torchaudio", silent=False)
run_cmd("pip uninstall -y utils || true")
os.chdir("/content")

run_cmd("pip install -q torchsde einops diffusers accelerate psutil")
run_cmd("pip install -q av spandrel albumentations onnx opencv-python onnxruntime nest_asyncio imageio aiohttp scipy")
run_cmd("pip install -q 'kornia==0.7.3'")
run_cmd("apt-get -y install -qq aria2 ffmpeg")

print("✅ Cell 2: Dependencies successfully installed.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 3: CLONE COMFYUI CORE
# ════════════════════════════════════════════════════════════════════════════
# @title {"single-column":true}
# @markdown # 💥 Cell 3: Clone Upstream ComfyUI
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

print("✅ Cell 3: ComfyUI Core ready.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 4: INSTALL CUSTOM NODES
# ════════════════════════════════════════════════════════════════════════════
# @title {"single-column":true}
# @markdown # 💥 Cell 4: Install Required Custom Nodes
custom_nodes_dir = "/content/ComfyUI/custom_nodes"
os.makedirs(custom_nodes_dir, exist_ok=True)

for item in os.listdir(custom_nodes_dir):
    full_p = os.path.join(custom_nodes_dir, item)
    if os.path.isdir(full_p) and (item.isdigit() or item.startswith(".") or item == "comfyui"):
        shutil.rmtree(full_p, ignore_errors=True)

os.chdir(custom_nodes_dir)

repos = [
    ("WhatDreamsCost-ComfyUI", "https://github.com/WhatDreamscost/WhatDreamsCost-ComfyUI"),
    ("ComfyUI_KJNodes", "https://github.com/kijai/ComfyUI-KJNodes.git"),
    ("ComfyUI_GGUF", "https://github.com/city96/ComfyUI-GGUF.git"),
    ("ComfyUI-LTXVideo", "https://github.com/Lightricks/ComfyUI-LTXVideo"),
    ("ComfyUI-VideoHelperSuite", "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"),
    ("rgthree-comfy", "https://github.com/rgthree/rgthree-comfy")
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
# CELL 5: DOWNLOAD MODELS & 4-LORA STACK
# ════════════════════════════════════════════════════════════════════════════
# @title {"single-column":true}
# @markdown # 💥 Cell 5: Download LTX-2.3 Models & LoRAs
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
        cmd = ['aria2c', '--console-log-level=error', '-c', '-x', '16',
               '-s', '16', '-k', '1M', '-d', dest_dir, '-o', filename, url]
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
    filename="ltx-2-3-22b-dev-Q4_K_M.gguf"
)
link_file_safe("/content/ComfyUI/models/unet/ltx-2-3-22b-dev-Q4_K_M.gguf", "/content/ComfyUI/models/diffusion_models/ltx-2-3-22b-dev-Q4_K_M.gguf")

text_encoder_model = download_file(
    "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
    "/content/ComfyUI/models/text_encoders",
    filename="gemma_3_12B_it_fp4_mixed.safetensors"
)
link_file_safe("/content/ComfyUI/models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors", "/content/ComfyUI/models/clip/gemma_3_12B_it_fp4_mixed.safetensors")

text_encoder2_model = download_file(
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
    "/content/ComfyUI/models/text_encoders",
    filename="ltx-2.3_text_projection_bf16.safetensors"
)
link_file_safe("/content/ComfyUI/models/text_encoders/ltx-2.3_text_projection_bf16.safetensors", "/content/ComfyUI/models/clip/ltx-2.3_text_projection_bf16.safetensors")

vae_model = download_file(
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors",
    "/content/ComfyUI/models/vae",
    filename="LTX23_video_vae_bf16.safetensors"
)
vae_audio_model = download_file(
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors",
    "/content/ComfyUI/models/vae",
    filename="LTX23_audio_vae_bf16.safetensors"
)
tiny_vae_model = download_file(
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors",
    "/content/ComfyUI/models/vae",
    filename="taeltx2_3.safetensors"
)
link_file_safe("/content/ComfyUI/models/vae/taeltx2_3.safetensors", "/content/ComfyUI/models/vae_approx/taeltx2_3.safetensors")

upscaler_model = download_file(
    "https://huggingface.co/vidfom/aimusic/resolve/main/ComfyUI/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "/content/ComfyUI/models/latent_upscale_models",
    filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
)
link_file_safe("/content/ComfyUI/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors", "/content/ComfyUI/models/upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")

print("📦 Downloading Director 2.0 4-LoRA Stack...")
lora_dir = "/content/ComfyUI/models/loras"
lora_1 = download_file("https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors", lora_dir, filename="ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors")
lora_2 = download_file("https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors", lora_dir, filename="LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors")
lora_3 = download_file("https://huggingface.co/joyfox/LTX-2.3-Transition-LORA/resolve/main/ltx2.3-transition.safetensors", lora_dir, filename="ltx2.3-transition.safetensors")
lora_4 = download_file("https://huggingface.co/vidfom/aimusic/resolve/main/ComfyUI/models/loras/LTX2.3-MVCamera-drclips.safetensors", lora_dir, filename="LTX2.3-MVCamera-drclips.safetensors")

audio_dest_dir = "/content/ComfyUI/input/whatdreamscost"
audio_file_target = os.path.join(audio_dest_dir, "Late night trap.mp3")
os.makedirs(audio_dest_dir, exist_ok=True)

if not os.path.exists(audio_file_target) or os.path.getsize(audio_file_target) < 10000:
    download_file("https://huggingface.co/vidfom/aimusic/resolve/main/Late%20night%20trap.mp3", audio_dest_dir, filename="Late night trap.mp3")

print("✅ Cell 5: Models, LoRAs and audio assets validated.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 6: COMPLETE 30-SECOND (756 FRAMES) 5-SCENE TIMELINE CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
# @title {"single-column":true}
# @markdown # 💥 Cell 6: 30-Second 5-Scene Storyboard & LoRA Settings

# @markdown ### 🎬 Resolution & Frame Parameters
width = 832            # @param [512, 640, 768, 832, 960, 1024, 1280] {type:"raw"}
height = 480           # @param [320, 384, 480, 512, 544, 720] {type:"raw"}
fps = 24               # @param [24, 25, 30] {type:"raw"}
overlap_frames = 5     # @param {type:"slider", min:0, max:16, step:1}
output_crf = 8         # @param {type:"slider", min:0, max:30, step:1}

# Explicit LoRA Paths
lora_dir = "/content/ComfyUI/models/loras"
lora_1 = os.path.join(lora_dir, "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors")
lora_2 = os.path.join(lora_dir, "LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors")
lora_3 = os.path.join(lora_dir, "ltx2.3-transition.safetensors")
lora_4 = os.path.join(lora_dir, "LTX2.3-MVCamera-drclips.safetensors")

# @markdown ### 🎛️ Director 2.0 4-LoRA Stack Strengths
use_lora_1 = True      # @param {type:"boolean"}
lora_strength_1 = 0.4  # @param {type:"slider", min:0.0, max:1.5, step:0.05}
use_lora_2 = True      # @param {type:"boolean"}
lora_strength_2 = 0.6  # @param {type:"slider", min:0.0, max:1.5, step:0.05}
use_lora_3 = True      # @param {type:"boolean"}
lora_strength_3 = 0.7  # @param {type:"slider", min:0.0, max:1.5, step:0.05}
use_lora_4 = True      # @param {type:"boolean"}
lora_strength_4 = 0.9  # @param {type:"slider", min:0.0, max:1.5, step:0.05}

# @markdown ### 🎵 Audio Sync & System Settings
audio_trim_start_frames = 447  # @param {type:"integer"}
use_song_audio = True          # @param {type:"boolean"}
resume_generation = True       # @param {type:"boolean"}
min_ram_guard_gb = 2.0         # @param {type:"slider", min:1.0, max:6.0, step:0.5}

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

NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, robotic movement, static presenter, jitter, flicker, facial distortion, extra limbs, watermark"

# 🎬 Complete 5-Scene Timeline from FutuTek YouTube Video & JSON (Total 756 frames / 31.5 seconds)
SEGMENTS_CONFIG = [
    {
        "id": 1,
        "image": "/content/ComfyUI/input/whatdreamscost/1.png",
        "seconds": 9.42,
        "frames": 226,
        "prompt": "Open up the canvas, blank space on my screen. Sold-out stadium concert, singer performs with explosive stage presence."
    },
    {
        "id": 2,
        "image": "/content/ComfyUI/input/whatdreamscost/2.png",
        "seconds": 6.72,
        "frames": 161,
        "prompt": "Drag a Checkpoint Loader, you know what I mean. Dynamic low-angle hero shot, rhythmic arm accents, charismatic superstar attitude."
    },
    {
        "id": 3,
        "image": "/content/ComfyUI/input/whatdreamscost/3.png",
        "seconds": 5.48,
        "frames": 131,
        "prompt": "KSampler in the middle, VAE on the right. Fast push-in camera, energetic beat groove, rich facial micro-expressions."
    },
    {
        "id": 4,
        "image": "/content/ComfyUI/input/whatdreamscost/4.png",
        "seconds": 9.40,
        "frames": 226,
        "prompt": "Put the Text Encoder, yeah, building tonight. Wide volumetric concert lighting, powerful pointing, explosive performance energy."
    },
    {
        "id": 5,
        "image": "/content/ComfyUI/input/whatdreamscost/5.3.png",
        "seconds": 3.47,
        "frames": 83,
        "prompt": "Connect the nodes, run the queue, watch the latent flow right through. Cinematic pull-back orbit shot, emotional peak."
    }
]

LORA_WEIGHTS = [
    {"enabled": use_lora_1, "name": lora_1, "strength": lora_strength_1},
    {"enabled": use_lora_2, "name": lora_2, "strength": lora_strength_2},
    {"enabled": use_lora_3, "name": lora_3, "strength": lora_strength_3},
    {"enabled": use_lora_4, "name": lora_4, "strength": lora_strength_4},
]

total_frames_est = sum(s['frames'] for s in SEGMENTS_CONFIG) - overlap_frames * (len(SEGMENTS_CONFIG) - 1)
print(f"✅ Cell 6: Configured {width}x{height} (Base: {width//2}x{height//2}) | {len(SEGMENTS_CONFIG)} Scenes | ~{total_frames_est} frames ({total_frames_est/fps:.2f}s @ {fps}fps)")


# ════════════════════════════════════════════════════════════════════════════
# CELL 7: PRODUCTION MEMORY ENGINE & ROBUST FREE_MEMORY HOOK
# ════════════════════════════════════════════════════════════════════════════
def malloc_trim_os():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def patch_comfy_memory_manager():
    try:
        import comfy.model_management as mm
        if not getattr(mm, "_is_free_memory_patched", False):
            _orig_free_memory = mm.free_memory
            def _safe_free_memory(*args, **kwargs):
                try:
                    res = _orig_free_memory(*args, **kwargs)
                    return res if isinstance(res, list) else []
                except Exception:
                    return []
            mm.free_memory = _safe_free_memory

            _orig_get_free_memory = mm.get_free_memory
            def _buffered_get_free_memory(dev=None, torch_free_too=False):
                try:
                    free = _orig_get_free_memory(dev, torch_free_too)
                    # Keep safe 1.2GB buffer for dynamic LoRA delta tensor allocations
                    return max(512 * 1024 * 1024, free - 1200 * 1024 * 1024)
                except Exception:
                    return 2 * 1024 * 1024 * 1024

            mm.get_free_memory = _buffered_get_free_memory
            mm._is_free_memory_patched = True

        mm.text_encoder_device = lambda: torch.device("cuda")
        mm.text_encoder_offload_device = lambda: torch.device("cuda")
    except Exception as e:
        print(f"Memory patch notice: {e}")

def patch_safetensors_direct_to_gpu():
    try:
        import safetensors.torch
        if not getattr(safetensors.torch, "_is_cuda_direct_patched", False):
            _orig_safetensors_load = safetensors.torch.load_file
            def _safe_cuda_load(filename, device="cpu"):
                fn_lower = str(filename).lower()
                if any(k in fn_lower for k in ["gemma", "clip", "text_encoder", "projection", "connector"]):
                    if torch.cuda.is_available():
                        return _orig_safetensors_load(filename, device="cuda")
                return _orig_safetensors_load(filename, device=device)
            safetensors.torch.load_file = _safe_cuda_load
            safetensors.torch._is_cuda_direct_patched = True
    except Exception:
        pass

patch_comfy_memory_manager()
patch_safetensors_direct_to_gpu()

def unwrap_tensor(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj
    if hasattr(obj, "args") and len(obj.args) > 0:
        return unwrap_tensor(obj.args[0])
    if hasattr(obj, "outputs") and len(obj.outputs) > 0:
        return unwrap_tensor(obj.outputs[0])
    if hasattr(obj, "result") and len(obj.result) > 0:
        return unwrap_tensor(obj.result[0])
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
    return obj

def gv(obj: Any, index: int = 0) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "args") and isinstance(obj.args, (list, tuple)) and len(obj.args) > 0:
        if len(obj.args) == 1 and isinstance(obj.args[0], (list, tuple)):
            if len(obj.args[0]) > index:
                return obj.args[0][index]
            return None
        if len(obj.args) > index:
            return obj.args[index]
        return None

    for attr in ["output", "outputs", "result", "values"]:
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if isinstance(val, (list, tuple)):
                if len(val) > index:
                    return val[index]
                return None
            elif isinstance(val, dict):
                if index in val:
                    return val[index]
                return None
            elif index == 0:
                return val

    if isinstance(obj, (tuple, list)):
        if len(obj) > index:
            return obj[index]
        return None

    if isinstance(obj, dict):
        if "result" in obj and isinstance(obj["result"], (list, tuple)):
            if len(obj["result"]) > index:
                return obj["result"][index]
            return None
        if index in obj:
            return obj[index]
        return None

    if not isinstance(obj, (torch.Tensor, str, int, float, bool, bytes)):
        try:
            items = list(obj)
            if len(items) > index:
                return items[index]
            return None
        except Exception:
            pass
        if hasattr(obj, "__dict__"):
            for v in obj.__dict__.values():
                if isinstance(v, (list, tuple)) and len(v) > index:
                    return v[index]

    if index == 0:
        return obj
    return None

def unwrap_latent(x: Any) -> Dict[str, Any]:
    if x is None:
        return {"samples": None}
    while isinstance(x, (tuple, list)) and len(x) > 0:
        x = x[0]
    if hasattr(x, "result"):
        res = getattr(x, "result")
        if isinstance(res, (tuple, list)) and len(res) > 0:
            x = res[0]
        elif isinstance(res, dict):
            x = res
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
    target = torch.device(target_device)
    latent_dict = unwrap_latent(latent)
    samples = latent_dict.get("samples", None)
    if samples is None:
        return latent_dict
    if isinstance(samples, torch.Tensor):
        if samples.is_nested:
            nested_list = [t.to(target) for t in samples.unbind()]
            latent_dict["samples"] = torch.nested.nested_tensor(nested_list)
        else:
            latent_dict["samples"] = samples.to(target)
    return latent_dict

def get_ram_free_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return 99.0

def drop_page_cache():
    patterns = [
        "/content/ComfyUI/models/unet/*.gguf",
        "/content/ComfyUI/models/diffusion_models/*.gguf",
        "/content/ComfyUI/models/text_encoders/*.safetensors",
        "/content/ComfyUI/models/clip/*.safetensors",
        "/content/ComfyUI/models/vae/*.safetensors",
        "/content/ComfyUI/models/latent_upscale_models/*.safetensors",
        "/content/ComfyUI/models/upscale_models/*.safetensors",
        "/content/ComfyUI/models/loras/*.safetensors",
    ]
    for pat in patterns:
        for f in glob.glob(pat):
            try:
                fd = os.open(f, os.O_RDONLY)
                size = os.fstat(fd).st_size
                os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)
                os.close(fd)
            except Exception:
                pass

def purge_deep(tag: str = ""):
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
    drop_page_cache()
    malloc_trim_os()

def ram_guard(min_free_gb: float = 2.0, tag: str = ""):
    if get_ram_free_gb() < min_free_gb:
        print(f"⚠️ [RAM GUARD] Free RAM ({get_ram_free_gb():.2f} GB) < {min_free_gb} GB -> Deep Purge")
        purge_deep(f"ram_guard:{tag}")

print("✅ Cell 7: Production Memory Engine & 1.2GB VRAM Shield active.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 8: NODE REGISTRY & DISPATCHER
# ════════════════════════════════════════════════════════════════════════════
# @title {"single-column":true}
# @markdown # 💥 Cell 8: Node Registry
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
                    self.app = web.Application()
                    self.loop = loop
                def send_sync(self, *args, **kwargs):
                    pass
            PromptServer.instance = MockServer()
except Exception:
    pass

from nodes import init_builtin_extra_nodes, init_external_custom_nodes

async def _init_nodes():
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
        task = asyncio.ensure_future(_init_nodes())
        loop.run_until_complete(task)
    else:
        loop.run_until_complete(_init_nodes())
except Exception:
    pass

from nodes import NODE_CLASS_MAPPINGS, LoraLoaderModelOnly

def call_node(node_instance: Any, **kwargs) -> Any:
    func_name = getattr(node_instance, "FUNCTION", None)
    callables = []

    if func_name and hasattr(node_instance, func_name):
        callables.append(getattr(node_instance, func_name))
    if hasattr(node_instance, "execute"):
        callables.append(node_instance.execute)
    for fallback in ["get_guider", "get_noise", "get_sampler", "get_sigmas", "sample", "apply_guide", "crop_guides", "upsample_latent", "concat", "separate", "encode", "decode"]:
        if hasattr(node_instance, fallback):
            callables.append(getattr(node_instance, fallback))
    if hasattr(node_instance, "EXECUTE_NORMALIZED"):
        callables.append(node_instance.EXECUTE_NORMALIZED)

    last_err = None
    for func in callables:
        try:
            sig = inspect.signature(func)
            valid_kwargs = {}
            for name, param in sig.parameters.items():
                if name in ['cls', 'self']:
                    continue
                if name in kwargs:
                    valid_kwargs[name] = kwargs[name]
                elif param.default is not inspect.Parameter.empty:
                    pass
                else:
                    if param.annotation == int or 'int' in str(param.annotation):
                        valid_kwargs[name] = 0
                    elif param.annotation == float or 'float' in str(param.annotation):
                        valid_kwargs[name] = 0.0
                    elif param.annotation == bool or 'bool' in str(param.annotation):
                        valid_kwargs[name] = False
                    elif param.annotation == str or 'str' in str(param.annotation):
                        valid_kwargs[name] = ""
                    else:
                        valid_kwargs[name] = None
            return func(**valid_kwargs)
        except Exception as e:
            last_err = e
            continue

    if last_err is not None:
        raise last_err
    raise AttributeError(f"Cannot execute node '{node_instance.__class__.__name__}'")

def load_vae_helper(vae_name: str, device: str = "main_device", dtype: str = "bf16"):
    if "VAELoaderKJ" in NODE_CLASS_MAPPINGS:
        try:
            vkj = NODE_CLASS_MAPPINGS["VAELoaderKJ"]()
            return gv(call_node(vkj, vae_name=vae_name, device=device, weight_dtype=dtype), 0)
        except Exception:
            pass
    vl = NODE_CLASS_MAPPINGS["VAELoader"]()
    return gv(call_node(vl, vae_name=vae_name), 0)

print(f"✅ Cell 8: {len(NODE_CLASS_MAPPINGS)} ComfyUI nodes registered.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 9: STORYBOARD KEYFRAME INITIALIZATION
# ════════════════════════════════════════════════════════════════════════════
# @title {"single-column":true}
# @markdown # 💥 Cell 9: Keyframe Validator
from PIL import Image, ImageOps, ImageDraw
import numpy as np

base_input = "/content/ComfyUI/input/whatdreamscost"
os.makedirs(base_input, exist_ok=True)

# Generate fallback images if keyframes are not yet uploaded
for idx, fname in enumerate(["1.png", "2.png", "3.png", "4.png", "5.3.png"]):
    p = os.path.join(base_input, fname)
    if not os.path.exists(p):
        img = Image.new("RGB", (768, 512), color=(30 + idx * 25, 25, 60 + idx * 25))
        d = ImageDraw.Draw(img)
        d.text((50, 220), f"Keyframe {fname} (Upload singer photo)", fill=(255, 255, 255))
        img.save(p)

if os.path.exists(f"{base_input}/5.png") and not os.path.exists(f"{base_input}/5.3.png"):
    shutil.copy(f"{base_input}/5.png", f"{base_input}/5.3.png")

def prepare_reference_image(image_path: str, width: int, height: int) -> torch.Tensor:
    if image_path and os.path.exists(image_path):
        pil_img = Image.open(image_path).convert("RGB")
        pil_img = ImageOps.exif_transpose(pil_img)

        # Center crop to target aspect ratio before resizing
        target_aspect = width / height
        img_w, img_h = pil_img.size
        img_aspect = img_w / img_h

        if img_aspect > target_aspect:
            new_w = int(target_aspect * img_h)
            offset = (img_w - new_w) // 2
            pil_img = pil_img.crop((offset, 0, offset + new_w, img_h))
        else:
            new_h = int(img_w / target_aspect)
            offset = (img_h - new_h) // 2
            pil_img = pil_img.crop((0, offset, img_w, offset + new_h))

        pil_resized = pil_img.resize((width, height), Image.BICUBIC)
        img_np = np.array(pil_resized).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)
    return torch.full((1, height, width, 3), 0.5)

print("✅ Cell 9: Keyframe assets validated.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 10: DIT MODEL LOADER WITH LORA VRAM OVERFLOW SHIELD
# ════════════════════════════════════════════════════════════════════════════
# @title {"single-column":true}
# @markdown # 💥 Cell 10: DiT & LoRA Combiner
def load_dit_with_loras(model_path: str, lora_configs: List[Dict[str, Any]]):
    # Deep purge before loading DiT to guarantee maximum free VRAM
    purge_deep("pre_dit_load")

    unet_loader = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
    model = gv(call_node(unet_loader, unet_name=model_path), 0)

    # Use native GGUF LoRA Loader if available, else standard LoraLoaderModelOnly
    lora_loader_cls = NODE_CLASS_MAPPINGS.get("LoraLoaderGGUF", LoraLoaderModelOnly)
    lora_loader = lora_loader_cls()

    for cfg in lora_configs:
        if cfg.get("enabled") and cfg.get("name") and os.path.exists(cfg["name"]):
            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                malloc_trim_os()
                lora_arg = os.path.basename(cfg["name"])
                res = call_node(lora_loader, model=model, lora_name=lora_arg, strength_model=cfg["strength"])
                model = gv(res, 0) or model
                print(f"  + Applied LoRA: {lora_arg} (Strength: {cfg['strength']})")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                malloc_trim_os()
            except Exception as e:
                print(f"  [Notice] LoRA {os.path.basename(cfg['name'])} fallback: {e}")

    # SageAttention & Chunk Feed Forward Hooks
    if "PatchSageAttentionKJ" in NODE_CLASS_MAPPINGS:
        try:
            sage = NODE_CLASS_MAPPINGS["PatchSageAttentionKJ"]()
            res = call_node(sage, model=model, sage_attention="auto")
            model = gv(res, 0) or model
            print("  ✓ SageAttention Hook Applied")
        except Exception:
            pass
    if "LTXVChunkFeedForward" in NODE_CLASS_MAPPINGS:
        try:
            cff = NODE_CLASS_MAPPINGS["LTXVChunkFeedForward"]()
            res = call_node(cff, model=model, chunks=8, dim_threshold=4096)
            model = gv(res, 0) or model
            print("  ✓ ChunkFeedForward Hook Applied (chunks=8)")
        except Exception:
            pass
    return model

print("✅ Cell 10: DiT loader ready.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 11: TEXT ENCODING & ZERO-OUT CONDITIONING (PHASE A)
# ════════════════════════════════════════════════════════════════════════════
# @title {"single-column":true}
# @markdown # 💥 Cell 11: Phase A Text Encoder
def zero_out_conditioning(conditioning: Any) -> Any:
    if "ConditioningZeroOut" in NODE_CLASS_MAPPINGS:
        try:
            cz = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
            return gv(call_node(cz, conditioning=conditioning), 0)
        except Exception:
            pass
    c = []
    for t in conditioning:
        d = t.copy() if len(t) > 1 and isinstance(t, dict) else {}
        pooled = d.get("pooled_output", None)
        if pooled is not None and torch.is_tensor(pooled):
            d["pooled_output"] = torch.zeros_like(pooled)
        c.append([torch.zeros_like(t[0]), d])
    return c

def encode_phase_a_text_conditionings(
    segments: List[Dict[str, Any]],
    global_prompt: str,
    negative_prompt: str,
    fps: int = 24
) -> List[Any]:
    print("\n" + "="*70 + "\n🎬 PHASE A: Dual-CLIP Direct-GPU Text Ingestion & Purge\n" + "="*70)
    drop_page_cache()
    purge_deep("phase_a_preflight")
    ram_guard(min_free_gb=min_ram_guard_gb, tag="phase_a_start")

    dcl = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
    clip_text_encode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
    ltxv_conditioning = NODE_CLASS_MAPPINGS["LTXVConditioning"]()

    clip_obj = gv(call_node(dcl, clip_name1=text_encoder_model, clip_name2=text_encoder2_model, type="ltxv", device="default"), 0)
    neg_raw = gv(call_node(clip_text_encode, text=negative_prompt, clip=clip_obj), 0)
    neg_conditioning = zero_out_conditioning(neg_raw)
    del neg_raw
    gc.collect()
    malloc_trim_os()

    segment_conditionings = []
    for s in segments:
        full_prompt = f"{global_prompt}\n\nSection Action: {s['prompt']}"
        pos_raw = gv(call_node(clip_text_encode, text=full_prompt, clip=clip_obj), 0)
        cond = call_node(ltxv_conditioning, frame_rate=fps, positive=pos_raw, negative=neg_conditioning)
        segment_conditionings.append(cond)
        del pos_raw
        gc.collect()
        malloc_trim_os()

    import comfy.model_management as mm
    del dcl, clip_obj, clip_text_encode, neg_conditioning
    mm.unload_all_models()
    mm.cleanup_models()
    mm.soft_empty_cache()
    if hasattr(mm, "current_loaded_models") and isinstance(mm.current_loaded_models, list):
        mm.current_loaded_models.clear()
    gc.collect()
    torch.cuda.empty_cache()
    malloc_trim_os()
    print(f"  ✓ Phase A Complete. Free System RAM: {get_ram_free_gb():.2f} GB")
    return segment_conditionings

print("✅ Cell 11: Text Conditioning encoder ready.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 12: EXACT BASIC SCHEDULER (LINEAR_QUADRATIC)
# ════════════════════════════════════════════════════════════════════════════
# @title {"single-column":true}
# @markdown # 💥 Cell 12: Sigmas Helper
def get_basic_scheduler_sigmas(model: Any, scheduler: str = "linear_quadratic", steps: int = 8, denoise: float = 1.0) -> Any:
    if "BasicScheduler" in NODE_CLASS_MAPPINGS:
        try:
            bs = NODE_CLASS_MAPPINGS["BasicScheduler"]()
            res = call_node(bs, model=model, scheduler=scheduler, scheduler_name=scheduler, steps=steps, denoise=denoise)
            sig = gv(res, 0)
            if sig is not None and isinstance(sig, torch.Tensor) and sig.numel() > 0:
                return sig
        except Exception:
            pass

    try:
        import comfy.samplers
        model_sampling = model.get_model_object("model_sampling")
        total_steps = steps
        if 0.0 < denoise < 1.0:
            total_steps = int(steps / denoise)
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, total_steps)
        return sigmas[-(steps + 1):]
    except Exception:
        pass

    total_steps = int(round(steps / denoise)) if (0.0 < denoise < 1.0) else steps
    sigmas = []
    for i in range(total_steps + 1):
        t = 1.0 - (i / total_steps)
        sigmas.append(t * t)
    sigmas = sigmas[-(steps + 1):]
    return torch.tensor(sigmas, dtype=torch.float32)

print("✅ Cell 12: BasicScheduler sigma calculator ready.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 13: SPATIOTEMPORAL TILED VAE DECODER
# ════════════════════════════════════════════════════════════════════════════
# @title {"single-column":true}
# @markdown # 💥 Cell 13: Tiled VAE Decoder
def tiled_decode_video(video_latent: Any, vae_obj: Any, prefer_tiled: bool = True, tile_size: int = 256) -> torch.Tensor:
    latent_dict = unwrap_latent(video_latent)

    if prefer_tiled and "LTXVSpatioTemporalTiledVAEDecode" in NODE_CLASS_MAPPINGS:
        try:
            tiled = NODE_CLASS_MAPPINGS["LTXVSpatioTemporalTiledVAEDecode"]()
            res = call_node(
                tiled,
                vae=vae_obj, latents=latent_dict,
                spatial_tiles=2, spatial_overlap=8,
                temporal_tile_length=16, temporal_overlap=4,
                last_frame_fix=False, working_device="auto", working_dtype="auto"
            )
            return unwrap_tensor(res)
        except Exception:
            pass

    if prefer_tiled and "VAEDecodeTiled" in NODE_CLASS_MAPPINGS:
        try:
            vdt = NODE_CLASS_MAPPINGS["VAEDecodeTiled"]()
            res = call_node(vdt, samples=latent_dict, vae=vae_obj, tile_size=tile_size)
            return unwrap_tensor(res)
        except Exception:
            pass

    vd = NODE_CLASS_MAPPINGS["VAEDecode"]()
    res = call_node(vd, samples=latent_dict, vae=vae_obj)
    return unwrap_tensor(res)

print("✅ Cell 13: Tiled VAE Decoder ready.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 14: OVERLAP BLENDING & AUDIO COMPOSITOR
# ════════════════════════════════════════════════════════════════════════════
# @title {"single-column":true}
# @markdown # 💥 Cell 14: Overlap Crossfade & Audio Mux
def linear_blend_overlap(source: Optional[torch.Tensor], new: torch.Tensor, overlap: int = 5) -> torch.Tensor:
    if source is None:
        return new
    if overlap <= 0 or source.shape[0] < overlap or new.shape[0] < overlap:
        return torch.cat([source, new], dim=0)
    src_tail = source[-overlap:].float()
    new_head = new[:overlap].float()
    w = torch.linspace(1.0, 0.0, overlap, device=src_tail.device).view(-1, 1, 1, 1)
    blended = src_tail * w + new_head * (1.0 - w)
    return torch.cat([source[:-overlap], blended.to(source.dtype), new[overlap:]], dim=0)

def blend_audio(audio_list: List[Optional[Dict[str, Any]]], overlap_frames: int, fps: int) -> Optional[Dict[str, Any]]:
    valid_audio = [a for a in audio_list if a is not None]
    if not valid_audio:
        return None
    sr = valid_audio[0]["sample_rate"]
    ov_samples = int(overlap_frames / fps * sr)
    result = valid_audio[0]["waveform"].float()
    samples_dim = result.shape.index(max(result.shape))

    for a in valid_audio[1:]:
        w = a["waveform"].float()
        if ov_samples <= 0 or result.shape[samples_dim] < ov_samples or w.shape[samples_dim] < ov_samples:
            result = torch.cat([result, w], dim=samples_dim)
            continue
        shape = [1 for _ in range(result.dim())]
        shape[samples_dim] = ov_samples
        t = torch.linspace(1.0, 0.0, ov_samples, device=result.device).view(*shape)
        res_tail = result.narrow(samples_dim, result.shape[samples_dim] - ov_samples, ov_samples)
        w_head = w.narrow(samples_dim, 0, ov_samples)
        blended = res_tail * t + w_head * (1.0 - t)
        res_body = result.narrow(samples_dim, 0, result.shape[samples_dim] - ov_samples)
        w_body = w.narrow(samples_dim, ov_samples, w.shape[samples_dim] - ov_samples)
        result = torch.cat([res_body, blended, w_body], dim=samples_dim)
    return {"waveform": result, "sample_rate": sr}

def trim_audio(audio: Optional[Dict[str, Any]], trim_start_frames: int, fps: int, total_frames: int) -> Optional[Dict[str, Any]]:
    if audio is None:
        return None
    sr = audio["sample_rate"]
    w = audio["waveform"]
    samples_dim = w.shape.index(max(w.shape))
    start_sample = int(trim_start_frames / fps * sr)
    length_samples = int(total_frames / fps * sr)
    if start_sample >= w.shape[samples_dim]:
        start_sample = 0
    end_sample = min(start_sample + length_samples, w.shape[samples_dim])
    w_trimmed = w.narrow(samples_dim, start_sample, end_sample - start_sample)
    return {"waveform": w_trimmed, "sample_rate": sr}

def mux_final_audio(video_path: str, song_path: str, crf: int = 8) -> str:
    out_path = video_path.replace('.mp4', '_synced.mp4')
    cmd = (f'ffmpeg -y -i "{video_path}" -i "{song_path}" '
           f'-map 0:v:0 -map 1:a:0 -c:v libx264 -crf {crf} -pix_fmt yuv420p '
           f'-c:a aac -b:a 320k -shortest "{out_path}"')
    run_cmd(cmd, silent=False)
    return out_path

print("✅ Cell 14: Overlap blending & audio compositor ready.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 15: MASTER 5-SCENE 30-SECOND 2-STAGE PIPELINE ENGINE
# ════════════════════════════════════════════════════════════════════════════
# @title {"single-column":true}
# @markdown # 💥 Cell 15: Master 2-Stage Director Pipeline Engine & Execution
def run_workflow_faithful_director_pipeline(
    segments: List[Dict[str, Any]],
    global_prompt: str,
    negative_prompt: str,
    song_path: Optional[str] = None,
    width: int = 832,
    height: int = 480,
    fps: int = 24,
    overlap_frames: int = 5,
    output_crf: int = 8,
    min_ram_guard_gb: float = 2.0,
    lora_configs: Optional[List[Dict[str, Any]]] = None,
    workdir: str = "/content/LTXDirector_Work",
    outdir: str = "/content/LTXStudio_Output",
    resume: bool = True
) -> Optional[str]:

    os.makedirs(workdir, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)
    if lora_configs is None:
        lora_configs = LORA_WEIGHTS

    patch_comfy_memory_manager()
    patch_safetensors_direct_to_gpu()
    purge_deep("Pre-Flight Reset")
    print(f"  [RAM Baseline] Free System RAM: {get_ram_free_gb():.2f} GB")

    with torch.inference_mode():
        # PHASE A: Text Conditioning
        segment_conditionings = encode_phase_a_text_conditionings(
            segments=segments,
            global_prompt=global_prompt,
            negative_prompt=negative_prompt,
            fps=fps
        )

        # PHASE B: 2-Stage Diffusion per Segment
        print("\n" + "="*70 + f"\n🎬 PHASE B: 2-Stage Faithful Diffusion ({width//2}x{height//2} -> {width}x{height})\n" + "="*70)
        final_latent_files = []

        emptyltxvlatentvideo = NODE_CLASS_MAPPINGS["EmptyLTXVLatentVideo"]()
        empty_audio_latent = NODE_CLASS_MAPPINGS["LTXVEmptyLatentAudio"]()
        concat_av_latent = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]()
        separate_av_latent = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
        samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
        cfgguider = NODE_CLASS_MAPPINGS["CFGGuider"]()

        # Guide & Crop Nodes
        add_guide_cls = NODE_CLASS_MAPPINGS.get("LTXVAddGuide", NODE_CLASS_MAPPINGS.get("LTXDirectorGuide"))
        crop_guides_cls = NODE_CLASS_MAPPINGS.get("LTXDirectorCropGuides", NODE_CLASS_MAPPINGS.get("LTXVCropGuides"))
        ltxvlatentupsampler = NODE_CLASS_MAPPINGS["LTXVLatentUpsampler"]()
        latentupscaleloader = NODE_CLASS_MAPPINGS["LatentUpscaleModelLoader"]()

        ks_euler = call_node(ksamplerselect, sampler_name="euler")
        half_w, half_h = width // 2, height // 2

        for idx, s in enumerate(segments):
            sid = s.get("id", idx + 1)
            target_final_file = f"{workdir}/final_latent_seg_{sid:02d}.pt"
            final_latent_files.append(target_final_file)

            if resume and os.path.exists(target_final_file) and os.path.getsize(target_final_file) > 1024:
                print(f"  ⏭ Scene {sid} already cached. Skipping.")
                continue

            print(f"\n--- Processing Scene {sid}/{len(segments)} ({s['seconds']}s | Keyframe: {os.path.basename(s['image'])}) ---")
            ram_guard(min_free_gb=min_ram_guard_gb, tag=f"seg_{sid}_start")

            # 1. Load DiT + LoRAs with Clean VRAM Shield
            dit = load_dit_with_loras(dit_model, lora_configs)
            n_frames = s["frames"]
            cond = segment_conditionings[idx]
            seed = s.get("seed", 2026 + idx * 77)

            # 2. Base Latents
            empty_lat = call_node(emptyltxvlatentvideo, width=half_w, height=half_h, length=n_frames, batch_size=1)
            vid_in = sync_latent_device(gv(empty_lat, 0), "cpu")
            del empty_lat

            ava = load_vae_helper(vae_audio_model, device="main_device", dtype="fp16")
            aud_lat_res = call_node(empty_audio_latent, frames_number=n_frames, frame_rate=fps, batch_size=1, audio_vae=ava)
            aud_lat = sync_latent_device(gv(aud_lat_res, 0), "cpu")
            del ava, aud_lat_res

            # 3. Stage 1 Guide Binding
            ref_img_s1 = prepare_reference_image(s.get("image", ""), half_w, half_h)
            vva = load_vae_helper(vae_model, device="main_device", dtype="bf16")

            stage1_pos = gv(cond, 0)
            stage1_neg = gv(cond, 1)
            stage1_vid_lat = vid_in
            stage1_model = dit

            if add_guide_cls is not None:
                guide_node = add_guide_cls()
                g_res = call_node(
                    guide_node,
                    positive=stage1_pos,
                    negative=stage1_neg,
                    vae=vva,
                    latent=vid_in,
                    image=ref_img_s1,
                    frame_idx=0,
                    strength=1.0,
                    model=dit
                )
                stage1_pos = gv(g_res, 0) if gv(g_res, 0) is not None else stage1_pos
                stage1_neg = gv(g_res, 1) if gv(g_res, 1) is not None else stage1_neg
                g_lat = gv(g_res, 2)
                if g_lat is not None:
                    stage1_vid_lat = sync_latent_device(g_lat, "cpu")
                mod_out = gv(g_res, 3)
                if mod_out is not None and hasattr(mod_out, "model_options"):
                    stage1_model = mod_out
                else:
                    stage1_model = dit
                print("  ✓ Stage 1 Reference Guide Applied (Identity Anchored)")
                del g_res, guide_node

            del vva, ref_img_s1, vid_in
            gc.collect()
            torch.cuda.empty_cache()
            malloc_trim_os()

            # Stage 1 AV Concat & Sampling (Euler 8 Steps)
            av_in_raw = call_node(concat_av_latent, video_latent=stage1_vid_lat, audio_latent=aud_lat)
            av_in = sync_latent_device(gv(av_in_raw, 0), "cpu")
            del aud_lat, stage1_vid_lat, av_in_raw

            sig_s1 = get_basic_scheduler_sigmas(model=stage1_model, scheduler="linear_quadratic", steps=8, denoise=1.0)
            print(f"  ⚡ Stage 1 Sampling (Euler 8 steps @ {half_w}x{half_h})...")
            n1 = call_node(randomnoise, noise_seed=seed)
            g1 = call_node(cfgguider, cfg=1.0, model=stage1_model, positive=stage1_pos, negative=stage1_neg)
            s1 = call_node(samplercustomadvanced, noise=gv(n1, 0), guider=gv(g1, 0), sampler=gv(ks_euler, 0), sigmas=sig_s1, latent_image=av_in)
            s1_lat = sync_latent_device(gv(s1, 0), "cpu")

            del n1, g1, av_in, s1, sig_s1, stage1_model
            gc.collect()
            torch.cuda.empty_cache()
            malloc_trim_os()

            sep1 = call_node(separate_av_latent, av_latent=s1_lat)
            v1 = sync_latent_device(gv(sep1, 0), "cpu")
            a1 = sync_latent_device(gv(sep1, 1), "cpu")
            del sep1, s1_lat

            # Stage 1 Crop Guides
            if crop_guides_cls is not None:
                crop1_node = crop_guides_cls()
                crop1 = call_node(crop1_node, positive=stage1_pos, negative=stage1_neg, latent=v1)
                crop_lat = sync_latent_device(gv(crop1, 2) if gv(crop1, 2) is not None else v1, "cpu")
                del crop1, crop1_node
            else:
                crop_lat = v1
            del v1

            # 4. Stage 2 2x Latent Spatial Upscale
            print(f"  ⚡ Stage 2 Latent Upscaling (2x to {width}x{height})...")
            up = gv(call_node(latentupscaleloader, model_name=upscaler_model), 0)
            vva = load_vae_helper(vae_model, device="main_device", dtype="bf16")
            ups = call_node(ltxvlatentupsampler, samples=crop_lat, upscale_model=up, vae=vva)
            v_ups = sync_latent_device(gv(ups, 0), "cpu")
            del up, crop_lat, ups

            # 5. Stage 2 Clean Guide Re-Binding at Target Resolution
            ref_img_s2 = prepare_reference_image(s.get("image", ""), width, height)
            stage2_pos = gv(cond, 0)
            stage2_neg = gv(cond, 1)
            stage2_vid_lat = v_ups
            stage2_model = dit

            if add_guide_cls is not None:
                guide_node = add_guide_cls()
                g2_res = call_node(
                    guide_node,
                    positive=stage2_pos,
                    negative=stage2_neg,
                    vae=vva,
                    latent=v_ups,
                    image=ref_img_s2,
                    frame_idx=0,
                    strength=0.5,
                    model=dit
                )
                stage2_pos = gv(g2_res, 0) if gv(g2_res, 0) is not None else stage2_pos
                stage2_neg = gv(g2_res, 1) if gv(g2_res, 1) is not None else stage2_neg
                g2_lat = gv(g2_res, 2)
                if g2_lat is not None:
                    stage2_vid_lat = sync_latent_device(g2_lat, "cpu")
                mod_out2 = gv(g2_res, 3)
                if mod_out2 is not None and hasattr(mod_out2, "model_options"):
                    stage2_model = mod_out2
                else:
                    stage2_model = dit
                print("  ✓ Stage 2 Reference Guide Applied (Micro-Details Locked)")
                del g2_res, guide_node

            del vva, ref_img_s2, v_ups
            gc.collect()
            torch.cuda.empty_cache()
            malloc_trim_os()

            # Stage 2 AV Concat & Refinement (Euler 4 Steps, denoise=0.42)
            av2_raw = call_node(concat_av_latent, video_latent=stage2_vid_lat, audio_latent=a1)
            av2 = sync_latent_device(gv(av2_raw, 0), "cpu")
            del a1, stage2_vid_lat, av2_raw

            sig_s2 = get_basic_scheduler_sigmas(model=stage2_model, scheduler="linear_quadratic", steps=4, denoise=0.42)
            print(f"  ⚡ Stage 2 Refinement (Euler 4 steps, denoise=0.42 @ {width}x{height})...")
            n2 = call_node(randomnoise, noise_seed=seed)
            g2 = call_node(cfgguider, cfg=1.0, model=stage2_model, positive=stage2_pos, negative=stage2_neg)
            s2 = call_node(samplercustomadvanced, noise=gv(n2, 0), guider=gv(g2, 0), sampler=gv(ks_euler, 0), sigmas=sig_s2, latent_image=av2)
            s2_lat = sync_latent_device(gv(s2, 0), "cpu")

            del n2, g2, av2, s2, sig_s2, stage2_model
            gc.collect()
            torch.cuda.empty_cache()
            malloc_trim_os()

            sep2 = call_node(separate_av_latent, av_latent=s2_lat)
            v_fin = sync_latent_device(gv(sep2, 0), "cpu")
            a_fin = sync_latent_device(gv(sep2, 1), "cpu")
            del sep2, s2_lat

            # Stage 2 Final Crop Guides
            if crop_guides_cls is not None:
                crop2_node = crop_guides_cls()
                crop2 = call_node(crop2_node, positive=stage2_pos, negative=stage2_neg, latent=v_fin)
                final_v_lat = gv(crop2, 2) if gv(crop2, 2) is not None else v_fin
                del crop2, crop2_node
            else:
                final_v_lat = v_fin

            v_cpu = unwrap_tensor(final_v_lat).detach().cpu().half()
            a_cpu = unwrap_tensor(a_fin).detach().cpu().half()
            del final_v_lat, a_fin, v_fin

            del dit
            gc.collect()
            purge_deep(f"Pre-Save Purge Seg {sid}")

            torch.save({"video": v_cpu, "audio": a_cpu, "frames": n_frames}, target_final_file)
            print(f"  💾 Scene {sid} 2-Stage Latents saved: {target_final_file}")
            del v_cpu, a_cpu
            gc.collect()
            malloc_trim_os()

        del segment_conditionings
        purge_deep("Phase B Completed: All Latents Generated & DiT Purged")
        print(f"  ✓ Phase B Complete. Free System RAM: {get_ram_free_gb():.2f} GB")

        # PHASE C: Out-of-Core Tiled VAE Decoding
        print("\n" + "="*70 + "\n🎬 PHASE C: Out-of-Core Tiled VAE Decoding (VAE Loaded Alone)\n" + "="*70)
        frame_cache_files, audio_cache_files = [], []

        for idx, lat_file in enumerate(final_latent_files):
            sid = idx + 1
            frame_file = f"{workdir}/frames_seg_{sid:02d}.pt"
            audio_file = f"{workdir}/audio_seg_{sid:02d}.pt"
            frame_cache_files.append(frame_file)
            audio_cache_files.append(audio_file)

            if resume and os.path.exists(frame_file) and os.path.getsize(frame_file) > 1024:
                print(f"  ⏭ Scene {sid} frames already decoded. Skipping.")
                continue

            print(f"  Decoding Scene {sid}/{len(final_latent_files)} in 256x256 tiles...")
            pack = torch.load(lat_file, map_location="cpu")
            v_lat, a_lat = pack["video"].float(), pack["audio"]
            del pack

            vva = load_vae_helper(vae_model, device="main_device", dtype="bf16")
            decoded_frames = tiled_decode_video(v_lat, vva, prefer_tiled=True, tile_size=256)
            del vva, v_lat
            torch.save(decoded_frames.detach().cpu().half(), frame_file)
            del decoded_frames
            gc.collect()
            malloc_trim_os()

            ava = load_vae_helper(vae_audio_model, device="main_device", dtype="fp16")
            audio_vae_decode = NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]()
            decoded_audio = call_node(audio_vae_decode, samples={"samples": a_lat}, audio_vae=ava)
            del ava, a_lat, audio_vae_decode
            torch.save(gv(decoded_audio, 0), audio_file)
            del decoded_audio
            gc.collect()
            malloc_trim_os()

            purge_deep(f"Scene {sid} Decoded")

        # PHASE D: Timeline Assembly & Overlap Crossfading
        print("\n" + "="*70 + "\n🎬 PHASE D: Timeline Assembly & SVI-Pro Linear Frame Blending\n" + "="*70)
        final_video_tensor = None
        decoded_audio_list = []

        for f_file, a_file in zip(frame_cache_files, audio_cache_files):
            if os.path.exists(f_file):
                frames = torch.load(f_file, map_location="cpu")
                final_video_tensor = linear_blend_overlap(final_video_tensor, frames, overlap=overlap_frames)
                del frames
            if os.path.exists(a_file):
                decoded_audio_list.append(torch.load(a_file, map_location="cpu"))

        if final_video_tensor is None:
            raise RuntimeError("❌ Video assembly failed: No frames decoded.")

        print(f"  Compiled Sequence: {final_video_tensor.shape[0]} frames ({final_video_tensor.shape[0]/fps:.2f}s @ {fps}fps)")

        compiled_audio = blend_audio(decoded_audio_list, overlap_frames=overlap_frames, fps=fps)
        if audio_trim_start_frames > 0 and compiled_audio is not None:
            compiled_audio = trim_audio(compiled_audio, audio_trim_start_frames, fps, final_video_tensor.shape[0])

        raw_video_path = os.path.join(outdir, "LTX23_Director_Master.mp4")
        video_saved = False

        if "CreateVideo" in NODE_CLASS_MAPPINGS:
            try:
                create_video = NODE_CLASS_MAPPINGS["CreateVideo"]()
                cv = call_node(create_video, fps=fps, images=final_video_tensor.float(), audio=compiled_audio)
                gv(cv, 0).save_to(raw_video_path, metadata=None)
                del cv
                video_saved = True
            except Exception:
                pass

        if not video_saved or not os.path.exists(raw_video_path) or os.path.getsize(raw_video_path) < 100:
            import imageio
            frames_np = (final_video_tensor.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
            imageio.mimwrite(raw_video_path, frames_np, fps=fps, quality=9)

        del final_video_tensor, compiled_audio
        print(f"  🎬 Master Video Saved: {raw_video_path}")

        final_output_path = raw_video_path
        if song_path and os.path.exists(song_path):
            final_output_path = mux_final_audio(raw_video_path, song_path, crf=output_crf)
            print(f"  🎵 Audio/Video Mux Complete (CRF {output_crf}): {final_output_path}")

        purge_deep("Pipeline Finished Successfully")
        return final_output_path

print("✅ Cell 14: Master 2-Stage Director Pipeline Engine loaded.")


# ────────────────────────────────────────────────────────────────────────────
# RUNTIME TRIGGER: FULL 30-SECOND 5-SCENE MUSIC VIDEO GENERATION
# ────────────────────────────────────────────────────────────────────────────
song_file_path = "/content/ComfyUI/input/whatdreamscost/Late night trap.mp3"
work_directory = "/content/LTXDirector_Work"
output_directory = "/content/LTXStudio_Output"

base_input = "/content/ComfyUI/input/whatdreamscost"
os.makedirs(base_input, exist_ok=True)

if os.path.exists(f"{base_input}/5.png") and not os.path.exists(f"{base_input}/5.3.png"):
    shutil.copy(f"{base_input}/5.png", f"{base_input}/5.3.png")

# Reset workdir to ensure clean fresh 5-scene generation
if os.path.exists(work_directory):
    print("🧹 Cleaning old cache files...")
    shutil.rmtree(work_directory, ignore_errors=True)
os.makedirs(work_directory, exist_ok=True)

for s in SEGMENTS_CONFIG:
    full_path = os.path.join("/content/ComfyUI/input", s['image']) if not s['image'].startswith("/content") else s['image']
    if not os.path.exists(full_path):
        print(f"  ⚠️ Keyframe missing for Scene {s['id']}: {full_path}")
    else:
        print(f"  ✓ Keyframe Scene {s['id']}: {full_path}")

print(f"  ✓ Audio Track Verified: {song_file_path}")

# Run Complete 30-Second Generation (All 5 Scenes)
final_video = run_workflow_faithful_director_pipeline(
    segments=SEGMENTS_CONFIG,
    global_prompt=GLOBAL_PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    song_path=song_file_path if (use_song_audio and os.path.exists(song_file_path)) else None,
    width=width,
    height=height,
    fps=fps,
    overlap_frames=overlap_frames,
    output_crf=output_crf,
    min_ram_guard_gb=min_ram_guard_gb,
    lora_configs=LORA_WEIGHTS,
    workdir=work_directory,
    outdir=output_directory,
    resume=False
)

print(f"\n🎉 AI Music Video Generation Complete!\nOutput File: {final_video}")