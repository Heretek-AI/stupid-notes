# ComfyUI on Proxmox VE — Modern ROCm 10+ & RDNA3 Hardware Acceleration

A production-tested guide for running **ComfyUI** inside an unprivileged Proxmox VE LXC container on AMD RDNA 3 / RDNA 3.5 hardware (e.g., Sapphire PULSE Radeon RX 7900 XTX `gfx1100` over OCuLink / PCIe alongside AMD Strix Halo `gfx1151`).

This setup enforces **modern ROCm 10+ standards** (avoiding legacy ROCm 6/7 fallbacks) and achieves hardware-level performance using **FeatherOps** (fused FP8/INT8 WMMA & Conv2d) and **RDNA3 FlashAttention-2**.

---

## 1. Architecture & Hardware Passthrough

When running discrete GPUs alongside AMD APUs (such as Minisforum MS-S1 MAX with Strix Halo + eGPU), the host kernel assigns DRM render nodes sequentially:
- `/dev/dri/renderD128`: Discrete GPU (e.g. RX 7900 XTX Navi 31 `gfx1100`)
- `/dev/dri/renderD129`: Integrated APU (e.g. Radeon 8060S Strix Halo `gfx1151`)
- `/dev/kfd`: AMD Kernel Fusion Driver (shared across all ROCm devices)

### The Proxmox KFD Aperture Trap
The ROCm KFD runtime discovers memory apertures using `AMDKFD_IOC_GET_PROCESS_APERTURES_NEW`. If the container only binds `renderD128` while omitting other nodes listed in host topology, `hsa_init` fails with:
```text
AMDKFD_IOC_SET_MEMORY_POLICY = -1 ESRCH
```
**Fix:** Pass both host DRM render nodes into the container so KFD can successfully map memory topologies, then strictly isolate the target GPU at the container runtime using `ROCR_VISIBLE_DEVICES=0`.

### Proxmox Container Configuration (`/etc/pve/lxc/<VMID>.conf`)
```ini
arch: amd64
cores: 8
memory: 16384
swap: 4096
ostype: ubuntu
net0: name=eth0,bridge=vmbr0,firewall=1,ip=dhcp,type=veth
rootfs: local-lvm:subvol-<VMID>-disk-0,size=80G

# AMD ROCm / DRM Device Passthrough
dev0: /dev/dri/renderD128,gid=992
dev1: /dev/dri/card1,gid=44
dev2: /dev/kfd,gid=992
dev3: /dev/dri/renderD129,gid=992

# Privileges and Memory Lock
lxc.prlimit.memlock: unlimited
lxc.cgroup2.devices.allow: a
lxc.cap.drop: 
```

*Note: Ensure the `render` group inside the container matches host GID 992 (`groupmod -g 992 render`).*

---

## 2. Installing Modern ROCm 10+ PyTorch Stack

AMD hosts official nightly ROCm 10+ wheels under `https://nightly.repo.amd.com/`.

> [!WARNING]
> Do **NOT** run generic `pip install --pre --index-url https://nightly.repo.amd.com/rocm/...`. Pip will backtrack through 50+ historical daily snapshots spanning months, resulting in dependency loops and hours of wasted downloads.
> Always pin the exact coordinated daily build timestamp across all packages.

### Step 1: Base Toolchain (in LXC)
```bash
apt-get update && apt-get install -y \
  build-essential git curl libgl1 libglib2.0-0 \
  python3 python3-pip python3-venv python3-dev
```

### Step 2: Create Python Virtual Environment
```bash
mkdir -p /opt/ComfyUI
python3 -m venv /opt/ComfyUI/venv
/opt/ComfyUI/venv/bin/pip install --upgrade pip setuptools wheel
```

### Step 3: Install ROCm Core SDK
ROCm nightly provides the core toolchain wheels and headers in `rocm/core/whl-next/`:
```bash
/opt/ComfyUI/venv/bin/pip install --no-cache-dir \
  --extra-index-url https://nightly.repo.amd.com/rocm/core/whl-next/ \
  https://nightly.repo.amd.com/rocm/core/whl-next/rocm/rocm-10.3.0a20261107.tar.gz \
  rocm-sdk-device-gfx1100==10.3.0a20261107 \
  rocm-sdk-devel==10.3.0a20261107
```
*(Installing `rocm-sdk-devel` provides `clang-24` and `hipcc` within the venv, required for building custom C++/HIP extensions).*

### Step 4: Install Pinned ROCm 10.3 PyTorch & Device Drivers
```bash
/opt/ComfyUI/venv/bin/pip install --no-cache-dir \
  --extra-index-url https://nightly.repo.amd.com/rocm/pytorch/whl-next/ \
  --extra-index-url https://nightly.repo.amd.com/rocm/core/whl-next/ \
  torch==2.16.0a0+rocm10.3.0a20261107 \
  torchvision==0.30.0a0+rocm10.3.0a20261107 \
  triton==3.9.0+gitb7ddcf86.rocm10.3.0a20261107 \
  amd-torch-device-gfx1100==2.16.0a0+rocm10.3.0a20261107 \
  amd-torchvision-device-gfx1100==0.30.0a0+rocm10.3.0a20261107
```

### Step 5: Verify PyTorch ROCm Detection
```bash
ROCR_VISIBLE_DEVICES=0 /opt/ComfyUI/venv/bin/python3 -c "
import torch
print('CUDA Available:', torch.cuda.is_available())
print('Device Count:', torch.cuda.device_count())
print('Device Name:', torch.cuda.get_device_name(0))
print('Arch:', torch.cuda.get_device_properties(0).gcnArchName)
"
```
**Expected Output:**
```text
CUDA Available: True
Device Count: 1
Device Name: AMD Radeon RX 7900 XTX
Arch: gfx1100
```

---

## 3. Deploying ComfyUI & Core Dependencies

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI/app
cd /opt/ComfyUI/app

# Install dependencies avoiding torch overwrite
/opt/ComfyUI/venv/bin/pip install -r requirements.txt --no-deps
/opt/ComfyUI/venv/bin/pip install \
  comfyui-frontend-package comfyui-workflow-templates comfyui-embedded-docs \
  torchsde einops transformers tokenizers sentencepiece safetensors aiohttp \
  yarl pyyaml scipy tqdm psutil alembic SQLAlchemy av comfy-aimdo comfy-kitchen \
  requests simpleeval blake3 kornia spandrel pydantic pydantic-settings PyOpenGL comfy-angle
```

---

## 4. RDNA3 Acceleration Kernels

### Accelerator A: `woct0rdho/feather-ops`
[FeatherOps](https://github.com/woct0rdho/feather-ops) provides specialized RDNA3/3.5 HIP kernels:
- **FP16-FP8 WMMA matmuls**: Loads FP8 directly into LDS/VGPR and upcasts to FP16 before WMMA, maximizing LDS occupancy.
- **INT8 ConvRot matmuls**: Fused Hadamard rotation for quantized INT8 models.
- **Conv2d implicit GEMM**: Optimized 3x3 convolutions.

#### Building Native HIP Extensions:
```bash
git clone https://github.com/woct0rdho/feather-ops.git /opt/ComfyUI/feather-ops
cd /opt/ComfyUI/feather-ops

# Build using the in-venv ROCm devel clang-24 toolchain
/opt/ComfyUI/venv/bin/pip install --no-build-isolation -e .
```
*(Build executes `hipcc` and `clang-24` with Link-Time Optimization (`ld.lld -O3 --lto-CGO3`) for `gfx1100` and `gfx1151`).*

#### Installing ComfyUI FeatherOps Node & Quantization Helpers:
```bash
git clone https://github.com/woct0rdho/ComfyUI-FeatherOps.git /opt/ComfyUI/app/custom_nodes/ComfyUI-FeatherOps
git clone https://github.com/Comfy-Org/comfy-quants.git /opt/ComfyUI/comfy-quants
/opt/ComfyUI/venv/bin/pip install --no-deps -e /opt/ComfyUI/comfy-quants
```

### Accelerator B: `chelokot/flash-attention-rdna3` (Triton FlashAttention-2)
Standard FlashAttention wheels lack RDNA3 ISA binary images (`hipErrorInvalidImage`). [flash-attention-rdna3](https://github.com/chelokot/flash-attention-rdna3) delivers a native **Triton FlashAttention-2** kernel specifically compiled for `gfx1100`:
```bash
git clone https://github.com/chelokot/flash-attention-rdna3.git /opt/ComfyUI/app/custom_nodes/RDNA3-Flash-Attention
/opt/ComfyUI/venv/bin/pip install --no-deps -e /opt/ComfyUI/app/custom_nodes/RDNA3-Flash-Attention
```

#### Enabling the Global FlashAttention Hook (Crucial):
By default, `flash-attention-rdna3` only defines a custom node (`ApplyRDNA3FlashAttention`) that must be manually inserted into every workflow. Without it, stock ComfyUI workflows fall back to unoptimized sub-quadratic attention (`[INFO] Using sub quadratic optimization for attention`).

To enable Triton FlashAttention-2 globally for **all** workflows automatically, update `/opt/ComfyUI/app/custom_nodes/RDNA3-Flash-Attention/__init__.py`:
```python
from pathlib import Path
import sys

repository_root = Path(__file__).resolve().parent
from .fa_rdna3.comfyui import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .fa_rdna3.sdpa import enable_rdna3_flash_attention

try:
    enable_rdna3_flash_attention()
    print("[RDNA3-Flash-Attention] Global SDPA hook enabled for RDNA3 (gfx1100). Standard workflows now use Triton FlashAttention-2 automatically!")
except Exception as e:
    print(f"[RDNA3-Flash-Attention] Warning: could not enable global SDPA hook: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

---

## 5. Systemd Service & VRAM Management

Create `/etc/systemd/system/comfyui.service`:
```ini
[Unit]
Description=ComfyUI AI Image Generation Service (RX 7900 XTX gfx1100 ROCm 10.3)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/ComfyUI/app
Environment="ROCR_VISIBLE_DEVICES=0"
Environment="HIP_VISIBLE_DEVICES=0"
Environment="HSA_OVERRIDE_GFX_VERSION=11.0.0"
Environment="PYTORCH_ROCM_ARCH=gfx1100"
Environment="TRITON_DEFAULT_FP_FUSION=1"
Environment="PATH=/opt/ComfyUI/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/opt/ComfyUI/venv/bin/python3 /opt/ComfyUI/app/main.py --listen 0.0.0.0 --port 8188 --highvram --fp8_e4m3fn-text-enc --enable-manager
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

### Why `--fp8_e4m3fn-text-enc` is Mandatory for eGPUs
Modern high-parameter models like Z-Image-Turbo / Lumina2 pair an 11.7 GB diffusion UNet with a 7.7 GB text encoder (`qwen_3_4b` in FP16).
1. **The PCIe Swap Penalty:** In full precision, $7.7\text{ GB} + 11.7\text{ GB} = 19.4\text{ GB}$. ComfyUI's memory manager detects that $19.4\text{ GB} + \text{activation buffers}$ exceeds the card's headroom, so it **offloads the text encoder to CPU RAM before loading the UNet on every single prompt**.
2. Over an OCuLink eGPU connection running at PCIe 4.0 x1 (~1.97 GB/s), shuttling 19.4 GB across the cable burns **11 seconds of pure PCIe wait time**!
3. **The Solution:** `--fp8_e4m3fn-text-enc` reduces Qwen 3.4B to **~3.8 GB** with zero perceptual loss. Total VRAM footprint drops to **~15.5 GB**, allowing ComfyUI to keep **both models permanently pinned in VRAM**. Subsequent prompts experience **0.00 seconds** of transfer latency.

Enable and start the service:
```bash
systemctl daemon-reload
systemctl enable --now comfyui.service
```

---

## 6. ComfyUI-Manager Integration (Modern Native Stack)

In modern ComfyUI (`0.44+`), ComfyUI-Manager is no longer required to be installed as an unmanaged `git clone` inside `custom_nodes/ComfyUI-Manager`. It has been integrated natively into core as a Python package (`comfyui-manager`), orchestrated via core startup hooks and native frontend panels ([ComfyUI Manager Install Docs](https://docs.comfy.org/manager/install)).

### Step 1: Install Manager Dependencies
ComfyUI core pins the matching manager package in `manager_requirements.txt`:
```bash
/opt/ComfyUI/venv/bin/pip install -r /opt/ComfyUI/app/manager_requirements.txt
```
*(Installs `comfyui_manager>=4.2.2`, `GitPython`, `PyGithub`, `uv`, and cryptographic verification libraries without touching ROCm or PyTorch).*

### Step 2: Startup Flags
ComfyUI core supports three dedicated CLI flags for manager orchestration:
- `--enable-manager`: Enables the ComfyUI-Manager backend and frontend UI.
- `--enable-manager-legacy-ui`: Retains the legacy modal manager interface while running on modern core.
- `--disable-manager-ui`: Runs background manager features/endpoints while suppressing frontend UI components.

Add `--enable-manager` and `Environment="COMFYUI_PATH=/opt/ComfyUI/app"` to `/etc/systemd/system/comfyui.service`.

### Step 3: Headless / LAN Server Security Configuration
When ComfyUI is hosted on a Proxmox LXC container bound to `--listen 0.0.0.0`, all web browser requests originate from remote LAN IPs (e.g., `192.168.31.x`). Under default settings, ComfyUI-Manager detects non-loopback access and locks down custom node installation and git operations.

To configure remote administrative access securely for your LAN, create `/opt/ComfyUI/app/user/__manager/config.ini`:
```ini
[default]
channel_url = https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main
share_option = all
bypass_ssl = False
file_logging = True
update_policy = stable-comfyui
network_mode = personal_cloud
security_level = normal
db_mode = cache
allow_git_url_install = True
allow_pip_install = True
```

#### Key Policy Directives:
- **`network_mode = personal_cloud`**: Informs the manager gatekeeper that remote connections across the local network are from trusted operator machines, unlocking node package installations.
- **`security_level`**:
  - `normal` (**Standard**, recommended): Gated validation against official registry.
  - `normal-` (**Relaxed**): Allows secondary registries and unreviewed releases.
  - `weak` (**Permissive**): Disables package verification guards.
- **`allow_git_url_install = True` & `allow_pip_install = True`**: Enables direct custom node repository URLs and PyPI package installation from the UI.

### Step 4: Verification Probes
Verify that ComfyUI-Manager is active and serving endpoints:
```bash
# Query active queue and manager status:
curl -s http://192.168.31.235:8188/v2/manager/queue/status
# Output: {"total_count": 0, "done_count": 0, "in_progress_count": 0, "pending_count": 0, "is_processing": false}

# Verify installed custom nodes via Manager API:
curl -s http://192.168.31.235:8188/v2/customnode/installed
# Output: Lists ComfyUI-FeatherOps, RDNA3-Flash-Attention with version hashes.
```

---

## 7. Field Traps & Troubleshooting

### Trap 1: `SafetensorError: Error while deserializing header: header too large`
- **Symptom:** ComfyUI fails immediately during `VAELoader` or `CLIPLoader` with:
  ```text
  safetensors._safetensors_rust.SafetensorError: Error while deserializing header: header too large
  ```
- **Root Cause:** The model file was downloaded from Hugging Face using `curl` without `-L` (follow redirects) or from a `/blob/` HTML view URL instead of `/resolve/main/`. The file on disk is an HTML error/landing page (`<!doctype html>`, ~130 KB) instead of binary weights. Safetensors reads the ASCII characters and interprets them as an impossible header byte-count integer.
- **Fix:** Download the true Git LFS binary with `curl -L` and verify key headers:
  ```bash
  curl -L -o /opt/ComfyUI/app/models/vae/ae.safetensors \
    https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors

  # Verify header integrity in Python:
  python3 -c "import safetensors.torch; print(len(safetensors.torch.load_file('/opt/ComfyUI/app/models/vae/ae.safetensors')))"
  # Output: 244 keys
  ```

### Trap 2: Argparse Flag Collision (`--gpu-only` vs `--highvram`)
- **Symptom:** ComfyUI service crashes with `status=2/INVALIDARGUMENT`:
  ```text
  main.py: error: argument --highvram: not allowed with argument --gpu-only
  ```
- **Fix:** In modern ComfyUI, these flags are mutually exclusive members of an argparse group `[--gpu-only | --highvram | --lowvram | --cpu]`. Use `--highvram --fp8_e4m3fn-text-enc`.

---

## 8. Real-World Performance Benchmarks

Tested on **Sapphire PULSE AMD Radeon RX 7900 XTX 24GB** on Minisforum MS-S1 MAX via OCuLink running **Z-Image-Turbo (11.7B Lumina2 + Qwen 3.4B)** at $1024 \times 1024$ resolution (8 steps):

| Scenario | Weight Transfer Time | 8-Step Diffusion Compute | Total Latency |
| :--- | :--- | :--- | :--- |
| **Default ComfyUI (No FP8 text-enc, sub-quadratic attention)** | 11.0 s (PCIe thrashing) | 6.6 s (0.83 s/step) | **17.63 s** |
| **Optimized (FP8 text-enc + Global Triton FlashAttention-2)** | **0.00 s** (VRAM resident) | **6.0 s** (**0.75 s/step**) | **7.05 s** |

- **Speedup:** **2.5× faster end-to-end generation** (dropping from 17.6s to 7.05s).
- **GPU Operating State:** 55°C core temperature, 14W idle power draw, link stable at PCIe 4.0 16 GT/s.
