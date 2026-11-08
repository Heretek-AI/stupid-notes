# ComfyUI on Proxmox VE — Modern ROCm 10+ & RDNA3 Hardware Acceleration

A production-tested guide for running **ComfyUI** inside an unprivileged Proxmox VE LXC container on AMD RDNA 3 / RDNA 3.5 hardware (e.g., Radeon RX 7900 XTX `gfx1100` over OCuLink / PCIe alongside AMD Strix Halo `gfx1151`).

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

## 3. Deploying ComfyUI & RDNA3 Accelerators

### Clone ComfyUI & Core Requirements
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

### Accelerator B: `chelokot/flash-attention-rdna3`
Standard FlashAttention wheels lack RDNA3 ISA binary images (`hipErrorInvalidImage`). [flash-attention-rdna3](https://github.com/chelokot/flash-attention-rdna3) delivers a native **Triton FlashAttention-2** kernel specifically compiled for `gfx1100`:
```bash
git clone https://github.com/chelokot/flash-attention-rdna3.git /opt/ComfyUI/app/custom_nodes/RDNA3-Flash-Attention
/opt/ComfyUI/venv/bin/pip install --no-deps -e /opt/ComfyUI/app/custom_nodes/RDNA3-Flash-Attention
```

---

## 5. Systemd Service Configuration

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
Environment="PATH=/opt/ComfyUI/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/opt/ComfyUI/venv/bin/python3 /opt/ComfyUI/app/main.py --listen 0.0.0.0 --port 8188 --highvram
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

> [!NOTE]
> In modern ComfyUI argparse, `--gpu-only` and `--highvram` are mutually exclusive. With 24 GB discrete VRAM, use `--highvram`.

Enable and start the service:
```bash
systemctl daemon-reload
systemctl enable --now comfyui.service
```

---

## 6. Verification & Health Probes

Query the ComfyUI API endpoint from any host on the network:
```bash
curl -s http://<CONTAINER_IP>:8188/system_stats | jq .
```

**Expected JSON Response:**
```json
{
  "system": {
    "os": "linux",
    "comfyui_version": "0.44.0",
    "pytorch_version": "2.16.0a0+rocm10.3.0a20261107",
    "argv": [
      "/opt/ComfyUI/app/main.py",
      "--listen",
      "0.0.0.0",
      "--port",
      "8188",
      "--highvram"
    ]
  },
  "devices": [
    {
      "name": "cuda:0 AMD Radeon RX 7900 XTX : native",
      "type": "cuda",
      "index": 0,
      "vram_total": 25753026560,
      "vram_free": 25553797120
    }
  ]
}
```

Check the node loader logs in `journalctl -u comfyui -n 30 --no-pager`:
```text
Import times for custom nodes:
[INFO]    0.0 seconds: /opt/ComfyUI/app/custom_nodes/websocket_image_save.py
[INFO]    0.0 seconds: /opt/ComfyUI/app/custom_nodes/RDNA3-Flash-Attention
[INFO]    0.1 seconds: /opt/ComfyUI/app/custom_nodes/ComfyUI-FeatherOps
[INFO] To see the GUI go to: http://0.0.0.0:8188
```

Both FeatherOps and RDNA3 FlashAttention load instantly, providing optimized generation passes on the dedicated 24 GB VRAM pool.
