# Minisforum MS-S1 MAX eGPU & OCuLink Architecture Guide

A comprehensive, field-tested guide for connecting and optimizing external GPUs (eGPUs) over **OCuLink (SFF-8611)** and **USB4 / Thunderbolt 5** on the **Minisforum MS-S1 MAX** (AMD Ryzen AI Max+ 395 "Strix Halo" / Radeon 8060S `gfx1151`).

---

## 1. Hardware Architecture & PCIe Topology

The Minisforum MS-S1 MAX features an internal PCIe x16 physical slot routed electrically as **PCIe 4.0 x4** (or native OCuLink port).

```
-[0000:00]-+-00.0  AMD Strix/Strix Halo Root Complex
           +-00.2  AMD Strix/Strix Halo IOMMU
           +-03.1-[65-67]----00.0-[66-67]----00.0-[67]--+-00.0  AMD Navi 31 [Radeon RX 7900 XTX] (eGPU)
           |                                            \-00.1  AMD Navi 31 HDMI/DP Audio
           +-03.2-[68-bf]----00.0-[69-bf]----...        Intel JHL9580 Thunderbolt 5 Barlow Ridge
           \-08.1-[c0]--+-00.0  AMD Strix Halo iGPU [Radeon 8060S Graphics gfx1151]
```

### The Root Port Chain:
- **`00:03.1`**: Host AMD Strix Halo GPP Bridge (`LnkCap: Speed 16GT/s, Width x4`).
- **`65:00.0`**: eGPU Upstream PCIe Switch Port (on AMD Navi cards: Navi 10 XL switch).
- **`66:00.0`**: eGPU Downstream PCIe Switch Port (`LnkSta: Speed 16GT/s, Width x16`).
- **`67:00.0`**: Discrete GPU Endpoint (Sapphire PULSE RX 7900 XTX 24GB).
- **`c0:00.0`**: On-die Strix Halo APU / iGPU (Radeon 8060S, 40 CU RDNA 3.5).

---

## 2. The Persistent PCIe Degradation Mystery (Root Causes)

If your external GPU over OCuLink persistently negotiates at **x1 width** or drops to **PCIe Gen 1 (2.5 GT/s)** speeds, here is why:

### Root Cause A: Boot-Time Power Sequencing & LTSSM Timeout (Locks at x1)
1. **The Race Condition**: During cold boot or reboot, the AMD Strix Halo SoC starts UEFI POST within milliseconds. The root port (`00:03.1`) initiates PCIe Link Training (LTSSM) at **timestamp ~0.385s**.
2. **Slow PSU / VRM Ramp**: High-power desktop eGPUs (like the RX 7900 XTX or RTX 4090/5090) and external ATX/SFX power supplies require time to charge massive input capacitors and stabilize 12V rails.
3. **The Downconfiguration Fallback**: During the LTSSM `Configuration` phase, the host sends training sets across all 4 lanes. **Lane 0** (adjacent to clock pairs on SFF-8611) locks first. If Lanes 1–3 fail to achieve frame lock before the 24ms LTSSM timeout expires, the Strix Halo controller automatically falls back to **Downconfigured Width (x1)** to avoid hanging POST.
4. **Kernel Compliance**: Once the BIOS locks `LnkSta: Width x1 (downgraded)`, the Linux kernel respects this configuration and **never autonomously retrains or widens the link to x4**.

### Root Cause B: ASPM L1 + Floating CLKREQ# (Drops from 16 GT/s to 2.5 GT/s)
1. By default, Active State Power Management (**ASPM L1**) is enabled on both `00:03.1` and `65:00.0`.
2. When the eGPU is idle, the link enters L1 sleep.
3. Upon waking, external OCuLink cables suffer clock jitter because Clock Power Management (`CLKREQ#`) is often unsynchronized across external docks. High-frequency Gen 4 (8 GHz) equalization fails, triggering a hardware safety failsafe that downclocks the bus to **PCIe Gen 1 (2.5 GT/s)** (~250 MB/s).

### Root Cause C: Signal Integrity & 8 GHz Insertion Loss Budget
- PCIe 4.0 operates at an **8 GHz Nyquist frequency** with a strict **-28 dB total insertion loss budget**.
- The MS-S1 MAX motherboard traces (~10 dB) + the GPU PCB traces (~11 dB) leave **less than 7 dB margin** for the OCuLink cable and dock combined.
- Cables longer than **50cm** or sharp bends (< 35mm radius) cause eye closure on outer lanes (lanes 2 & 3).

---

## 3. The Linux DRM Node Inversion & LXC Container Trap

When an eGPU is attached to a Strix Halo host running Proxmox VE / Linux:

### 1. DRM Render Node Inversion
Linux DRM enumerates the discrete PCIe GPU first:
- `/dev/dri/renderD128` (minor 128) -> **External eGPU** (RX 7900 XTX)
- `/dev/dri/renderD129` (minor 129) -> **Strix Halo iGPU** (Radeon 8060S)
- `/dev/dri/card0` -> Strix Halo
- `/dev/dri/card1` -> External eGPU

Existing LXCs that hardcode `dev0: /dev/dri/renderD128` will inadvertently grab the external eGPU instead of the onboard Strix Halo!

### 2. The KFD `SET_MEMORY_POLICY` ESRCH Trap
If you attempt to isolate the eGPU by passing *only* `/dev/dri/renderD129` into an LXC container while passing `/dev/kfd`, ROCm/PyTorch fails with:
```text
hsa api call failure: HSA_STATUS_ERROR_OUT_OF_RESOURCES
AMDKFD_IOC_SET_MEMORY_POLICY = -1 ESRCH (No such process)
```
- **Why**: `/dev/kfd` is a system-wide character device. When `hsa_init` discovers the topology, it attempts to acquire VM apertures on all kernel-reported nodes. When it hits Node 1 (the eGPU) and finds no corresponding `/dev/dri/renderD128` device node, KFD returns `-ESRCH`, crashing the entire ROCm runtime inside the container.
- **The Clean Solution**:
  Mount both render nodes so KFD can initialize process apertures, and pass **`ROCR_VISIBLE_DEVICES=1`** (or your Strix Halo node index) in the container's environment. PyTorch/ROCm will report `CUDA count: 1`, bind strictly to `Radeon 8060S Graphics (gfx1151)`, and **never touch or allocate memory on the eGPU**.

---

## 4. Complete Remediation & Implementation Guide

### Step 1: Update Host Kernel Boot Parameters
Edit `/etc/default/grub` on the host:
```bash
GRUB_CMDLINE_LINUX_DEFAULT="quiet iommu=pt amdgpu.gttsize=114688 ttm.pages_limit=28672000 pcie_aspm=off amdgpu.aspm=0 pci=pcie_bus_perf amdgpu.ppfeaturemask=0xffffffff"
```

| Parameter | Purpose |
| :--- | :--- |
| `pcie_aspm=off` | Disables PCIe ASPM across all root ports, stopping L1 sleep drops to 2.5 GT/s. |
| `amdgpu.aspm=0` | Prohibits the AMD GPU driver from requesting dynamic link downclocking. |
| `pci=pcie_bus_perf` | Configures Max Payload Size (MPS) and Read Request for maximum throughput. |
| `amdgpu.ppfeaturemask=0xffffffff` | Unlocks full PowerPlay control (allowing 350W PPT tuning on 7900 XTX). |
| `amdgpu.gttsize=114688` | Allocates 112GB GTT memory for the Strix Halo unified memory pool. |

Update GRUB:
```bash
update-grub
```

### Step 2: Power Sequencing (Crucial for eGPU Docks)
1. **Always switch ON the eGPU power supply (ATX/SFX) first**.
2. If using a dock like the **Minisforum DEG1**, ensure the power switch / sync jumper is in the **Always On / Direct** mode so the GPU has stable 12V rails and clock lock before the host powers up.
3. Power on the Minisforum MS-S1 MAX.

### Step 3: BIOS Settings (MS-S1 MAX AMI UEFI)
1. Press `Del` or `F7` on boot to enter BIOS.
2. Navigate to **Advanced -> AMD PBS / CBS -> PCIe Configuration**.
3. Force the PCIe/OCuLink slot link speed from **[Auto]** to **[Gen 4]**.
4. Set ASPM for the slot to **[Disabled]**.
5. *Note on Resizable BAR*:
   - For **AMD GPUs (RX 7000 series)**: Keep Resizable BAR **Enabled**. The MS-S1 MAX properly allocates a 32GB BAR (`BAR 0: current size: 32GB`), enabling full Smart Access Memory.
   - For **NVIDIA GPUs (RTX 5090 / RTX 6000 Ada / Blackwell WS)**: Disable ReBAR in BIOS if MMIO allocation conflicts occur.

### Step 4: LXC Container GPU Isolation & Config
In your Proxmox LXC config (`/etc/pve/lxc/<vmid>.conf`):
```ini
# Pass DRM nodes
dev0: /dev/dri/renderD129,gid=993
dev1: /dev/dri/card0,gid=44
dev2: /dev/kfd,gid=993
dev3: /dev/accel/accel0,gid=993

# If KFD requires render node presence for initialization, pass renderD128 as auxiliary:
dev4: /dev/dri/renderD128,gid=993
```

In the container environment file (e.g., `/opt/kyojin/kyojin.env` or `/etc/environment`):
```bash
# Pin container exclusively to Strix Halo (Node 1 in KFD)
ROCR_VISIBLE_DEVICES=1
```

---

## 5. Verification & Diagnostic Commands

### Check Negotiated PCIe Speed and Width
```bash
# Check host root port
lspci -s 00:03.1 -vv | grep -E "LnkCap:|LnkSta:"

# Check eGPU upstream switch
lspci -s 65:00.0 -vv | grep -E "LnkCap:|LnkSta:"

# Check GPU endpoint
lspci -s 67:00.0 -vv | grep -E "LnkCap:|LnkSta:"
```
*Expected Healthy Output*: `Speed 16GT/s, Width x4` on `00:03.1` and `65:00.0`.

### Check Resizable BAR Allocation
```bash
lspci -s 67:00.0 -vv | grep -A 3 "Physical Resizable BAR"
```
*Output*: `BAR 0: current size: 32GB`.

### Check Power Envelope & Thermals (7900 XTX)
```bash
# View temperatures and power
cat /sys/kernel/debug/dri/1/amdgpu_pm_info

# Maximize power cap from 303W to 350W
echo 350000000 > /sys/class/drm/card1/device/hwmon/hwmon*/power1_cap
```

### Verify Container Isolation (inside LXC)
```bash
python3 -c "import torch; print('CUDA count:', torch.cuda.device_count()); print('Device 0:', torch.cuda.get_device_name(0))"
```
*Output*:
```text
CUDA count: 1
Device 0: Radeon 8060S Graphics
```
*(Confirms container sees ONLY Strix Halo and has zero access to the eGPU).*

---

## 6. Thunderbolt 5 Complex Rescan Script (Optional USB4 / TB5 Setup)

If you are also running GPUs over the Barlow Ridge Thunderbolt 5 controller (`00:03.2`), rescan the bus after boot:

```bash
# Remove TB5 bridge
echo 1 > /sys/bus/pci/devices/0000:00:03.2/remove
sleep 1
# Rescan PCI bus
echo 1 > /sys/bus/pci/rescan
```
