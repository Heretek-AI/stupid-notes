I'm running 4 RTX PRO 6000 Blackwell WS cards (2x USB4v1 1xUSB4v2 1xOculink) using the same minipc and had the same BAR issue. Here's what I had to do:

A) disable resizable bar in BIOS (this was specifically needed for RTX PRO 6000 Blackwell WS and RTX 5090)

B) update kernel boot params. Here's the settings in my `/etc/default/grub`:

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash pcie_aspm=off pcie_port_pm=off amd_iommu=on pci=realloc thunderbolt.host_reset=false"
```

C) After booting, you need to force a rescan of the thunderbolt complex. I put this in a systemd script.

1) To find the device ID, inspect the output of `lspci -t -vv`. Look for the part with Thunderbolt 5:

```
-[0000:00]-+-00.0  Advanced Micro Devices, Inc. [AMD] Strix/Strix Halo Root Complex
           +-03.2-[65-bc]----00.0-[66-bc]--+-00.0-[67]----00.0  Intel Corporation JHL9580 Thunderbolt 5 80/120G NHI [Barlow Ridge Host 80G 2023]
```

2) remove the thunderbolt complex:

```bash
echo 1 | sudo tee /sys/bus/pci/devices/0000\:00\:03.2/remove
```

3) wait a moment and rescan:

```bash
echo 1 | sudo tee /sys/bus/pci/rescan
```
