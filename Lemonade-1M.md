# Lemonade / llama.cpp — Qwen 27B 1M Context on AMD Strix Halo (Vulkan)

### Context & Problem
Running large models (like `Qwen3.8-27B-UD-Q8_K_XL`) at ultra-long context windows (1M tokens / `1,048,576`) on a heterogeneous AMD system (discrete RX 7900 XTX 24GB + Strix Halo 128GB Unified APU) triggers several failure modes if unmanaged:
1. **Multi-GPU KV Cache Overflow:** Layer-splitting (`--split-mode layer`) shards KV cache across devices; a 1M context at Q4 needs ~64 GB KV cache, instantly overflowing the 24 GB discrete card and hard-locking the kernel memory subsystem.
2. **ROCm Missing Fatbins (`invalid kernel file`):** Prebuilt ROCm fatbins often lack `gfx1151` targets for the Strix Halo APU, failing on forward compute dispatches (`GET_ROWS`).
3. **MTP Draft Collision:** Auto-injected speculative decoding (`--spec-type draft-mtp`) asserts on unallocated draft buffers with custom contexts, causing server crashes.
4. **Context Capping:** `llama-server` automatically clamps requested context down to `n_ctx_train` (256k) unless overridden in metadata.

---

### The Solution: Isolate to Strix Halo Unified Memory with Vulkan

Using **Vulkan** (`Mesa RADV`) targets the Strix Halo APU (`Vulkan1` / Radeon 8060S) directly via runtime SPIR-V JIT. This unlocks the entire 128 GB unified memory pool for both the 30.2 GB Q8 model weights and the ~64 GB 1M KV cache on a single, coherent memory bus.

---

### Working Configuration (`recipe_options.json`)

```json
  "user.Qwen3.8-27B": {
    "ctx_size": 1048576,
    "llamacpp_args": "-ngl 99 -np 1 -fa on -ctk q4_0 -ctv q4_0 -b 2048 -ub 512 --override-kv qwen35.context_length=int:1048576 --rope-scaling yarn --rope-scale 4.0 --yarn-orig-ctx 262144 --spec-type none --chat-template-kwargs '{\"preserve_thinking\": true}'",
    "llamacpp_backend": "vulkan",
    "llamacpp_device": "Vulkan1",
    "merge_args": true,
    "pinned": false
  }
