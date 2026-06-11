#!/bin/bash
# Apply vLLM patches.
# Copies patched Python files into the installed vLLM package.
#
# Patches:
#   model_manager.py (vllm/lora/model_manager.py):
#     - Assert → warning for empty supported_lora_modules
#     - _post_create_module_hooks list for init-time adapter injection
#     - model.lora_manager = self moved before _create_lora_modules
#     - set_adapter_weights dispatch in activate_adapter
#     - Relaxed register_module type check
#
#   qwen3_5_config.py (vllm/transformers_utils/configs/qwen3_5.py):
#     - Fix list→set for ignore_keys_at_rope_validation (transformers 5.2.0 compat)
#
#   gpu_worker.py (vllm/v1/worker/gpu_worker.py):
#     - Memory-profiling assert -> clamp+warning when free memory INCREASES
#       during profiling (sibling process exiting at a sweep tail is benign;
#       previously crashed the engine boot)
#
# Usage: bash vllm_patches/apply.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"

if [ ! -d "$SITE_PACKAGES/vllm" ]; then
    echo "Error: vllm not found in $SITE_PACKAGES"
    exit 1
fi

cp "$SCRIPT_DIR/model_manager.py" "$SITE_PACKAGES/vllm/lora/model_manager.py"
echo "Patched vllm/lora/model_manager.py"

cp "$SCRIPT_DIR/qwen3_5_config.py" "$SITE_PACKAGES/vllm/transformers_utils/configs/qwen3_5.py"
echo "Patched vllm/transformers_utils/configs/qwen3_5.py"

cp "$SCRIPT_DIR/gpu_worker.py" "$SITE_PACKAGES/vllm/v1/worker/gpu_worker.py"
echo "Patched vllm/v1/worker/gpu_worker.py"

echo "All vLLM patches applied."
