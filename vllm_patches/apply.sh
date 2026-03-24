#!/bin/bash
# Apply vLLM patches for MLP adapter support.
# Copies patched Python files into the installed vLLM package.
#
# Patches (all in vllm/lora/model_manager.py):
#   - Assert → warning for empty supported_lora_modules
#   - _post_create_module_hooks list for init-time adapter injection
#   - model.lora_manager = self moved before _create_lora_modules
#   - set_adapter_weights dispatch in activate_adapter
#   - Relaxed register_module type check
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
echo "Applied vLLM patches to $SITE_PACKAGES/vllm/lora/model_manager.py"
