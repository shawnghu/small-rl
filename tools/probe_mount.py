"""One-off probe: what does a fresh Modal container actually see in /repo?

Reuses the exact image + copy=False repo mount from modal_train_gr.py, then
prints the container's view of vllm_utils.MLP_PRESETS and the mtime/hash of the
mounted files. Definitively answers whether the mount is serving stale code.

    .venv/bin/python -m modal run tools/probe_mount.py
"""
import modal

from tools.modal_train_gr import app, image, REPO_REMOTE


@app.function(image=image, timeout=120)
def probe():
    import os
    import subprocess
    import sys

    os.chdir(REPO_REMOTE)
    sys.path.insert(0, REPO_REMOTE)

    import vllm_utils
    keys = sorted(vllm_utils.MLP_PRESETS.keys())
    has_new = {k: (k in vllm_utils.MLP_PRESETS) for k in ("m29f3", "m24f8")}

    def stamp(path):
        p = os.path.join(REPO_REMOTE, path)
        try:
            st = os.stat(p)
            h = subprocess.run(["md5sum", p], capture_output=True, text=True).stdout.split()[0]
            return f"mtime={st.st_mtime:.0f} size={st.st_size} md5={h[:12]}"
        except Exception as e:
            return f"ERR {e}"

    return {
        "vllm_utils_preset_keys": keys,
        "has_new_presets": has_new,
        "vllm_utils.py": stamp("vllm_utils.py"),
        "train.py": stamp("train.py"),
    }


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(probe.remote(), indent=2))
