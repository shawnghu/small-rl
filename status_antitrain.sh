#!/bin/bash
# Status snapshot for the 3 anti-train sweeps.
cd /workspace/small-rl-antitrain
echo "=== $(date '+%H:%M:%S') ==="
echo "orchestrators alive: $(pgrep -fc 'sweep.py --name matrix_gr_7envs_antitrain')/3"
tot=0; done=0
for w in wm01 wm05 wm10; do
  d=output/matrix_gr_7envs_antitrain_$w
  started=$(ls -d $d/*/ 2>/dev/null | grep -v sweep_graphs | wc -l)
  # a run is 'finished' if its train.log shows the final checkpoint save / max step reached
  fin=0
  for lg in $d/*/train.log; do
    [ -f "$lg" ] || continue
    if grep -qE "Training (completed|finished)|\[Save\] (final|checkpoint).*(1000|2000)|100%\|.*(1000/1000|2000/2000)" "$lg" 2>/dev/null; then fin=$((fin+1)); fi
  done
  maxstep=$(grep -hoE "timing @[0-9]+" $d/*/train.log 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1)
  echo "  $w: started=$started/21 finished~=$fin maxstep=${maxstep:-0}"
  done=$((done+fin)); tot=$((tot+21))
done
echo "vLLM deaths: $(grep -c 'died during startup' sweep_*.sweeplog 2>/dev/null | awk -F: '{s+=$2}END{print s}')"
echo "tracebacks in train.logs: $(grep -lE 'Traceback' output/matrix_gr_7envs_antitrain_*/*/train.log 2>/dev/null | wc -l)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  gpu/'
echo "global finished ~= $done/63"
