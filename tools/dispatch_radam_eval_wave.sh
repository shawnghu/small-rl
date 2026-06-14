#!/bin/bash
# Dispatch forget-scale evals for newly finished radam runs (idempotent).
# Compares wandb-finished runs against /tmp/dispatched.txt; always absolute paths.
set -u
REPO=/workspace/small-rl
cd /tmp && WANDB_API_KEY=$WANDB_API_KEY timeout 120 $REPO/.venv/bin/python /tmp/finished_runs.py 2>&1 | grep -v "Loaded credentials" | tr ',' '\n' > /tmp/finished_now.txt
NEW=$(comm -23 <(sort /tmp/finished_now.txt) <(sort /tmp/dispatched.txt) | paste -sd,)
if [ -z "$NEW" ]; then echo "no new finished runs"; exit 0; fi
echo "dispatching: $NEW"
cp /tmp/finished_now.txt /tmp/dispatched.txt
cd $REPO && nohup $REPO/.venv/bin/python -m modal run --detach $REPO/tools/modal_train_gr.py::launch_modal_eval_canonical_radam --only "$NEW" > /tmp/radam_eval_wave_latest.log 2>&1 &
sleep 20
grep -E "dispatching|Traceback|Error" /tmp/radam_eval_wave_latest.log | head -3
