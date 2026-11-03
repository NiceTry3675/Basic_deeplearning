#!/bin/bash
# 사용법: ./runq.sh jobs_file N
# jobs_file 각 줄: "<name> <script.py> <인자...>"
#   → results/runs/<name>.json 또는 <name>_s0.json 존재 시 skip
cd "$(dirname "$0")"
JOBS=$1; N=${2:-2}
grep -v '^\s*#' "$JOBS" | grep -v '^\s*$' | xargs -P "$N" -L 1 bash -c '
  name=$0; script=$1; shift
  if [ -f ../results/runs/$name.json ] || [ -f ../results/runs/${name}_s0.json ]; then echo "skip $name"; exit 0; fi
  ../../.venv/bin/python $script --name $name "$@" > ../results/logs/$name.log 2>&1
  tail -1 ../results/logs/$name.log'
