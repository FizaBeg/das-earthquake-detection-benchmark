#!/usr/bin/env bash
# Run a list of training jobs (one per line in $1) with 2 parallel single-thread workers; skip finished ones.
cd /home/claude/work
jobs_file=$1; out=$2
grep -v '^\s*$' "$jobs_file" | while read -r model frac seed; do
  tag="${model}_f${frac}_s${seed}"
  [ -f "$out/$tag.json" ] && continue
  echo "$model $frac $seed"
done | xargs -P 2 -L 1 bash -c 'python3 -W ignore train.py --model "$0" --frac "$1" --seed "$2" --epochs 20 --out '"$out"' >> '"$out"'/log.txt 2>&1'
