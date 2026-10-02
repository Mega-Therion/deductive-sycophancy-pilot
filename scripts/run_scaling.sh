#!/bin/sh
# Probe a range of Qwen3 sizes on one GPU. Used for the Hugging Face Jobs run.
set -e
pip install -q "transformers>=4.51" scikit-learn accelerate
for m in Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-14B Qwen/Qwen3-32B; do
  echo "=== $m"
  python -m dsp.probe --model "$m" --dtype bfloat16 --device cuda
  rm -f runs/*/activations.npz
done
echo "=== ALL RESULTS JSON"
for f in runs/*/probe_results.json; do echo "--- $f"; cat "$f"; echo; done
