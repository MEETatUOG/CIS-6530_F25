#!/bin/bash
set -euo pipefail

GHIDRA_HEADLESS="/opt/ghidra/support/analyzeHeadless"
PROJECT_DIR="/tmp/ghidra"
SCRIPT_PATH="/home/vmd/Documents/submission_3/scripts"
SAMPLE_DIR="/home/vmd/Documents/submission_3/samples"
OUTPUT_DIR="/home/vmd/Documents/submission_3/outputs"

MAX_JOBS=$(nproc)   # or set manually, e.g. 4

mkdir -p "$PROJECT_DIR" "$SCRIPT_PATH" "$SAMPLE_DIR" "$OUTPUT_DIR/results"
chmod 700 "$SAMPLE_DIR"

process_sample() {
  local sample="$1"
  local base
  base=$(basename "$sample")
  local name="${base%.*}"
  local proj_name="proj_${name}_$RANDOM"

  echo "[INFO] Processing: $sample"
  "$GHIDRA_HEADLESS" "$PROJECT_DIR" "$proj_name" \
    -import "$sample" \
    -postScript dump_opcodes_fast27.py "$OUTPUT_DIR" "$OUTPUT_DIR/results" \
    -scriptPath "$SCRIPT_PATH" \
    -noanalysis \
    -deleteProject \
    >"$OUTPUT_DIR/${name}.log" 2>&1 || echo "[WARN] Failed: $sample" >&2
}

export -f process_sample
export GHIDRA_HEADLESS PROJECT_DIR SCRIPT_PATH OUTPUT_DIR

find "$SAMPLE_DIR" -type f -name "*.bin" | \
  parallel -j "$MAX_JOBS" process_sample {}

echo "✅ Done. Check:"
echo "  - Results: $OUTPUT_DIR/results"
echo "  - Logs:    $OUTPUT_DIR/*.log"
