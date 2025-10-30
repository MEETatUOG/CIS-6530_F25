#!/bin/bash
set -euo pipefail

GHIDRA_HEADLESS="/opt/ghidra/support/analyzeHeadless"
PROJECT_DIR="/tmp/ghidra"
SCRIPT_PATH="/home/vmd/Documents/submission_3/scripts"
SAMPLE_DIR="/home/vmd/Documents/submission_3/samples/"
OUTPUT_DIR="/home/vmd/Documents/submission_3/outputs"

mkdir -p "$PROJECT_DIR" "$SCRIPT_PATH" "$SAMPLE_DIR" "$OUTPUT_DIR"
chmod 700 "$SAMPLE_DIR"

for sample in "$SAMPLE_DIR"/*.bin; do
  [ -e "$sample" ] || { echo "No .bin files in $SAMPLE_DIR"; break; }
  base=$(basename "$sample")
  name="${base%.*}"
  proj_name="proj_${name}_$RANDOM"
  echo "Processing: $sample -> $OUTPUT_DIR/results"
  "$GHIDRA_HEADLESS" "$PROJECT_DIR" "$proj_name" \
    -import "$sample" \
    -postScript dump_opcodes.py "$OUTPUT_DIR" "$OUTPUT_DIR/results" \
    -scriptPath "$SCRIPT_PATH" \
    -noanalysis \
    -deleteProject
done

echo "Done. Check $OUTPUT_DIR/results for CSV files and $OUTPUT_DIR for logs."