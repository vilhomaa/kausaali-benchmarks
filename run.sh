#!/usr/bin/env bash
# End-to-end cross-framework forest benchmark.
# Run from the repo root:  bash run.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"

echo "════════ Step 1 — Scala (this project): datasets + all forests ════════"
sbt -batch "runMain benchmark.GrfBenchmark"

echo ""
echo "════════ Step 2 — Python / EconML ════════"
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install -q -r "$SCRIPT_DIR/requirements.txt"
fi
"$VENV_DIR/bin/python" "$SCRIPT_DIR/bench_econml.py"

echo ""
echo "════════ Step 3 — R / grf ════════"
# grf lives in a personal library because the system R library is read-only.
export R_LIBS_USER="${R_LIBS_USER:-$HOME/Library/R/arm64/4.2/library-grf}"
Rscript -e 'if (!requireNamespace("grf", quietly=TRUE)) install.packages("grf", repos="https://cloud.r-project.org", type="binary")'
Rscript "$SCRIPT_DIR/bench_grf.R"

echo ""
echo "════════ Step 4 — aggregate ════════"
"$VENV_DIR/bin/python" "$SCRIPT_DIR/aggregate.py"
