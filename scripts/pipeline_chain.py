"""
pipeline_chain.py -- Auto-chained pipeline executor
=====================================================
Polls the prepare_data.py task output file for a completion marker,
then automatically launches run_sentiment_nlp.py inference.

Usage:
    python scripts/pipeline_chain.py <task_output_file>
"""

import subprocess
import sys
import time
from pathlib import Path

# Output file of the background prepare_data.py task (passed as CLI argument)
if len(sys.argv) < 2:
    print("[ERROR] Usage: python scripts/pipeline_chain.py <task_output_file>")
    sys.exit(1)

WATCH_FILE   = Path(sys.argv[1])
ROOT         = Path(__file__).parent.parent
PYTHON       = str(ROOT / ".venv" / "Scripts" / "python.exe")
INFER_SCRIPT = str(ROOT / "scripts" / "run_sentiment_nlp.py")

# prepare_data.py prints one of these strings when it finishes
DONE_MARKERS = [
    "Command: python scripts/run_sentiment_nlp.py",
    "All texts have been inferred",
    "Data preparation complete",
    "Step 5/5",   # final step header signals data collection is done
]

POLL_INTERVAL = 30  # seconds between polls

print("=" * 60)
print("[CHAIN] Pipeline watcher started")
print(f"  Watch file   : {WATCH_FILE}")
print(f"  Infer script : {INFER_SCRIPT}")
print(f"  Poll interval: {POLL_INTERVAL}s")
print("=" * 60)

while True:
    time.sleep(POLL_INTERVAL)

    if not WATCH_FILE.exists():
        print("[CHAIN] Waiting for output file to appear ...")
        continue

    try:
        content = WATCH_FILE.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[CHAIN] Failed to read output file: {e}")
        continue

    # Check for completion markers
    found = next((m for m in DONE_MARKERS if m in content), None)
    if found:
        print(f"\n[CHAIN] Completion marker detected: '{found}'")
        print("[CHAIN] prepare_data.py finished — waiting 10s to confirm stability ...")
        time.sleep(10)
        break
    else:
        # Show last 2 non-empty lines as progress preview
        lines   = [l for l in content.splitlines() if l.strip()]
        preview = " | ".join(lines[-2:])[-120:] if lines else "(empty)"
        print(f"[CHAIN] Waiting ... {preview}")

print("\n" + "=" * 60)
print("[CHAIN] Launching NolBERT inference ...")
print(f"  Command: {PYTHON} {INFER_SCRIPT}")
print("=" * 60 + "\n")

result = subprocess.run(
    [PYTHON, INFER_SCRIPT],
    cwd=str(ROOT),
)

print(f"\n[CHAIN] Inference complete — exit code: {result.returncode}")
