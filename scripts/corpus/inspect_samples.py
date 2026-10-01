#!/usr/bin/env python3
"""Convenience script to run Gate C1 sample inspection.

Usage
-----
    python3 scripts/corpus/inspect_samples.py data/samples/gate_c1/
    python3 scripts/corpus/inspect_samples.py data/samples/gate_c1/carolina/representative.parquet
"""

import os
from pathlib import Path
import sys

# Ensure src directory is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cambacica.corpus.cli import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] not in ("sample", "inspect", "compare", "-h", "--help"):
        args = ["inspect"] + args
    elif not args:
        args = ["inspect", "data/samples/gate_c1/"]
    sys.stdout.flush()
    sys.stderr.flush()
    ret = main(args)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(ret)
