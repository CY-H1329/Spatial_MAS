#!/usr/bin/env python3
"""Alias : utilisez `infer_mindcube_qwen3vl.py` (ou les autres backends). Ce script délègue à Qwen3-VL."""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "infer_mindcube_qwen3vl.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
