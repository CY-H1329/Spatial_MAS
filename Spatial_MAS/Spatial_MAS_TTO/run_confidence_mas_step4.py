#!/usr/bin/env python3
"""
Jupyter / Python 실행용 — Confidence MAS v3 step4 실행 스크립트.

실행: python run_confidence_mas_step4.py
또는 Jupyter에서 이 파일 내용을 셀에 붙여넣기 후 실행.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent  # 서버: Path("/home/jovyan/CY/Spatial_MAS")
sys.path.insert(0, str(PROJECT_ROOT))

from test_confidence_mas_v3_step4 import run_confidence_mas_test_step4, build_runners_for_confidence

head_gen, spec_gen, reason_gen = build_runners_for_confidence(
    specialist_device="cuda",
    use_vlm_reasoning=True,
)

results = run_confidence_mas_test_step4(
    head_gen, spec_gen, reason_gen,
    benchmark="cvbench",
    max_samples=50,
    T=10.0,
    kappa=1.0,
    gamma=0.1,
)

print(f"\nAccuracy: {results['correct']}/{results['total']} = {100*results['accuracy']:.1f}%")
