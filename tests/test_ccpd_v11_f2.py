"""验证 CCPD-v1.1 F2 层选择、校准和成对指标。"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "experiments" / "ccpd_v11_know_act" / "f2_probe"


def load_module(name: str, filename: str):
    """从实验目录加载单文件模块。"""
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


EXTRACT = load_module("extract_native_hidden", "extract_native_hidden.py")
ANALYZE = load_module("analyze_probes", "analyze_probes.py")


class F2ProbeTests(unittest.TestCase):
    """覆盖预注册层和基础校准计算。"""

    def test_layer_indices_are_representative_and_unique(self) -> None:
        """28 层模型应选择 embedding、14、21 与 28 层输出。"""
        self.assertEqual(
            EXTRACT.layer_indices(28),
            {"embedding": 0, "middle": 14, "late": 21, "last": 28},
        )

    def test_ece_is_zero_for_perfect_binary_confidence(self) -> None:
        """完全正确的 0/1 概率应具有零 ECE。"""
        labels = np.asarray([0, 1, 0, 1])
        probabilities = np.asarray([0.0, 1.0, 0.0, 1.0])
        self.assertEqual(ANALYZE.expected_calibration_error(labels, probabilities), 0.0)

    def test_ece_detects_uninformative_overconfidence(self) -> None:
        """全预测为高正类概率时应产生明显校准误差。"""
        labels = np.asarray([0, 1])
        probabilities = np.asarray([0.9, 0.9])
        self.assertAlmostEqual(
            ANALYZE.expected_calibration_error(labels, probabilities), 0.4
        )


if __name__ == "__main__":
    unittest.main()
