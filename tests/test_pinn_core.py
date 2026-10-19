"""自動微分による物理残差と、評価用の誤差指標を検証するテスト。"""

import math
import sys
from pathlib import Path

# このファイルを直接実行する場合も、リポジトリ直下の main.py を読み込めるようにする。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import main


def test_second_derivative_residual_matches_harmonic_oscillator_equation():
    """cos(t) の1階・2階微分と、x'' + x = 0 の成立を確認する。引数・戻り値はない。"""
    # 既知の解析解を使って微分処理そのものを検証する。学習用データではない。
    t = torch.tensor([[0.0], [math.pi / 2]], dtype=torch.float32, requires_grad=True)
    x = torch.cos(t)

    residual, dx_dt, d2x_dt2 = main.harmonic_oscillator_residual(t, x)

    # 速度は -sin(t)、加速度は -cos(t) となり、浮動小数点の誤差範囲で残差が0になる。
    assert torch.allclose(dx_dt, -torch.sin(t), atol=1e-6)
    assert torch.allclose(d2x_dt2, -torch.cos(t), atol=1e-6)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)


def test_relative_l2_error_is_zero_for_exact_solution():
    """予測と真値が一致すると相対 L2 誤差が0になることを確認する。引数・戻り値はない。"""
    # 分母のノルムが0にならない真値を用意し、誤差指標の基本的な性質を検証する。
    truth = torch.tensor([[1.0], [0.0], [-1.0]])
    prediction = truth.clone()

    assert main.relative_l2_error(prediction, truth) == 0.0


if __name__ == "__main__":
    # pytest 経由に加えて、このファイルの直接実行でも2つの検証を行える。
    test_second_derivative_residual_matches_harmonic_oscillator_equation()
    test_relative_l2_error_is_zero_for_exact_solution()
