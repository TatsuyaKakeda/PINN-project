import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import main


def test_second_derivative_residual_matches_harmonic_oscillator_equation():
    t = torch.tensor([[0.0], [math.pi / 2]], dtype=torch.float32, requires_grad=True)
    x = torch.cos(t)

    residual, dx_dt, d2x_dt2 = main.harmonic_oscillator_residual(t, x)

    assert torch.allclose(dx_dt, -torch.sin(t), atol=1e-6)
    assert torch.allclose(d2x_dt2, -torch.cos(t), atol=1e-6)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)


def test_relative_l2_error_is_zero_for_exact_solution():
    truth = torch.tensor([[1.0], [0.0], [-1.0]])
    prediction = truth.clone()

    assert main.relative_l2_error(prediction, truth) == 0.0


if __name__ == "__main__":
    test_second_derivative_residual_matches_harmonic_oscillator_equation()
    test_relative_l2_error_is_zero_for_exact_solution()
