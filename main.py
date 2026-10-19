from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


@dataclass
class Config:
    seed: int = 42
    hidden_layers: tuple[int, ...] = (32, 32, 32)
    learning_rate: float = 1.0e-3
    epochs: int = 5000
    n_collocation: int = 128
    n_eval: int = 1000
    print_every: int = 500
    w_physics: float = 1.0
    w_ic_position: float = 1.0
    w_ic_velocity: float = 1.0
    output_dir: Path = Path("outputs")


def set_seed(seed: int) -> None:
    """乱数シードを固定し、同じ条件で再実行しやすくする。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class PINN(nn.Module):
    """時刻 t を入力し、位置 x(t) を出力する多層パーセプトロン。"""

    def __init__(self, hidden_layers: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = 1

        for width in hidden_layers:
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.Tanh())
            in_features = width

        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


def derivative(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """自動微分で dy/dt を計算する。"""
    return torch.autograd.grad(
        y,
        t,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def harmonic_oscillator_residual(
    t: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    調和振動子 d²x/dt² = -x を残差 x'' + x = 0 として評価する。

    戻り値:
        residual: 物理残差 r(t) = x''(t) + x(t)
        dx_dt: 速度 dx/dt
        d2x_dt2: 加速度 d²x/dt²
    """
    dx_dt = derivative(x, t)
    d2x_dt2 = derivative(dx_dt, t)
    residual = d2x_dt2 + x
    return residual, dx_dt, d2x_dt2


def physics_loss(model: PINN, t_collocation: torch.Tensor) -> torch.Tensor:
    """微分方程式 x'' + x = 0 の残差 r(t) の平均二乗誤差。"""
    x = model(t_collocation)
    residual, _, _ = harmonic_oscillator_residual(t_collocation, x)
    return torch.mean(residual**2)


def initial_condition_loss(model: PINN) -> tuple[torch.Tensor, torch.Tensor]:
    """
    初期条件 x(0)=1, x'(0)=0 の損失。

    解析解 cos(t) はここでは使わず、指定された初期値だけを使う。
    """
    t0 = torch.zeros((1, 1), dtype=torch.float32, requires_grad=True)
    x0 = model(t0)
    dx_dt0 = derivative(x0, t0)

    loss_x0 = torch.mean((x0 - 1.0) ** 2)
    loss_v0 = torch.mean((dx_dt0 - 0.0) ** 2)
    return loss_x0, loss_v0


def relative_l2_error(prediction: torch.Tensor, truth: torch.Tensor) -> float:
    """相対 L2 誤差 ||予測-真値||₂ / ||真値||₂。"""
    numerator = torch.linalg.norm(prediction - truth)
    denominator = torch.linalg.norm(truth)
    return (numerator / denominator).item()


def train(config: Config) -> tuple[PINN, dict[str, list[float]]]:
    set_seed(config.seed)

    model = PINN(config.hidden_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学習点: 微分方程式の残差を小さくするためのコロケーション点。
    # 解析解 cos(t) は学習に使わない。
    t_collocation = torch.linspace(
        0.0,
        2.0 * math.pi,
        config.n_collocation,
        dtype=torch.float32,
    ).reshape(-1, 1)
    t_collocation.requires_grad_(True)

    history: dict[str, list[float]] = {
        "total": [],
        "physics": [],
        "ic_position": [],
        "ic_velocity": [],
    }

    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad()

        # L_physics = mean((x''(t) + x(t))²)
        loss_physics = physics_loss(model, t_collocation)

        # L_IC = (x(0)-1)² + (x'(0)-0)²
        loss_x0, loss_v0 = initial_condition_loss(model)

        loss = (
            config.w_physics * loss_physics
            + config.w_ic_position * loss_x0
            + config.w_ic_velocity * loss_v0
        )

        loss.backward()
        optimizer.step()

        history["total"].append(loss.item())
        history["physics"].append(loss_physics.item())
        history["ic_position"].append(loss_x0.item())
        history["ic_velocity"].append(loss_v0.item())

        if epoch == 1 or epoch % config.print_every == 0:
            print(
                f"epoch {epoch:5d} | total={loss.item():.6e} "
                f"physics={loss_physics.item():.6e} "
                f"ic_x={loss_x0.item():.6e} ic_v={loss_v0.item():.6e}"
            )

    return model, history


def evaluate(model: PINN, config: Config) -> dict[str, torch.Tensor | float]:
    """
    学習点とは独立した評価点で、解析解 cos(t) と比較する。
    解析解は評価専用であり、学習損失には使わない。
    """
    t_eval = torch.linspace(
        0.0,
        2.0 * math.pi,
        config.n_eval,
        dtype=torch.float32,
    ).reshape(-1, 1)

    with torch.no_grad():
        x_pred = model(t_eval)
        x_exact = torch.cos(t_eval)
        abs_error = torch.abs(x_pred - x_exact)

    return {
        "t": t_eval,
        "x_pred": x_pred,
        "x_exact": x_exact,
        "abs_error": abs_error,
        "relative_l2": relative_l2_error(x_pred, x_exact),
        "max_abs_error": torch.max(abs_error).item(),
    }


def save_plots(
    history: dict[str, list[float]],
    evaluation: dict[str, torch.Tensor | float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    t = evaluation["t"].detach().cpu().numpy()
    x_pred = evaluation["x_pred"].detach().cpu().numpy()
    x_exact = evaluation["x_exact"].detach().cpu().numpy()
    abs_error = evaluation["abs_error"].detach().cpu().numpy()

    plt.figure(figsize=(8, 5))
    plt.plot(t, x_exact, label="exact cos(t)", linewidth=2)
    plt.plot(t, x_pred, "--", label="PINN prediction", linewidth=2)
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title("1D harmonic oscillator: exact vs PINN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "solution_comparison.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(t, abs_error, color="tab:red", linewidth=2)
    plt.xlabel("t")
    plt.ylabel("|x_PINN(t) - cos(t)|")
    plt.title("Absolute error on independent evaluation points")
    plt.tight_layout()
    plt.savefig(output_dir / "error_distribution.png", dpi=200)
    plt.close()

    epochs = np.arange(1, len(history["total"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.semilogy(epochs, history["total"], label="total")
    plt.semilogy(epochs, history["physics"], label="physics")
    plt.semilogy(epochs, history["ic_position"], label="x(0)")
    plt.semilogy(epochs, history["ic_velocity"], label="x'(0)")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_curve.png", dpi=200)
    plt.close()


def main() -> None:
    config = Config()
    model, history = train(config)
    evaluation = evaluate(model, config)
    save_plots(history, evaluation, config.output_dir)

    print("\n=== 評価結果 ===")
    print(f"final training loss: {history['total'][-1]:.6e}")
    print(f"relative L2 error : {evaluation['relative_l2']:.6e}")
    print(f"max absolute error: {evaluation['max_abs_error']:.6e}")
    print(f"plots saved to    : {config.output_dir.resolve()}")


if __name__ == "__main__":
    main()
