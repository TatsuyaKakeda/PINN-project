"""PINNで調和振動子 x''(t) = -x(t) の初期値問題を解く。

微分方程式と初期条件だけで学習し、解析解 cos(t) は学習後の精度評価に使う。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# 画面を開かずに画像を保存できるよう、描画用のバックエンドを指定する。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


@dataclass
class Config:
    """学習・評価・画像保存の設定。各フィールドを引数で上書きできる。"""

    seed: int = 42
    # 各要素が隠れ層の幅を表す。既定では32ユニットの隠れ層を3層使う。
    hidden_layers: tuple[int, ...] = (32, 32, 32)
    learning_rate: float = 1.0e-3
    epochs: int = 5000
    # 方程式を課す学習点と、学習後に誤差を測る評価点を別々に設定する。
    n_collocation: int = 128
    n_eval: int = 1000
    print_every: int = 500
    # 方程式・初期位置・初期速度の損失が、全損失に寄与する重み。
    w_physics: float = 1.0
    w_ic_position: float = 1.0
    w_ic_velocity: float = 1.0
    output_dir: Path = Path("outputs")


def set_seed(seed: int) -> None:
    """整数 seed で各ライブラリの乱数を固定する。戻り値はない。"""
    # 重みの初期化などを再現し、同じ環境・設定での結果を比較しやすくする。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class PINN(nn.Module):
    """時刻 t から位置 x(t) を近似する全結合ネットワーク。

    入出力はともに形状 (N, 1)。重みとバイアスを、物理法則と初期条件に
    合うように学習することで、N 個の時刻における解を表現する。
    """

    def __init__(self, hidden_layers: tuple[int, ...]) -> None:
        """hidden_layers の各要素を隠れ層のユニット数として層を構築する。"""
        super().__init__()
        layers: list[nn.Module] = []
        # 入力は時刻という1変数。既定の構造は 1 → 32 → 32 → 32 → 1。
        in_features = 1

        for width in hidden_layers:
            layers.append(nn.Linear(in_features, width))
            # Tanh は滑らかなので、運動方程式に必要な2階微分を計算できる。
            layers.append(nn.Tanh())
            in_features = width

        # 出力層には活性化関数を置かず、位置の値域を制限しない。
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """形状 (N, 1) の時刻 t を受け取り、同じ形状の予測位置を返す。"""
        return self.net(t)


def derivative(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """t から計算した y の導関数 dy/dt を、t と同じ形状で返す。

    t は requires_grad=True、y は t につながる計算グラフを持つ必要がある。
    各 y[i] が対応する t[i] だけに依存する、このネットワークの構造を前提とする。
    """
    return torch.autograd.grad(
        y,
        t,
        # 全出力の和の勾配を計算する。各時刻が独立なので各点の dy/dt が得られる。
        grad_outputs=torch.ones_like(y),
        # 微分の計算もグラフに残し、2階微分と損失からのパラメータ更新を可能にする。
        create_graph=True,
        retain_graph=True,  # 後続の微分・逆伝播でも元のグラフを利用する。
    )[0]


def harmonic_oscillator_residual(
    t: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    調和振動子 d²x/dt² = -x を残差 x'' + x = 0 として評価する。

    入力:
        t: requires_grad=True の時刻テンソル (N, 1)
        x: t から計算され、計算グラフがつながっている位置 (N, 1)
    戻り値:
        residual: 物理残差 r(t) = x''(t) + x(t)（各戻り値の形状は (N, 1)）
        dx_dt: 速度 dx/dt
        d2x_dt2: 加速度 d²x/dt²
    """
    # 位置を時刻で2回微分し、速度と加速度を求める。
    dx_dt = derivative(x, t)
    d2x_dt2 = derivative(dx_dt, t)
    # 復元力による加速度が -x となる方程式なので、x'' + x が0なら物理法則を満たす。
    residual = d2x_dt2 + x
    return residual, dx_dt, d2x_dt2


def physics_loss(model: PINN, t_collocation: torch.Tensor) -> torch.Tensor:
    """model の予測から、方程式の残差の平均二乗をスカラーとして返す。

    t_collocation は方程式を課す時刻 (N, 1) で、requires_grad=True が必要。
    """
    x = model(t_collocation)
    residual, _, _ = harmonic_oscillator_residual(t_collocation, x)
    # 教師となる位置データの代わりに、各点の残差を0へ近づける。
    return torch.mean(residual**2)


def initial_condition_loss(model: PINN) -> tuple[torch.Tensor, torch.Tensor]:
    """
    model を受け取り、初期条件 x(0)=1, x'(0)=0 の損失を返す。

    解析解 cos(t) はここでは使わず、指定された初期値だけを使う。
    戻り値は初期位置と初期速度の二乗誤差を表す、2つのスカラーテンソル。
    """
    # 初期速度にも自動微分が必要なため、t=0 での計算グラフを作る。
    # 時間についての初期値問題なので、終端 t=2π に別の境界条件は課さない。
    t0 = torch.zeros((1, 1), dtype=torch.float32, requires_grad=True)
    x0 = model(t0)
    dx_dt0 = derivative(x0, t0)

    # 方程式だけでは解を一意に選べないため、初期位置と初期速度も制約する。
    loss_x0 = torch.mean((x0 - 1.0) ** 2)
    loss_v0 = torch.mean((dx_dt0 - 0.0) ** 2)
    return loss_x0, loss_v0


def relative_l2_error(prediction: torch.Tensor, truth: torch.Tensor) -> float:
    """同じ形状の予測と真値から、相対 L2 誤差を Python の float で返す。

    ||予測-真値||₂ / ||真値||₂ により、誤差を真値全体の大きさで規格化する。
    真値のノルムが0でないことを前提とする。
    """
    numerator = torch.linalg.norm(prediction - truth)
    denominator = torch.linalg.norm(truth)
    return (numerator / denominator).item()


def train(config: Config) -> tuple[PINN, dict[str, list[float]]]:
    """config の設定で学習し、学習済みモデルと各エポックの損失履歴を返す。

    history の total は重み付き全損失、その他の項目は重みを掛ける前の各損失。
    """
    set_seed(config.seed)

    model = PINN(config.hidden_layers)
    # Adam が更新するのは、位置関数を表すネットワークの重みとバイアス。
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学習点: 微分方程式の残差を小さくするためのコロケーション点。
    # 区間 [0, 2π] の両端を含む等間隔の点を、全エポックで共通して使う。
    # 解析解 cos(t) は学習に使わない。
    t_collocation = torch.linspace(
        0.0,
        2.0 * math.pi,
        config.n_collocation,
        dtype=torch.float32,
    ).reshape(-1, 1)
    # パラメータの微分に加え、入力時刻に関する微分を自動微分で求められるようにする。
    t_collocation.requires_grad_(True)

    history: dict[str, list[float]] = {
        "total": [],
        "physics": [],
        "ic_position": [],
        "ic_velocity": [],
    }

    for epoch in range(1, config.epochs + 1):
        # 前回のパラメータ勾配を消し、このエポックの損失から計算し直す。
        optimizer.zero_grad()

        # L_physics = mean((x''(t) + x(t))²)
        loss_physics = physics_loss(model, t_collocation)

        # L_IC = (x(0)-1)² + (x'(0)-0)²
        loss_x0, loss_v0 = initial_condition_loss(model)

        # 方程式と2つの初期条件を同時に満たすよう、3項の重み付き和を最小化する。
        loss = (
            config.w_physics * loss_physics
            + config.w_ic_position * loss_x0
            + config.w_ic_velocity * loss_v0
        )

        loss.backward()  # 時刻微分の計算も経由し、全損失のパラメータ勾配を求める。
        optimizer.step()  # 求めた勾配に基づいて重みとバイアスを更新する。

        # 更新前に評価した損失を数値として保存し、学習の収束を後で可視化する。
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
    学習済み model を config.n_eval 個の評価点で解析解 cos(t) と比較する。
    解析解は評価専用であり、学習損失には使わない。

    戻り値は時刻・予測・解析解・絶対誤差のテンソル (N, 1) と、
    相対 L2 誤差・最大絶対誤差の float をまとめた辞書。
    """
    # 学習用とは別に評価用の等間隔格子を作る。両端などは学習点と重なり得る。
    t_eval = torch.linspace(
        0.0,
        2.0 * math.pi,
        config.n_eval,
        dtype=torch.float32,
    ).reshape(-1, 1)

    # 評価では導関数やパラメータ更新を求めないため、計算グラフの記録を省く。
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
    """損失履歴 history と評価結果 evaluation から3種類の図を保存する。

    output_dir は保存先ディレクトリ。画像をファイルに書き出し、戻り値はない。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 描画用に勾配の追跡を外し、CPU上の NumPy 配列へ変換する。
    t = evaluation["t"].detach().cpu().numpy()
    x_pred = evaluation["x_pred"].detach().cpu().numpy()
    x_exact = evaluation["x_exact"].detach().cpu().numpy()
    abs_error = evaluation["abs_error"].detach().cpu().numpy()

    # 波形を重ねることで、予測の振幅や位相が解析解に合っているかを確認する。
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

    # 各時刻の絶対誤差を示し、近似の精度が低い区間を確認する。
    plt.figure(figsize=(8, 5))
    plt.plot(t, abs_error, color="tab:red", linewidth=2)
    plt.xlabel("t")
    plt.ylabel("|x_PINN(t) - cos(t)|")
    plt.title("Absolute error on independent evaluation points")
    plt.tight_layout()
    plt.savefig(output_dir / "error_distribution.png", dpi=200)
    plt.close()

    # 桁の異なる損失を対数軸で比較し、方程式・初期条件それぞれの収束を確認する。
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
    """既定設定で学習・評価・図の保存を実行し、誤差を表示する。引数・戻り値はない。"""
    config = Config()
    model, history = train(config)
    evaluation = evaluate(model, config)
    save_plots(history, evaluation, config.output_dir)

    # 学習損失と解析解に対する誤差を併記し、物理制約への適合と予測精度を確認する。
    print("\n=== 評価結果 ===")
    print(f"final training loss: {history['total'][-1]:.6e}")
    print(f"relative L2 error : {evaluation['relative_l2']:.6e}")
    print(f"max absolute error: {evaluation['max_abs_error']:.6e}")
    print(f"plots saved to    : {config.output_dir.resolve()}")


if __name__ == "__main__":
    # テストなどから import したときには学習を始めず、直接実行した場合だけ動かす。
    main()
