# PINNによる1次元調和振動子

PyTorchで Physics-Informed Neural Networks (PINNs) を実装し、1次元調和振動子

```text
d²x/dt² = -x
x(0) = 1
dx/dt(0) = 0
0 <= t <= 2π
```

を解く学習用プロジェクトです。解析解 `x(t)=cos(t)` は学習には使わず、学習後の評価と可視化にだけ使います。

## 実行方法

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## 出力

実行後、`outputs/` に以下の図が保存されます。

- `solution_comparison.png`: 解析解 `cos(t)` と PINN 予測の比較
- `error_distribution.png`: 評価点での絶対誤差分布
- `training_curve.png`: 全損失、物理残差、初期条件損失の学習曲線

## 変更しやすい設定

`main.py` 冒頭の `Config` を変更すると、ネットワーク構造や学習率を調整できます。

- `hidden_layers`: 隠れ層のユニット数
- `learning_rate`: Adam の学習率
- `epochs`: 学習エポック数
- `n_collocation`: 物理残差を評価する学習点数
- `n_eval`: 学習点とは独立した評価点数
