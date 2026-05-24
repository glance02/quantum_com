import argparse
import time

import numpy as np
from scipy.optimize import minimize

from qml_models import (
    MODEL_SPECS,
    artifact_path,
    data_path,
    evaluate_numpy,
    fit_preprocessor,
    load_csv,
    save_artifacts,
    sigmoid_np,
    simulate_quantum_features,
    stratified_split,
    transform_features,
)


def fit_readout(q_train, y_train, l2=0.02):
    def objective(vector):
        logits = q_train @ vector[:-1] + vector[-1]
        loss = np.mean(np.logaddexp(0.0, logits) - y_train * logits)
        return loss + l2 * np.mean(vector[:-1] ** 2)

    result = minimize(
        objective,
        np.zeros(q_train.shape[1] + 1, dtype=np.float64),
        method="BFGS",
        options={"maxiter": 300, "gtol": 1e-6},
    )
    return result.x.astype(np.float32), float(result.fun)


def readout_accuracy(q, y, readout):
    prob = sigmoid_np(q @ readout[:-1] + readout[-1])
    return float(np.mean((prob >= 0.5).astype(np.int64) == y.astype(np.int64)))


def train_one_model(name, x_train, y_train, x_val, y_val, seed=42):
    spec = MODEL_SPECS[name]
    rng = np.random.default_rng(seed)
    theta = np.zeros(spec["param_count"], dtype=np.float64)
    if name == "lightweight":
        theta = rng.normal(0.0, 0.35, size=spec["param_count"])

    steps = spec["epochs"]
    batch_size = min(192, len(x_train))
    best_val = -1.0
    best_params = None
    started = time.time()

    for step in range(1, steps + 1):
        batch_idx = rng.choice(len(x_train), size=batch_size, replace=False)
        delta = rng.choice([-1.0, 1.0], size=theta.shape[0])
        c = 0.22 / (step ** 0.101)
        lr = spec["lr"] / (step ** 0.2)

        theta_plus = theta + c * delta
        theta_minus = theta - c * delta

        q_plus = simulate_quantum_features(
            x_train[batch_idx], theta_plus, spec["num_qubits"], spec["layers"]
        )
        q_minus = simulate_quantum_features(
            x_train[batch_idx], theta_minus, spec["num_qubits"], spec["layers"]
        )
        _, loss_plus = fit_readout(q_plus, y_train[batch_idx])
        _, loss_minus = fit_readout(q_minus, y_train[batch_idx])
        grad = (loss_plus - loss_minus) / (2.0 * c) * delta
        theta -= lr * grad
        theta = (theta + np.pi) % (2.0 * np.pi) - np.pi

        q_train = simulate_quantum_features(x_train, theta, spec["num_qubits"], spec["layers"])
        readout, train_loss = fit_readout(q_train, y_train)
        q_val = simulate_quantum_features(x_val, theta, spec["num_qubits"], spec["layers"])
        train_acc = readout_accuracy(q_train, y_train, readout)
        val_acc = readout_accuracy(q_val, y_val, readout)

        if val_acc >= best_val:
            best_val = val_acc
            best_params = {
                f"{name}_theta": theta.astype(np.float32),
                f"{name}_weight": readout[:-1].reshape(1, -1).astype(np.float32),
                f"{name}_bias": np.array([readout[-1]], dtype=np.float32),
            }

        print(
            f"{name:11s} step={step:03d} "
            f"loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    return best_params


def main():
    parser = argparse.ArgumentParser(description="Train two DR quantum machine learning classifiers.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    x, y = load_csv(data_path("train.csv"))
    train_idx, val_idx = stratified_split(y, seed=7)
    stats = fit_preprocessor(x[train_idx])
    x_train = transform_features(x[train_idx], stats)
    x_val = transform_features(x[val_idx], stats)
    y_train = y[train_idx]
    y_val = y[val_idx]

    baseline_params = train_one_model("baseline", x_train, y_train, x_val, y_val, seed=args.seed)
    lightweight_params = train_one_model("lightweight", x_train, y_train, x_val, y_val, seed=args.seed + 1)

    save_artifacts(stats, baseline_params, lightweight_params)
    acc_b, _ = evaluate_numpy(baseline_params, x_val, y_val, "baseline")
    acc_l, _ = evaluate_numpy(lightweight_params, x_val, y_val, "lightweight")
    print(f"saved={artifact_path()}")
    print(f"validation_acc_baseline={acc_b:.4f}")
    print(f"validation_acc_lightweight={acc_l:.4f}")


if __name__ == "__main__":
    main()
