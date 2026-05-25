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


def quantum_features(name, x, theta):
    spec = MODEL_SPECS[name]
    return simulate_quantum_features(
        x,
        theta,
        spec["num_qubits"],
        spec["layers"],
        spec.get("active_qubits", spec["num_qubits"]),
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


def readout_loss(q, y, readout, l2=0.02):
    logits = q @ readout[:-1] + readout[-1]
    loss = np.mean(np.logaddexp(0.0, logits) - y * logits)
    return float(loss + l2 * np.mean(readout[:-1] ** 2))


def readout_accuracy(q, y, readout):
    prob = sigmoid_np(q @ readout[:-1] + readout[-1])
    return float(np.mean((prob >= 0.5).astype(np.int64) == y.astype(np.int64)))


def pack_params(name, theta, readout):
    return {
        f"{name}_theta": theta.astype(np.float32),
        f"{name}_weight": readout[:-1].reshape(1, -1).astype(np.float32),
        f"{name}_bias": np.array([readout[-1]], dtype=np.float32),
    }


def train_one_model(name, x_train, y_train, x_val=None, y_val=None, seed=42):
    spec = MODEL_SPECS[name]
    has_validation = x_val is not None and y_val is not None
    starts = spec["starts"] if has_validation else 1
    steps = spec["epochs"] if has_validation else spec["final_epochs"]
    batch_size = min(spec["batch_size"], len(x_train))
    best_metric = -np.inf
    best_params = None
    best_val = None
    started = time.time()

    for start in range(starts):
        rng = np.random.default_rng(seed + 1009 * start)
        theta = rng.normal(0.0, spec["init_scale"], size=spec["param_count"])
        q_train = quantum_features(name, x_train, theta)
        readout, train_loss = fit_readout(q_train, y_train)
        train_acc = readout_accuracy(q_train, y_train, readout)

        if has_validation:
            q_val = quantum_features(name, x_val, theta)
            val_acc = readout_accuracy(q_val, y_val, readout)
            metric = val_acc
            metric_text = f"val_acc={val_acc:.4f}"
        else:
            val_acc = None
            metric = -train_loss
            metric_text = "val_acc=NA"

        if metric >= best_metric:
            best_metric = metric
            best_val = val_acc
            best_params = pack_params(name, theta, readout)

        print(
            f"{name:11s} start={start + 1:02d}/{starts:02d} step=000 "
            f"loss={train_loss:.4f} train_acc={train_acc:.4f} {metric_text} "
            f"theta_std={theta.std():.3f} elapsed={time.time() - started:.1f}s",
            flush=True,
        )

        for step in range(1, steps + 1):
            q_current = quantum_features(name, x_train, theta)
            readout, train_loss = fit_readout(q_current, y_train)
            batch_idx = rng.choice(len(x_train), size=batch_size, replace=False)
            grad = np.zeros_like(theta)
            c = spec["spsa_c"] / (step ** 0.101)

            for _ in range(spec["spsa_repeats"]):
                delta = rng.choice([-1.0, 1.0], size=theta.shape[0])
                theta_plus = theta + c * delta
                theta_minus = theta - c * delta
                q_plus = quantum_features(name, x_train[batch_idx], theta_plus)
                q_minus = quantum_features(name, x_train[batch_idx], theta_minus)
                loss_plus = readout_loss(q_plus, y_train[batch_idx], readout)
                loss_minus = readout_loss(q_minus, y_train[batch_idx], readout)
                grad += (loss_plus - loss_minus) / (2.0 * c) * delta

            grad /= float(spec["spsa_repeats"])
            grad_norm = float(np.linalg.norm(grad))
            if grad_norm > spec["grad_clip"]:
                grad *= spec["grad_clip"] / grad_norm

            lr = spec["lr"] / (step ** 0.2)
            theta -= lr * grad
            theta = (theta + np.pi) % (2.0 * np.pi) - np.pi

            q_train = quantum_features(name, x_train, theta)
            readout, train_loss = fit_readout(q_train, y_train)
            train_acc = readout_accuracy(q_train, y_train, readout)

            if has_validation:
                q_val = quantum_features(name, x_val, theta)
                val_acc = readout_accuracy(q_val, y_val, readout)
                metric = val_acc
                metric_text = f"val_acc={val_acc:.4f}"
            else:
                val_acc = None
                metric = -train_loss
                metric_text = "val_acc=NA"

            if metric >= best_metric:
                best_metric = metric
                best_val = val_acc
                best_params = pack_params(name, theta, readout)

            print(
                f"{name:11s} start={start + 1:02d}/{starts:02d} step={step:03d} "
                f"loss={train_loss:.4f} train_acc={train_acc:.4f} {metric_text} "
                f"theta_std={theta.std():.3f} elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    return best_params, best_val


def refit_readout_on_full_train(name, theta, x_full, y_full):
    q_full = quantum_features(name, x_full, theta)
    readout, _ = fit_readout(q_full, y_full)
    return pack_params(name, theta, readout)


def main():
    parser = argparse.ArgumentParser(description="Train two DR quantum machine learning classifiers.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    x, y = load_csv(data_path("train.csv"))
    train_idx, val_idx = stratified_split(y, seed=7)
    stats = fit_preprocessor(x)
    x_train = transform_features(x[train_idx], stats)
    x_val = transform_features(x[val_idx], stats)
    y_train = y[train_idx]
    y_val = y[val_idx]

    baseline_params, baseline_val = train_one_model("baseline", x_train, y_train, x_val, y_val, seed=args.seed)
    lightweight_params, lightweight_val = train_one_model("lightweight", x_train, y_train, x_val, y_val, seed=0)

    x_full = transform_features(x, stats)
    baseline_params = refit_readout_on_full_train(
        "baseline", baseline_params["baseline_theta"], x_full, y
    )
    lightweight_params = refit_readout_on_full_train(
        "lightweight", lightweight_params["lightweight_theta"], x_full, y
    )

    save_artifacts(stats, baseline_params, lightweight_params)
    x_val_final = transform_features(x[val_idx], stats)
    acc_b, _ = evaluate_numpy(baseline_params, x_val_final, y_val, "baseline")
    acc_l, _ = evaluate_numpy(lightweight_params, x_val_final, y_val, "lightweight")
    print(f"saved={artifact_path()}")
    print(f"selection_val_acc_baseline={baseline_val:.4f}")
    print(f"selection_val_acc_lightweight={lightweight_val:.4f}")
    print(f"validation_acc_baseline={acc_b:.4f}")
    print(f"validation_acc_lightweight={acc_l:.4f}")


if __name__ == "__main__":
    main()
