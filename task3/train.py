import argparse
import csv
import json
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


LOG_FIELDS = [
    "phase",
    "model",
    "start",
    "step",
    "loss",
    "train_acc",
    "cv_acc",
    "l2",
    "theta_std",
    "theta_absmax",
    "elapsed",
    "is_best",
]


class TrainingLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.log_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=LOG_FIELDS)
        self.writer.writeheader()
        self.file.flush()

    def log_step(self, **row):
        clean = {field: row.get(field, "") for field in LOG_FIELDS}
        self.writer.writerow(clean)
        self.file.flush()

    def close(self):
        self.file.close()


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


def make_cv_splits(y, seeds=(5, 7), val_per_class=64):
    return [stratified_split(y, val_per_class=val_per_class, seed=seed) for seed in seeds]


def fit_best_readout(q, y, l2_grid):
    best_readout = None
    best_loss = np.inf
    best_l2 = None
    for l2 in l2_grid:
        readout, loss = fit_readout(q, y, l2=l2)
        if loss < best_loss:
            best_loss = loss
            best_readout = readout
            best_l2 = l2
    return best_readout, best_loss, best_l2


def evaluate_candidate(name, theta, x, y, cv_splits, l2_grid):
    q_all = quantum_features(name, x, theta)
    best_l2 = l2_grid[0]
    best_cv = -np.inf

    for l2 in l2_grid:
        fold_scores = []
        for train_idx, val_idx in cv_splits:
            readout, _ = fit_readout(q_all[train_idx], y[train_idx], l2=l2)
            fold_scores.append(readout_accuracy(q_all[val_idx], y[val_idx], readout))
        cv_acc = float(np.mean(fold_scores))
        if cv_acc > best_cv:
            best_cv = cv_acc
            best_l2 = l2

    readout, train_loss = fit_readout(q_all, y, l2=best_l2)
    train_acc = readout_accuracy(q_all, y, readout)
    return {
        "cv_acc": best_cv,
        "train_acc": train_acc,
        "train_loss": train_loss,
        "readout": readout,
        "l2": best_l2,
    }


def pack_params(name, theta, readout):
    return {
        f"{name}_theta": theta.astype(np.float32),
        f"{name}_weight": readout[:-1].reshape(1, -1).astype(np.float32),
        f"{name}_bias": np.array([readout[-1]], dtype=np.float32),
    }


def train_one_model(name, x_train, y_train, cv_splits, seed=42, logger=None):
    spec = MODEL_SPECS[name]
    starts = spec["starts"]
    steps = spec["epochs"]
    batch_size = min(spec["batch_size"], len(x_train))
    l2_grid = spec["l2_grid"]
    best_metric = -np.inf
    best_params = None
    best_info = None
    started = time.time()

    for start in range(starts):
        rng = np.random.default_rng(seed + 1009 * start)
        theta = rng.normal(0.0, spec["init_scale"], size=spec["param_count"])
        info = evaluate_candidate(name, theta, x_train, y_train, cv_splits, l2_grid)
        metric = info["cv_acc"]

        is_best = metric >= best_metric
        if is_best:
            best_metric = metric
            best_info = info
            best_params = pack_params(name, theta, info["readout"])

        elapsed = time.time() - started
        print(
            f"{name:11s} start={start + 1:02d}/{starts:02d} step=000 "
            f"loss={info['train_loss']:.4f} train_acc={info['train_acc']:.4f} "
            f"cv_acc={info['cv_acc']:.4f} l2={info['l2']:.3g} "
            f"theta_std={theta.std():.3f} elapsed={elapsed:.1f}s",
            flush=True,
        )
        if logger is not None:
            logger.log_step(
                phase="candidate",
                model=name,
                start=start + 1,
                step=0,
                loss=f"{info['train_loss']:.8f}",
                train_acc=f"{info['train_acc']:.8f}",
                cv_acc=f"{info['cv_acc']:.8f}",
                l2=info["l2"],
                theta_std=f"{theta.std():.8f}",
                theta_absmax=f"{np.abs(theta).max():.8f}",
                elapsed=f"{elapsed:.3f}",
                is_best=int(is_best),
            )

        for step in range(1, steps + 1):
            q_current = quantum_features(name, x_train, theta)
            readout, _, _ = fit_best_readout(q_current, y_train, l2_grid)
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

            info = evaluate_candidate(name, theta, x_train, y_train, cv_splits, l2_grid)
            metric = info["cv_acc"]

            is_best = metric >= best_metric
            if is_best:
                best_metric = metric
                best_info = info
                best_params = pack_params(name, theta, info["readout"])

            elapsed = time.time() - started
            print(
                f"{name:11s} start={start + 1:02d}/{starts:02d} step={step:03d} "
                f"loss={info['train_loss']:.4f} train_acc={info['train_acc']:.4f} "
                f"cv_acc={info['cv_acc']:.4f} l2={info['l2']:.3g} "
                f"theta_std={theta.std():.3f} elapsed={elapsed:.1f}s",
                flush=True,
            )
            if logger is not None:
                logger.log_step(
                    phase="spsa",
                    model=name,
                    start=start + 1,
                    step=step,
                    loss=f"{info['train_loss']:.8f}",
                    train_acc=f"{info['train_acc']:.8f}",
                    cv_acc=f"{info['cv_acc']:.8f}",
                    l2=info["l2"],
                    theta_std=f"{theta.std():.8f}",
                    theta_absmax=f"{np.abs(theta).max():.8f}",
                    elapsed=f"{elapsed:.3f}",
                    is_best=int(is_best),
                )

    return best_params, best_info


def refit_readout_on_full_train(name, theta, x_full, y_full, l2):
    q_full = quantum_features(name, x_full, theta)
    readout, _ = fit_readout(q_full, y_full, l2=l2)
    return pack_params(name, theta, readout)


def main():
    parser = argparse.ArgumentParser(description="Train two DR quantum machine learning classifiers.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    log_path = artifact_path().parent / "train_log.csv"
    summary_path = artifact_path().parent / "train_summary.json"
    logger = TrainingLogger(log_path)

    x, y = load_csv(data_path("train.csv"))
    stats = fit_preprocessor(x)
    x_full = transform_features(x, stats)
    cv_splits = make_cv_splits(y)

    try:
        baseline_params, baseline_info = train_one_model("baseline", x_full, y, cv_splits, seed=args.seed, logger=logger)
        lightweight_params, lightweight_info = train_one_model("lightweight", x_full, y, cv_splits, seed=0, logger=logger)
    finally:
        logger.close()

    baseline_params = refit_readout_on_full_train(
        "baseline", baseline_params["baseline_theta"], x_full, y, baseline_info["l2"]
    )
    lightweight_params = refit_readout_on_full_train(
        "lightweight", lightweight_params["lightweight_theta"], x_full, y, lightweight_info["l2"]
    )

    save_artifacts(stats, baseline_params, lightweight_params)
    acc_b, _ = evaluate_numpy(baseline_params, x_full, y, "baseline")
    acc_l, _ = evaluate_numpy(lightweight_params, x_full, y, "lightweight")
    summary = {
        "artifact_path": str(artifact_path()),
        "log_path": str(log_path),
        "baseline": {
            "selection_cv_acc": baseline_info["cv_acc"],
            "selection_l2": baseline_info["l2"],
            "train_acc": acc_b,
            "theta_std": float(baseline_params["baseline_theta"].std()),
            "theta_absmax": float(np.abs(baseline_params["baseline_theta"]).max()),
        },
        "lightweight": {
            "selection_cv_acc": lightweight_info["cv_acc"],
            "selection_l2": lightweight_info["l2"],
            "train_acc": acc_l,
            "theta_std": float(lightweight_params["lightweight_theta"].std()),
            "theta_absmax": float(np.abs(lightweight_params["lightweight_theta"]).max()),
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"saved={artifact_path()}")
    print(f"log={log_path}")
    print(f"summary={summary_path}")
    print(f"selection_cv_acc_baseline={baseline_info['cv_acc']:.4f} l2={baseline_info['l2']:.3g}")
    print(f"selection_cv_acc_lightweight={lightweight_info['cv_acc']:.4f} l2={lightweight_info['l2']:.3g}")
    print(f"train_acc_baseline={acc_b:.4f}")
    print(f"train_acc_lightweight={acc_l:.4f}")


if __name__ == "__main__":
    main()
