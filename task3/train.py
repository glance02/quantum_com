import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


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

FEATURE_COLUMNS = [
    "ma1",
    "ma2",
    "ma3",
    "ma4",
    "ma5",
    "ma6",
    "exudate1",
    "exudate2",
    "exudate3",
    "exudate5",
    "exudate6",
    "exudate7",
    "exudate8",
    "macula_opticdisc_distance",
    "opticdisc_diameter",
    "am_fm_classification",
]

MODEL_SPECS = {
    "baseline": {
        "num_qubits": 8,
        "active_qubits": 8,
        "feature_stride": 3,
        "layers": 3,
        "param_count": 80,
        "epochs": 1,
        "final_epochs": 4,
        "starts": 16,
        "batch_size": 256,
        "lr": 0.12,
        "init_scale": 0.5,
        "spsa_c": 0.18,
        "spsa_repeats": 1,
        "grad_clip": 2.5,
        "l2_grid": [0.005, 0.01, 0.02],
        "shots": 30000,
    },
    "lightweight": {
        "num_qubits": 4,
        "active_qubits": 4,
        "feature_stride": 3,
        "layers": 2,
        "param_count": 24,
        "epochs": 3,
        "final_epochs": 6,
        "starts": 50,
        "batch_size": 256,
        "lr": 0.14,
        "init_scale": 0.55,
        "spsa_c": 0.2,
        "spsa_repeats": 3,
        "grad_clip": 2.5,
        "l2_grid": [0.005, 0.01, 0.02],
        "shots": 30000,
    },
}

_PAIR_CACHE = {}
_CNOT_CACHE = {}
_OBS_CACHE = {}


def project_root():
    return Path(__file__).resolve().parent


def data_path(name):
    return project_root() / "data" / name


def artifacts_dir():
    return project_root() / "artifacts"


def model_artifact_path(name):
    return artifacts_dir() / f"{name}_model.npz"


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    raw_x = np.array([[float(row[col]) for col in FEATURE_COLUMNS] for row in rows], dtype=np.float64)
    x = augment_features(raw_x)
    y = np.array([int(row["class"]) for row in rows], dtype=np.float64)
    return x, y


def augment_features(x):
    ma = x[:, 0:6]
    exudate = x[:, 6:13]
    distance = x[:, 13]
    diameter = x[:, 14]
    am_fm = x[:, 15]
    eps = 1e-6
    ma_sum = ma.sum(axis=1)
    ma_mean = ma.mean(axis=1)
    exudate_sum = exudate.sum(axis=1)
    exudate_tail_sum = exudate[:, 3:].sum(axis=1)
    anatomy_ratio = distance / (diameter + eps)

    engineered = np.stack(
        [
            ma_sum,
            ma[:, 0],
            ma[:, 0] - ma[:, -1],
            ma[:, 0] / (ma_mean + eps),
            exudate_sum,
            exudate_tail_sum,
            exudate_tail_sum / (exudate_sum + eps),
            ma_sum / (exudate_sum + eps),
            anatomy_ratio,
            am_fm * anatomy_ratio,
        ],
        axis=1,
    )
    return np.concatenate([x, engineered], axis=1)


def fit_preprocessor(x):
    lo = np.quantile(x, 0.01, axis=0)
    hi = np.quantile(x, 0.99, axis=0)
    clipped = np.clip(x, lo, hi)
    mean = clipped.mean(axis=0)
    scale = clipped.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return {"lo": lo, "hi": hi, "mean": mean, "scale": scale}


def transform_features(x, stats):
    clipped = np.clip(x, stats["lo"], stats["hi"])
    standardized = (clipped - stats["mean"]) / stats["scale"]
    return ((np.clip(standardized, -3.0, 3.0) + 3.0) / 6.0 * math.pi).astype(np.float32)


def stratified_split(y, val_per_class=86, seed=7):
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    for label in (0, 1):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        val_parts.append(idx[:val_per_class])
        train_parts.append(idx[val_per_class:])
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _qubit_pairs(num_qubits, qubit):
    key = (num_qubits, qubit)
    if key not in _PAIR_CACHE:
        zero_idx = np.array(
            [idx for idx in range(1 << num_qubits) if ((idx >> qubit) & 1) == 0],
            dtype=np.int64,
        )
        _PAIR_CACHE[key] = (zero_idx, zero_idx | (1 << qubit))
    return _PAIR_CACHE[key]


def _apply_ry(state, num_qubits, qubit, theta):
    zero_idx, one_idx = _qubit_pairs(num_qubits, qubit)
    a = state[:, zero_idx].copy()
    b = state[:, one_idx].copy()
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    if np.ndim(c):
        c = c[:, None]
        s = s[:, None]
    state[:, zero_idx] = c * a - s * b
    state[:, one_idx] = s * a + c * b


def _apply_rz(state, num_qubits, qubit, theta):
    zero_idx, one_idx = _qubit_pairs(num_qubits, qubit)
    e0 = np.exp(-0.5j * theta)
    e1 = np.exp(0.5j * theta)
    if np.ndim(e0):
        e0 = e0[:, None]
        e1 = e1[:, None]
    state[:, zero_idx] *= e0
    state[:, one_idx] *= e1


def _apply_rx(state, num_qubits, qubit, theta):
    zero_idx, one_idx = _qubit_pairs(num_qubits, qubit)
    a = state[:, zero_idx].copy()
    b = state[:, one_idx].copy()
    c = np.cos(theta / 2.0)
    s = -1j * np.sin(theta / 2.0)
    if np.ndim(c):
        c = c[:, None]
        s = s[:, None]
    state[:, zero_idx] = c * a + s * b
    state[:, one_idx] = s * a + c * b


def _apply_cnot(state, num_qubits, control, target):
    key = (num_qubits, control, target)
    if key not in _CNOT_CACHE:
        zero_target = np.array(
            [
                idx
                for idx in range(1 << num_qubits)
                if ((idx >> control) & 1) == 1 and ((idx >> target) & 1) == 0
            ],
            dtype=np.int64,
        )
        _CNOT_CACHE[key] = (zero_target, zero_target | (1 << target))
    zero_idx, one_idx = _CNOT_CACHE[key]
    tmp = state[:, zero_idx].copy()
    state[:, zero_idx] = state[:, one_idx]
    state[:, one_idx] = tmp


def _observable_matrix(num_qubits):
    if num_qubits not in _OBS_CACHE:
        basis = np.arange(1 << num_qubits)
        z_terms = np.stack([1 - 2 * ((basis >> q) & 1) for q in range(num_qubits)], axis=1)
        zz_terms = np.stack([z_terms[:, q] * z_terms[:, (q + 1) % num_qubits] for q in range(num_qubits)], axis=1)
        _OBS_CACHE[num_qubits] = np.concatenate([z_terms, zz_terms], axis=1).astype(np.float64)
    return _OBS_CACHE[num_qubits]


def _feature_index(base, feature_dim, stride):
    return (base * stride + base // feature_dim) % feature_dim


def simulate_quantum_features(x, theta, num_qubits, layers, active_qubits=None, feature_stride=1):
    active_qubits = num_qubits if active_qubits is None else active_qubits
    batch = x.shape[0]
    state = np.zeros((batch, 1 << num_qubits), dtype=np.complex128)
    state[:, 0] = 1.0

    for q in range(active_qubits):
        _apply_ry(state, num_qubits, q, x[:, _feature_index(q, x.shape[1], feature_stride)])
        _apply_rz(state, num_qubits, q, x[:, _feature_index(q + active_qubits, x.shape[1], feature_stride)])

    p = 0
    for layer in range(layers):
        for q in range(active_qubits):
            _apply_ry(
                state,
                num_qubits,
                q,
                0.5 * x[:, _feature_index(2 * layer * active_qubits + q, x.shape[1], feature_stride)],
            )
            _apply_rz(
                state,
                num_qubits,
                q,
                0.5 * x[:, _feature_index(2 * layer * active_qubits + active_qubits + q, x.shape[1], feature_stride)],
            )

        for q in range(num_qubits):
            _apply_ry(state, num_qubits, q, theta[p])
            p += 1
            _apply_rz(state, num_qubits, q, theta[p])
            p += 1
            _apply_rx(state, num_qubits, q, theta[p])
            p += 1

        entangle_order = range(active_qubits) if layer % 2 == 0 else reversed(range(active_qubits))
        for q in entangle_order:
            _apply_cnot(state, num_qubits, q, (q + 1) % active_qubits)

    while p < len(theta):
        _apply_ry(state, num_qubits, p % num_qubits, theta[p])
        p += 1

    probabilities = state.real * state.real + state.imag * state.imag
    diagonal_features = probabilities @ _observable_matrix(num_qubits)
    x_features = []
    for q in range(num_qubits):
        zero_idx, one_idx = _qubit_pairs(num_qubits, q)
        coherence = np.conj(state[:, zero_idx]) * state[:, one_idx]
        x_features.append(2.0 * np.real(coherence).sum(axis=1))
    y_features = []
    for q in range(num_qubits):
        zero_idx, one_idx = _qubit_pairs(num_qubits, q)
        coherence = np.conj(state[:, zero_idx]) * state[:, one_idx]
        y_features.append(2.0 * np.imag(coherence).sum(axis=1))
    basis = np.arange(1 << num_qubits)
    xx_features = []
    for q in range(num_qubits):
        mask = (1 << q) | (1 << ((q + 1) % num_qubits))
        xx_features.append(np.real(np.sum(np.conj(state) * state[:, basis ^ mask], axis=1)))
    return np.concatenate(
        [
            diagonal_features,
            np.stack(x_features, axis=1),
            np.stack(y_features, axis=1),
            np.stack(xx_features, axis=1),
        ],
        axis=1,
    ).astype(np.float32)


def accuracy_from_prob(prob, y):
    return float(np.mean((prob >= 0.5).astype(np.int64) == y.astype(np.int64)))


def evaluate_numpy(params, x, y, name):
    spec = MODEL_SPECS[name]
    q = simulate_quantum_features(
        x,
        params[f"{name}_theta"],
        spec["num_qubits"],
        spec["layers"],
        spec.get("active_qubits", spec["num_qubits"]),
        spec.get("feature_stride", 1),
    )
    bias = float(params[f"{name}_bias"].reshape(-1)[0])
    prob = sigmoid_np(q @ params[f"{name}_weight"].reshape(-1) + bias)
    return accuracy_from_prob(prob, y), prob


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def save_artifacts(stats, baseline_params, lightweight_params):
    artifacts_dir().mkdir(parents=True, exist_ok=True)
    common_stats = {
        "lo": stats["lo"].astype(np.float32),
        "hi": stats["hi"].astype(np.float32),
        "mean": stats["mean"].astype(np.float32),
        "scale": stats["scale"].astype(np.float32),
    }
    np.savez(model_artifact_path("baseline"), **common_stats, **baseline_params)
    np.savez(model_artifact_path("lightweight"), **common_stats, **lightweight_params)


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
        spec.get("feature_stride", 1),
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
        rng = np.random.default_rng(seed + start)
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

    log_path = artifacts_dir() / "train_log.csv"
    summary_path = artifacts_dir() / "train_summary.json"
    logger = TrainingLogger(log_path)

    x, y = load_csv(data_path("train.csv"))
    stats = fit_preprocessor(x)
    x_full = transform_features(x, stats)
    cv_splits = make_cv_splits(y)

    try:
        baseline_params, baseline_info = train_one_model("baseline", x_full, y, cv_splits, seed=0, logger=logger)
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
        "baseline_model_path": str(model_artifact_path("baseline")),
        "lightweight_model_path": str(model_artifact_path("lightweight")),
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

    print(f"saved_baseline={model_artifact_path('baseline')}")
    print(f"saved_lightweight={model_artifact_path('lightweight')}")
    print(f"log={log_path}")
    print(f"summary={summary_path}")
    print(f"selection_cv_acc_baseline={baseline_info['cv_acc']:.4f} l2={baseline_info['l2']:.3g}")
    print(f"selection_cv_acc_lightweight={lightweight_info['cv_acc']:.4f} l2={lightweight_info['l2']:.3g}")
    print(f"train_acc_baseline={acc_b:.4f}")
    print(f"train_acc_lightweight={acc_l:.4f}")


if __name__ == "__main__":
    main()
