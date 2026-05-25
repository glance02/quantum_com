import csv
import math
from pathlib import Path

import numpy as np
import pyqpanda3.core as pq
from pyvqnet.nn import Linear, Module
from pyvqnet.qnn.pq3.quantumlayer import QuantumLayerV3
from pyvqnet.tensor import QTensor, log, mean, sigmoid
from pyvqnet.utils.initializer import zeros


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
        "starts": 8,
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
        "feature_stride": 1,
        "layers": 2,
        "param_count": 24,
        "epochs": 3,
        "final_epochs": 6,
        "starts": 25,
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


def artifact_path():
    return project_root() / "artifacts" / "qml_artifacts.npz"


def model_artifact_path(name):
    return project_root() / "artifacts" / f"{name}_model.npz"


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


def quantum_feature_dim(num_qubits):
    return 5 * num_qubits


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


def build_pqanda3_program(x, theta, num_qubits, layers, active_qubits=None, feature_stride=1):
    active_qubits = num_qubits if active_qubits is None else active_qubits
    circuit = pq.QCircuit(num_qubits)
    for q in range(active_qubits):
        circuit << pq.RY(q, x[_feature_index(q, len(x), feature_stride)])
        circuit << pq.RZ(q, x[_feature_index(q + active_qubits, len(x), feature_stride)])

    p = 0
    for layer in range(layers):
        for q in range(active_qubits):
            circuit << pq.RY(q, 0.5 * x[_feature_index(2 * layer * active_qubits + q, len(x), feature_stride)])
            circuit << pq.RZ(q, 0.5 * x[_feature_index(2 * layer * active_qubits + active_qubits + q, len(x), feature_stride)])

        for q in range(num_qubits):
            circuit << pq.RY(q, theta[p])
            p += 1
            circuit << pq.RZ(q, theta[p])
            p += 1
            circuit << pq.RX(q, theta[p])
            p += 1

        entangle_order = range(active_qubits) if layer % 2 == 0 else reversed(range(active_qubits))
        for q in entangle_order:
            circuit << pq.CNOT(q, (q + 1) % active_qubits)

    while p < len(theta):
        circuit << pq.RY(p % num_qubits, theta[p])
        p += 1

    program = pq.QProg()
    program << circuit
    return program


def make_quantum_layer(name):
    spec = MODEL_SPECS[name]
    num_qubits = spec["num_qubits"]
    active_qubits = spec.get("active_qubits", num_qubits)
    feature_stride = spec.get("feature_stride", 1)
    layers = spec["layers"]
    observables = [{f"Z{i}": 1.0} for i in range(num_qubits)]
    observables += [{f"Z{i} Z{(i + 1) % num_qubits}": 1.0} for i in range(num_qubits)]
    observables += [{f"X{i}": 1.0} for i in range(num_qubits)]
    observables += [{f"Y{i}": 1.0} for i in range(num_qubits)]
    observables += [{f"X{i} X{(i + 1) % num_qubits}": 1.0} for i in range(num_qubits)]

    def qfun(x, theta):
        return build_pqanda3_program(x, theta, num_qubits, layers, active_qubits, feature_stride)

    return QuantumLayerV3(
        qfun,
        spec["param_count"],
        "cpu",
        pauli_str_dict=observables,
        shots=spec["shots"],
        initializer=zeros,
        name=f"{name}_vqc",
    )


class HybridQuantumClassifier(Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        spec = MODEL_SPECS[name]
        self.num_qubits = spec["num_qubits"]
        self.layers = spec["layers"]
        self.param_count = spec["param_count"]
        self.quantum = make_quantum_layer(name)
        self.readout = Linear(quantum_feature_dim(self.num_qubits), 1, weight_initializer=zeros, bias_initializer=zeros)

    def forward(self, x):
        q = self.quantum(x)
        return sigmoid(self.readout(q))

    def predict_numpy(self, x):
        prob = self.forward(QTensor(x.astype(np.float32))).to_numpy().reshape(-1)
        return (prob >= 0.5).astype(np.int64), prob


def binary_cross_entropy(prob, target):
    return mean(-(target * log(prob + 1e-7) + (1.0 - target) * log(1.0 - prob + 1e-7)))


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
    prob = sigmoid_np(q @ params[f"{name}_weight"].reshape(-1) + float(params[f"{name}_bias"]))
    return accuracy_from_prob(prob, y), prob


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def extract_params(model):
    return {
        f"{model.name}_theta": model.quantum.m_para.to_numpy().astype(np.float32),
        f"{model.name}_weight": model.readout.weights.to_numpy().astype(np.float32),
        f"{model.name}_bias": model.readout.bias.to_numpy().astype(np.float32),
    }


def load_artifacts(path=None):
    path = artifact_path() if path is None else Path(path)
    baseline_path = model_artifact_path("baseline")
    lightweight_path = model_artifact_path("lightweight")
    if path == artifact_path() and baseline_path.exists() and lightweight_path.exists():
        base = np.load(baseline_path)
        light = np.load(lightweight_path)
        center_key = "mean" if "mean" in base.files else "median"
        stats = {key: base[key] for key in ("lo", "hi", "scale")}
        stats["mean"] = base[center_key]
        params = {}
        params.update({key: base[key] for key in base.files if key not in stats})
        params.update({key: light[key] for key in light.files if key not in stats})
        return stats, params

    data = np.load(path)
    center_key = "mean" if "mean" in data.files else "median"
    stats = {key: data[key] for key in ("lo", "hi", "scale")}
    stats["mean"] = data[center_key]
    params = {key: data[key] for key in data.files if key not in stats}
    return stats, params


def save_artifacts(stats, baseline_params, lightweight_params, path=None):
    path = artifact_path() if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    common_stats = {
        "lo": stats["lo"].astype(np.float32),
        "hi": stats["hi"].astype(np.float32),
        "mean": stats["mean"].astype(np.float32),
        "scale": stats["scale"].astype(np.float32),
    }
    np.savez(model_artifact_path("baseline"), **common_stats, **baseline_params)
    np.savez(model_artifact_path("lightweight"), **common_stats, **lightweight_params)
    np.savez(
        path,
        **common_stats,
        **baseline_params,
        **lightweight_params,
    )


def score(acc_b, acc_l):
    q_b = MODEL_SPECS["baseline"]["num_qubits"]
    p_b = MODEL_SPECS["baseline"]["param_count"]
    q_l = MODEL_SPECS["lightweight"]["num_qubits"]
    p_l = MODEL_SPECS["lightweight"]["param_count"]
    compression = 0.5 * ((q_b - q_l) / q_b + (p_b - p_l) / p_b)
    return 22 * acc_b + 22 * acc_l + 6 * compression * min(acc_l / acc_b, 1.0)
