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
    "baseline": {"num_qubits": 8, "layers": 3, "param_count": 80, "epochs": 18, "lr": 0.035, "shots": 5000},
    "lightweight": {"num_qubits": 4, "layers": 2, "param_count": 24, "epochs": 28, "lr": 0.05, "shots": 5000},
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
    x = np.array([[float(row[col]) for col in FEATURE_COLUMNS] for row in rows], dtype=np.float64)
    y = np.array([int(row["class"]) for row in rows], dtype=np.float64)
    return x, y


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
    return (np.clip(standardized, -3.0, 3.0) / 3.0 * math.pi).astype(np.float32)


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
    return 2 * num_qubits


def simulate_quantum_features(x, theta, num_qubits, layers):
    batch = x.shape[0]
    state = np.zeros((batch, 1 << num_qubits), dtype=np.complex128)
    state[:, 0] = 1.0

    for q in range(num_qubits):
        _apply_ry(state, num_qubits, q, x[:, q % x.shape[1]])
        _apply_rz(state, num_qubits, q, x[:, (q + num_qubits) % x.shape[1]])

    p = 0
    for layer in range(layers):
        for q in range(num_qubits):
            _apply_ry(state, num_qubits, q, 0.5 * x[:, (2 * layer * num_qubits + q) % x.shape[1]])
            _apply_rz(state, num_qubits, q, 0.5 * x[:, (2 * layer * num_qubits + num_qubits + q) % x.shape[1]])

        for q in range(num_qubits):
            _apply_ry(state, num_qubits, q, theta[p])
            p += 1
            _apply_rz(state, num_qubits, q, theta[p])
            p += 1
            _apply_rx(state, num_qubits, q, theta[p])
            p += 1

        entangle_order = range(num_qubits) if layer % 2 == 0 else reversed(range(num_qubits))
        for q in entangle_order:
            _apply_cnot(state, num_qubits, q, (q + 1) % num_qubits)

    while p < len(theta):
        _apply_ry(state, num_qubits, p % num_qubits, theta[p])
        p += 1

    probabilities = state.real * state.real + state.imag * state.imag
    return (probabilities @ _observable_matrix(num_qubits)).astype(np.float32)


def build_pqanda3_program(x, theta, num_qubits, layers):
    circuit = pq.QCircuit(num_qubits)
    for q in range(num_qubits):
        circuit << pq.RY(q, x[q % len(x)])
        circuit << pq.RZ(q, x[(q + num_qubits) % len(x)])

    p = 0
    for layer in range(layers):
        for q in range(num_qubits):
            circuit << pq.RY(q, 0.5 * x[(2 * layer * num_qubits + q) % len(x)])
            circuit << pq.RZ(q, 0.5 * x[(2 * layer * num_qubits + num_qubits + q) % len(x)])

        for q in range(num_qubits):
            circuit << pq.RY(q, theta[p])
            p += 1
            circuit << pq.RZ(q, theta[p])
            p += 1
            circuit << pq.RX(q, theta[p])
            p += 1

        entangle_order = range(num_qubits) if layer % 2 == 0 else reversed(range(num_qubits))
        for q in entangle_order:
            circuit << pq.CNOT(q, (q + 1) % num_qubits)

    while p < len(theta):
        circuit << pq.RY(p % num_qubits, theta[p])
        p += 1

    program = pq.QProg()
    program << circuit
    return program


def make_quantum_layer(name):
    spec = MODEL_SPECS[name]
    num_qubits = spec["num_qubits"]
    layers = spec["layers"]
    observables = [{f"Z{i}": 1.0} for i in range(num_qubits)]
    observables += [{f"Z{i} Z{(i + 1) % num_qubits}": 1.0} for i in range(num_qubits)]

    def qfun(x, theta):
        return build_pqanda3_program(x, theta, num_qubits, layers)

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
    q = simulate_quantum_features(x, params[f"{name}_theta"], spec["num_qubits"], spec["layers"])
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
