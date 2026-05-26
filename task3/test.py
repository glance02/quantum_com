import csv
import math
from pathlib import Path

import numpy as np
import pyqpanda3.core as pq
from pyvqnet.nn import Linear, Module
from pyvqnet.qnn.pq3.quantumlayer import QuantumLayerV3
from pyvqnet.tensor import QTensor, sigmoid
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


def transform_features(x, stats):
    clipped = np.clip(x, stats["lo"], stats["hi"])
    standardized = (clipped - stats["mean"]) / stats["scale"]
    return ((np.clip(standardized, -3.0, 3.0) + 3.0) / 6.0 * math.pi).astype(np.float32)


def quantum_feature_dim(num_qubits):
    return 5 * num_qubits


def _feature_index(base, feature_dim, stride):
    return (base * stride + base // feature_dim) % feature_dim


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


def load_artifacts():
    baseline_path = model_artifact_path("baseline")
    lightweight_path = model_artifact_path("lightweight")
    base = np.load(baseline_path)
    light = np.load(lightweight_path)
    center_key = "mean" if "mean" in base.files else "median"
    stats = {key: base[key] for key in ("lo", "hi", "scale")}
    stats["mean"] = base[center_key]
    params = {}
    params.update({key: base[key] for key in base.files if key not in stats})
    params.update({key: light[key] for key in light.files if key not in stats})
    return stats, params


def score(acc_b, acc_l):
    q_b = MODEL_SPECS["baseline"]["num_qubits"]
    p_b = MODEL_SPECS["baseline"]["param_count"]
    q_l = MODEL_SPECS["lightweight"]["num_qubits"]
    p_l = MODEL_SPECS["lightweight"]["param_count"]
    compression = 0.5 * ((q_b - q_l) / q_b + (p_b - p_l) / p_b)
    return 22 * acc_b + 22 * acc_l + 6 * compression * min(acc_l / acc_b, 1.0)


def evaluate_vqnet(params, x, y, name, repeats=5):
    model = HybridQuantumClassifier(name)
    model.quantum.m_para.init_from_tensor(QTensor(params[f"{name}_theta"].astype("float32")))
    model.readout.weights.init_from_tensor(QTensor(params[f"{name}_weight"].astype("float32")))
    model.readout.bias.init_from_tensor(QTensor(params[f"{name}_bias"].astype("float32")))
    prob_sum = None
    for _ in range(repeats):
        _, prob = model.predict_numpy(x)
        prob_sum = prob if prob_sum is None else prob_sum + prob
    pred = (prob_sum / repeats >= 0.5).astype("int64")
    return float((pred == y.astype("int64")).mean())


def main():
    baseline_path = model_artifact_path("baseline")
    lightweight_path = model_artifact_path("lightweight")
    if not baseline_path.exists() or not lightweight_path.exists():
        raise FileNotFoundError(
            f"Missing trained model files: {baseline_path} and {lightweight_path}. Please run train.py first."
        )

    stats, params = load_artifacts()
    x_test, y_test = load_csv(data_path("test.csv"))
    x_test = transform_features(x_test, stats)

    acc_b = evaluate_vqnet(params, x_test, y_test, "baseline")
    acc_l = evaluate_vqnet(params, x_test, y_test, "lightweight")
    q_b = MODEL_SPECS["baseline"]["num_qubits"]
    p_b = MODEL_SPECS["baseline"]["param_count"]
    q_l = MODEL_SPECS["lightweight"]["num_qubits"]
    p_l = MODEL_SPECS["lightweight"]["param_count"]
    final_score = score(acc_b, acc_l)

    print(f"{final_score:.2f},{acc_b:.2f},{acc_l:.2f},{q_b},{p_b},{q_l},{p_l}")


if __name__ == "__main__":
    main()
