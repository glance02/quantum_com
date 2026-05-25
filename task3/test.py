from qml_models import (
    MODEL_SPECS,
    HybridQuantumClassifier,
    artifact_path,
    data_path,
    load_artifacts,
    load_csv,
    score,
    transform_features,
)
from pyvqnet.tensor import QTensor


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
    if not artifact_path().exists():
        raise FileNotFoundError(f"Missing trained artifact: {artifact_path()}. Please run train.py first.")

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
