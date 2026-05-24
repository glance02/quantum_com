from math import cos, pi, sin

from pyqpanda3.core import CNOT, CPUQVM, H, QProg, RY, X, Z, measure, qif


# 题目要求制备的未知态为：
# |psi> = cos(pi / 6)|0> + sin(pi / 6)|1>
#
# RY(theta) 作用在 |0> 上时会得到：
# RY(theta)|0> = cos(theta / 2)|0> + sin(theta / 2)|1>
#
# 因此这里 theta 需要取 pi / 3，这样 theta / 2 = pi / 6。
UNKNOWN_STATE_THETA = pi / 3

# 原始未知态的两个振幅，后面用于和 Bob 恢复出的量子态计算保真度。
ALPHA = cos(pi / 6)
BETA = sin(pi / 6)


def prepare_bell_state(q1, q2):
    """在 q1、q2 上制备贝尔态 (|00> + |11>) / sqrt(2)。"""
    circuit = QProg()

    # 第一步：对 q1 施加 H 门。
    # |0> 经过 H 门后会变成 (|0> + |1>) / sqrt(2)，即进入叠加态。
    #
    # 第二步：以 q1 为控制比特、q2 为目标比特施加 CNOT。
    # 当 q1 是 |1> 时翻转 q2，当 q1 是 |0> 时不变。
    # 这样 q1 和 q2 就从普通叠加态变成纠缠态：
    # (|00> + |11>) / sqrt(2)。
    circuit << H(q1) << CNOT(q1, q2)
    return circuit


def teleportation_circuit(q0, q1, q2):
    """构建完整的量子隐形传态线路。

    参数含义：
    q0：Alice 手中的未知量子态，是要被传送的量子比特。
    q1：Alice 手中的纠缠比特。
    q2：Bob 手中的纠缠比特，最终应该恢复成 q0 原来的状态。
    """
    # QProg 中使用整数编号表示量子比特和经典位。
    # 这里根据传入的最大编号自动创建足够数量的量子比特。
    prog = QProg(max(q0, q1, q2) + 1)

    # 1. 在 q0 上制备题目指定的未知量子态：
    # cos(pi / 6)|0> + sin(pi / 6)|1>。
    prog << RY(q0, UNKNOWN_STATE_THETA)

    # 2. 在 q1、q2 上制备共享贝尔态。
    # q1 属于 Alice，q2 属于 Bob。
    prog << prepare_bell_state(q1, q2)

    # 3. Alice 对 q0 和 q1 做联合操作。
    # 标准隐形传态中，先做 CNOT(q0, q1)，再对 q0 做 H 门。
    # 这一步会把原始态的信息转移到后续测量结果和 Bob 的 q2 上。
    prog << CNOT(q0, q1) << H(q0)

    # 4. Alice 测量 q0、q1。
    # 这里把 q0 的测量结果存入经典位 q0，把 q1 的测量结果存入经典位 q1。
    # 由于 pyqpanda3 可以用整数编号表示经典位，这里直接复用同名编号。
    prog << measure(q0, q0) << measure(q1, q1)

    # 5. Bob 根据 Alice 发送来的经典测量结果修正 q2。
    #
    # 如果 q1 的测量结果为 1，需要对 q2 做 X 门修正。
    # X 门相当于经典比特翻转：|0> <-> |1>。
    prog << qif([q1]).then(QProg(max(q0, q1, q2) + 1) << X(q2)).qendif()

    # 如果 q0 的测量结果为 1，需要对 q2 做 Z 门修正。
    # Z 门会给 |1> 分量加一个负号，用来修正相位。
    prog << qif([q0]).then(QProg(max(q0, q1, q2) + 1) << Z(q2)).qendif()

    return prog


def run_program(prog, shots=1):
    """运行量子程序并返回模拟器结果。"""
    # CPUQVM 是 pyqpanda3 提供的 CPU 量子虚拟机。
    # shots 表示重复运行次数。这里主要看一次运行后的状态向量，所以默认取 1。
    qvm = CPUQVM()
    qvm.run(prog, shots)
    return qvm.result()


def extract_single_qubit_state(state_vector, qubit):
    """从完整状态向量中提取某个单量子比特的纯态。

    量子线路中一共有 3 个量子比特，所以模拟器返回的是 2^3 = 8 维状态向量。
    但题目关心的是 Bob 的 q2 最终状态，因此需要从完整状态向量里取出 q2
    对应的 |0> 和 |1> 振幅。

    注意：测量 q0、q1 以后，系统会坍缩到某一个测量分支。
    在完成条件修正后，q2 应该处在和原始 q0 相同的纯态上。
    """
    amp0 = 0j
    amp1 = 0j

    for index, amplitude in enumerate(state_vector):
        # 跳过数值上近似为 0 的振幅，避免浮点误差干扰判断。
        if abs(amplitude) < 1e-12:
            continue

        # pyqpanda3 的状态向量索引可以按二进制理解。
        # (index >> qubit) & 1 用来判断当前基态中该量子比特是 0 还是 1。
        if (index >> qubit) & 1:
            amp1 = amplitude
        else:
            amp0 = amplitude

    # 提取出来后做归一化，保证 |amp0|^2 + |amp1|^2 = 1。
    norm = (abs(amp0) ** 2 + abs(amp1) ** 2) ** 0.5
    if norm == 0:
        raise ValueError("Cannot extract a zero-norm state.")

    return amp0 / norm, amp1 / norm


def single_qubit_probabilities(state_vector, qubit):
    """计算某个单量子比特测得 0 和 1 的概率。"""
    prob0 = 0.0
    prob1 = 0.0

    for index, amplitude in enumerate(state_vector):
        # 量子态中某个基态的测量概率等于该基态振幅模长的平方。
        probability = abs(amplitude) ** 2

        # 根据当前基态里 qubit 这一位是 0 还是 1，把概率累加到对应结果上。
        if (index >> qubit) & 1:
            prob1 += probability
        else:
            prob0 += probability

    return {"0": prob0, "1": prob1}


def fidelity(state_a, state_b):
    """计算两个单量子比特纯态的保真度。

    对纯态 |psi> 和 |phi>，保真度为：
    F = |<psi|phi>|^2

    F 越接近 1，说明两个量子态越接近；F = 1 表示完全相同。
    """
    inner_product = state_a[0].conjugate() * state_b[0] + state_a[1].conjugate() * state_b[1]
    return abs(inner_product) ** 2


def format_complex(value):
    """把复数振幅格式化成更容易阅读的字符串。"""
    if abs(value.imag) < 1e-12:
        return f"{value.real:.12f}"
    return f"{value.real:.12f}{value.imag:+.12f}j"


def main():
    # 约定三个量子比特的编号：
    # q0 是 Alice 要传送的未知态；
    # q1 是 Alice 拥有的纠缠比特；
    # q2 是 Bob 拥有的纠缠比特。
    q0, q1, q2 = 0, 1, 2

    # 单独运行贝尔态制备，验证 prepare_bell_state 是否正确。
    bell_result = run_program(QProg(3) << prepare_bell_state(q1, q2))

    # 运行完整的量子隐形传态线路。
    teleportation_result = run_program(teleportation_circuit(q0, q1, q2))

    # 获取完整系统的最终状态向量，后续从中分析 Bob 的 q2。
    final_state_vector = teleportation_result.get_state_vector()

    # 原始态和恢复态。
    original_state = (complex(ALPHA), complex(BETA))
    recovered_state = extract_single_qubit_state(final_state_vector, q2)

    # Bob 的 q2 最终测量概率，以及恢复态和原始态的保真度。
    recovered_probabilities = single_qubit_probabilities(final_state_vector, q2)
    state_fidelity = fidelity(original_state, recovered_state)

    # 输出贝尔态振幅。
    # 对 q1、q2 来说，非零振幅应该对应 |00> 和 |11> 两项。
    print("Bell state amplitudes:")
    print(bell_result.get_state_vector())
    print()

    # 输出 Alice 的测量结果。
    # 由于量子测量具有随机性，每次运行得到的经典比特结果可能不同。
    print("Alice measurement counts:")
    print(teleportation_result.get_counts())
    print()

    # 输出 Bob 的 q2 概率。
    # 正确结果应接近：P(0)=0.75，P(1)=0.25。
    print("Bob q2 probabilities:")
    print(recovered_probabilities)
    print()

    # 输出原始态。
    print("Original state:")
    print(f"|psi> = {ALPHA:.12f}|0> + {BETA:.12f}|1>")
    print()

    # 输出 Bob 恢复后的 q2 状态。
    print("Recovered q2 state:")
    print(f"|phi> = {format_complex(recovered_state[0])}|0> + {format_complex(recovered_state[1])}|1>")
    print()

    # 输出保真度。
    # 如果协议实现正确，保真度应该非常接近 1。
    print(f"Fidelity = {state_fidelity:.12f}")


if __name__ == "__main__":
    main()
