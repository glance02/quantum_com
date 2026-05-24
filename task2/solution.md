# 量子隐形传态解题思路

## 1. 贝尔态制备

贝尔态

$$
|\psi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)
$$

可以用两个门完成：

1. 对 `q1` 施加 Hadamard 门，使其变为 $\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$；
2. 以 `q1` 为控制比特、`q2` 为目标比特施加 CNOT 门，得到纠缠态。

因此 `prepare_bell_state(q1, q2)` 中的核心线路为：

```python
H(q1)
CNOT(q1, q2)
```

## 2. 未知量子态制备

题目要求制备

$$
|\psi\rangle = \cos(\pi / 6)|0\rangle + \sin(\pi / 6)|1\rangle
$$

`RY(theta)` 作用在 $|0\rangle$ 上时满足：

$$
R_y(\theta)|0\rangle = \cos(\theta / 2)|0\rangle + \sin(\theta / 2)|1\rangle
$$

所以这里取

$$
\theta = \pi / 3
$$

即可得到目标未知态。

## 3. 隐形传态线路

完整线路按标准协议构建：

1. 对 `q0` 使用 `RY(pi / 3)` 制备未知态；
2. 对 `q1`, `q2` 调用 `prepare_bell_state(q1, q2)` 制备共享贝尔态；
3. Alice 对 `q0`, `q1` 执行 `CNOT(q0, q1)` 和 `H(q0)`；
4. 测量 `q0`, `q1`，把测量结果分别写入同编号经典位；
5. Bob 根据测量结果修正 `q2`：
   - 如果 `q1` 的测量结果为 1，则对 `q2` 施加 `X`；
   - 如果 `q0` 的测量结果为 1，则对 `q2` 施加 `Z`。

对应代码中的条件修正为：

```python
qif([q1]).then(QProg(...) << X(q2)).qendif()
qif([q0]).then(QProg(...) << Z(q2)).qendif()
```

## 4. 保真度计算

原始态为：

$$
|\psi\rangle = \alpha |0\rangle + \beta |1\rangle
$$

Bob 恢复后的态记为：

$$
|\phi\rangle = a |0\rangle + b |1\rangle
$$

纯态保真度计算公式为：

$$
F = |\langle \psi | \phi \rangle|^2
$$

即：

$$
F = |\alpha^* a + \beta^* b|^2
$$

程序从最终状态向量中提取 `q2` 的单比特态，然后与原始态计算保真度。

## 5. 运行结果

使用 `quantum_com` 环境运行：

```powershell
C:\Users\glance\miniforge3\envs\quantum_com\python.exe task2\teleportation.py
```

一次运行的输出示例：

```text
Bell state amplitudes:
[(0.7071067811865476+0j), 0j, 0j, 0j, 0j, 0j, (0.7071067811865476+0j), 0j]

Alice measurement counts:
{'11': 1}

Bob q2 probabilities:
{'0': 0.7500000000000012, '1': 0.25000000000000033}

Original state:
|psi> = 0.866025403784|0> + 0.500000000000|1>

Recovered q2 state:
|phi> = 0.866025403784|0> + 0.500000000000|1>

Fidelity = 1.000000000000
```

由于 Alice 的测量结果具有随机性，`Alice measurement counts` 每次运行可能不同；但条件修正后 Bob 的 `q2` 都会恢复为原始态。最终 `q2` 的测量概率约为 $P(0)=0.75$、$P(1)=0.25$，与原始态一致，保真度为 1，说明量子隐形传态协议成功实现。
