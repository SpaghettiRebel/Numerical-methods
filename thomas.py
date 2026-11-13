import numpy as np

from common import Vector


def forward_sweep(a: Vector, b: Vector, c: Vector, d: Vector) -> tuple[Vector, Vector]:
    """
    Прямой ход: P_i = -c_i / (b_i + a_i*P_(i-1)),
    Q_i = (d_i - a_i*Q_(i-1)) / (b_i + a_i*P_(i-1)).
    """
    n = len(b)
    p = np.zeros(n, dtype=b.dtype)
    q = np.zeros(n, dtype=b.dtype)
    previous_p = 0
    previous_q = 0

    for i in range(n):
        denominator = b[i] + a[i] * previous_p
        if denominator == 0:
            raise ValueError(f"нулевой знаменатель на шаге {i + 1}: прогонка неприменима")
        p[i] = -c[i] / denominator
        q[i] = (d[i] - a[i] * previous_q) / denominator
        previous_p = p[i]
        previous_q = q[i]

    return p, q


def backward_sweep(p: Vector, q: Vector) -> Vector:
    """Обратный ход: x_n = Q_n, x_i = P_i*x_(i+1) + Q_i."""
    n = len(p)
    x = np.zeros(n, dtype=p.dtype)
    x[n - 1] = q[n - 1]
    for i in reversed(range(n - 1)):
        x[i] = p[i] * x[i + 1] + q[i]
    return x


def solve(a: Vector, b: Vector, c: Vector, d: Vector) -> Vector:
    """Решение трёхдиагональной системы методом прогонки (прямой и обратный ход)."""
    p, q = forward_sweep(a, b, c, d)
    return backward_sweep(p, q)
