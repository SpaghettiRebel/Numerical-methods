import numpy as np

from common import Matrix, Vector


def grid_of(left: float, right: float, step: float) -> Vector:
    """Равномерная сетка; шаг должен укладываться целое число раз."""
    count = (right - left) / step
    if abs(count - round(count)) > 1e-9:
        raise ValueError(f"шаг {step} не укладывается целое число раз в [{left}, {right}]")
    return left + step * np.arange(round(count) + 1)


def euler(rhs, left: float, right: float, step: float, start: Vector) -> tuple[Vector, Matrix]:
    """Метод Эйлера: u(k+1) = u(k) + h*f(x(k), u(k)); первый порядок точности."""
    x = grid_of(left, right, step)
    values = np.zeros((len(x), len(start)))
    values[0] = start
    for k in range(len(x) - 1):
        values[k + 1] = values[k] + step * rhs(x[k], values[k])
    return x, values


def runge_kutta(rhs, left: float, right: float, step: float,
                start: Vector) -> tuple[Vector, Matrix]:
    """Метод Рунге-Кутты 4-го порядка: усреднение четырёх пробных наклонов."""
    x = grid_of(left, right, step)
    values = np.zeros((len(x), len(start)))
    values[0] = start
    for k in range(len(x) - 1):
        u = values[k]
        k1 = rhs(x[k], u)
        k2 = rhs(x[k] + step / 2.0, u + step / 2.0 * k1)
        k3 = rhs(x[k] + step / 2.0, u + step / 2.0 * k2)
        k4 = rhs(x[k] + step, u + step * k3)
        values[k + 1] = u + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x, values


def adams(rhs, left: float, right: float, step: float, start: Vector) -> tuple[Vector, Matrix]:
    """Явный метод Адамса 4-го порядка; первые три шага считаются методом Рунге-Кутты."""
    x = grid_of(left, right, step)
    if len(x) < 5:
        raise ValueError("для метода Адамса нужно не менее четырёх шагов сетки")
    values = np.zeros((len(x), len(start)))
    values[:4] = runge_kutta(rhs, left, left + 3.0 * step, step, start)[1]

    derivatives = [rhs(x[k], values[k]) for k in range(4)]
    for k in range(3, len(x) - 1):
        values[k + 1] = values[k] + step / 24.0 * (55.0 * derivatives[k]
                                                   - 59.0 * derivatives[k - 1]
                                                   + 37.0 * derivatives[k - 2]
                                                   - 9.0 * derivatives[k - 3])
        derivatives.append(rhs(x[k + 1], values[k + 1]))
    return x, values


def runge_romberg(coarse: Vector, fine: Vector, order: int) -> tuple[Vector, Vector]:
    """Уточнение по Рунге-Ромбергу и оценка погрешности решения с меньшим шагом."""
    refined = fine + (fine - coarse) / (2.0 ** order - 1.0)
    return refined, np.abs(refined - fine)
