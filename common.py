import json
import os
import sys
from fractions import Fraction
from typing import Any

import numpy as np

Number = int | float | Fraction | complex
Vector = np.ndarray
Matrix = np.ndarray


def data_path(default_name: str) -> str:
    """Путь к файлу данных: аргумент командной строки или файл из папки data."""
    for argument in sys.argv[1:]:
        if not argument.startswith("-"):
            return argument
    folder = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(folder, "data", default_name)


def read_data(path: str) -> dict:
    """Читает JSON-файл с исходными данными."""
    if not os.path.exists(path):
        raise ValueError(f"файл '{path}' не найден")
    with open(path, encoding="utf-8") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(f"файл '{path}' не является корректным JSON: {error}")
    if not isinstance(data, dict):
        raise ValueError(f"файл '{path}' должен содержать JSON-объект")
    return data


def read_number(raw: Any, where: str) -> Fraction:
    """Число из файла в виде точной дроби; допускаются 12, -0.75 и "-3/4"."""
    if isinstance(raw, bool):
        raise ValueError(f"{where}: логическое значение не является числом")
    if isinstance(raw, int):
        return Fraction(raw)
    if isinstance(raw, float):
        return Fraction(str(raw))
    if isinstance(raw, str):
        try:
            return Fraction(raw.strip().replace(",", "."))
        except (ValueError, ZeroDivisionError):
            raise ValueError(f"{where}: '{raw}' не является числом")
    raise ValueError(f"{where}: '{raw}' не является числом")


def read_matrix(data: dict, key: str = "A", square: bool = True) -> Matrix:
    """Читает матрицу точных дробей (массив dtype=object) и проверяет её размеры."""
    if key not in data:
        raise ValueError(f"в исходных данных нет ключа '{key}'")
    rows = data[key]
    if not isinstance(rows, list) or len(rows) == 0:
        raise ValueError(f"'{key}' должен быть непустым списком строк матрицы")
    matrix = []
    for i, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != len(rows[0]):
            raise ValueError(f"'{key}': строка {i + 1} должна быть списком из {len(rows[0])} чисел")
        matrix.append([read_number(value, f"{key}[{i + 1}][{j + 1}]")
                       for j, value in enumerate(row)])
    if square and len(matrix) != len(matrix[0]):
        raise ValueError(f"'{key}': матрица должна быть квадратной, получено "
                         f"{len(matrix)}x{len(matrix[0])}")
    return np.array(matrix, dtype=object)


def read_vector(data: dict, key: str, length: int | None = None) -> Vector:
    """Читает вектор точных дробей и при необходимости проверяет его длину."""
    if key not in data:
        raise ValueError(f"в исходных данных нет ключа '{key}'")
    values = data[key]
    if not isinstance(values, list) or len(values) == 0:
        raise ValueError(f"'{key}' должен быть непустым списком чисел")
    if length is not None and len(values) != length:
        raise ValueError(f"'{key}': {len(values)} элементов вместо {length}")
    return np.array([read_number(value, f"{key}[{i + 1}]") for i, value in enumerate(values)],
                    dtype=object)


def read_matrix_list(data: dict) -> list[tuple[str, Matrix]]:
    """Список пар (заголовок, матрица) по ключу "matrices" либо одна матрица "A"."""
    if "matrices" in data:
        items = data["matrices"]
        if not isinstance(items, list) or len(items) == 0:
            raise ValueError("'matrices' должен быть непустым списком матриц")
        result = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"'matrices[{i + 1}]' должен быть объектом с ключом 'A'")
            result.append((item.get("title", f"Матрица №{i + 1}"), read_matrix(item)))
        return result
    return [(data.get("title", "Матрица A"), read_matrix(data))]


def read_epsilon(data: dict, default: float = 0.01) -> float:
    """Читает заданную точность вычислений."""
    if "epsilon" not in data:
        return default
    epsilon = float(read_number(data["epsilon"], "epsilon"))
    if epsilon <= 0:
        raise ValueError("точность eps должна быть положительной")
    return epsilon


def read_scalar(data: dict, key: str) -> float:
    """Читает одно вещественное число по ключу."""
    if key not in data:
        raise ValueError(f"в исходных данных нет ключа '{key}'")
    return float(read_number(data[key], key))


def read_int(data: dict, key: str, default: int) -> int:
    """Читает натуральное число по ключу."""
    value = data.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"'{key}' должно быть натуральным числом")
    return value


def read_max_iterations(data: dict, default: int = 1000) -> int:
    """Читает предельное число итераций."""
    return read_int(data, "max_iterations", default)


def comment(data: dict) -> str:
    """Пояснение к данным из файла (если оно есть)."""
    return data.get("comment", "")


def check_square(matrix: Matrix, name: str = "матрица") -> int:
    """Проверяет, что матрица квадратная, и возвращает её порядок."""
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        shape = "x".join(map(str, matrix.shape))
        raise ValueError(f"{name} должна быть квадратной, получено {shape}")
    return matrix.shape[0]


def norm(matrix: Matrix) -> Number:
    """Бесконечная норма матрицы (наибольшая сумма модулей по строке)."""
    return np.abs(matrix).sum(axis=1).max()


def is_symmetric(matrix: Matrix, tolerance: float = 1e-9) -> bool:
    """Проверяет симметричность матрицы."""
    check_square(matrix)
    return np.abs(matrix - matrix.T).max() <= tolerance


def enable_utf8() -> None:
    """Переводит вывод в UTF-8, чтобы протокол корректно писался в файл под Windows."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass


def number(value: Number, precision: int = 6) -> str:
    """Число в виде строки: одинаково для float, Fraction и complex."""
    if isinstance(value, Fraction):
        value = float(value)
    if isinstance(value, complex):
        sign = "-" if value.imag < 0 else "+"
        return f"{number(value.real, precision)} {sign} {number(abs(value.imag), precision)}i"

    value = float(value)
    if abs(value) < 0.5 * 10 ** (-precision):
        value = 0.0
    return f"{value:.{precision}f}"


def exact(value: Number) -> str:
    """Точная запись числа: целое, обыкновенная дробь или обычное число."""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (int, np.integer, Fraction)):
        return str(value)
    return number(value)


def print_header(title: str) -> None:
    """Заголовок программы или её крупной части."""
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_section(title: str) -> None:
    """Заголовок пункта вывода."""
    print()
    print(title)
    print("-" * len(title))


def print_matrix(matrix: Matrix, title: str | None = None,
                 use_exact: bool = False, precision: int = 6) -> None:
    """Печатает матрицу с выравниванием по столбцам."""
    if title:
        print(title)
    if matrix.size == 0:
        print("  (пустая матрица)")
        return
    cells = [[exact(v) if use_exact else number(v, precision) for v in row] for row in matrix]
    width = max(len(cell) for row in cells for cell in row)
    for row in cells:
        print("  " + "  ".join(cell.rjust(width) for cell in row))


def print_vector(vector: Vector, title: str | None = None, name: str | None = None,
                 use_exact: bool = False, precision: int = 6) -> None:
    """Печатает вектор строкой или в виде x1 = ..., x2 = ..."""
    if title:
        print(title)
    if len(vector) == 0:
        print("  (пустой вектор)")
        return
    cells = [exact(v) if use_exact else number(v, precision) for v in vector]
    width = max(len(cell) for cell in cells)
    if name is None:
        print("  " + "  ".join(cell.rjust(width) for cell in cells))
        return
    label_width = len(f"{name}{len(vector)}")
    for i, cell in enumerate(cells, start=1):
        print(f"  {f'{name}{i}':<{label_width}} = {cell.rjust(width)}")


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Печатает таблицу с выравниванием по ширине столбцов."""
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    print("  " + "  ".join(headers[i].rjust(widths[i]) for i in range(len(headers))))
    print("  " + "  ".join("-" * width for width in widths))
    for row in rows:
        print("  " + "  ".join(row[i].rjust(widths[i]) for i in range(len(row))))


def short_table(rows: list[list[str]], max_rows: int = 25) -> list[list[str]]:
    """Сокращает длинную таблицу: начало, многоточие и последние строки."""
    if len(rows) <= max_rows:
        return rows
    return rows[:max_rows - 6] + [["..."] * len(rows[0])] + rows[-5:]


def print_check(title: str, deviation: float, tolerance: float = 1e-8) -> None:
    """Печатает результат проверки по наибольшему отклонению."""
    verdict = "ВЫПОЛНЕНО" if deviation <= tolerance else "НЕ ВЫПОЛНЕНО"
    print(f"  {title}: макс. отклонение = {deviation:.3e}  ->  {verdict}")


def check_matrices(title: str, a: Matrix, b: Matrix, tolerance: float = 1e-8) -> None:
    """Сравнивает две матрицы (или два вектора) и печатает вердикт."""
    print_check(title, float(np.abs(a - b).max()), tolerance)


def print_source(path: str, note: str) -> None:
    """Печатает, откуда взяты исходные данные."""
    print(f"Исходные данные: {path}")
    if note:
        print(f"Описание: {note}")


def print_system(matrix: Matrix, rhs: Vector,
                 title: str = "Расширенная матрица системы (A | b):") -> None:
    """Печатает расширенную матрицу системы."""
    print(title)
    cells = [[exact(value) for value in row] for row in matrix]
    free = [exact(value) for value in rhs]
    width = max(len(cell) for row in cells for cell in row)
    free_width = max(len(cell) for cell in free)
    for row, value in zip(cells, free):
        body = "  ".join(cell.rjust(width) for cell in row)
        print("  " + body + "  |  " + value.rjust(free_width))
