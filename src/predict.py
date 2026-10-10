"""Прогноз урожайности финальной моделью «Цифровой агроном» (TorchScript, экспорт из NNTrack).

Модель принимает 6 признаков в том виде, как в датасете (значения 0–1),
и возвращает hg/ha_yield в шкале датасета (1.0 = 100 000 гг/га = 10 т/га).

Примеры:
    python src/predict.py --values 0.5 0.333 0.952 0.45 0.012 0.62
    python src/predict.py --input examples/input_example.csv --output predictions.csv
"""
import argparse
import csv
import io
import math
import os
import sys

import torch

FEATURES = ["Area", "Item", "Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
TARGET = "hg/ha_yield"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(ROOT, "model", "model.torchscript")
# Коды Item (после масштабирования 0–1 шаг 1/9) — культуры исходного датасета FAO, по алфавиту
CROPS = ["маниок", "кукуруза", "бананы-плантайн", "картофель", "рис", "сорго", "соя", "батат", "пшеница", "ямс"]


class InputError(Exception):
    """Ошибка во входных данных — понятное сообщение без трассировки."""


def load_model(path):
    if not os.path.isfile(path):
        raise InputError(f"файл модели не найден: {path}")
    # Читаем через память: на Windows torch.jit.load не открывает пути с кириллицей.
    with open(path, "rb") as f:
        return torch.jit.load(io.BytesIO(f.read()), map_location="cpu").eval()


def parse_features(values, where):
    """Проверяет один набор признаков; возвращает (числа, предупреждения)."""
    if len(values) != len(FEATURES):
        raise InputError(f"{where}: ожидается {len(FEATURES)} признаков ({', '.join(FEATURES)}), "
                         f"получено {len(values)}")
    nums, warnings = [], []
    for name, raw in zip(FEATURES, values):
        try:
            v = float(str(raw).strip().replace(",", "."))
        except ValueError:
            raise InputError(f"{where}: признак {name} = «{raw}» — не число") from None
        if math.isnan(v) or math.isinf(v):
            raise InputError(f"{where}: признак {name} пустой или бесконечный")
        if not 0.0 <= v <= 1.0:
            warnings.append(f"{where}: {name} = {v} вне диапазона 0–1, на котором обучалась модель; "
                            f"прогноз может быть ненадёжным")
        nums.append(v)
    return nums, warnings


def read_csv(path):
    """Читает CSV: 6 столбцов признаков или 7 (последний — фактическая урожайность)."""
    if not os.path.isfile(path):
        raise InputError(f"файл не найден: {path}")
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.reader(f) if any(c.strip() for c in r)]
    if not rows:
        raise InputError(f"{path}: файл пустой")
    try:
        float(rows[0][0])
    except ValueError:
        rows = rows[1:]  # первая строка — заголовок
    if not rows:
        raise InputError(f"{path}: в файле только заголовок, нет данных")
    feats, actual, warnings = [], [], []
    for i, row in enumerate(rows, start=1):
        where = f"строка {i}"
        if len(row) == len(FEATURES) + 1:
            row, fact = row[:-1], row[-1]
            try:
                actual.append(float(fact))
            except ValueError:
                raise InputError(f"{where}: фактическая урожайность «{fact}» — не число") from None
        elif actual:
            raise InputError(f"{where}: в файле смешаны строки с фактом и без")
        nums, w = parse_features(row, where)
        feats.append(nums)
        warnings += w
    return feats, (actual or None), warnings


def crop_name(item):
    k = round(item * 9)
    return CROPS[k] if abs(item * 9 - k) < 0.05 and 0 <= k < len(CROPS) else "?"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Прогноз урожайности моделью TorchScript из NNTrack")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--values", nargs="+", metavar="X", help="6 признаков одной строки через пробел")
    src.add_argument("--input", help="CSV с признаками (6 столбцов) или признаками и фактом (7 столбцов)")
    ap.add_argument("--output", help="куда сохранить прогнозы (CSV)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="путь к model.torchscript")
    args = ap.parse_args(argv)

    try:
        if args.values:
            nums, warnings = parse_features(args.values, "ввод")
            feats, actual = [nums], None
        else:
            feats, actual, warnings = read_csv(args.input)
        model = load_model(args.model)
    except InputError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 2
    for w in warnings:
        print(f"Предупреждение: {w}", file=sys.stderr)

    with torch.no_grad():
        pred = model(torch.tensor(feats, dtype=torch.float32)).reshape(-1).tolist()

    print(f"{'№':>3}  {'культура':16s} {'прогноз hg/ha_yield':>20} {'т/га':>7}" + ("  факт т/га  ошибка" if actual else ""))
    for i, (x, p) in enumerate(zip(feats, pred), start=1):
        line = f"{i:>3}  {crop_name(x[1]):16s} {p:>20.4f} {max(p, 0) * 10:>7.2f}"
        if actual:
            line += f"  {actual[i - 1] * 10:>9.2f}  {(p - actual[i - 1]) * 10:>+6.2f}"
        print(line)
    if actual and len(actual) > 1:
        n = len(actual)
        mae = sum(abs(p - a) for p, a in zip(pred, actual)) / n
        rmse = math.sqrt(sum((p - a) ** 2 for p, a in zip(pred, actual)) / n)
        mean = sum(actual) / n
        ss = sum((a - mean) ** 2 for a in actual)
        r2 = 1 - sum((p - a) ** 2 for p, a in zip(pred, actual)) / ss if ss > 0 else float("nan")
        print(f"\nСтрок: {n}   R2 = {r2:.4f}   MAE = {mae:.4f}   RMSE = {rmse:.4f}  (в шкале датасета)")

    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(FEATURES + ["predicted_" + TARGET, "predicted_t_per_ha"] + ([TARGET] if actual else []))
            for i, (x, p) in enumerate(zip(feats, pred)):
                w.writerow(x + [round(p, 5), round(max(p, 0) * 10, 3)] + ([actual[i]] if actual else []))
        print(f"\nПрогнозы сохранены: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
