"""Оценка финальной модели и анализ ошибок.

Воспроизводит разбиение NNTrack 2.3.1 для проектов «Большие данные»:
random_split(dataset, [train, val], generator=torch.Generator().manual_seed(123)), val = int(N * 0.2).
R² на проверке совпадает с тем, что показывает NNTrack (0.9846).

Запуск:
    python src/evaluate.py
    python src/evaluate.py --save docs/error_analysis.txt
"""
import argparse
import io
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "data", "yield_df_training_scaled.csv")
MODEL = os.path.join(ROOT, "model", "model.torchscript")
SEED, VAL_PART = 123, 0.2
CROPS = ["Маниок", "Кукуруза", "Бананы-плантайн", "Картофель", "Рис", "Сорго", "Соя", "Батат", "Пшеница", "Ямс"]


def metrics(pred, true):
    err = pred - true
    r2 = 1 - (err ** 2).sum() / ((true - true.mean()) ** 2).sum()
    return r2, np.abs(err).mean(), np.sqrt((err ** 2).mean())


def nntrack_split(n):
    val = int(n * VAL_PART)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEED)).tolist()
    return np.array(perm[: n - val]), np.array(perm[n - val:])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Метрики и анализ ошибок финальной модели")
    ap.add_argument("--data", default=DATASET)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--save", help="сохранить отчёт в текстовый файл")
    args = ap.parse_args(argv)

    data = np.loadtxt(args.data, delimiter=",", skiprows=1, dtype=np.float32)
    X, y = data[:, :6], data[:, 6]
    with open(args.model, "rb") as f:  # через память: пути с кириллицей на Windows
        model = torch.jit.load(io.BytesIO(f.read()), map_location="cpu").eval()
    with torch.no_grad():
        p = model(torch.from_numpy(X)).numpy().ravel()

    tr, va = nntrack_split(len(X))
    item = np.round(X[:, 1] * 9).astype(int)
    year = 1990 + np.round(X[:, 2] * 21).astype(int)  # восстановлено по шагу нормировки (2003 г. в данных нет)

    out = []
    w = out.append
    w(f"Строк: {len(X)}; обучение: {len(tr)}; проверка: {len(va)} (random_split NNTrack, seed={SEED})")
    w("Единицы: шкала датасета hg/ha_yield (1.0 = 10 т/га)\n")
    w("Метрики:")
    for name, idx in (("обучение", tr), ("проверка", va), ("весь датасет", np.arange(len(X)))):
        r2, mae, rmse = metrics(p[idx], y[idx])
        w(f"  {name:13s} R2={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")
    base = y[tr].mean()
    w(f"  база «всегда среднее» на проверке: MAE={np.abs(y[va] - base).mean():.4f}  "
      f"RMSE={np.sqrt(((y[va] - base) ** 2).mean()):.4f}\n")

    w("Ошибки по культурам (проверка):")
    w(f"  {'культура':16s} {'строк':>6} {'ср. урож., т/га':>16} {'MAE, т/га':>10} {'MAE, % урожая':>14}")
    rows = []
    for k in range(10):
        idx = va[item[va] == k]
        _, mae, _ = metrics(p[idx], y[idx])
        rows.append((CROPS[k], len(idx), y[idx].mean() * 10, mae * 10, mae / y[idx].mean()))
    for name, cnt, mean, mae, rel in sorted(rows, key=lambda r: -r[3]):
        w(f"  {name:16s} {cnt:>6} {mean:>16.2f} {mae:>10.2f} {rel:>13.1%}")

    w("\nКачество по годам (проверка):")
    for yr in sorted(set(year[va])):
        idx = va[year[va] == yr]
        r2, mae, _ = metrics(p[idx], y[idx])
        w(f"  {yr}  строк={len(idx):4d}  R2={r2:.4f}  MAE={mae * 10:.2f} т/га")

    err = p[va] - y[va]
    w("\nКрупнейшие ошибки (проверка), т/га:")
    w(f"  {'культура':16s} {'год':>5} {'факт':>7} {'прогноз':>8} {'ошибка':>8}")
    for j in np.argsort(-np.abs(err))[:10]:
        i = va[j]
        w(f"  {CROPS[item[i]]:16s} {year[i]:>5} {y[i] * 10:>7.1f} {p[i] * 10:>8.1f} {err[j] * 10:>+8.1f}")
    ae = np.abs(err) * 10
    w("\nАбсолютная ошибка на проверке, т/га: "
      + ", ".join(f"{q}% строк < {np.percentile(ae, q):.2f}" for q in (50, 75, 90, 95)))
    w(f"Отрицательных прогнозов на всём датасете: {(p < 0).sum()} из {len(p)}")

    text = "\n".join(out)
    print(text)
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\nОтчёт сохранён: {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
