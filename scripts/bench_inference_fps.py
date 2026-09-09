#!/usr/bin/env python3
"""Чистая пропускная способность модели на CPU — то, что стоит в таблице README.

Меряет только вызов инференса: синтетический вход нужной модели формы, прогрев,
затем серия замеров. Декодирование видео, препроцессинг (grayscale + resize) и
разбор выхода сюда не входят — сквозной конвейер всегда медленнее, потому что
делит те же ядра с декодером; на GridV3 разница выходит примерно в 10%.

Модель грузится заново на каждый повтор: так в замер попадает и разброс от
того, как OpenVINO разложит сеть по потокам, а он между загрузками заметен.
Итог — медиана повторов, она устойчивее среднего к одиночному выбросу.

    uv run scripts/bench_inference_fps.py models/*.onnx
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_ov_models import V2  # noqa: E402  (нужен sys.path выше)

DEFAULT_WARMUP = 10
DEFAULT_RUNS = 60
DEFAULT_REPEATS = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("models", nargs="+", help="Пути к .xml или .onnx")
    parser.add_argument("--device", default="CPU", help="CPU, GPU, AUTO")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help=f"Прогревочных вызовов (по умолчанию {DEFAULT_WARMUP})")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help=f"Замеров в одном повторе (по умолчанию {DEFAULT_RUNS})")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help=f"Повторов с перезагрузкой модели (по умолчанию {DEFAULT_REPEATS})")
    return parser.parse_args()


def bench_model(path, device, warmup, runs, repeats, rng):
    fps_per_repeat = []
    params = None
    for _ in range(repeats):
        compiled, _, output_layer, params = V2.load_model(str(path), device=device)
        tensor = rng.random(
            (1, params["seq"], params["input_height"], params["input_width"])
        ).astype(np.float32)
        for _ in range(warmup):
            compiled(tensor)[output_layer]
        laps = []
        for _ in range(runs):
            started = time.perf_counter()
            compiled(tensor)[output_layer]
            laps.append(time.perf_counter() - started)
        # Один вызов обрабатывает seq кадров разом, отсюда и кадры в секунду.
        fps_per_repeat.append(params["seq"] / statistics.median(laps))
        del compiled
    return fps_per_repeat, params


def main():
    args = parse_args()
    rng = np.random.default_rng(0)
    print(f"Устройство: {args.device}, прогрев {args.warmup}, "
          f"замеров {args.runs} x {args.repeats} повторов\n")
    print(f"{'модель':<54} {'вход':>10} {'повторы, кадр/с':>26} {'медиана':>9}")
    print("-" * 104)
    for model_path in args.models:
        fps_runs, params = bench_model(
            model_path, args.device, args.warmup, args.runs, args.repeats, rng
        )
        shape = f"{params['input_width']}x{params['input_height']}"
        runs_text = " ".join(f"{value:.1f}" for value in fps_runs)
        print(f"{Path(model_path).name:<54} {shape:>10} {runs_text:>26} "
              f"{statistics.median(fps_runs):>9.1f}", flush=True)


if __name__ == "__main__":
    main()
