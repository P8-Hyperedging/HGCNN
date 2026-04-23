import time
import itertools
import dataclasses

from data.n_preprocessing import FeatureConfig
from model.QualityHGNN_V2.train import Train_QHGNN_v2

# Short labels for each flag, in field order
FLAG_LABELS = {
    "use_review_count":  "rc",
    "use_location":      "loc",
    "use_opening_hours": "oh",
    "use_categories":    "cat",
    "use_is_open":       "io",
    "use_state_onehot":  "st",
    "use_city_freq":     "cf",
}

def config_label(config: FeatureConfig) -> str:
    parts = [short for field, short in FLAG_LABELS.items() if getattr(config, field)]
    return "+".join(parts) if parts else "none"

def all_configs() -> list[FeatureConfig]:
    fields = list(FLAG_LABELS.keys())
    configs = []
    for values in itertools.product([False, True], repeat=len(fields)):
        configs.append(FeatureConfig(**dict(zip(fields, values))))
    return configs

def print_results_table(results: list[tuple[str, float, float]]):
    results_sorted = sorted(results, key=lambda r: r[1], reverse=True)
    col_w = max(len(r[0]) for r in results_sorted)
    sep = "-" * (col_w + 32)
    print(f"\n{'='*len(sep)}")
    print(f"  ABLATION RESULTS  ({len(results)} runs)")
    print(f"{'='*len(sep)}")
    print(f"  {'Features':<{col_w}}  {'Val Acc':>8}  {'F1':>8}  {'Time':>8}")
    print(sep)
    for label, acc, f1, t in results_sorted:
        print(f"  {label:<{col_w}}  {acc:>7.2f}%  {f1:>8.4f}  {t:>6.1f}s")
    print(sep)
    best = results_sorted[0]
    print(f"\n  Best: {best[0]}  ->  {best[1]:.2f}% acc")

if __name__ == "__main__":
    epochs = 500
    seed = 42  # fixed seed so all configs are comparable

    trainer = Train_QHGNN_v2()  # load reviews once

    configs = all_configs()
    print(f"Running {len(configs)} ablation configurations...\n")

    results = []
    for i, config in enumerate(configs):
        label = config_label(config)
        print(f"\n[{i+1}/{len(configs)}] {label}")

        run_start = time.time()
        result = trainer.train(
            num_epochs=epochs,
            model_name=f"QHGNN_v2[{label}]",
            feature_config=config,
            seed=seed,
        )
        elapsed = time.time() - run_start

        results.append((label, result.valid_acc, result.valid_f1, elapsed))

    print_results_table(results)
