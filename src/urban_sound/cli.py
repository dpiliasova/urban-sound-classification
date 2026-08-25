"""Command-line entry point for reproducible experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from urban_sound.config import AudioConfig, TrainingConfig
from urban_sound.experiment import run_cross_validation, run_fold


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--seed", type=int, default=1_337)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--head-epochs", type=int, default=9)
    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples-per-split", type=int)
    parser.add_argument(
        "--preload-features",
        action="store_true",
        help="Keep cached single-channel spectrograms in memory across folds.",
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision.")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not download or load ImageNet weights; intended for pipeline smoke tests.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    development = subparsers.add_parser("train-dev", help="Run folds 1–8 / 9 / 10")
    _add_common_arguments(development)

    cross_validation = subparsers.add_parser(
        "cross-validate", help="Run rotating validation and test folds"
    )
    _add_common_arguments(cross_validation)
    cross_validation.add_argument("--folds", nargs="+", type=int, default=list(range(1, 11)))
    cross_validation.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    audio_config = AudioConfig()
    training_config = TrainingConfig(
        seed=arguments.seed,
        batch_size=arguments.batch_size,
        head_epochs=arguments.head_epochs,
        finetune_epochs=arguments.finetune_epochs,
        patience=arguments.patience,
        num_workers=arguments.num_workers,
        use_amp=not arguments.no_amp,
    )
    common = {
        "data_root": arguments.data_root,
        "output_dir": arguments.output_dir,
        "audio_config": audio_config,
        "training_config": training_config,
        "pretrained": not arguments.no_pretrained,
        "max_samples_per_split": arguments.max_samples_per_split,
        "cache_dir": arguments.cache_dir,
        "preload_features": arguments.preload_features,
    }

    if arguments.command == "train-dev":
        metrics = run_fold(test_fold=10, val_fold=9, **common)
        print(f"Test accuracy: {float(metrics['test_accuracy']):.2%}")
    else:
        summary = run_cross_validation(
            test_folds=tuple(arguments.folds),
            resume=arguments.resume,
            **common,
        )
        print(
            f"10-fold accuracy: {float(summary['mean_accuracy']):.2%} "
            f"± {float(summary['standard_deviation']):.2%}"
        )


if __name__ == "__main__":
    main()
