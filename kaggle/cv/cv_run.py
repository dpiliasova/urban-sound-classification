"""Kaggle GPU entry point for corrected official-fold cross-validation."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
PROJECT_ROOT = WORKING_ROOT / "urban-sound-classification"
OUTPUT_DIR = WORKING_ROOT / "corrected_cross_validation"
CACHE_DIR = WORKING_ROOT / "spectrogram_cache"
REPOSITORY_URL = "https://github.com/dpiliasova/urban-sound-classification.git"
COMPATIBLE_TORCH_VERSION = "2.7.1"
COMPATIBLE_TORCHVISION_VERSION = "0.22.1"


def run(command: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, flush=True)
    return completed.stdout.strip()


def find_dataset_root() -> Path:
    for metadata_path in INPUT_ROOT.rglob("UrbanSound8K.csv"):
        if metadata_path.parent.name == "metadata":
            candidate = metadata_path.parent.parent
            if (candidate / "audio" / "fold1").is_dir():
                return candidate
        candidate = metadata_path.parent
        if (candidate / "fold1").is_dir():
            return candidate
    raise FileNotFoundError(
        "UrbanSound8K.csv and fold directories were not found under /kaggle/input"
    )


def package_versions() -> dict[str, str]:
    names = ["torch", "torchvision", "librosa", "numpy", "pandas", "scikit-learn"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def ensure_gpu_compatible_pytorch() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability(0)
    required_architecture = f"sm_{major}{minor}"
    compiled_architectures = set(torch.cuda.get_arch_list())
    if required_architecture in compiled_architectures:
        return
    if os.environ.get("URBAN_SOUND_TORCH_RESTARTED") == "1":
        raise RuntimeError(
            f"Installed PyTorch still does not support {required_architecture}: "
            f"{sorted(compiled_architectures)}"
        )
    print(
        f"PyTorch {torch.__version__} does not support {required_architecture}; "
        f"installing torch {COMPATIBLE_TORCH_VERSION} for the assigned GPU.",
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--upgrade",
            f"torch=={COMPATIBLE_TORCH_VERSION}",
            f"torchvision=={COMPATIBLE_TORCHVISION_VERSION}",
            "--index-url",
            "https://download.pytorch.org/whl/cu126",
        ],
        check=True,
    )
    environment = os.environ.copy()
    environment["URBAN_SOUND_TORCH_RESTARTED"] = "1"
    os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve())], environment)


def main() -> None:
    started_at = time.time()
    ensure_gpu_compatible_pytorch()
    if PROJECT_ROOT.exists():
        shutil.rmtree(PROJECT_ROOT)
    run(["git", "clone", "--depth", "1", REPOSITORY_URL, str(PROJECT_ROOT)])
    commit = run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).splitlines()[-1]
    run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(PROJECT_ROOT)])
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from urban_sound.config import AudioConfig, TrainingConfig
    from urban_sound.experiment import run_cross_validation

    dataset_root = find_dataset_root()

    import torch

    runtime = {
        "project_commit": commit,
        "dataset_root": str(dataset_root),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": package_versions(),
        "started_unix": started_at,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print(json.dumps(runtime, indent=2), flush=True)

    summary = run_cross_validation(
        data_root=dataset_root,
        output_dir=OUTPUT_DIR,
        audio_config=AudioConfig(),
        training_config=TrainingConfig(
            seed=1_337,
            batch_size=64,
            head_epochs=9,
            finetune_epochs=50,
            patience=15,
            num_workers=0,
            use_amp=True,
        ),
        pretrained=True,
        cache_dir=CACHE_DIR,
        preload_features=True,
    )

    runtime["finished_unix"] = time.time()
    runtime["elapsed_minutes"] = (runtime["finished_unix"] - started_at) / 60
    runtime["mean_accuracy"] = summary["mean_accuracy"]
    runtime["standard_deviation"] = summary["standard_deviation"]
    (OUTPUT_DIR / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")

    for checkpoint in OUTPUT_DIR.glob("fold_*/best_model.pth"):
        checkpoint.unlink()
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
