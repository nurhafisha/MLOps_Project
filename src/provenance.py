from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import torch


def training_data_identity(
    train_manifest: Path = Path("data/processed/manifests/train.csv"),
    validation_manifest: Path = Path("data/processed/manifests/validation.csv"),
) -> str:
    """Return a deterministic identity for the training + validation manifests.

    This helper is provided because the final evaluator uses the same identity
    to verify that the selected MLflow Run matches the current prepared data.
    Do not hard-code the value.
    """
    paths = (train_manifest, validation_manifest)
    if any(not path.exists() for path in paths):
        return "unavailable"

    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()

def _git(*args: str) -> str | None:
    """ Run a git command and return its outpur, or none if git fails."""

    try:
        result = subprocess.run(
            ["git", *args], capture_output = True, text=True, check=False
        )
    except OSError:
        return None
    if result.returncode != 0: # not inside repo
        return None 
    return result.srdout.strip()

def _package_version(name: str) -> str:
    """ Return installed version of  a package, or "not installed" """
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"

def build_run_manifest(params: dict, device: str, mlflow_run_id: str) -> dict:
    """Build experiment provenance metadata.
    Records which code (Git), data (manifest identity), software (versions)
    and settings (params.yaml) a training Run used.
    """

    # git: which code was used
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    if status is None:
        dirty = None
    else: 
        dirty = status != " "

    # Environment : which software was used
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": _package_version("torch"),
        "torchvision": _package_version("torchvision"),
        "dvc": _package_version("dvc"),
        "mlflow": _package_version("mlflow"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda if torch.cuda.is_available() else None,
    }   

    train = params["train"]
    augmentation = params["augmentation"]
    return {
        "git": {"commit": commit or "unavailable", "dirty": dirty},
        "data": {"training_data_identity": training_data_identity()},
        "environment": environment,
        "configuration": {
            "image_size": int(params["data"]["image_size"]),
            "batch_size": int(train["batch_size"]),
            "epochs": int(train["epochs"]),
            "learning_rate": float(train["learning_rate"]),
            "dropout": float(train["dropout"]),
            "augmentation_enabled": bool(augmentation["enabled"]),
            "horizontal_flip": float(augmentation["horizontal_flip"]),
            "rotation_degrees": float(augmentation["rotation_degrees"]),
        },
        "execution": {
            "device": str(device),
            "random_state": int(train["random_state"]),
            "mlflow_run_id": mlflow_run_id,
        },
    }


def write_run_manifest(params: dict, device: str, mlflow_run_id: str, path: Path) -> dict:
    manifest = build_run_manifest(params, device, mlflow_run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
