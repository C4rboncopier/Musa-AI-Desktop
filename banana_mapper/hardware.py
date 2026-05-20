from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass


PYTORCH_SETUP_URL = "https://pytorch.org/get-started/locally/"
NVIDIA_DRIVER_URL = "https://www.nvidia.com/Download/index.aspx"


@dataclass(frozen=True)
class GpuInfo:
    name: str
    status: str
    detail: str = ""


@dataclass(frozen=True)
class HardwareStatus:
    cpu_name: str
    cpu_cores: int
    gpus: tuple[GpuInfo, ...]
    torch_installed: bool
    torch_version: str
    cuda_available: bool
    cuda_version: str
    recommended_device: str
    device_label: str
    issue_title: str
    issue_detail: str
    setup_steps: tuple[str, ...]

    @property
    def has_compatible_gpu(self) -> bool:
        return self.cuda_available and any(gpu.status == "Available" for gpu in self.gpus)


def detect_hardware() -> HardwareStatus:
    cpu_name = _cpu_name()
    cpu_cores = os.cpu_count() or 1
    torch_installed = False
    torch_version = "Not installed"
    cuda_available = False
    cuda_version = "Unavailable"
    cuda_gpus: list[GpuInfo] = []

    try:
        import torch  # type: ignore

        torch_installed = True
        torch_version = str(getattr(torch, "__version__", "Installed"))
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = str(getattr(torch.version, "cuda", "") or "Unavailable")
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                cuda_gpus.append(
                    GpuInfo(
                        name=str(torch.cuda.get_device_name(index)),
                        status="Available",
                        detail=f"CUDA device {index}",
                    )
                )
    except Exception:
        pass

    if cuda_available and cuda_gpus:
        return HardwareStatus(
            cpu_name=cpu_name,
            cpu_cores=cpu_cores,
            gpus=tuple(cuda_gpus),
            torch_installed=torch_installed,
            torch_version=torch_version,
            cuda_available=True,
            cuda_version=cuda_version,
            recommended_device="gpu",
            device_label=f"GPU ({cuda_gpus[0].name})",
            issue_title="GPU acceleration is ready",
            issue_detail="AI mapping can use CUDA acceleration for YOLO inference.",
            setup_steps=(),
        )

    driver_gpus = _nvidia_smi_gpus()
    gpus = tuple(driver_gpus) if driver_gpus else (GpuInfo("No compatible CUDA GPU detected", "Missing"),)

    if not torch_installed:
        issue_title = "PyTorch is missing"
        issue_detail = (
            "AI mapping can run on CPU, but GPU acceleration needs PyTorch with CUDA support. "
            "Install the GPU-enabled PyTorch package for this Python environment."
        )
        setup_steps = (
            "Install or update the NVIDIA driver if this computer has an NVIDIA GPU.",
            "Open the PyTorch setup guide and choose the CUDA build for Windows.",
            "Restart Musa AI after installation, then run Hardware Check again.",
        )
    elif driver_gpus:
        issue_title = "CUDA is not available to PyTorch"
        issue_detail = (
            "An NVIDIA GPU was detected, but the installed PyTorch package cannot use CUDA. "
            "This usually means the CPU-only PyTorch build is installed or the NVIDIA driver is incompatible."
        )
        setup_steps = (
            "Update the NVIDIA display driver.",
            "Install a CUDA-enabled PyTorch build in the same Python environment used by Musa AI.",
            "Restart Musa AI after installation, then run Hardware Check again.",
        )
    else:
        issue_title = "No compatible GPU was detected"
        issue_detail = (
            "AI mapping will run on CPU. GPU acceleration is recommended because YOLO scans many image tiles, "
            "and CUDA-capable NVIDIA GPUs can process those model predictions much faster."
        )
        setup_steps = (
            "Use a computer with a CUDA-capable NVIDIA GPU for accelerated AI mapping.",
            "Install the latest NVIDIA driver.",
            "Install CUDA-enabled PyTorch in the same Python environment used by Musa AI.",
        )

    return HardwareStatus(
        cpu_name=cpu_name,
        cpu_cores=cpu_cores,
        gpus=gpus,
        torch_installed=torch_installed,
        torch_version=torch_version,
        cuda_available=False,
        cuda_version=cuda_version,
        recommended_device="cpu",
        device_label="CPU",
        issue_title=issue_title,
        issue_detail=issue_detail,
        setup_steps=setup_steps,
    )


def inference_device_arg(device_preference: str, status: HardwareStatus | None = None) -> str:
    status = status or detect_hardware()
    if device_preference == "gpu" and status.has_compatible_gpu:
        return "0"
    return "cpu"


def _cpu_name() -> str:
    name = platform.processor() or platform.machine() or "CPU"
    return name.strip() or "CPU"


def _nvidia_smi_gpus() -> list[GpuInfo]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    try:
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=4,
            creationflags=creationflags,
            check=False,
        )
    except Exception:
        return []

    gpus: list[GpuInfo] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if not parts or not parts[0]:
            continue
        detail = f"NVIDIA driver {parts[1]}" if len(parts) > 1 and parts[1] else "NVIDIA driver detected"
        gpus.append(GpuInfo(parts[0], "Installed", detail))
    return gpus
