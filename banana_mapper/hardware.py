from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


ProgressCallback = Callable[[int, str, str], None]


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
    cpu_threads: int
    cpu_clock: str
    ram_total: str
    ram_available: str
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


def detect_hardware(progress_callback: ProgressCallback | None = None) -> HardwareStatus:
    _report(progress_callback, 5, "Initializing hardware diagnostics...", "info")
    _report(progress_callback, 18, "Checking CPU information...", "info")
    cpu_name = _cpu_name()
    cpu_cores, cpu_threads, cpu_clock = _cpu_topology()
    _report(progress_callback, 28, f"CPU detected: {cpu_name}", "ok")

    _report(progress_callback, 36, "Checking RAM information...", "info")
    ram_total, ram_available = _ram_info()
    _report(progress_callback, 44, f"RAM detected: {ram_total} total", "ok")

    torch_installed = False
    torch_version = "Not installed"
    cuda_available = False
    cuda_version = "Unavailable"
    cuda_gpus: list[GpuInfo] = []

    _report(progress_callback, 54, "Checking PyTorch installation...", "info")
    try:
        import torch  # type: ignore

        torch_installed = True
        torch_version = str(getattr(torch, "__version__", "Installed"))
        _report(progress_callback, 64, f"PyTorch installed: {torch_version}", "ok")
        _report(progress_callback, 72, "Testing CUDA availability...", "info")
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = str(getattr(torch.version, "cuda", "") or "Unavailable")
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                memory_gb = getattr(props, "total_memory", 0) / (1024 ** 3)
                cuda_gpus.append(
                    GpuInfo(
                        name=str(torch.cuda.get_device_name(index)),
                        status="Available",
                        detail=f"CUDA device {index}; {memory_gb:.1f} GB VRAM",
                    )
                )
            _report(progress_callback, 84, "GPU detected successfully.", "ok")
        else:
            _report(progress_callback, 78, "CUDA drivers not available to PyTorch.", "warning")
    except Exception:
        _report(progress_callback, 64, "PyTorch is not installed or could not be loaded.", "warning")
        pass

    if cuda_available and cuda_gpus:
        _report(progress_callback, 100, "Hardware check completed successfully.", "ok")
        return HardwareStatus(
            cpu_name=cpu_name,
            cpu_cores=cpu_cores,
            cpu_threads=cpu_threads,
            cpu_clock=cpu_clock,
            ram_total=ram_total,
            ram_available=ram_available,
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

    _report(progress_callback, 82, "Checking installed display adapters...", "info")
    driver_gpus = _nvidia_smi_gpus()
    if not driver_gpus:
        driver_gpus = _display_adapter_gpus()
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

    if driver_gpus:
        _report(progress_callback, 92, "GPU adapter detected, but CUDA is not ready.", "warning")
    else:
        _report(progress_callback, 92, "No CUDA-capable GPU was detected.", "warning")
    _report(progress_callback, 100, "Hardware check completed with setup guidance.", "warning")
    return HardwareStatus(
        cpu_name=cpu_name,
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
        cpu_clock=cpu_clock,
        ram_total=ram_total,
        ram_available=ram_available,
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
    if sys.platform.startswith("win"):
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0])
                if name.strip():
                    return " ".join(name.split())
        except OSError:
            pass
    name = platform.processor() or _linux_cpu_name() or platform.machine() or "CPU"
    return name.strip() or "CPU"


def _cpu_topology() -> tuple[int, int, str]:
    threads = os.cpu_count() or 1
    cores = threads
    clock = "Unavailable"

    if sys.platform.startswith("win"):
        rows = _run_csv_command(
            [
                "wmic",
                "cpu",
                "get",
                "NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed",
                "/format:csv",
            ],
            timeout=4,
        )
        if rows:
            core_values = [_safe_int(row.get("NumberOfCores")) for row in rows]
            thread_values = [_safe_int(row.get("NumberOfLogicalProcessors")) for row in rows]
            clock_values = [_safe_int(row.get("MaxClockSpeed")) for row in rows]
            cores = sum(value for value in core_values if value > 0) or cores
            threads = sum(value for value in thread_values if value > 0) or threads
            max_clock = max(clock_values or [0])
            if max_clock > 0:
                clock = f"Up to {max_clock / 1000:.2f} GHz"
        else:
            cores = _windows_physical_cores() or cores
            mhz = _windows_cpu_mhz_registry()
            if mhz:
                clock = f"Current {mhz / 1000:.2f} GHz"
    else:
        cores = _linux_physical_cores() or cores
        mhz = _linux_cpu_mhz()
        if mhz:
            clock = f"Up to {mhz / 1000:.2f} GHz"

    return max(1, cores), max(threads, cores), clock


def _ram_info() -> tuple[str, str]:
    if sys.platform.startswith("win"):
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return _bytes_label(status.ullTotalPhys), _bytes_label(status.ullAvailPhys)
        except Exception:
            pass

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return _bytes_label(pages * page_size), _bytes_label(available_pages * page_size)
    except (AttributeError, OSError, ValueError):
        return "Unavailable", "Unavailable"


def _windows_physical_cores() -> int:
    try:
        import ctypes
        from ctypes import wintypes

        relation_processor_core = 0
        buffer_size = wintypes.DWORD(0)
        ctypes.windll.kernel32.GetLogicalProcessorInformation(None, ctypes.byref(buffer_size))
        if not buffer_size.value:
            return 0

        class SystemLogicalProcessorInformation(ctypes.Structure):
            _fields_ = [
                ("ProcessorMask", ctypes.c_size_t),
                ("Relationship", wintypes.DWORD),
                ("Data", ctypes.c_byte * 16),
            ]

        item_size = ctypes.sizeof(SystemLogicalProcessorInformation)
        array_type = SystemLogicalProcessorInformation * (buffer_size.value // item_size)
        buffer = array_type()
        if not ctypes.windll.kernel32.GetLogicalProcessorInformation(buffer, ctypes.byref(buffer_size)):
            return 0
        return sum(1 for item in buffer if item.Relationship == relation_processor_core)
    except Exception:
        return 0


def _windows_cpu_mhz_registry() -> int:
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            return _safe_int(str(winreg.QueryValueEx(key, "~MHz")[0]))
    except OSError:
        return 0


def _nvidia_smi_gpus() -> list[GpuInfo]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    try:
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,driver_version,memory.total",
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
        parts = [part.strip() for part in line.split(",")]
        if not parts or not parts[0]:
            continue
        details = []
        if len(parts) > 1 and parts[1]:
            details.append(f"NVIDIA driver {parts[1]}")
        if len(parts) > 2 and parts[2]:
            details.append(parts[2])
        detail = "; ".join(details) if details else "NVIDIA driver detected"
        gpus.append(GpuInfo(parts[0], "Installed", detail))
    return gpus


def _display_adapter_gpus() -> list[GpuInfo]:
    if not sys.platform.startswith("win"):
        return []
    rows = _run_csv_command(
        [
            "wmic",
            "path",
            "win32_VideoController",
            "get",
            "Name,AdapterRAM,DriverVersion",
            "/format:csv",
        ],
        timeout=4,
    )
    gpus: list[GpuInfo] = []
    for row in rows:
        name = (row.get("Name") or "").strip()
        if not name:
            continue
        ram = _safe_int(row.get("AdapterRAM"))
        detail_parts = []
        if ram > 0:
            detail_parts.append(f"{_bytes_label(ram)} VRAM")
        driver = (row.get("DriverVersion") or "").strip()
        if driver:
            detail_parts.append(f"driver {driver}")
        gpus.append(GpuInfo(name, "Installed", "; ".join(detail_parts)))
    return gpus


def _run_csv_command(args: list[str], timeout: int = 4) -> list[dict[str, str]]:
    try:
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=creationflags,
            check=False,
        )
    except Exception:
        return []
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return []
    header = [part.strip() for part in lines[0].split(",")]
    rows = []
    for line in lines[1:]:
        values = [part.strip() for part in line.split(",")]
        if len(values) < len(header):
            continue
        rows.append(dict(zip(header, values)))
    return rows


def _linux_cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return ""
    return ""


def _linux_physical_cores() -> int:
    cores: set[tuple[str, str]] = set()
    physical_id = "0"
    core_id = ""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                if core_id:
                    cores.add((physical_id, core_id))
                physical_id, core_id = "0", ""
                continue
            if line.startswith("physical id"):
                physical_id = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                core_id = line.split(":", 1)[1].strip()
    except OSError:
        return 0
    return len(cores)


def _linux_cpu_mhz() -> int:
    try:
        values = []
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("cpu mhz"):
                values.append(float(line.split(":", 1)[1].strip()))
        return int(max(values)) if values else 0
    except (OSError, ValueError):
        return 0


def _safe_int(value: str | None) -> int:
    try:
        return int(str(value or "").strip())
    except ValueError:
        return 0


def _bytes_label(value: int | float) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


def _report(callback: ProgressCallback | None, percent: int, message: str, level: str) -> None:
    if callback is not None:
        callback(percent, message, level)
