"""Fail-closed GPU identity, memory preflight, and concurrency controls."""

from __future__ import annotations

import ctypes
import fcntl
import os
import re
import uuid as uuidlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cambacica.benchmark.config import HARD_GPU_MEMORY_BUDGET_MIB, BenchmarkConfig

EXPECTED_GPU_UUID = "GPU-7562e34d-b0a2-a725-2ad5-ed1327be3cb0"
MIB = 1024 * 1024


class GpuSafetyError(RuntimeError):
    """The device or memory state does not satisfy the approved C3 policy."""

    def __init__(self, message: str, *, diagnostics: dict[str, object] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class BenchmarkLockError(GpuSafetyError):
    """Another Cambacica C3 benchmark holds the host-local lock."""


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _normalise_gpu_uuid(value: Any) -> str:
    if isinstance(value, bytes) and len(value) == 16:
        raw = str(uuidlib.UUID(bytes=value))
    else:
        raw = _decode(value).strip().lower()
    if raw.startswith("mig-"):
        return raw
    return raw.removeprefix("gpu-")


def _memory_mib(value: Any) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value < 0 or value == (2**64 - 1):
        return None
    return value // MIB


def _normalise_pci_bus_id(value: Any) -> str | None:
    if value is None:
        return None
    raw = _decode(value).strip().lower()
    match = re.fullmatch(
        r"(?:0x)?([0-9a-f]+):([0-9a-f]+):([0-9a-f]+)\.([0-9a-f]+)", raw
    )
    if match is None:
        raise GpuSafetyError(f"invalid PCI bus ID: {raw!r}")
    domain, bus, device, function = (int(part, 16) for part in match.groups())
    if domain > 0xFFFFFFFF or bus > 0xFF or device > 0x1F or function > 7:
        raise GpuSafetyError(f"invalid PCI bus ID: {raw!r}")
    return f"{domain:x}:{bus:02x}:{device:02x}.{function:x}"


def _cuda_pci_bus_id(properties: Any) -> str | None:
    bus_id = getattr(properties, "pci_bus_id", getattr(properties, "pciBusId", None))
    if isinstance(bus_id, (str, bytes)) and not bus_id.isdigit():
        return _normalise_pci_bus_id(bus_id)
    if bus_id is None:
        return None
    try:
        bus = int(bus_id)
        device = int(properties.pci_device_id)
        domain = int(getattr(properties, "pci_domain_id", 0))
        function = int(getattr(properties, "pci_function_id", 0))
    except (AttributeError, TypeError, ValueError) as exc:
        raise GpuSafetyError("invalid PyTorch PCI bus ID properties") from exc
    return _normalise_pci_bus_id(f"{domain:x}:{bus:x}:{device:x}.{function:x}")


def _cuda_driver_pci_bus_id() -> str:
    """Identify CUDA-visible logical device 0 without creating a CUDA context."""

    try:
        driver = ctypes.CDLL("libcuda.so.1")
        cu_init = driver.cuInit
        cu_init.argtypes = [ctypes.c_uint]
        cu_init.restype = ctypes.c_int
        cu_device_get = driver.cuDeviceGet
        cu_device_get.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        cu_device_get.restype = ctypes.c_int
        cu_device_get_pci = driver.cuDeviceGetPCIBusId
        cu_device_get_pci.argtypes = [
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_int,
            ctypes.c_int,
        ]
        cu_device_get_pci.restype = ctypes.c_int

        status = cu_init(0)
        if status != 0:
            raise GpuSafetyError(f"CUDA Driver API cuInit failed: {status}")
        device = ctypes.c_int(-1)
        status = cu_device_get(ctypes.byref(device), 0)
        if status != 0 or device.value < 0:
            raise GpuSafetyError(f"CUDA Driver API cuDeviceGet failed: {status}")
        bus_id = ctypes.create_string_buffer(32)
        status = cu_device_get_pci(bus_id, len(bus_id), device.value)
        if status != 0:
            raise GpuSafetyError(
                f"CUDA Driver API cuDeviceGetPCIBusId failed: {status}"
            )
        if b"\0" not in bus_id.raw:
            raise GpuSafetyError("CUDA Driver API returned an unterminated PCI bus ID")
        pci_bus_id = _normalise_pci_bus_id(bus_id.value)
        if pci_bus_id is None:
            raise GpuSafetyError("CUDA Driver API returned no PCI bus ID")
        return pci_bus_id
    except GpuSafetyError:
        raise
    except Exception as exc:
        raise GpuSafetyError(f"CUDA Driver API PCI bus ID query failed: {exc}") from exc


def _nvml_processes(nvml: Any, handle: Any) -> list[dict[str, int | None]]:
    function = getattr(nvml, "nvmlDeviceGetComputeRunningProcesses_v3", None)
    if function is None:
        function = getattr(nvml, "nvmlDeviceGetComputeRunningProcesses", None)
    if function is None:
        raise GpuSafetyError("NVML process-level memory monitoring is unavailable")
    try:
        rows = function(handle)
    except Exception as exc:
        raise GpuSafetyError(f"NVML could not read GPU process memory: {exc}") from exc
    result = []
    for process in rows:
        result.append(
            {
                "pid": int(process.pid),
                "used_memory_mib": _memory_mib(getattr(process, "usedGpuMemory", None)),
            }
        )
    return sorted(result, key=lambda item: item["pid"] or -1)


@dataclass(frozen=True)
class GpuSnapshot:
    physical_index: int
    uuid: str
    name: str
    pci_bus_id: str
    total_memory_mib: int
    free_memory_mib: int
    used_memory_mib: int
    driver_version: str
    mapping_method: str
    processes: tuple[dict[str, int | None], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "physical_index": self.physical_index,
            "uuid": self.uuid,
            "name": self.name,
            "pci_bus_id": self.pci_bus_id,
            "total_memory_mib": self.total_memory_mib,
            "free_memory_mib": self.free_memory_mib,
            "used_memory_mib": self.used_memory_mib,
            "driver_version": self.driver_version,
            "mapping_method": self.mapping_method,
            "processes": list(self.processes),
        }


class BenchmarkLock:
    """An advisory host-local lock; the kernel releases it on process exit."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream = None

    def acquire(self) -> BenchmarkLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a+")
        try:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._stream.close()
            self._stream = None
            raise BenchmarkLockError(
                f"Another Cambacica benchmark holds {self.path}; no process was changed"
            ) from exc
        self._stream.seek(0)
        self._stream.truncate()
        self._stream.write(f"pid={os.getpid()}\n")
        self._stream.flush()
        return self

    def release(self) -> None:
        if self._stream is not None:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            self._stream.close()
            self._stream = None

    def __enter__(self) -> BenchmarkLock:  # noqa: PYI034 - Python 3.10 has no typing.Self
        return self.acquire()

    def __exit__(self, *_exc: object) -> None:
        self.release()


def preflight_gpu(
    torch_module: Any,
    nvml: Any,
    config: BenchmarkConfig,
    *,
    environ: dict[str, str] | None = None,
) -> tuple[GpuSnapshot, Any]:
    """Verify that visible cuda:0 maps to the scientist-pinned physical GPU 0."""

    config.validate()
    env = os.environ if environ is None else environ
    if env.get("CUDA_VISIBLE_DEVICES") != "0":
        raise GpuSafetyError(
            "CUDA_VISIBLE_DEVICES must be exactly 0; device remapping is refused"
        )
    if not torch_module.cuda.is_available():
        raise GpuSafetyError("CUDA is unavailable; no fallback device will be used")
    if torch_module.cuda.device_count() != 1:
        raise GpuSafetyError(
            "CUDA_VISIBLE_DEVICES=0 must expose exactly one CUDA device"
        )

    try:
        nvml.nvmlInit()
        if int(nvml.nvmlDeviceGetCount()) < 1:
            raise GpuSafetyError("NVML reports no physical GPUs")
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        observed_uuid = _normalise_gpu_uuid(nvml.nvmlDeviceGetUUID(handle))
        expected_uuid_value = _normalise_gpu_uuid(EXPECTED_GPU_UUID)
        if observed_uuid != expected_uuid_value:
            raise GpuSafetyError(
                f"physical GPU 0 UUID mismatch: expected {EXPECTED_GPU_UUID}, got {observed_uuid}",
                diagnostics={
                    "expected_gpu_uuid": EXPECTED_GPU_UUID,
                    "observed_gpu_uuid": observed_uuid,
                    "nvml_physical_index": 0,
                },
            )
        uuid = f"GPU-{observed_uuid}"
        pci_bus_id = _normalise_pci_bus_id(nvml.nvmlDeviceGetPciInfo(handle).busId)
        properties = torch_module.cuda.get_device_properties(0)
        visible_uuid = getattr(properties, "uuid", None)
        visible_pci = _cuda_pci_bus_id(properties)
        if visible_uuid is not None:
            if _normalise_gpu_uuid(visible_uuid) != expected_uuid_value:
                raise GpuSafetyError(
                    "visible cuda:0 UUID does not match the approved physical GPU 0"
                )
            mapping_method = "cuda_uuid"
        elif visible_pci is not None and pci_bus_id is not None:
            if visible_pci != pci_bus_id:
                raise GpuSafetyError(
                    "visible cuda:0 PCI bus ID does not match physical GPU 0"
                )
            mapping_method = "pci_bus_id"
        elif visible_pci is None and pci_bus_id is not None:
            if _cuda_driver_pci_bus_id() != pci_bus_id:
                raise GpuSafetyError(
                    "CUDA Driver API cuda:0 PCI bus ID does not match physical GPU 0"
                )
            mapping_method = "cuda_driver_pci_bus_id"
        else:
            raise GpuSafetyError(
                "CUDA-to-NVML device mapping cannot be established reliably"
            )
        memory = nvml.nvmlDeviceGetMemoryInfo(handle)
        total_mib = _memory_mib(memory.total)
        free_mib = _memory_mib(memory.free)
        used_mib = _memory_mib(memory.used)
        if total_mib is None or free_mib is None or used_mib is None:
            raise GpuSafetyError("NVML returned unavailable GPU memory values")
        processes = _nvml_processes(nvml, handle)
        driver = _decode(nvml.nvmlSystemGetDriverVersion())
    except GpuSafetyError:
        raise
    except Exception as exc:
        raise GpuSafetyError(f"GPU identity or NVML preflight failed: {exc}") from exc

    if config.gpu_memory_budget_mib > HARD_GPU_MEMORY_BUDGET_MIB:
        raise GpuSafetyError("configured GPU budget exceeds 40960 MiB")
    snapshot = GpuSnapshot(
        physical_index=0,
        uuid=uuid,
        name=_decode(getattr(properties, "name", "unknown")),
        pci_bus_id=pci_bus_id or "unavailable",
        total_memory_mib=total_mib,
        free_memory_mib=free_mib,
        used_memory_mib=used_mib,
        driver_version=driver,
        mapping_method=mapping_method,
        processes=tuple(processes),
    )
    if free_mib < config.gpu_memory_budget_mib:
        raise GpuSafetyError(
            f"insufficient free GPU memory: {free_mib} MiB available; "
            f"{config.gpu_memory_budget_mib} MiB is required for budget and margin",
            diagnostics={"startup_gpu": snapshot.to_dict()},
        )
    return snapshot, handle


class GpuSafetyMonitor:
    """Read-only NVML sampling with hard-budget and initial-target checks."""

    def __init__(
        self,
        nvml: Any,
        handle: Any,
        config: BenchmarkConfig,
        *,
        pid: int | None = None,
    ) -> None:
        self.nvml = nvml
        self.handle = handle
        self.config = config.validate()
        self.pid = os.getpid() if pid is None else pid
        self.peak_process_memory_mib = 0
        self.last_sample: dict[str, object] | None = None

    def sample(self, *, enforce_target: bool = True) -> dict[str, object]:
        try:
            memory = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
            free_mib = _memory_mib(memory.free)
            processes = _nvml_processes(self.nvml, self.handle)
        except GpuSafetyError:
            raise
        except Exception as exc:
            raise GpuSafetyError(
                f"runtime NVML memory monitoring failed: {exc}"
            ) from exc
        if free_mib is None:
            raise GpuSafetyError("runtime NVML free-memory reading is unavailable")
        current = next((row for row in processes if row["pid"] == self.pid), None)
        if current is None or current["used_memory_mib"] is None:
            raise GpuSafetyError(
                "NVML cannot report this process's GPU memory; benchmark stopped"
            )
        process_mib = int(current["used_memory_mib"])
        self.peak_process_memory_mib = max(self.peak_process_memory_mib, process_mib)
        sample = {
            "free_memory_mib": free_mib,
            "process_memory_mib": process_mib,
            "processes": processes,
        }
        self.last_sample = sample
        if free_mib < self.config.safety_margin_mib:
            raise GpuSafetyError(
                f"free GPU memory fell below the {self.config.safety_margin_mib} MiB "
                "safety margin",
                diagnostics={"last_gpu_memory_sample": sample},
            )
        if process_mib > self.config.gpu_memory_budget_mib:
            raise GpuSafetyError(
                f"Cambacica process used {process_mib} MiB, above the "
                f"{self.config.gpu_memory_budget_mib} MiB hard budget",
                diagnostics={"last_gpu_memory_sample": sample},
            )
        if enforce_target and process_mib > self.config.peak_target_mib:
            raise GpuSafetyError(
                f"Cambacica process used {process_mib} MiB, above the "
                f"{self.config.peak_target_mib} MiB initial peak target",
                diagnostics={"last_gpu_memory_sample": sample},
            )
        return self.last_sample


def apply_allocator_limit(
    torch_module: Any, config: BenchmarkConfig, total_memory_mib: int
) -> float:
    """Cap the PyTorch caching allocator below the total-process peak target."""

    config.validate()
    backend_getter = getattr(torch_module.cuda, "get_allocator_backend", None)
    if backend_getter is None:
        raise GpuSafetyError("PyTorch allocator backend cannot be identified")
    try:
        allocator_backend = backend_getter()
    except Exception as exc:
        raise GpuSafetyError(f"PyTorch allocator backend check failed: {exc}") from exc
    if allocator_backend != "native":
        raise GpuSafetyError(
            f"unsupported PyTorch allocator backend {allocator_backend!r}; "
            "the C3 memory fraction requires the native caching allocator"
        )
    setter = getattr(torch_module.cuda, "set_per_process_memory_fraction", None)
    if setter is None:
        raise GpuSafetyError("PyTorch allocator memory limit is unavailable")
    fraction = config.allocator_limit_mib / total_memory_mib
    if not 0 < fraction < 1:
        raise GpuSafetyError("computed PyTorch allocator fraction is invalid")
    try:
        setter(fraction, 0)
    except Exception as exc:
        raise GpuSafetyError(f"could not apply PyTorch allocator limit: {exc}") from exc
    return fraction


def numeric_stability_check(loss_finite: bool, gradients_finite: bool) -> None:
    if not loss_finite:
        raise GpuSafetyError("non-finite loss detected")
    if not gradients_finite:
        raise GpuSafetyError("non-finite gradients detected")
