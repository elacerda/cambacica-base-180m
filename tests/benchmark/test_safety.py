from __future__ import annotations

import ctypes
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cambacica.benchmark import safety
from cambacica.benchmark.config import BenchmarkConfig
from cambacica.benchmark.safety import (
    EXPECTED_GPU_UUID,
    BenchmarkLock,
    BenchmarkLockError,
    GpuSafetyError,
    GpuSafetyMonitor,
    apply_allocator_limit,
    numeric_stability_check,
    preflight_gpu,
)

MIB = 1024 * 1024


@pytest.fixture(autouse=True)
def driver_loader(monkeypatch):
    """Every safety test must explicitly provide a fake driver to use the fallback."""
    loader = Mock(side_effect=OSError("driver library unavailable in CPU-only tests"))
    monkeypatch.setattr(safety.ctypes, "CDLL", loader)
    return loader


class FakeCudaDriver:
    def __init__(self, pci: bytes = b"0000:27:00.0"):
        self.pci = pci
        self.cuInit = Mock(return_value=0)
        self.cuDeviceGet = Mock(side_effect=self._get_device)
        self.cuDeviceGetPCIBusId = Mock(side_effect=self._get_pci_bus_id)

    def _get_device(self, device, ordinal):
        assert ordinal == 0
        ctypes.cast(device, ctypes.POINTER(ctypes.c_int)).contents.value = 7
        return 0

    def _get_pci_bus_id(self, buffer, length, device):
        # CUdevice is the handle returned by cuDeviceGet, not an assumed ordinal.
        assert device == 7
        assert length == len(buffer)
        buffer.value = self.pci
        return 0


@pytest.fixture
def cuda_driver(driver_loader):
    driver = FakeCudaDriver()
    driver_loader.side_effect = None
    driver_loader.return_value = driver
    return driver


class FakeCuda:
    def __init__(
        self,
        *,
        count: int = 1,
        uuid: str | None = EXPECTED_GPU_UUID,
        pci: str | None = None,
    ):
        self.count = count
        self.properties = SimpleNamespace(
            name="NVIDIA H100 PCIe", uuid=uuid, pci_bus_id=pci
        )

    def is_available(self):
        return True

    def device_count(self):
        return self.count

    def get_device_properties(self, _index):
        assert _index == 0
        return self.properties


class FakeTorch:
    def __init__(self, cuda: FakeCuda | None = None):
        self.cuda = cuda or FakeCuda()


class FakeNVML:
    def __init__(
        self,
        *,
        uuid: str = EXPECTED_GPU_UUID,
        free_mib: int = 50_000,
        pci: bytes | None = b"00000000:3b:00.0",
    ):
        self.uuid = uuid
        self.free_mib = free_mib
        self.pci = pci
        self.process_rows = []
        self.handle = object()

    def nvmlInit(self):
        return None

    def nvmlDeviceGetCount(self):
        return 2

    def nvmlDeviceGetHandleByIndex(self, index):
        assert index == 0
        return self.handle

    def nvmlDeviceGetUUID(self, _handle):
        return self.uuid

    def nvmlDeviceGetPciInfo(self, _handle):
        return SimpleNamespace(busId=self.pci)

    def nvmlDeviceGetMemoryInfo(self, _handle):
        total = 81_559 * MIB
        free = self.free_mib * MIB
        return SimpleNamespace(total=total, free=free, used=total - free)

    def nvmlDeviceGetComputeRunningProcesses_v3(self, _handle):
        return list(self.process_rows)

    def nvmlSystemGetDriverVersion(self):
        return "550.54.14"


def make_config(**overrides) -> BenchmarkConfig:
    values = {"architecture": "A", "context_length": 2048}
    values.update(overrides)
    return BenchmarkConfig(**values)


def test_preflight_confirms_physical_gpu_zero_and_available_memory() -> None:
    nvml = FakeNVML()
    snapshot, handle = preflight_gpu(
        FakeTorch(), nvml, make_config(), environ={"CUDA_VISIBLE_DEVICES": "0"}
    )
    assert snapshot.uuid == EXPECTED_GPU_UUID
    assert snapshot.physical_index == 0
    assert snapshot.mapping_method == "cuda_uuid"
    assert snapshot.driver_version == "550.54.14"
    assert handle is nvml.handle


@pytest.mark.parametrize(
    "torch_module,nvml,environ,message",
    [
        (
            FakeTorch(),
            FakeNVML(uuid="GPU-OTHER"),
            {"CUDA_VISIBLE_DEVICES": "0"},
            "UUID mismatch",
        ),
        (FakeTorch(), FakeNVML(), {"CUDA_VISIBLE_DEVICES": "1"}, "exactly 0"),
        (
            FakeTorch(FakeCuda(count=2)),
            FakeNVML(),
            {"CUDA_VISIBLE_DEVICES": "0"},
            "exactly one",
        ),
        (
            FakeTorch(),
            FakeNVML(free_mib=39_999),
            {"CUDA_VISIBLE_DEVICES": "0"},
            "insufficient free",
        ),
    ],
)
def test_preflight_fails_closed_without_identity_or_memory(
    torch_module, nvml, environ, message: str
) -> None:
    with pytest.raises(GpuSafetyError, match=message):
        preflight_gpu(torch_module, nvml, make_config(), environ=environ)


def test_visible_device_mapping_can_use_pci_bus_id() -> None:
    cuda = FakeCuda(uuid=None, pci="0000:3b:00.0")
    snapshot, _ = preflight_gpu(
        FakeTorch(cuda),
        FakeNVML(),
        make_config(),
        environ={"CUDA_VISIBLE_DEVICES": "0"},
    )
    assert snapshot.mapping_method == "pci_bus_id"


def test_visible_cuda_uuid_without_nvml_prefix_matches_expected_gpu() -> None:
    cuda = FakeCuda(uuid=EXPECTED_GPU_UUID.removeprefix("GPU-"))
    snapshot, _ = preflight_gpu(
        FakeTorch(cuda),
        FakeNVML(),
        make_config(),
        environ={"CUDA_VISIBLE_DEVICES": "0"},
    )
    assert snapshot.uuid == EXPECTED_GPU_UUID


def test_visible_device_mapping_can_use_numeric_torch_pci_fields() -> None:
    cuda = FakeCuda(uuid=None)
    cuda.properties.pci_bus_id = 0x3B
    cuda.properties.pci_device_id = 0
    cuda.properties.pci_domain_id = 0
    snapshot, _ = preflight_gpu(
        FakeTorch(cuda),
        FakeNVML(),
        make_config(),
        environ={"CUDA_VISIBLE_DEVICES": "0"},
    )
    assert snapshot.mapping_method == "pci_bus_id"


@pytest.mark.parametrize("attributes_absent", [False, True])
@pytest.mark.parametrize(
    "cuda_pci,nvml_pci,canonical",
    [
        (b"0000:27:00.0", b"00000000:27:00.0", "0:27:00.0"),
        (b"00000000:27:00.0", b"0000:27:00.0", "0:27:00.0"),
        (b" 000A:0B:01.2 ", b"0000000a:b:1.2", "a:0b:01.2"),
    ],
)
def test_cuda_driver_fallback_matches_normalised_pci_bus_id(
    cuda_driver, driver_loader, attributes_absent, cuda_pci, nvml_pci, canonical
) -> None:
    cuda_driver.pci = cuda_pci
    cuda = FakeCuda(uuid=None)
    if attributes_absent:
        cuda.properties = SimpleNamespace(name="NVIDIA H100 PCIe")
    nvml = FakeNVML(pci=nvml_pci)
    snapshot, handle = preflight_gpu(
        FakeTorch(cuda), nvml, make_config(), environ={"CUDA_VISIBLE_DEVICES": "0"}
    )
    assert handle is nvml.handle
    assert snapshot.physical_index == 0
    assert snapshot.uuid == EXPECTED_GPU_UUID
    assert snapshot.pci_bus_id == canonical
    assert snapshot.to_dict()["mapping_method"] == "cuda_driver_pci_bus_id"
    driver_loader.assert_called_once_with("libcuda.so.1")
    cuda_driver.cuInit.assert_called_once_with(0)
    assert cuda_driver.cuDeviceGet.call_count == 1
    assert cuda_driver.cuDeviceGet.call_args.args[1] == 0
    assert cuda_driver.cuDeviceGetPCIBusId.call_count == 1


def test_cuda_driver_pci_mismatch_never_retries_another_gpu(cuda_driver) -> None:
    # Both GPUs have the same name; only the pinned physical GPU's PCI can match.
    nvml = FakeNVML()
    nvml.nvmlDeviceGetHandleByIndex = Mock(wraps=nvml.nvmlDeviceGetHandleByIndex)
    with pytest.raises(GpuSafetyError, match="PCI bus ID does not match"):
        preflight_gpu(
            FakeTorch(FakeCuda(uuid=None)),
            nvml,
            make_config(),
            environ={"CUDA_VISIBLE_DEVICES": "0"},
        )
    nvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(0)
    assert cuda_driver.cuDeviceGet.call_count == 1
    assert cuda_driver.cuDeviceGet.call_args.args[1] == 0
    assert cuda_driver.cuDeviceGetPCIBusId.call_count == 1


def test_cuda_driver_library_unavailable_is_rejected(driver_loader) -> None:
    with pytest.raises(GpuSafetyError, match="driver library unavailable"):
        preflight_gpu(
            FakeTorch(FakeCuda(uuid=None)),
            FakeNVML(),
            make_config(),
            environ={"CUDA_VISIBLE_DEVICES": "0"},
        )
    driver_loader.assert_called_once_with("libcuda.so.1")


@pytest.mark.parametrize("symbol", ["cuInit", "cuDeviceGet", "cuDeviceGetPCIBusId"])
def test_cuda_driver_missing_symbol_is_rejected(cuda_driver, symbol) -> None:
    delattr(cuda_driver, symbol)
    with pytest.raises(GpuSafetyError, match=f"CUDA Driver API.*{symbol}"):
        preflight_gpu(
            FakeTorch(FakeCuda(uuid=None)),
            FakeNVML(),
            make_config(),
            environ={"CUDA_VISIBLE_DEVICES": "0"},
        )


@pytest.mark.parametrize("symbol", ["cuInit", "cuDeviceGet", "cuDeviceGetPCIBusId"])
def test_cuda_driver_api_failure_stops_immediately(cuda_driver, symbol) -> None:
    function = getattr(cuda_driver, symbol)
    function.side_effect = None
    function.return_value = 101
    with pytest.raises(GpuSafetyError, match=f"{symbol} failed: 101"):
        preflight_gpu(
            FakeTorch(FakeCuda(uuid=None)),
            FakeNVML(),
            make_config(),
            environ={"CUDA_VISIBLE_DEVICES": "0"},
        )
    cuda_driver.cuInit.assert_called_once_with(0)
    if symbol == "cuInit":
        cuda_driver.cuDeviceGet.assert_not_called()
    else:
        assert cuda_driver.cuDeviceGet.call_count == 1
        assert cuda_driver.cuDeviceGet.call_args.args[1] == 0
    if symbol != "cuDeviceGetPCIBusId":
        cuda_driver.cuDeviceGetPCIBusId.assert_not_called()


def test_cuda_driver_requires_device_handle_even_on_success(cuda_driver) -> None:
    cuda_driver.cuDeviceGet.side_effect = None
    cuda_driver.cuDeviceGet.return_value = 0
    with pytest.raises(GpuSafetyError, match="cuDeviceGet failed"):
        preflight_gpu(
            FakeTorch(FakeCuda(uuid=None)),
            FakeNVML(),
            make_config(),
            environ={"CUDA_VISIBLE_DEVICES": "0"},
        )
    cuda_driver.cuDeviceGetPCIBusId.assert_not_called()


def test_cuda_driver_rejects_unterminated_pci_id(cuda_driver) -> None:
    def unterminated(buffer, length, _device):
        buffer.raw = b"a" * length
        return 0

    cuda_driver.cuDeviceGetPCIBusId.side_effect = unterminated
    with pytest.raises(GpuSafetyError, match="unterminated PCI bus ID"):
        preflight_gpu(
            FakeTorch(FakeCuda(uuid=None)),
            FakeNVML(),
            make_config(),
            environ={"CUDA_VISIBLE_DEVICES": "0"},
        )


@pytest.mark.parametrize("source", ["driver", "nvml", "both"])
@pytest.mark.parametrize(
    "pci",
    [
        b"",
        b"   ",
        b"not-a-pci-id",
        b"0000:27:00",
        b"0000:gg:00.0",
        b"100000000:27:00.0",
        b"0000:100:00.0",
        b"0000:27:20.0",
        b"0000:27:00.8",
        b"\xff",
    ],
)
def test_cuda_driver_mapping_rejects_invalid_pci_ids(cuda_driver, source, pci) -> None:
    if source in {"driver", "both"}:
        cuda_driver.pci = pci
    nvml_pci = pci if source in {"nvml", "both"} else b"00000000:27:00.0"
    with pytest.raises(GpuSafetyError):
        preflight_gpu(
            FakeTorch(FakeCuda(uuid=None)),
            FakeNVML(pci=nvml_pci),
            make_config(),
            environ={"CUDA_VISIBLE_DEVICES": "0"},
        )


def test_cuda_driver_mapping_requires_nvml_pci_id(driver_loader) -> None:
    with pytest.raises(GpuSafetyError, match="cannot be established reliably"):
        preflight_gpu(
            FakeTorch(FakeCuda(uuid=None)),
            FakeNVML(pci=None),
            make_config(),
            environ={"CUDA_VISIBLE_DEVICES": "0"},
        )
    driver_loader.assert_not_called()


@pytest.mark.parametrize("identity_source", ["nvml_uuid", "torch_uuid", "torch_pci"])
def test_identity_mismatch_never_uses_driver_fallback(
    cuda_driver, driver_loader, identity_source
) -> None:
    cuda_driver.pci = b"0000:3b:00.0"
    cuda = FakeCuda(uuid=None)
    nvml = FakeNVML()
    if identity_source == "nvml_uuid":
        nvml.uuid = "GPU-OTHER"
    elif identity_source == "torch_uuid":
        cuda.properties.uuid = "GPU-OTHER"
    else:
        cuda.properties.pci_bus_id = "0000:27:00.0"
    with pytest.raises(GpuSafetyError, match="mismatch|does not match"):
        preflight_gpu(
            FakeTorch(cuda), nvml, make_config(), environ={"CUDA_VISIBLE_DEVICES": "0"}
        )
    driver_loader.assert_not_called()
    cuda_driver.cuInit.assert_not_called()


@pytest.mark.parametrize("cuda", [FakeCuda(), FakeCuda(uuid=None, pci="0000:3b:00.0")])
def test_existing_identity_paths_do_not_load_driver(driver_loader, cuda) -> None:
    preflight_gpu(
        FakeTorch(cuda),
        FakeNVML(),
        make_config(),
        environ={"CUDA_VISIBLE_DEVICES": "0"},
    )
    driver_loader.assert_not_called()


@pytest.mark.parametrize("pci", ["", "not-a-pci-id", "0000:27:20.0", 0x27])
def test_invalid_torch_pci_never_uses_driver_fallback(driver_loader, pci) -> None:
    cuda = FakeCuda(uuid=None)
    cuda.properties.pci_bus_id = pci
    with pytest.raises(GpuSafetyError, match="invalid.*PCI bus ID"):
        preflight_gpu(
            FakeTorch(cuda),
            FakeNVML(),
            make_config(),
            environ={"CUDA_VISIBLE_DEVICES": "0"},
        )
    driver_loader.assert_not_called()


def test_cuda_driver_fallback_preserves_startup_memory_budget(cuda_driver) -> None:
    with pytest.raises(GpuSafetyError, match="insufficient free GPU memory"):
        preflight_gpu(
            FakeTorch(FakeCuda(uuid=None)),
            FakeNVML(pci=b"00000000:27:00.0", free_mib=40_959),
            make_config(),
            environ={"CUDA_VISIBLE_DEVICES": "0"},
        )


def test_unavailable_nvml_process_api_is_rejected() -> None:
    nvml = FakeNVML()
    nvml.nvmlDeviceGetComputeRunningProcesses_v3 = None
    nvml.nvmlDeviceGetComputeRunningProcesses = None
    with pytest.raises(GpuSafetyError, match="process-level memory monitoring"):
        preflight_gpu(
            FakeTorch(), nvml, make_config(), environ={"CUDA_VISIBLE_DEVICES": "0"}
        )


def test_allocator_limit_stays_below_36_gib_peak_target() -> None:
    calls = []
    cuda = SimpleNamespace(
        get_allocator_backend=lambda: "native",
        set_per_process_memory_fraction=lambda fraction, device: calls.append(
            (fraction, device)
        ),
    )
    torch_module = SimpleNamespace(cuda=cuda)
    fraction = apply_allocator_limit(
        torch_module, make_config(), total_memory_mib=81_559
    )
    assert fraction == pytest.approx(35_840 / 81_559)
    assert calls == [(fraction, 0)]


def test_allocator_limit_rejects_unenforceable_backend() -> None:
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(
            get_allocator_backend=lambda: "cudaMallocAsync",
            set_per_process_memory_fraction=lambda *_: None,
        )
    )
    with pytest.raises(GpuSafetyError, match="requires the native caching allocator"):
        apply_allocator_limit(torch_module, make_config(), total_memory_mib=81_559)


def test_monitor_rejects_budget_target_and_unavailable_process_memory() -> None:
    nvml = FakeNVML()
    config = make_config()
    monitor = GpuSafetyMonitor(nvml, nvml.handle, config, pid=123)
    nvml.process_rows = [SimpleNamespace(pid=123, usedGpuMemory=35_000 * MIB)]
    assert monitor.sample()["process_memory_mib"] == 35_000
    nvml.process_rows = [SimpleNamespace(pid=123, usedGpuMemory=36_865 * MIB)]
    with pytest.raises(GpuSafetyError, match="initial peak target"):
        monitor.sample()
    nvml.process_rows = [SimpleNamespace(pid=123, usedGpuMemory=-1)]
    with pytest.raises(GpuSafetyError, match="cannot report this process"):
        monitor.sample()


def test_monitor_rejects_free_memory_below_safety_margin() -> None:
    nvml = FakeNVML(free_mib=4095)
    nvml.process_rows = [SimpleNamespace(pid=123, usedGpuMemory=100 * MIB)]
    monitor = GpuSafetyMonitor(nvml, nvml.handle, make_config(), pid=123)
    with pytest.raises(GpuSafetyError, match="safety margin"):
        monitor.sample()


def test_nonfinite_loss_or_gradient_stops_run() -> None:
    numeric_stability_check(True, True)
    with pytest.raises(GpuSafetyError, match="non-finite loss"):
        numeric_stability_check(False, True)
    with pytest.raises(GpuSafetyError, match="non-finite gradients"):
        numeric_stability_check(True, False)


def test_active_lock_is_rejected_without_removing_lock_file(tmp_path: Path) -> None:
    path = tmp_path / "c3.lock"
    first = BenchmarkLock(path).acquire()
    try:
        with pytest.raises(BenchmarkLockError, match="Another Cambacica benchmark"):
            BenchmarkLock(path).acquire()
        assert path.is_file()
    finally:
        first.release()
