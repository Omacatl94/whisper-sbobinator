import sbobinator


def test_get_device_returns_cuda_when_available(fake_cuda):
    assert sbobinator.get_device() == "cuda"


def test_get_device_returns_cpu_when_unavailable(fake_no_cuda):
    assert sbobinator.get_device() == "cpu"


def test_get_device_info_with_cuda(fake_cuda):
    info = sbobinator.get_device_info()
    assert info["device"] == "cuda"
    assert "T1000" in info["name"]
    assert info["vram_gb"] == 4.0


def test_get_device_info_no_cuda(fake_no_cuda):
    info = sbobinator.get_device_info()
    assert info["device"] == "cpu"
    assert info["name"] == "CPU"
    assert info["vram_gb"] is None
