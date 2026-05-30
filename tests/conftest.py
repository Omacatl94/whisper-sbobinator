import pytest


@pytest.fixture
def fake_cuda(monkeypatch):
    """Forza torch.cuda.is_available() a True per i test."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "NVIDIA T1000")

    class _FakeProps:
        total_memory = 4 * 1024 ** 3  # 4 GB

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i=0: _FakeProps())
    return True


@pytest.fixture
def fake_no_cuda(monkeypatch):
    """Forza torch.cuda.is_available() a False per i test."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return False
