import pytest
import torch

from master.models import losses


@pytest.fixture
def dummy_params():
    camera_params = {"image_width": 4, "image_height": 4, "focal_length": 800.0}
    hapke_params = {"w": 0.6, "B0": 0.4, "h": 0.1, "phase_fun": "hg", "xi": 0.1}
    return camera_params, hapke_params


@pytest.fixture(autouse=True)
def mock_renderer(monkeypatch):
    def _mock_render_shading_batched(dem_tensor_flat, meta_flat, camera, hapke_model, device):
        # Return deterministic reflectance to avoid heavy renderer dependency
        return dem_tensor_flat * 0.0 + 0.5

    monkeypatch.setattr(losses, "_render_shading_batched", _mock_render_shading_batched)
    yield


def test_compute_reflectance_map_from_dem_shape_and_finite(dummy_params):
    camera_params, hapke_params = dummy_params
    device = torch.device("cpu")
    B, H, W = 2, 4, 4

    dem = torch.randn(B, 1, H, W, device=device, requires_grad=True)
    meta = torch.zeros(B, 5, 5, device=device)

    refl = losses.compute_reflectance_map_from_dem(dem, meta, device, camera_params, hapke_params)

    assert refl.shape == (B, 5, H, W)
    assert torch.isfinite(refl).all().item()


def test_calculate_total_loss_no_nan_and_backward(dummy_params):
    camera_params, hapke_params = dummy_params
    device = torch.device("cpu")
    B, H, W = 2, 4, 4

    outputs = torch.randn(B, 1, H, W, device=device, requires_grad=True)
    targets = torch.randn(B, 1, H, W, device=device)
    meta = torch.zeros(B, 5, 5, device=device)
    target_refl = torch.full((B, 5, H, W), 0.5, device=device)

    total_loss = losses.calculate_total_loss(
        outputs,
        targets,
        target_refl,
        meta,
        hapke_params=hapke_params,
        device=device,
        camera_params=camera_params,
        w_mse=0.05,
        w_grad=1.0,
        w_refl=1000.0,
    )

    assert torch.isfinite(total_loss).item()
    assert hasattr(total_loss, "loss_components")
    assert all(torch.isfinite(torch.tensor(v)).item() for v in total_loss.loss_components.values())

    total_loss.backward()

    assert outputs.grad is not None
    assert torch.isfinite(outputs.grad).all().item()


def test_calculate_total_loss_handles_nan_from_renderer(dummy_params, monkeypatch):
    camera_params, hapke_params = dummy_params
    device = torch.device("cpu")
    B, H, W = 1, 2, 2

    outputs = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
    targets = torch.zeros_like(outputs)
    meta = torch.zeros(B, 5, 5, device=device)
    target_refl = torch.zeros(B, 5, H, W, device=device)

    def _nan_render(dem_tensor_flat, meta_flat, camera, hapke_model, device):
        bad = torch.tensor([[float("nan"), float("inf")], [-float("inf"), 0.5]], device=device)
        return bad.unsqueeze(0).repeat(dem_tensor_flat.shape[0], 1, 1)

    monkeypatch.setattr(losses, "_render_shading_batched", _nan_render)

    total_loss = losses.calculate_total_loss(
        outputs,
        targets,
        target_refl,
        meta,
        hapke_params=hapke_params,
        device=device,
        camera_params=camera_params,
        w_mse=1.0,
        w_grad=1.0,
        w_refl=1.0,
    )

    assert torch.isfinite(total_loss).item()

    total_loss.backward()

    # Gradients should still propagate for finite locations
    assert outputs.grad is not None
    assert torch.isfinite(outputs.grad).all().item()
