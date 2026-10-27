import sys

import pytest
import torch

from pipeline.export.parity import mean_cosine
from pipeline.export.to_coreml import export_coreml
from pipeline.export.to_executorch import export_executorch

# Core ML models only run on macOS; ExecuTorch's runtime also runs on Linux
EXPORTERS = [
    pytest.param(export_coreml, marks=pytest.mark.skipif(sys.platform != "darwin", reason="Core ML runs on macOS only")),
    export_executorch,
]


class Identity(torch.nn.Module):
    """Returns its input as a unit vector per row, like an embedder that changes nothing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x.flatten(1), dim=-1)


class Reversed(Identity):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).flip(-1)


def inputs() -> torch.Tensor:
    return torch.rand(4, 3, 8, 8, generator=torch.Generator().manual_seed(0))


@pytest.mark.parametrize("export", EXPORTERS)
def test_parity_on_identity_model(export, tmp_path):
    exported = export(Identity(), torch.zeros(1, 3, 8, 8), tmp_path / "identity")

    assert mean_cosine(Identity(), exported, inputs()) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("export", EXPORTERS)
def test_parity_catches_an_export_that_computes_something_else(export, tmp_path):
    exported = export(Reversed(), torch.zeros(1, 3, 8, 8), tmp_path / "reversed")

    assert mean_cosine(Identity(), exported, inputs()) < 0.99


def test_parity_refuses_an_export_from_another_embedder(tmp_path):
    from types import SimpleNamespace

    from pipeline.export.model import check_sidecar, write_sidecar

    exported = tmp_path / "model.pte"
    write_sidecar(exported, SimpleNamespace(signature="dinov3-vits16-224px-aaaaaaaa"))

    check_sidecar(exported, SimpleNamespace(signature="dinov3-vits16-224px-aaaaaaaa"))
    with pytest.raises(SystemExit, match="Export it again"):
        check_sidecar(exported, SimpleNamespace(signature="dinov3-vits16-224px-bbbbbbbb"))
