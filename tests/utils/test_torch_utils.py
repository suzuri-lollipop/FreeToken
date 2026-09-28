"""Default-dtype context manager (utils/torch_utils.py).

torch_dtype temporarily moves torch's global default dtype; if the finally
ever goes missing, default tensors created after a raising model build would
silently change precision - no error, just silently wrong weights.
"""

from __future__ import annotations

import torch
import pytest

from freetoken.utils.torch_utils import torch_dtype


def test_torch_dtype_sets_and_restores():
    original = torch.get_default_dtype()
    with torch_dtype(torch.float64):
        assert torch.get_default_dtype() == torch.float64
    assert torch.get_default_dtype() == original


def test_torch_dtype_restores_after_exception():
    original = torch.get_default_dtype()
    with pytest.raises(RuntimeError):
        with torch_dtype(torch.float16):
            assert torch.get_default_dtype() == torch.float16
            raise RuntimeError("boom")
    assert torch.get_default_dtype() == original


def test_torch_dtype_nested_restores_each_level():
    original = torch.get_default_dtype()
    with torch_dtype(torch.float64):
        with torch_dtype(torch.float16):
            assert torch.get_default_dtype() == torch.float16
        assert torch.get_default_dtype() == torch.float64
    assert torch.get_default_dtype() == original