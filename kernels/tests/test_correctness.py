"""kernels/tests/test_correctness.py — Phase 2 (STUB).

Validate the fused dequant+GEMV kernel against harness/reference_gemv.py across
multiple shapes and seeds, within tolerance. **MUST pass before any benchmark.**

This file is the gate: §7 says no timing number is ever reported without a
passing correctness test in the same run.

TODO(Phase 2):
  - parametrize over shapes (N, K), group sizes, seeds,
  - assert torch.allclose(fused, reference, rtol=..., atol=...) with a
    justified tolerance budget (see docs/02_kernel_design.md).
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Phase 2 not implemented yet — see docs/02_kernel_design.md")
def test_fused_matches_reference():
    raise NotImplementedError
