"""Compose match EPnP rotation with MLP refine delta."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gmp.diff_epnp import _gram_schmidt_so3


class PoseComposer(nn.Module):
    compose_modes = ('refine_only', 'match_only', 'match_then_refine')

    @staticmethod
    def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dim=-1)

    @staticmethod
    def _project_so3(R: torch.Tensor) -> torch.Tensor:
        """Gram-Schmidt SO(3) projection — avoids second SVD in mat→quat path."""
        return _gram_schmidt_so3(R)

    @staticmethod
    def _mat_to_quat(R: torch.Tensor) -> torch.Tensor:
        """Batch 3×3 -> quaternion (B,4), numerically stable."""
        R = PoseComposer._project_so3(R)
        b = R.shape[0]
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        q = torch.zeros(b, 4, device=R.device, dtype=R.dtype)
        s0 = (trace + 1.0).clamp(min=1e-6).sqrt() * 2.0
        q0 = torch.stack([
            0.25 * s0,
            (R[:, 2, 1] - R[:, 1, 2]) / s0.clamp(min=1e-6),
            (R[:, 0, 2] - R[:, 2, 0]) / s0.clamp(min=1e-6),
            (R[:, 1, 0] - R[:, 0, 1]) / s0.clamp(min=1e-6),
        ], dim=-1)
        s1 = (1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]).clamp(min=1e-6).sqrt() * 2.0
        q1 = torch.stack([
            (R[:, 2, 1] - R[:, 1, 2]) / s1.clamp(min=1e-6),
            0.25 * s1,
            (R[:, 0, 1] + R[:, 1, 0]) / s1.clamp(min=1e-6),
            (R[:, 0, 2] + R[:, 2, 0]) / s1.clamp(min=1e-6),
        ], dim=-1)
        s2 = (1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2]).clamp(min=1e-6).sqrt() * 2.0
        q2 = torch.stack([
            (R[:, 0, 2] - R[:, 2, 0]) / s2.clamp(min=1e-6),
            (R[:, 0, 1] + R[:, 1, 0]) / s2.clamp(min=1e-6),
            0.25 * s2,
            (R[:, 1, 2] + R[:, 2, 1]) / s2.clamp(min=1e-6),
        ], dim=-1)
        s3 = (1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1]).clamp(min=1e-6).sqrt() * 2.0
        q3 = torch.stack([
            (R[:, 1, 0] - R[:, 0, 1]) / s3.clamp(min=1e-6),
            (R[:, 0, 2] + R[:, 2, 0]) / s3.clamp(min=1e-6),
            (R[:, 1, 2] + R[:, 2, 1]) / s3.clamp(min=1e-6),
            0.25 * s3,
        ], dim=-1)
        cond0 = trace > 0.0
        cond1 = (~cond0) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
        cond2 = (~cond0) & (~cond1) & (R[:, 1, 1] > R[:, 2, 2])
        q = torch.where(cond0.unsqueeze(-1), q0, q)
        q = torch.where(cond1.unsqueeze(-1), q1, q)
        q = torch.where(cond2.unsqueeze(-1), q2, q)
        q = torch.where((~cond0 & ~cond1 & ~cond2).unsqueeze(-1), q3, q)
        return F.normalize(q, dim=-1, eps=1e-6)

    def forward(
        self,
        q_refine: torch.Tensor,
        R_match: torch.Tensor | None,
        mode: str = 'match_then_refine',
    ) -> tuple[torch.Tensor, dict]:
        mode = mode if mode in self.compose_modes else 'match_then_refine'
        meta = {'compose_mode': mode}

        if mode == 'refine_only' or R_match is None:
            q_pred = F.normalize(q_refine, dim=-1, eps=1e-6)
            meta['used_match'] = False
            return q_pred, meta

        q_match = self._mat_to_quat(R_match)
        meta['q_match'] = q_match.detach()
        meta['used_match'] = True

        if mode == 'match_only':
            q_pred = q_match
        else:
            q_pred = self._quat_mul(q_refine, q_match)
        q_pred = F.normalize(q_pred, dim=-1, eps=1e-6)
        meta['q_refine'] = q_refine.detach()
        return q_pred, meta
