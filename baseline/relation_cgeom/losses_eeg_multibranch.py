from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from baseline.relation_cgeom.losses_complex_geom import hierarchical_contrastive_loss_complex_geom


@dataclass
class EEGMultiBranchLossConfig:
    lambda_cgeom: float = 1.0
    lambda_tf_align: float = 0.3
    lambda_raw_mask: float = 0.0
    use_cgeom_main: bool = True
    use_tf_align: bool = True
    use_raw_mask: bool = False
    tf_align_type: str = 'cosine'
    raw_mask_type: str = 'cosine'


def complex_geom_main_loss(z1: torch.Tensor, z2: torch.Tensor, temporal_unit: int = 0) -> torch.Tensor:
    return hierarchical_contrastive_loss_complex_geom(z1, z2, temporal_unit=temporal_unit)


def time_frequency_alignment_loss(
    raw_repr: torch.Tensor,
    complex_repr: torch.Tensor,
    loss_type: str = 'cosine',
) -> torch.Tensor:
    if raw_repr is None or complex_repr is None:
        raise ValueError('Alignment loss requires both raw and complex representations.')

    raw_repr = F.normalize(raw_repr, dim=-1)
    complex_repr = F.normalize(complex_repr, dim=-1)

    if loss_type == 'cosine':
        return (1.0 - (raw_repr * complex_repr).sum(dim=-1)).mean()
    if loss_type == 'mse':
        return F.mse_loss(raw_repr, complex_repr)
    raise ValueError(f'Unsupported tf_align_type={loss_type}')


def raw_masked_representation_loss(
    masked_repr: torch.Tensor,
    full_repr: torch.Tensor,
    loss_type: str = 'cosine',
) -> torch.Tensor:
    if masked_repr is None or full_repr is None:
        raise ValueError('Raw mask loss requires both masked and full raw representations.')

    full_repr = full_repr.detach()

    if loss_type == 'cosine':
        masked_repr = F.normalize(masked_repr, dim=-1)
        full_repr = F.normalize(full_repr, dim=-1)
        return (1.0 - (masked_repr * full_repr).sum(dim=-1)).mean()
    if loss_type == 'mse':
        return F.mse_loss(masked_repr, full_repr)
    raise ValueError(f'Unsupported raw_mask_type={loss_type}')


class EEGMultiBranchLoss(nn.Module):
    def __init__(self, config: EEGMultiBranchLossConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        view1: Dict[str, torch.Tensor],
        view2: Dict[str, torch.Tensor],
        temporal_unit: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        main_ref = view1.get('main_repr', None)
        if main_ref is None:
            main_ref = next(v for v in view1.values() if isinstance(v, torch.Tensor))
        total_loss = torch.tensor(0.0, device=main_ref.device)
        loss_terms: Dict[str, float] = {}

        if self.config.use_cgeom_main:
            loss_cgeom = complex_geom_main_loss(view1['main_repr'], view2['main_repr'], temporal_unit=temporal_unit)
            total_loss = total_loss + self.config.lambda_cgeom * loss_cgeom
            loss_terms['loss_cgeom'] = float(loss_cgeom.detach().item())

        if self.config.use_tf_align:
            raw1 = view1.get('raw_align')
            raw2 = view2.get('raw_align')
            complex1 = view1.get('complex_align')
            complex2 = view2.get('complex_align')
            if raw1 is not None and raw2 is not None and complex1 is not None and complex2 is not None:
                loss_align_1 = time_frequency_alignment_loss(raw1, complex1, self.config.tf_align_type)
                loss_align_2 = time_frequency_alignment_loss(raw2, complex2, self.config.tf_align_type)
                loss_tf_align = 0.5 * (loss_align_1 + loss_align_2)
                total_loss = total_loss + self.config.lambda_tf_align * loss_tf_align
                loss_terms['loss_tf_align'] = float(loss_tf_align.detach().item())

        if self.config.use_raw_mask:
            masked1 = view1.get('raw_masked_repr')
            masked2 = view2.get('raw_masked_repr')
            full1 = view1.get('raw_full_repr')
            full2 = view2.get('raw_full_repr')
            if masked1 is not None and masked2 is not None and full1 is not None and full2 is not None:
                loss_raw_mask_1 = raw_masked_representation_loss(masked1, full1, self.config.raw_mask_type)
                loss_raw_mask_2 = raw_masked_representation_loss(masked2, full2, self.config.raw_mask_type)
                loss_raw_mask = 0.5 * (loss_raw_mask_1 + loss_raw_mask_2)
                total_loss = total_loss + self.config.lambda_raw_mask * loss_raw_mask
                loss_terms['loss_raw_mask'] = float(loss_raw_mask.detach().item())

        loss_terms['loss_total'] = float(total_loss.detach().item())
        return total_loss, loss_terms
