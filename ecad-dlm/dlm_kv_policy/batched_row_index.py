"""Per-request row indexing. -1 denotes a dummy row and is never written."""
from __future__ import annotations
import torch


def pack_rows(selected: torch.Tensor) -> torch.Tensor:
    """Ascending positions, padded to the largest request with -1.

    One explicit device-to-host count synchronization per distinct row set.
    """
    count = int(selected.sum(-1).max().item())
    positions = torch.arange(selected.shape[1], device=selected.device).expand_as(selected)
    packed = positions.masked_fill(~selected, selected.shape[1]).sort(-1).values[:, :count]
    return packed.masked_fill(packed.eq(selected.shape[1]), -1)


def gather_rows(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    if positions.ndim == 1:
        return values.index_select(1, positions)
    values = values.expand(positions.shape[0], -1, -1)
    return values.gather(1, positions.clamp_min(0)[..., None].expand(-1, -1, values.shape[-1]))


def scatter_rows(target: torch.Tensor, positions: torch.Tensor, values: torch.Tensor) -> None:
    # Boolean indexing excludes dummy writes entirely (including duplicate dummy indices).
    batch = torch.arange(positions.shape[0], device=positions.device)[:, None].expand_as(positions)
    valid = positions.ge(0)
    target[batch[valid], positions[valid]] = values[valid]
