"""Read J-space (or the logit lens, J = I) at every layer application and position."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

from inkling_jlens.model import FINAL, Inkling


@dataclass
class Lens:
    J: torch.Tensor  # [n_sources, d, d] float32 on device
    labels: list[str]
    sources: list[int]
    n_prompts: int
    passes: int = 3
    J_msg: torch.Tensor | None = None  # [n_msg, 4, d, d] float32 on device (message lens), if fitted
    msg_labels: list[str] | None = None

    @classmethod
    def load(cls, path: str, device) -> Lens:
        s = torch.load(path, map_location="cpu", weights_only=False)
        J_msg = s.get("J_msg")
        return cls(J=s["J"].float().to(device), labels=s["labels"], sources=s["sources"], n_prompts=s["n_prompts"],
                   passes=s.get("passes", 3), J_msg=None if J_msg is None else J_msg.float().to(device),
                   msg_labels=s.get("msg_labels"))


CHANNELS = ("q", "k", "v", "mlp")


@torch.no_grad()
def message_logits(model: Inkling, lens: Lens, ids: torch.Tensor, positions: list[int]):
    """Message lens: what each pass tells the next. Returns logits [n_msg, 4, P, V] (CPU) for the
    message each loop layer receives at the start of passes 2..N, per channel (q, k, v, mlp)."""
    keys = [("msg", i) for i in range(len(model.msg_sites))]
    with model.record(keys) as acts:
        model.forward(ids)
    pos = torch.tensor(positions, device=model.device)
    out = []
    for i, k in enumerate(keys):
        m = acts[k][0][pos].float()  # [P, 4, d]
        z = torch.einsum("pcd,ced->cpe", m, lens.J_msg[i])  # J_c @ m_c  -> [4, P, d]
        out.append(model.unembed(z).float().cpu())
    return torch.stack(out)


@torch.no_grad()
def lens_logits(model: Inkling, lens: Lens | None, ids: torch.Tensor, positions: list[int] | None = None):
    """Returns (logits [n_apps, P, V] float32 on CPU, model_logits [P, V]) at `positions` (default: all).

    lens=None gives the logit lens. Applications without a fitted J (if any) are left as NaN.
    """
    if lens is not None:
        assert model.loop[2] == lens.passes, f"lens fitted with {lens.passes} passes; call model.set_passes({lens.passes})"
    apps = lens.sources if lens is not None else list(range(model.n_apps))
    with model.record([*apps, FINAL]) as acts:
        model.forward(ids)
    pos = torch.arange(ids.shape[1]) if positions is None else torch.tensor(positions)
    pos = pos.to(model.device)
    final = model.unembed(acts[FINAL][0, pos]).float().cpu()
    out = torch.full((model.n_apps, len(pos), final.shape[-1]), float("nan"))
    for i, a in enumerate(apps):
        h = acts[a][0, pos].float()
        z = h @ lens.J[i].T if lens is not None else h
        out[a] = model.unembed(z).float().cpu()
    return out, final


def decode(model: Inkling, token_id: int) -> str:
    return model.tokenizer.decode([token_id])
