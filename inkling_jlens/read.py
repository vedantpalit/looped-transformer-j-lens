"""Print what the Jacobian lens reads on each pass of the loop, word by word.

usage: python -m inkling_jlens.read --model DIR --lens lens_main.pt \
           [--exit-lenses lens_p1.pt lens_p2.pt] [--baseline data/prompts.jsonl] \
           --prompt "The chemical symbol of the metal with atomic number 79 is" [--k 5]

Stages: before the loop (layers 0-5), pass 1, pass 2, pass 3 (loop layers 6-17), after the loop (18-23).
Pass p < 3 is read with the exit lens fitted on the model cut to p passes (fit.py --passes p), if given;
otherwise with the full-model lens. Scores are log-probabilities averaged over the stage's layers; with
--baseline, each layer's mean log-probability over ordinary text is subtracted first (contrastive readout).
"""

from __future__ import annotations

import argparse
import json

import torch

from inkling_jlens.model import Inkling
from inkling_jlens.readout import Lens, lens_logits


def stage_apps(model: Inkling) -> dict[str, list[int]]:
    start, end, passes = model.loop
    n = end - start
    out = {"before loop": list(range(start))}
    for p in range(1, passes + 1):
        out[f"pass {p}"] = list(range(start + (p - 1) * n, start + p * n))
    out["after loop"] = list(range(start + passes * n, model.n_apps))
    return out


@torch.no_grad()
def baseline(model: Inkling, lens: Lens, texts: list[str], skip_first: int = 16, n_pos: int = 32) -> torch.Tensor:
    """Mean log-probability per layer application over ordinary text: [n_apps, V]."""
    total, n = None, 0
    for t in texts:
        ids = model.encode_text(t, max_length=128)
        pos = list(range(skip_first, min(ids.shape[1], skip_first + n_pos)))
        if not pos:
            continue
        lg, _ = lens_logits(model, lens, ids, pos)
        lp = torch.log_softmax(lg, -1).sum(1)
        total = lp if total is None else total + lp
        n += len(pos)
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", required=True, help="lens fitted on the full model (all passes)")
    ap.add_argument("--exit-lenses", nargs="*", default=[], help="exit lenses for passes 1, 2, ... in order")
    ap.add_argument("--baseline", help="jsonl with a 'text' field; enables contrastive scores")
    ap.add_argument("--n-baseline", type=int, default=16)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    model = Inkling(args.model)
    tok = model.tokenizer
    ids = model.encode_text(args.prompt)
    words = [tok.decode([t]) for t in ids[0].tolist()]
    texts = [json.loads(l)["text"] for l in open(args.baseline)][: args.n_baseline] if args.baseline else []

    lenses = [(Lens.load(args.lens, model.device), None)]
    for p, path in enumerate(args.exit_lenses, start=1):
        lenses.append((Lens.load(path, model.device), p))

    stages: dict[str, torch.Tensor] = {}
    for lens, only_pass in lenses:
        model.set_passes(lens.passes)
        lg, _ = lens_logits(model, lens, ids)
        lp = torch.log_softmax(lg, -1)
        if texts:
            lp = lp - baseline(model, lens, texts)[:, None, :]
        for name, apps in stage_apps(model).items():
            if only_pass is not None and name != f"pass {only_pass}":
                continue
            stages[name] = lp[apps].mean(0)  # [P, V]
    model.set_passes(model.trained_passes)

    order = ["before loop"] + [f"pass {p}" for p in range(1, model.trained_passes + 1)] + ["after loop"]
    for i, w in enumerate(words):
        print(f"{i:3d} {w!r}")
        for name in order:
            top = stages[name][i].topk(args.k).indices.tolist()
            print(f"      {name:12s} " + ", ".join(repr(tok.decode([t]).strip()) for t in top))


if __name__ == "__main__":
    main()
