"""Fit the Jacobian lens on Inkling-2-Mini, one shard of the prompt corpus per job.

Estimator (as in anthropics/jacobian-lens, jlens/fitting.py): for each prompt, one forward on the
prompt replicated `dim_batch` times, then ceil(d_model / dim_batch) backward passes; batch element b
carries a one-hot cotangent on output dim (start + b) of the target at every valid position. The
gradient at source position p is then sum_{p' >= p} d target[p'] / d h_a[p], averaged over valid p.
Sources are layer *applications* (48: the loop band counts once per pass), not modules, plus the
message sites (the gated messages each loop layer receives from the previous pass; 4 channels each).
--passes N fits the early-exit lens: the model run with N loop passes, target its own final state.

usage: python -m inkling_jlens.fit --model DIR --prompts prompts.jsonl --out DIR [--shard i --n-shards n]
Writes out/shard{i}.pt (running sums, resumable) and logs per-prompt time and convergence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from inkling_jlens.model import FINAL, Inkling


def valid_positions(seq_len: int, skip_first: int) -> torch.Tensor:
    """Skip the attention-sink prefix and the last position (no next-token target)."""
    return torch.arange(skip_first, seq_len - 1)


def zeros_for(key, d):
    """Applications: J [d, d]. Message sites ("msg", i): J [4, d, d], one per (q, k, v, mlp) channel."""
    return torch.zeros(4, d, d) if isinstance(key, tuple) else torch.zeros(d, d)


def jacobian_for_ids(model: Inkling, ids: torch.Tensor, sources: list, target, dim_batch: int, skip_first: int):
    d = model.d_model
    positions = valid_positions(ids.shape[1], skip_first).to(model.device)
    J = {a: zeros_for(a, d) for a in sources}
    n_passes = math.ceil(d / dim_batch)
    with model.record([*sources, target], grad_root=True) as acts, torch.enable_grad():
        model.forward(ids.expand(dim_batch, -1))
        out = acts[target]
        ins = [acts[a] for a in sources]
        rows_idx = torch.arange(dim_batch, device=out.device)
        cot = torch.zeros_like(out)
        for i, start in enumerate(range(0, d, dim_batch)):
            n = min(dim_batch, d - start)
            cot.zero_()
            cot[rows_idx[:n, None], positions[None, :], start + rows_idx[:n, None]] = 1.0
            grads = torch.autograd.grad(out, ins, grad_outputs=cot, retain_graph=i < n_passes - 1)
            for a, g in zip(sources, grads, strict=True):
                rows = g[:n, positions].float().mean(1).cpu()  # [n, d] or [n, 4, d] for messages
                if rows.dim() == 3:
                    J[a][:, start : start + n] = rows.transpose(0, 1)
                else:
                    J[a][start : start + n] = rows
            del grads
    return J


def atomic_save(obj, path):
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, help="jsonl with a 'text' field per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--n-prompts", type=int, default=None, help="use only the first N prompts of the corpus")
    ap.add_argument("--target", default=FINAL, help="'final' (input to the unembedding) or an application index")
    ap.add_argument("--dim-batch", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--skip-first", type=int, default=16)
    ap.add_argument("--checkpoint-every", type=int, default=8, help="a checkpoint is ~3GB")
    ap.add_argument("--passes", type=int, default=None, help="loop passes to run (default: as trained, 3)")
    ap.add_argument("--no-messages", action="store_true", help="skip the message lens")
    args = ap.parse_args()

    torch.manual_seed(0)
    prompts = [json.loads(l)["text"] for l in open(args.prompts)]
    if args.n_prompts:
        prompts = prompts[: args.n_prompts]
    mine = list(range(args.shard, len(prompts), args.n_shards))

    model = Inkling(args.model)
    if args.passes:
        model.set_passes(args.passes)
    target = FINAL if args.target == FINAL else int(args.target)
    last_source = model.n_apps if target == FINAL else target
    sources = list(range(last_source))
    if target == FINAL and not args.no_messages:
        sources += [("msg", i) for i in range(len(model.msg_sites))]
    print(f"passes={model.loop[2]} apps={model.n_apps} msg_sites={len(model.msg_sites)} sources={len(sources)} target={target} "
          f"shard {args.shard}/{args.n_shards}: {len(mine)} prompts", flush=True)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"shard{args.shard}.pt")
    meta = dict(sources=sources, target=target, labels=model.labels, msg_labels=model.msg_labels, passes=model.loop[2],
                skip_first=args.skip_first, max_seq_len=args.max_seq_len)
    if os.path.exists(path):
        state = torch.load(path, weights_only=False)
        assert all(state[k] == v for k, v in meta.items()), "checkpoint was fitted with different settings"
        # Extending a fit (larger --n-prompts) is safe only if this shard's prompt list starts with the
        # prompts already done, i.e. same --n-shards and the same prompts file. Otherwise prompts would be
        # mixed or counted twice.
        done_positions = {i: mine.index(i) for i in state["done_ids"] if i in mine}
        assert len(done_positions) == len(state["done_ids"]) and all(p < state["next"] for p in done_positions.values()), (
            "checkpoint's completed prompts are not a prefix of this shard's prompt list: "
            "keep --n-shards and the prompts file unchanged when extending a fit")
        print(f"resuming: {state['n_done']} done, next {state['next']}", flush=True)
    else:
        d = model.d_model
        state = dict(meta, J_sum={a: zeros_for(a, d) for a in sources}, n_done=0, next=0, done_ids=[])

    sqrt_d = math.sqrt(model.d_model)
    for j, idx in enumerate(mine):
        if j < state["next"]:
            continue
        ids = model.encode_text(prompts[idx], max_length=args.max_seq_len)
        if ids.shape[1] <= args.skip_first + 1:
            state["next"] = j + 1
            continue
        t0 = time.perf_counter()
        J = jacobian_for_ids(model, ids, sources, target, args.dim_batch, args.skip_first)
        norm = max(J[a].norm().item() for a in sources) / sqrt_d
        if state["n_done"]:
            n = state["n_done"]
            shift = max(((J[a] - state["J_sum"][a] / n).norm() / ((n + 1) * (state["J_sum"][a] / n).norm())).item() for a in sources)
        else:
            shift = float("nan")
        for a in sources:
            state["J_sum"][a] += J[a]
        state["n_done"] += 1
        state["next"] = j + 1
        state["done_ids"].append(idx)
        print(
            f"prompt {j + 1}/{len(mine)} (#{idx}) len={ids.shape[1]} {time.perf_counter() - t0:.0f}s "
            f"max||J||/sqrt(d)={norm:.3f} mean_shift={shift:.2e} peak_mem={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB",
            flush=True,
        )
        if state["next"] % args.checkpoint_every == 0:
            atomic_save(state, path)
    atomic_save(state, path)
    print(f"done: {state['n_done']} prompts -> {path}", flush=True)


if __name__ == "__main__":
    main()
