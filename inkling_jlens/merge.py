"""Merge fit shards into one lens file.

usage: python -m inkling_jlens.merge FIT_DIR OUT.pt
OUT.pt holds (fp16, prompt-weighted means over shards):
  J      [n_apps, d, d]      output lens per layer application (labels)
  J_msg  [n_msg, 4, d, d]    message lens per message site (msg_labels) and channel (q, k, v, mlp); absent if not fitted
plus labels, msg_labels, passes, target, n_prompts.
"""

import glob
import os
import sys

import torch


def main():
    fit_dir, out = sys.argv[1], sys.argv[2]
    shards = [torch.load(p, weights_only=False) for p in sorted(glob.glob(os.path.join(fit_dir, "shard*.pt")))]
    shards = [s for s in shards if s["n_done"]]
    first = shards[0]
    for s in shards[1:]:
        assert s["sources"] == first["sources"] and s["target"] == first["target"], "shards disagree"
    n = sum(s["n_done"] for s in shards)
    mean = {k: sum(s["J_sum"][k] for s in shards) / n for k in first["sources"]}
    apps = [k for k in first["sources"] if not isinstance(k, tuple)]
    msgs = sorted((k for k in first["sources"] if isinstance(k, tuple)), key=lambda k: k[1])
    lens = dict(J=torch.stack([mean[a] for a in apps]).half(), labels=first["labels"], sources=apps,
                msg_labels=first.get("msg_labels", []), passes=first.get("passes", 3), target=first["target"], n_prompts=n,
                skip_first=first["skip_first"], max_seq_len=first["max_seq_len"])
    if msgs:
        lens["J_msg"] = torch.stack([mean[k] for k in msgs]).half()
    if len(shards) >= 2:
        # split-half reliability: lens from even shards vs odd shards; small = more prompts won't change it
        halves = [shards[0::2], shards[1::2]]
        h = [{k: sum(s["J_sum"][k] for s in half) / sum(s["n_done"] for s in half) for k in apps} for half in halves]
        rel = {apps_k: ((h[0][apps_k] - h[1][apps_k]).norm() / mean[apps_k].norm()).item() for apps_k in apps}
        worst = max(rel, key=rel.get)
        lens["split_half_rel_diff"] = rel
        print(f"split-half rel. diff (even vs odd shards): median {sorted(rel.values())[len(rel) // 2]:.3f}, "
              f"max {rel[worst]:.3f} at {first['labels'][worst]}")
    torch.save(lens, out)
    print(f"{len(shards)} shards, {n} prompts, J {tuple(lens['J'].shape)}"
          + (f", J_msg {tuple(lens['J_msg'].shape)}" if msgs else "") + f" -> {out}")


if __name__ == "__main__":
    main()
