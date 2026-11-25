"""Build the fitting corpus: 128-token-ish chunks of WikiText-103 (test + validation), pretraining-like
text as in the paper. Chunks are cut on paragraph boundaries and spread across articles.

usage: python -m inkling_jlens.prepare_corpus PARQUET [PARQUET ...] --out data/prompts.jsonl [--n 1024]
"""

import argparse
import json
import random

import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquets", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--min-chars", type=int, default=700, help="~150+ tokens, so a 128-token cut is full")
    args = ap.parse_args()

    lines = [l for p in args.parquets for l in pq.read_table(p).column("text").to_pylist()]
    chunks, buf = [], ""
    for line in lines:
        s = line.strip()
        if s.startswith("= ") and s.endswith(" ="):  # heading: close the current chunk
            if len(buf) >= args.min_chars:
                chunks.append(buf)
            buf = ""
            continue
        if not s:
            continue
        buf = (buf + " " + s).strip()
        if len(buf) >= args.min_chars:
            chunks.append(buf)
            buf = ""
    clean = [c.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".") for c in chunks]
    random.Random(0).shuffle(clean)
    with open(args.out, "w") as f:
        for c in clean[: args.n]:
            f.write(json.dumps({"text": c}) + "\n")
    print(f"{len(clean)} chunks, wrote {min(args.n, len(clean))} -> {args.out}")


if __name__ == "__main__":
    main()
