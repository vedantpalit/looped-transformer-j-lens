"""Inkling-2-Mini with its residual stream exposed per layer *application*.

The 24 decoder layers run as 48 applications: layers 0-5 once, the loop band 6-17 three times with
shared weights (passes p1, p2, p3; each later pass restarts from the layer-5 state and reads
"messages" from the previous pass), then layers 18-23 once. A forward hook on a looped layer fires
once per pass, so activations are keyed by application index (call order), never by module.

Two things differ from a plain residual stack and matter for the lens:
- Layer inputs are attention-weighted mixes over earlier block states ("attention residuals"), so a
  layer's output reaches the logits through many paths. The Jacobian with respect to that output
  accounts for all of them; the logit lens (J = I) does not.
- The unembedding reads the output of `final_input` (one more attention residual over all block
  states), not the last layer's output. That mix is the lens target, "final".
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator

import torch
from transformers import AutoTokenizer
from transformers.models.inkling2_loop import Inkling2LoopForConditionalGeneration

BEGIN_OF_TEXT = "<|begin_of_text|>"
FINAL = "final"
# The template ends after the user turn; the model opens its turn with one of these.
PREFILL_ANSWER = "<|message_model|><|content_text|>"  # answer directly: no visible thinking
PREFILL_THINK = "<|message_model|><|content_thinking|>"  # visible thinking first


class Inkling:
    def __init__(self, path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.hf = Inkling2LoopForConditionalGeneration.from_pretrained(path, dtype=dtype, device_map=device)
        self.hf.eval()
        for param in self.hf.parameters():
            param.requires_grad_(False)
        self.text = self.hf.model.language_model
        self.d_model: int = self.text.config.hidden_size
        self.trained_passes: int = self.text.config.loop_num_passes
        self.set_passes(self.trained_passes)

    def set_passes(self, passes: int) -> None:
        """Number of loop passes to run (the model is trained with 3). Rebuilds the application and
        message-site indexing. Fewer passes = an early exit from the loop."""
        config = self.text.config
        config.loop_num_passes = passes
        start, end = config.loop_layer_index_start, config.loop_layer_index_end
        n_layers = config.num_hidden_layers

        # application index -> (layer, pass); pass is 0 outside the loop band
        self.apps: list[tuple[int, int]] = [(l, 0) for l in range(start)]
        self.apps += [(l, p) for p in range(1, passes + 1) for l in range(start, end)]
        self.apps += [(l, 0) for l in range(end, n_layers)]
        self.labels = [f"L{l:02d}" + (f".p{p}" if p else "") for l, p in self.apps]
        self.n_apps = len(self.apps)
        self._app_of_call = {}  # (layer, k-th call of that layer within one forward) -> application index
        calls: dict[int, int] = {}
        for a, (l, _) in enumerate(self.apps):
            self._app_of_call[(l, calls.get(l, 0))] = a
            calls[l] = calls.get(l, 0) + 1
        # message sites: the messages layer l receives at the start of pass p >= 2, in call order of
        # `message_input`; each is [B, T, 4, d] for the (query, key, value, mlp) inputs
        self.msg_sites: list[tuple[int, int]] = [(l, p) for p in range(2, passes + 1) for l in range(start, end)]
        self.msg_labels = [f"M{l:02d}.p{p}" for l, p in self.msg_sites]
        self.loop = (start, end, passes)

    @property
    def device(self) -> torch.device:
        return self.text.embed_tokens.weight.device

    def encode_text(self, text: str, max_length: int | None = None) -> torch.Tensor:
        """Raw text, prefixed with <|begin_of_text|> (the tokenizer adds no BOS itself)."""
        ids = self.tokenizer(BEGIN_OF_TEXT + text, add_special_tokens=False).input_ids
        if max_length is not None:
            ids = ids[:max_length]
        return torch.tensor([ids], device=self.device)

    def encode_chat(self, messages: list[dict], effort: str | float | None = None, prefill: str = "", **kwargs) -> torch.Tensor:
        """Chat prompt ending after the user turn. `prefill` is appended verbatim as the start of the
        model's turn: PREFILL_ANSWER skips the thinking block (hidden reasoning), PREFILL_THINK opens it."""
        ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, reasoning_effort=effort, return_dict=False, **kwargs
        )
        if isinstance(ids, dict) or hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        ids = list(ids)
        if prefill:
            ids += self.tokenizer(prefill, add_special_tokens=False).input_ids
        return torch.tensor([ids], device=self.device)

    def forward(self, input_ids: torch.Tensor):
        """Text decoder only (no LM head). Builds a graph when grad is enabled."""
        return self.text(input_ids=input_ids, use_cache=False)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        """final norm -> lm_head -> logit softcap; exactly the model's own head on the `final` state."""
        # same order as the model: norm on the (fp32) mixed state, then cast to the head's dtype
        x = self.text.norm(residual).to(self.hf.lm_head.weight.dtype)
        return self.hf.logit_softcap(self.hf.lm_head(x).float())

    @contextlib.contextmanager
    def edit(self, edits: dict) -> Iterator[None]:
        """During forwards inside the context, replace the output of application a by edits[a](output)."""
        calls: dict[int, int] = {}
        handles = []

        def reset(module, args, kwargs):
            calls.clear()

        def layer_hook(layer_idx):
            def hook(module, args, output):
                k = calls.get(layer_idx, 0)
                calls[layer_idx] = k + 1
                a = self._app_of_call[(layer_idx, k)]
                if a in edits:
                    return edits[a](output)

            return hook

        try:
            handles.append(self.text.register_forward_pre_hook(reset, with_kwargs=True))
            for l, layer in enumerate(self.text.layers):
                handles.append(layer.register_forward_hook(layer_hook(l)))
            yield
        finally:
            for h in handles:
                h.remove()

    @contextlib.contextmanager
    def edit_messages(self, edits: dict) -> Iterator[None]:
        """During forwards inside the context, replace message site i's output ([B, T, 4, d]) by
        edits[i](output). Sites are indexed as self.msg_sites (call order of `message_input`)."""
        calls = [0]
        handles = []

        def reset(module, args, kwargs):
            calls[0] = 0

        def hook(module, args, output):
            i = calls[0]
            calls[0] += 1
            if i in edits:
                return edits[i](output)

        try:
            handles.append(self.text.register_forward_pre_hook(reset, with_kwargs=True))
            handles.append(self.text.message_input.register_forward_hook(hook))
            yield
        finally:
            for h in handles:
                h.remove()

    @contextlib.contextmanager
    def record(self, at: Iterable[int | str | tuple], *, grad_root: bool = False) -> Iterator[dict]:
        """Capture, during the next forward: layer applications (ints), FINAL, and message sites
        (("msg", i), i indexing self.msg_sites; tensor [B, T, 4, d]).

        With grad_root, the embedding output is made to require grad so every captured tensor is in
        an autograd graph (all parameters have requires_grad=False).
        """
        want = set(at)
        acts: dict = {}
        calls: dict[int, int] = {}
        msg_calls = [0]
        handles = []

        def reset(module, args, kwargs):
            calls.clear()
            msg_calls[0] = 0

        def msg_hook(module, args, output):
            i = msg_calls[0]
            msg_calls[0] += 1
            if ("msg", i) in want:
                acts[("msg", i)] = output

        def layer_hook(layer_idx):
            def hook(module, args, output):
                k = calls.get(layer_idx, 0)
                calls[layer_idx] = k + 1
                a = self._app_of_call[(layer_idx, k)]
                if a in want:
                    acts[a] = output if torch.is_tensor(output) else output[0]

            return hook

        def final_hook(module, args, output):
            acts[FINAL] = output[0]

        def embed_hook(module, args, output):
            return output.requires_grad_(True)

        try:
            handles.append(self.text.register_forward_pre_hook(reset, with_kwargs=True))
            for l, layer in enumerate(self.text.layers):
                handles.append(layer.register_forward_hook(layer_hook(l)))
            if FINAL in want:
                handles.append(self.text.final_input.register_forward_hook(final_hook))
            if any(isinstance(k, tuple) and k[0] == "msg" for k in want):
                handles.append(self.text.message_input.register_forward_hook(msg_hook))
            if grad_root:
                handles.append(self.text.embed_tokens.register_forward_hook(embed_hook))
            yield acts
        finally:
            for h in handles:
                h.remove()
