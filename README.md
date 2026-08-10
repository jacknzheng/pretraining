# pretraining

Pretraining architectures implemented from scratch in PyTorch. Each one is built up from
first principles with notes on the reasoning, time/space complexity, and correctness tests
against a reference where one exists.

## Layout

```
notebooks/     exploratory implementations, one per topic
pretraining/   package for implementations promoted out of the notebooks
```

## `notebooks/attention.ipynb`

|                                | status                           |
| ------------------------------ | -------------------------------- |
| Dot product attention          | done, verified vs. reference     |
| MHA (looped + einsum)          | done, verified                   |
| MHA with KV cache              | done, prefill == decode verified |
| GQA                            | done, prefill == decode verified |
| MLA                            | done, prefill == decode verified |
| MLA with RoPE                  | done, prefill == decode verified |
| MLA with lightning indexer     | not started                      |
| Linear attention (naive)       | done                             |
| Linear attention with chunking | not started                      |
| DeepSeek sparse attention      | not started                      |

The RoPE variant uses decoupled positional embeddings: the head splits into a "nope" half
that gets absorbed into the query at decode time (`q·(Wc) = (qᵀW)·c`, so the cache is never
expanded) and a small rope half kept in a separate `pe_cache`. RoPE and the up-projection
don't commute, which is why the split exists.

`wkv_b` must be `bias=False` — the absorption identity only holds for a pure linear map, so
a bias makes prefill and decode compute different functions. Verified: prefill vs. six
one-token decode steps agree to 6.6e-07.

Also has notes on arithmetic intensity of prefill vs. decode.

## `notebooks/moe.ipynb`

|                                            | status                                              |
| ------------------------------------------ | --------------------------------------------------- |
| `Expert`, `Router`                         | done                                                |
| `MoeBlock` — top-k routing, shared experts | done                                                |
| Router load-balancing loss                 | done — computed per block, summed across the stack  |
| `MoeTransformer`                           | done — pre-norm MLA + MoeBlock with residuals       |
| `MoeGPT`                                   | done — handwritten forward, returns loss + lb_loss  |
| `DeepseekRouter`                           | done                                                |
| Latent MoE (Kimi K3)                       | not started                                         |

MLA and RoPE are copy-pasted into this notebook rather than imported from
`attention.ipynb`, so the two copies can drift.

## `notebooks/pos_encodings.ipynb`

The search for good positional embeddings, in the order the field found them:

| | |
| --- | --- |
| Integer / absolute | done, with notes on why it fails |
| Binary | done — motivates the move to continuous |
| Sinusoidal | done, plus plots of magnitude and cosine similarity vs. distance |
| Learned | done |
| Relative | done |
| Rotary (RoPE) | done |
| YaRN | done — angle rescaling for context extension |

Includes primers on `torch.polar`, `torch.stack`, and `torch.gather`.

## `notebooks/tokenizer.ipynb`

Byte-pair encoding from scratch — `BPETokenizer` with `get_stats`, `merge_seq`, `merge`,
`encode`, `decode`.

Note on encode order: merges must be applied in ascending index order, since `257 = (256, t)`
needs its `256` already formed. A single ascending pass is sufficient — a merge replaces a
pair with a brand-new id, so it can only ever create work for *higher*-indexed rules, never
lower ones. Confirmed empirically over 200k randomized trials against the repeatedly-rescan
(`min`-index) form: zero disagreements. The `min` form is still preferred in practice, for
speed on short inputs and independence from dict insertion order.

## `notebooks/gpt2.ipynb`

|                                        | status                                  |
| -------------------------------------- | --------------------------------------- |
| `RMSNorm`, `LayerNorm`, `BatchNorm`     | done                                    |
| `Cache`                                | in progress — `__init__` signature only |
| Dataloader (tinyshakespeare → `.bin`)  | done                                    |
| `Tokenizer`                            | in progress                             |
| GPT-2 block / config / model           | not started                             |
| LR scheduler, optimizer, dropout       | markdown notes only                     |

The dataloader keeps one flat 1-D stream of token ids and slices fixed windows out of it —
which is why pretraining needs no PAD token: every window is `max_seq_len` by construction.
Stored as `uint16`, which holds GPT-2's 50257 but wraps silently above 65535, so it asserts.

## `notebooks/algorithms.ipynb`

PEFT / fine-tuning. All stubs: LoRA, DPO, SFT loss. RL section empty.

## `notebooks/puzzles.ipynb`

Scratch space for PyTorch indexing semantics — `nonzero(as_tuple=True)`, `index_add_`.

## `pretraining/`

|                 | status                     |
| --------------- | -------------------------- |
| `deepseekv3.py` | not started — comment only |

## Setup

```bash
uv sync
```

Python 3.12. Deps: torch, einops, numpy, matplotlib, transformers, datasets, nbformat,
nbimporter.
