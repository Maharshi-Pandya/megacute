"""Llama-3.2-1B decode with megacute: one persistent kernel launch per generated token.

    python examples/llama/llama1b.py                    # greedy-generate from a prompt, report tok/s
    python examples/llama/llama1b.py --batch 16 --tokens 256   # B identical sequences (throughput)
    python examples/llama/llama1b.py --check            # compare tokens with Hugging Face transformers

Loads the Hugging Face checkpoint, converts it to megacute's layout (fused QKV, interleaved
gate|up, per-layer stacked parameters), builds `model.llama_decode` (this folder) and runs every step --
all 16 layers, KV-cache append, final norm, tied lm_head and the position increment -- as a
single launch. The prompt goes through the same decode program one token at a time.
Needs a Blackwell (sm_100a) GPU plus `transformers`, `safetensors` and `huggingface_hub`.
"""

import argparse
import json
import math
import pathlib
import sys
import time

import torch

import megacute as mc

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parent))   # this example's model.py
from model import llama_decode  # noqa: E402

BF = torch.bfloat16


def load(repo):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    path = pathlib.Path(snapshot_download(repo, allow_patterns=["*.json", "*.safetensors"]))
    cfg = json.loads((path / "config.json").read_text())
    w = {}
    for f in sorted(path.glob("*.safetensors")):
        w.update(load_file(str(f), device="cuda"))
    return path, cfg, {k: v.to(BF) for k, v in w.items()}


def params(cfg, w):
    """Hugging Face Llama weights -> model.llama_decode parameters (layers stacked)."""
    n, F_ = cfg["num_hidden_layers"], cfg["intermediate_size"]
    L = lambda i, name: w[f"model.layers.{i}.{name}.weight"]
    cat = lambda f: torch.cat([f(i) for i in range(n)]).contiguous()
    return {
        "attn_norm": cat(lambda i: L(i, "input_layernorm")[None]),
        "wqkv": cat(lambda i: torch.cat([L(i, "self_attn.q_proj"), L(i, "self_attn.k_proj"), L(i, "self_attn.v_proj")])),
        "wo": cat(lambda i: L(i, "self_attn.o_proj")),
        "mlp_norm": cat(lambda i: L(i, "post_attention_layernorm")[None]),
        "wgu": cat(lambda i: mc.swiglu_weight(L(i, "mlp.gate_proj"), L(i, "mlp.up_proj"))),
        "wd": cat(lambda i: L(i, "mlp.down_proj")),
        "final_norm": w["model.norm.weight"][None].contiguous(),
        "lm_head": w.get("lm_head.weight", w["model.embed_tokens.weight"]),     # tied in Llama-3.2-1B
    }


def rope_tables(cfg, max_seq):
    """cos/sin (max_seq, head_dim), rotate-half convention, with Llama-3 frequency scaling."""
    D = cfg["hidden_size"] // cfg["num_attention_heads"]
    inv = 1.0 / cfg["rope_theta"] ** (torch.arange(0, D, 2, dtype=torch.float64, device="cuda") / D)
    sc = cfg.get("rope_scaling") or {}
    if sc.get("rope_type", sc.get("type")) == "llama3":
        f, lo, hi, orig = sc["factor"], sc["low_freq_factor"], sc["high_freq_factor"], sc["original_max_position_embeddings"]
        wl = 2 * math.pi / inv
        smooth = ((orig / wl - lo) / (hi - lo)).clamp(0, 1)
        inv = torch.where(wl > orig / lo, inv / f, torch.where(wl < orig / hi, inv, (1 - smooth) * inv / f + smooth * inv))
    ang = torch.arange(max_seq, dtype=torch.float64, device="cuda")[:, None] * inv[None]
    ang = torch.cat([ang, ang], -1)
    return ang.cos().to(BF).contiguous(), ang.sin().to(BF).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--prompt", default="The three laws of thermodynamics are")
    ap.add_argument("--tokens", type=int, default=128, help="tokens to generate")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--check", action="store_true", help="compare greedy tokens with transformers")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    path, cfg, w = load(a.model)
    tok = AutoTokenizer.from_pretrained(str(path)) if (path / "tokenizer.json").exists() else AutoTokenizer.from_pretrained(a.model)
    E, H, KVH = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    n, F_, V, B, Lm = cfg["num_hidden_layers"], cfg["intermediate_size"], cfg["vocab_size"], a.batch, a.max_seq
    prompt = tok(a.prompt, return_tensors="pt").input_ids[0].tolist()
    assert len(prompt) + a.tokens <= Lm, "--max-seq too small"

    t0 = time.time()
    prog = mc.compile(llama_decode(B, E, H, KVH, F_, n, Lm, V, eps=cfg["rms_norm_eps"]))
    print(f"compiled in {time.time() - t0:.1f}s: {len(prog.sched.insts)} instructions, {len(prog.nodes)} nodes", flush=True)
    cos, sin = rope_tables(cfg, Lm)
    kvd = KVH * (E // H)
    st = dict(x=torch.empty(B, E, device="cuda", dtype=BF), pos=torch.zeros(1, 1, dtype=torch.int32, device="cuda"),
              k_cache=torch.zeros(n * B * Lm, kvd, device="cuda", dtype=BF),
              v_cache=torch.zeros(n * B * Lm, kvd, device="cuda", dtype=BF))
    p = params(cfg, w)
    emb = w["model.embed_tokens.weight"]

    def step(ids):              # ids: (B,) int64 on the GPU -> next ids (B,), no host sync
        st["x"].copy_(emb.index_select(0, ids))
        return prog(cos=cos, sin=sin, **st, **p).argmax(-1)

    ids = torch.tensor(prompt[:1] * B, device="cuda")
    for t in prompt[1:]:        # prompt: one decode step per token (no separate prefill)
        step(ids)
        ids = torch.full((B,), t, device="cuda")
    out = []
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(a.tokens):
        ids = step(ids)
        out.append(ids)
    torch.cuda.synchronize()
    dt = time.time() - t0
    gen = torch.stack(out, 1)[0].tolist()
    print(f"{a.tokens} tokens x batch {B}: {dt / a.tokens * 1e3:.3f} ms/step, {a.tokens / dt:.0f} tok/s per sequence, "
          f"{a.tokens * B / dt:.0f} tok/s total (context {len(prompt)}..{len(prompt) + a.tokens})")
    print(repr(a.prompt + tok.decode(gen)))

    if a.check:
        from transformers import AutoModelForCausalLM
        ref = AutoModelForCausalLM.from_pretrained(str(path), dtype=BF).cuda()
        want = ref.generate(torch.tensor([prompt], device="cuda"), max_new_tokens=a.tokens, do_sample=False)[0, len(prompt):]
        same = next((i for i, (x, y) in enumerate(zip(gen, want.tolist())) if x != y), len(gen))
        print(f"check vs transformers: first {same} of {a.tokens} greedy tokens identical "
              f"(bf16 runs drift apart eventually; a mismatch at token 0-2 would mean a bug)")
        print(repr(a.prompt + tok.decode(want)))


if __name__ == "__main__":
    main()
