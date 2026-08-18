#!/usr/bin/env python3
"""
Self-contained hierarchical beam search for a pretrain_ar_v2 (use_3head, byte-level
SID) checkpoint. Predicts the NEXT item's SID given a user's interaction history.

ONLY dependencies: torch + transformers (Qwen3-MoE support) + numpy. No repo imports.

What it does
------------
The model is a Qwen3-MoE backbone with vocab substituted to 3*8192 = 24576 (byte-level
SID tokens) plus 3 external linear heads (one per SID byte level). Each item occupies
4 tokens: [ctx, sid0, sid1, sid2], where
    sid0 = byte0 + 0*8192   (byte0 in [0,8191])
    sid1 = byte1 + 1*8192
    sid2 = byte2 + 2*8192
and ctx is a placeholder token (id 0); this checkpoint was trained use_sideinfo=false,
so the ctx slot carries no feature injection — it is just embed_tokens(0).

Beam search runs 3 autoregressive steps over the byte levels:
    step 0: head_0 at the trailing ctx slot           -> top `bf0` byte0 candidates
    step 1: feed each byte0, head_1                    -> top `bf1` byte1 per beam, keep `beam`
    step 2: feed each byte1(+offset), head_2           -> top `bf2` byte2 per beam, keep `beam`
The score of a triple is the sum of the 3 log-softmax values (a log-probability).

Defaults: branching factor 8/8/8, beam size 64 -> outputs 64 (sa,sb,sc) triples + scores.

Checkpoint layout expected (rank-0 / single-EP, as saved by SimpleCheckpointCallback):
    <ckpt>/_model_rank0.pt      backbone Megatron state_dict (headless)
    <ckpt>/external_rank0.pt    {'lm_heads': {'heads.0.weight',...}, ...}
    <config_json>               base HF config.json of the model architecture

Usage
-----
    python recif_beam_search_standalone.py \
        --ckpt   /mnt/bn/tt-ecom-foundation-nas-1t/checkpoints/pretrain_ar_v2/recif_product_train1k/iter_0001000 \
        --config /mnt/bn/tt-ecom-foundation-nas-1t/models/Qwen3-MoE-tiny-sid-randinit/config.json \
        --history 598080194427,628177754964,755993681678 \
        --bf 8,8,8 --beam 64 --device cuda:0

`--history` is a comma-separated list of the user's historical item SID int64 values
(same packing as training: sid = sa | sb<<14 | sc<<28). If omitted, a demo history is
used. Output: 64 lines `rank  (sa, sb, sc)  sid_int64  log_prob`, plus optional --out json.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------- #
# Megatron -> HF weight mapping (single-rank EP=1; verified against the ckpt).
# ----------------------------------------------------------------------------- #
def _split_qkv(qkv_w, num_heads, num_kv_heads, head_dim, hidden):
    """Megatron interleaved [nkv, (hpg+2)*hd, hidden] -> q, k, v."""
    hpg = num_heads // num_kv_heads
    group = (hpg + 2) * head_dim
    qs = hpg * head_dim
    qkv = qkv_w.reshape(num_kv_heads, group, hidden)
    q = qkv[:, :qs, :].reshape(-1, hidden).contiguous()
    k = qkv[:, qs:qs + head_dim, :].reshape(-1, hidden).contiguous()
    v = qkv[:, qs + head_dim:, :].reshape(-1, hidden).contiguous()
    return q, k, v


def _split_gated(fc1):
    """Megatron GatedMLP fc1 [2*inter, hidden] -> gate, up."""
    inter = fc1.shape[0] // 2
    return fc1[:inter, :].contiguous(), fc1[inter:, :].contiguous()


def megatron_to_hf(sd, arch):
    """Map a (single-rank) Megatron backbone state_dict to HF Qwen3-MoE keys."""
    L = arch['num_hidden_layers']
    nh, nkv, hd, hs = (arch['num_attention_heads'], arch['num_key_value_heads'],
                       arch['head_dim'], arch['hidden_size'])
    ne = arch['num_experts']
    hf = {}
    hf['model.embed_tokens.weight'] = sd['embedding.word_embeddings.weight']
    hf['model.norm.weight'] = sd['decoder.final_layernorm.weight']
    for i in range(L):
        pm, ph = f'decoder.layers.{i}', f'model.layers.{i}'
        q, k, v = _split_qkv(sd[f'{pm}.self_attention.linear_qkv.weight'], nh, nkv, hd, hs)
        hf[f'{ph}.self_attn.q_proj.weight'] = q
        hf[f'{ph}.self_attn.k_proj.weight'] = k
        hf[f'{ph}.self_attn.v_proj.weight'] = v
        hf[f'{ph}.self_attn.o_proj.weight'] = sd[f'{pm}.self_attention.linear_proj.weight']
        hf[f'{ph}.self_attn.q_norm.weight'] = sd[f'{pm}.self_attention.q_layernorm.weight']
        hf[f'{ph}.self_attn.k_norm.weight'] = sd[f'{pm}.self_attention.k_layernorm.weight']
        hf[f'{ph}.input_layernorm.weight'] = sd[f'{pm}.self_attention.linear_qkv.layer_norm_weight']
        hf[f'{ph}.post_attention_layernorm.weight'] = sd[f'{pm}.pre_mlp_layernorm.weight']
        hf[f'{ph}.mlp.gate.weight'] = sd[f'{pm}.mlp.router.weight']
        for e in range(ne):
            g, u = _split_gated(sd[f'{pm}.mlp.experts.linear_fc1.weight{e}'])
            hf[f'{ph}.mlp.experts.{e}.gate_proj.weight'] = g
            hf[f'{ph}.mlp.experts.{e}.up_proj.weight'] = u
            hf[f'{ph}.mlp.experts.{e}.down_proj.weight'] = \
                sd[f'{pm}.mlp.experts.linear_fc2.weight{e}']
    return hf


# ----------------------------------------------------------------------------- #
# Loading
# ----------------------------------------------------------------------------- #
VOCAB = 8192  # per-byte SID vocab


def load_model_and_heads(ckpt_dir, config_json, device, dtype=torch.bfloat16):
    from transformers import Qwen3MoeConfig, Qwen3MoeModel

    base = json.load(open(config_json))
    arch = {k: base[k] for k in (
        'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads',
        'head_dim', 'hidden_size', 'num_experts')}

    # Backbone (headless), vocab forced to the substituted 3*8192 = 24576.
    cfg = Qwen3MoeConfig(**base)
    cfg.vocab_size = 3 * VOCAB
    model = Qwen3MoeModel(cfg).to(dtype)

    raw = torch.load(os.path.join(ckpt_dir, '_model_rank0.pt'),
                     map_location='cpu', weights_only=False)
    sd = {k: v for k, v in raw.items() if not k.endswith('_extra_state')}
    hf_sd = megatron_to_hf(sd, arch)
    # Qwen3MoeModel is the bare backbone (it *is* `model`), so its submodule
    # keys have no `model.` prefix — strip the prefix our HF map produced.
    hf_sd = {(k[len('model.'):] if k.startswith('model.') else k): v
             for k, v in hf_sd.items()}
    missing, unexpected = model.load_state_dict(hf_sd, strict=False)
    # rotary_emb buffers / non-persistent buffers are expected-missing; flag real gaps.
    real_missing = [m for m in missing if 'rotary' not in m and 'inv_freq' not in m]
    if real_missing:
        raise RuntimeError(f"Missing backbone weights: {real_missing[:10]}")
    if unexpected:
        raise RuntimeError(f"Unexpected backbone keys: {unexpected[:10]}")
    model = model.to(device).eval()

    # 3 external SID heads: Linear(hidden, 8192, bias=False) per byte level.
    ext = torch.load(os.path.join(ckpt_dir, 'external_rank0.pt'),
                     map_location='cpu', weights_only=False)
    head_sd = ext['lm_heads']
    H = base['hidden_size']
    heads = torch.nn.ModuleList([torch.nn.Linear(H, VOCAB, bias=False) for _ in range(3)])
    for k in range(3):
        heads[k].weight.data.copy_(head_sd[f'heads.{k}.weight'])
    heads = heads.to(device, dtype=dtype).eval()
    return model, heads


# ----------------------------------------------------------------------------- #
# SID <-> bytes
# ----------------------------------------------------------------------------- #
def sid_to_bytes(v):
    """int64 sid -> (sa, sb, sc), each in [0, 8191]. sid = sa | sb<<14 | sc<<28."""
    v = int(v)
    return v & 0x1FFF, (v >> 14) & 0x1FFF, (v >> 28) & 0x1FFF


def bytes_to_sid(sa, sb, sc):
    return int(sa) | (int(sb) << 14) | (int(sc) << 28)


def build_prefix(history_sids, device):
    """User history (list of int64 SIDs) -> token tensor [1, 4K+1].

    Per item: [ctx=0, sa, sb+8192, sc+16384]. Trailing ctx=0 = target slot.
    """
    toks = []
    for v in history_sids:
        sa, sb, sc = sid_to_bytes(v)
        toks += [0, sa, sb + VOCAB, sc + 2 * VOCAB]
    toks += [0]  # target item's ctx slot
    return torch.tensor([toks], dtype=torch.long, device=device)


# ----------------------------------------------------------------------------- #
# Beam search (no KV cache: re-run the prefix each of the 3 steps — robust &
# cheap for short sequences + small beams; avoids cache-reorder API drift).
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def beam_search(model, heads, prefix, bf=(8, 8, 8), beam=64):
    device = prefix.device

    def hidden_last(seq):
        out = model(input_ids=seq, use_cache=False)
        return out.last_hidden_state[:, -1, :]  # [B, H]

    def logp(h, level):
        lin = heads[level]
        return F.log_softmax(lin(h.to(lin.weight.dtype)).float(), dim=-1)  # [B, 8192]

    # step 0: byte0
    lp0 = logp(hidden_last(prefix), 0)                      # [1, 8192]
    s0, sa = lp0.topk(bf[0], dim=-1)                        # [1, bf0]
    s0, sa = s0.squeeze(0), sa.squeeze(0)                   # [bf0]

    # step 1: byte1 (feed sa at sid0 slot, offset 0)
    seq1 = torch.cat([prefix.expand(bf[0], -1), sa[:, None]], dim=1)  # [bf0, 4K+2]
    lp1 = logp(hidden_last(seq1), 1)                        # [bf0, 8192]
    s1, sb = lp1.topk(bf[1], dim=-1)                        # [bf0, bf1]
    joint1 = s0[:, None] + s1                               # [bf0, bf1]
    flat = joint1.reshape(-1)
    keep = min(beam, flat.numel())
    top_s, top_i = flat.topk(keep)                         # [keep]
    par = top_i // bf[1]
    sa_k = sa[par]                                         # [keep]
    sb_k = sb.reshape(-1)[top_i]                           # [keep]

    # step 2: byte2 (feed sb at sid1 slot, offset +8192)
    seq2 = torch.cat([prefix.expand(keep, -1), sa_k[:, None], (sb_k + VOCAB)[:, None]], dim=1)
    lp2 = logp(hidden_last(seq2), 2)                       # [keep, 8192]
    s2, sc = lp2.topk(bf[2], dim=-1)                       # [keep, bf2]
    joint2 = top_s[:, None] + s2                           # [keep, bf2]
    flat2 = joint2.reshape(-1)
    final = min(beam, flat2.numel())
    fin_s, fin_i = flat2.topk(final)
    par2 = fin_i // bf[2]
    sa_f = sa_k[par2]
    sb_f = sb_k[par2]
    sc_f = sc.reshape(-1)[fin_i]
    return (torch.stack([sa_f, sb_f, sc_f], dim=1).cpu().numpy(),  # [final, 3]
            fin_s.cpu().numpy())                                    # [final]


DEMO_HISTORY = [598080194427, 628177754964, 755993681678, 620899127444, 1644725458491]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='/mnt/bn/tt-ecom-foundation-nas-1t/checkpoints/'
                    'pretrain_ar_v2/recif_product_train1k/iter_0001000')
    ap.add_argument('--config', default='/mnt/bn/tt-ecom-foundation-nas-1t/models/'
                    'Qwen3-MoE-tiny-sid-randinit/config.json')
    ap.add_argument('--history', default='', help='comma-separated int64 SIDs (user history)')
    ap.add_argument('--bf', default='8,8,8', help='branching factors per byte level')
    ap.add_argument('--beam', type=int, default=64)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='', help='optional path to write results as JSON')
    args = ap.parse_args()

    bf = tuple(int(x) for x in args.bf.split(','))
    history = ([int(x) for x in args.history.split(',') if x.strip()]
               if args.history else DEMO_HISTORY)

    # bf16 on GPU; float32 on CPU (bf16 CPU kernels are incomplete).
    dtype = torch.bfloat16 if str(args.device).startswith('cuda') else torch.float32
    model, heads = load_model_and_heads(args.ckpt, args.config, args.device, dtype=dtype)
    prefix = build_prefix(history, args.device)
    print(f"[info] history={len(history)} items, prefix tokens={prefix.shape[1]}, "
          f"bf={bf}, beam={args.beam}, device={args.device}")

    triples, scores = beam_search(model, heads, prefix, bf=bf, beam=args.beam)

    print(f"\nTop-{len(triples)} predicted SID triples (sorted by log-prob):")
    print(f"{'rank':>4}  {'(sa, sb, sc)':>20}  {'sid_int64':>16}  {'log_prob':>10}")
    results = []
    for r, ((sa, sb, sc), sc_score) in enumerate(zip(triples, scores)):
        sid = bytes_to_sid(sa, sb, sc)
        print(f"{r:>4}  {f'({sa}, {sb}, {sc})':>20}  {sid:>16}  {sc_score:>10.4f}")
        results.append({'rank': r, 'sid': [int(sa), int(sb), int(sc)],
                        'sid_int64': sid, 'log_prob': float(sc_score)})
    if args.out:
        json.dump(results, open(args.out, 'w'), indent=2)
        print(f"\n[info] wrote {len(results)} results to {args.out}")


if __name__ == '__main__':
    main()