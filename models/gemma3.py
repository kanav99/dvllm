"""
Model export for Gemma 3 (google/gemma-3-*-it, google/gemma-3-*-pt).

Writes model.bin in the same "legacy" raw-header, combined-per-layer-block
layout that common.hpp's read_checkpoint()/memory_map_weights() already
know how to parse for Llama, just with the extra fields/arrays Gemma 3
needs (explicit head_dim, QK-norm, sandwich norm, dual RoPE theta,
sliding window). See include/common_gemma3.hpp for the reader side.
"""
import os
import struct
import argparse

import torch


def serialize_fp32(file, tensor):
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    file.write(struct.pack(f"{len(d)}f", *d))


def permute_rope_2d(w, n_heads, head_dim):
    """
    HF applies RoPE with the "rotate_half" convention (first/second half of
    head_dim rotated as a pair), while the C++ forward pass here (like
    llama2.c) uses the "interleaved pairs" convention ((0,1), (2,3), ...).
    Permute q_proj/k_proj's output rows, per head, from the former to the
    latter so the existing interleaved-pair RoPE code is correct unchanged.
    """
    in_dim = w.shape[1]
    return w.view(n_heads, 2, head_dim // 2, in_dim).transpose(1, 2).reshape(n_heads * head_dim, in_dim)


def permute_rope_1d(w, head_dim):
    """Same permutation as permute_rope_2d, for the 1-D q_norm/k_norm weight."""
    return w.view(2, head_dim // 2).transpose(0, 1).reshape(head_dim)


# Gemma 3's tokenizer (vocab + merges) is byte-identical across every
# released size (270M/1B/4B/12B/27B) and has exactly this many real pieces.
# The 4B/12B/27B *models* report a larger config.vocab_size (e.g. 262208)
# because their embedding table is padded with a handful of extra rows for
# multimodal-only pseudo-tokens (e.g. the image placeholder, id 262144) that
# a text tokenizer never produces -- so we always export exactly this many
# rows/logits, matching what tokenizer.bin actually contains.
TEXT_VOCAB_SIZE = 262144


def export_model(model_id, output_path):
    from transformers import AutoModelForCausalLM

    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")
    full_config = model.config
    state_dict = model.state_dict()

    # 4B/12B/27B are multimodal (Gemma3ForConditionalGeneration): the text
    # backbone's hyperparameters live under `config.text_config`, and its
    # weights are prefixed `model.language_model.*` instead of `model.*`.
    # 270M/1B are plain Gemma3ForCausalLM and use the flat config/prefix.
    config = getattr(full_config, "text_config", full_config)
    prefix = "model.language_model." if f"model.language_model.embed_tokens.weight" in state_dict else "model."
    print(f"Detected weight key prefix: '{prefix}' (config: {type(config).__name__})")

    dim = config.hidden_size
    hidden_dim = config.intermediate_size
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    vocab_size = min(config.vocab_size, TEXT_VOCAB_SIZE)
    seq_len = config.max_position_embeddings
    head_dim = config.head_dim
    sliding_window = config.sliding_window
    sliding_window_pattern = getattr(config, "sliding_window_pattern", None) \
        or getattr(config, "_sliding_window_pattern", None) or 6

    query_pre_attn_scalar = float(config.query_pre_attn_scalar)
    # rope theta/scaling moved from flat `rope_theta`/`rope_local_base_freq`/
    # `rope_scaling` attributes into `config.rope_parameters[layer_type]` in
    # newer transformers; support both so this doesn't silently break on a
    # transformers upgrade/downgrade. Linear RoPE scaling (used by 4B/12B/27B
    # to extend context length) only ever applies to the global/full-attention
    # layers -- local/sliding-window layers keep factor=1.0.
    rope_params = getattr(config, "rope_parameters", None)
    if rope_params is not None:
        rope_theta_global = float(rope_params["full_attention"]["rope_theta"])
        rope_theta_local = float(rope_params["sliding_attention"]["rope_theta"])
        rope_scaling_factor = float(rope_params["full_attention"].get("factor", 1.0))
    else:
        rope_theta_global = float(config.rope_theta)
        rope_theta_local = float(getattr(config, "rope_local_base_freq", 10000.0))
        rope_scaling_factor = float((getattr(config, "rope_scaling", None) or {}).get("factor", 1.0))

    lm_head_key = f"{prefix}lm_head.weight" if f"{prefix}lm_head.weight" in state_dict else "lm_head.weight"
    embed_key = f"{prefix}embed_tokens.weight"
    assert lm_head_key not in state_dict or torch.equal(state_dict[lm_head_key], state_dict[embed_key]), \
        "expected tied embeddings for Gemma 3 -- this checkpoint has a distinct lm_head.weight"
    assert getattr(config, "attn_logit_softcapping", None) in (None, 0.0), "attn logit softcapping not supported"
    assert getattr(config, "final_logit_softcapping", None) in (None, 0.0), "final logit softcapping not supported"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_file = open(output_path, "wb")

    # header: 10 int64 + 4 double, then int32 shared_weights (always 1: tied embeddings)
    header = struct.pack(
        "10q4d",
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len,
        head_dim, sliding_window, sliding_window_pattern,
        query_pre_attn_scalar, rope_theta_local, rope_theta_global, rope_scaling_factor,
    )
    out_file.write(header)
    out_file.write(struct.pack("i", 1))  # shared_weights

    def norm(name):
        # Gemma's RMSNorm scales by (1 + weight); baking the +1 into the
        # exported weight lets us reuse the existing unmodified rmsnorm().
        return state_dict[name] + 1.0

    def layer(i, suffix):
        return state_dict[f"{prefix}layers.{i}.{suffix}"]

    print("Writing token embeddings...")
    serialize_fp32(out_file, state_dict[embed_key][:vocab_size])

    for i in range(n_layers):
        serialize_fp32(out_file, norm(f"{prefix}layers.{i}.input_layernorm.weight"))
    for i in range(n_layers):
        serialize_fp32(out_file, norm(f"{prefix}layers.{i}.post_attention_layernorm.weight"))
    for i in range(n_layers):
        serialize_fp32(out_file, norm(f"{prefix}layers.{i}.pre_feedforward_layernorm.weight"))
    for i in range(n_layers):
        serialize_fp32(out_file, norm(f"{prefix}layers.{i}.post_feedforward_layernorm.weight"))
    for i in range(n_layers):
        serialize_fp32(out_file, permute_rope_1d(norm(f"{prefix}layers.{i}.self_attn.q_norm.weight"), head_dim))
    for i in range(n_layers):
        serialize_fp32(out_file, permute_rope_1d(norm(f"{prefix}layers.{i}.self_attn.k_norm.weight"), head_dim))

    print("Writing attention weights...")
    for i in range(n_layers):
        wqkv = torch.cat([
            permute_rope_2d(layer(i, "self_attn.q_proj.weight"), n_heads, head_dim),
            permute_rope_2d(layer(i, "self_attn.k_proj.weight"), n_kv_heads, head_dim),
            layer(i, "self_attn.v_proj.weight"),
        ], dim=0)
        serialize_fp32(out_file, wqkv)
    for i in range(n_layers):
        serialize_fp32(out_file, layer(i, "self_attn.o_proj.weight"))

    print("Writing MLP weights...")
    for i in range(n_layers):
        w1w3 = torch.cat([
            layer(i, "mlp.gate_proj.weight"),
            layer(i, "mlp.up_proj.weight"),
        ], dim=0)
        serialize_fp32(out_file, w1w3)
    for i in range(n_layers):
        serialize_fp32(out_file, layer(i, "mlp.down_proj.weight"))

    serialize_fp32(out_file, norm(f"{prefix}norm.weight"))

    out_file.close()
    print(f"wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="google/gemma-3-1b-it", help="huggingface model id or path")
    args = parser.parse_args()

    output_path = os.path.expanduser(f"~/.dvllm/{args.model}/model.bin")
    export_model(args.model, output_path)
