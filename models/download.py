"""
This script has functions and utilties for model export.
Basically, we have a bunch of versions of the model, and we
want to export them to .bin files to be read from and inferenced in C.

Among the "input" versions of PyTorch files/models:
- Official Llama 2 weights released by Meta
- Huggingface weights available on the hub
- llama2.c (this repo) trained models

Among the "output" versions of .bin files:
- v0: Legacy files of the original llama2.c repo (will eventually be DEPRECATED)
- v1-vN: Improved .bin files with a proper header, cache alignment, etc.

This script aspires to provide all of these conversions.
"""

import argparse
import gzip
import json
import os
import shutil
import struct
import gc
from pathlib import Path

import numpy as np
import torch

from model import ModelArgs, Transformer, precompute_freqs_cis

# -----------------------------------------------------------------------------
# common utilities


def serialize_fp32(file, tensor):
    """writes one fp32 tensor to file that is open in wb mode"""
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f"{len(d)}f", *d)
    file.write(b)


def serialize_int8(file, tensor):
    """writes one int8 tensor to file that is open in wb mode"""
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    b = struct.pack(f"{len(d)}b", *d)
    file.write(b)


def quantize_q80(w, group_size):
    """
    takes a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
    assert w.numel() % group_size == 0
    w = w.float()  # convert to float32
    w = w.reshape(-1, group_size)
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 127.0
    # scale into range [-127, 127]
    quant = w / scale[:, None]
    # round to nearest integer
    int8val = torch.round(quant).to(torch.int8)
    # dequantize by rescaling
    fp32val = (int8val.float() * scale[:, None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    # calculate the max error in each group
    err = torch.abs(fp32valr - w).max(dim=1).values
    # find the max error across all groups
    maxerr = err.max().item()
    return int8val, scale, maxerr


def get_hidden_dim(p):
    """Computes hidden_dim if not explicitly set in params."""
    if hasattr(p, "hidden_dim") and p.hidden_dim is not None:
        return p.hidden_dim
    hidden_dim = 4 * p.dim
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = p.multiple_of * ((hidden_dim + p.multiple_of - 1) // p.multiple_of)
    return hidden_dim


# -----------------------------------------------------------------------------
# legacy


def legacy_export(provider, filepath):
    """Original export of llama2.c bin files, i.e. version v0"""
    out_file = open(filepath, "wb")
    p = provider.params
    hidden_dim = get_hidden_dim(p)
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads

    vocab_size = p.vocab_size if provider.shared_classifier else -p.vocab_size

    header = struct.pack(
        "qqqqqqq",
        p.dim,
        hidden_dim,
        p.n_layers,
        p.n_heads,
        n_kv_heads,
        vocab_size,
        p.max_seq_len,
    )
    out_file.write(header)

    serialize_fp32(out_file, provider.get_tok_embeddings())

    for layer in range(p.n_layers): serialize_fp32(out_file, provider.get_attention_norm(layer))
    # for layer in range(p.n_layers): serialize_fp32(out_file, provider.get_wq(layer))
    # for layer in range(p.n_layers): serialize_fp32(out_file, provider.get_wk(layer))
    # for layer in range(p.n_layers): serialize_fp32(out_file, provider.get_wv(layer))
    for layer in range(p.n_layers):
        serialize_fp32(out_file, provider.get_wq(layer))
        serialize_fp32(out_file, provider.get_wk(layer))
        serialize_fp32(out_file, provider.get_wv(layer))
    for layer in range(p.n_layers): serialize_fp32(out_file, provider.get_wo(layer))

    for layer in range(p.n_layers): serialize_fp32(out_file, provider.get_ffn_norm(layer))
    # for layer in range(p.n_layers): serialize_fp32(out_file, provider.get_w1(layer))
    # for layer in range(p.n_layers): serialize_fp32(out_file, provider.get_w2(layer))
    # for layer in range(p.n_layers): serialize_fp32(out_file, provider.get_w3(layer))
    for layer in range(p.n_layers):
        serialize_fp32(out_file, provider.get_w1(layer))
        serialize_fp32(out_file, provider.get_w3(layer))
    
    # w2 can stay in its own loop or go after
    for layer in range(p.n_layers): serialize_fp32(out_file, provider.get_w2(layer))

    serialize_fp32(out_file, provider.get_norm())

    freqs_cos, freqs_sin = precompute_freqs_cis(p.dim // p.n_heads, p.max_seq_len)
    serialize_fp32(out_file, freqs_cos[: p.max_seq_len])
    serialize_fp32(out_file, freqs_sin[: p.max_seq_len])

    if not provider.shared_classifier:
        serialize_fp32(out_file, provider.get_output())

    out_file.close()
    print(f"wrote {filepath}")


# -----------------------------------------------------------------------------
# new version


def version1_export(provider, filepath):
    """
    Export the model weights in full float32 .bin file to be read from C.
    This is same as legacy_export, but with a proper header.
    """
    version = 1
    out_file = open(filepath, "wb")
    
    out_file.write(struct.pack("I", 0x616B3432)) # magic
    out_file.write(struct.pack("i", version))

    p = provider.params
    hidden_dim = get_hidden_dim(p)
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    
    header = struct.pack(
        "qqqqqqq",
        p.dim,
        hidden_dim,
        p.n_layers,
        p.n_heads,
        n_kv_heads,
        p.vocab_size,
        p.max_seq_len,
    )
    out_file.write(header)

    out_file.write(struct.pack("B", int(provider.shared_classifier)))
    pad = 256 - out_file.tell() 
    assert pad >= 0
    out_file.write(b"\0" * pad)

    for layer in range(p.n_layers): serialize_fp32(out_file, provider.get_attention_norm(layer))
    for layer in range(p.n_layers): serialize_fp32(out_file, provider.get_ffn_norm(layer))
    
    serialize_fp32(out_file, provider.get_norm())
    serialize_fp32(out_file, provider.get_tok_embeddings())
    
    for layer in range(p.n_layers): serialize_fp32(out_file, provider.get_wq(layer))
    for layer in range(p.n_layers): serialize_fp32(out_file, provider.get_wk(layer))
    for layer in range(p.n_layers): serialize_fp32(out_file, provider.get_wv(layer))
    for layer in range(p.n_layers): serialize_fp32(out_file, provider.get_wo(layer))
    for layer in range(p.n_layers): serialize_fp32(out_file, provider.get_w1(layer))
    for layer in range(p.n_layers): serialize_fp32(out_file, provider.get_w2(layer))
    for layer in range(p.n_layers): serialize_fp32(out_file, provider.get_w3(layer))
    
    if not provider.shared_classifier:
        serialize_fp32(out_file, provider.get_output())

    out_file.close()
    print(f"wrote {filepath}")


def version2_export(provider, filepath, group_size=64):
    """
    Export the model weights in Q8_0 into .bin file to be read from C.
    """
    version = 2
    p = provider.params

    while p.dim % group_size != 0:
        group_size //= 2
        print(f"BACKOFF: reducing group size to {group_size} to fit hidden_dim")

    out_file = open(filepath, "wb")
    out_file.write(struct.pack("I", 0x616B3432))
    out_file.write(struct.pack("i", version))

    hidden_dim = get_hidden_dim(p)
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack(
        "qqqqqqq",
        p.dim,
        hidden_dim,
        p.n_layers,
        p.n_heads,
        n_kv_heads,
        p.vocab_size,
        p.max_seq_len,
    )
    out_file.write(header)
    out_file.write(struct.pack("B", int(provider.shared_classifier)))
    out_file.write(struct.pack("i", group_size))  
    pad = 256 - out_file.tell()
    assert pad >= 0
    out_file.write(b"\0" * pad)

    # Norms remain in fp32
    for layer in range(p.n_layers): serialize_fp32(out_file, provider.get_attention_norm(layer))
    for layer in range(p.n_layers): serialize_fp32(out_file, provider.get_ffn_norm(layer))
    serialize_fp32(out_file, provider.get_norm())

    max_err = 0.0
    total_weights = 1 + 7 * p.n_layers + (0 if provider.shared_classifier else 1)
    current_weight = 1

    def _process_and_write(w):
        nonlocal max_err, current_weight
        q, s, err = quantize_q80(w, group_size)
        serialize_int8(out_file, q)
        serialize_fp32(out_file, s)
        if err > max_err:
            max_err = err
        print(f"{current_weight}/{total_weights} quantized to Q8_0 with max error {err}")
        current_weight += 1

    # Sequentially fetch, quantize, write, and drop
    _process_and_write(provider.get_tok_embeddings())
    for layer in range(p.n_layers): _process_and_write(provider.get_wq(layer))
    for layer in range(p.n_layers): _process_and_write(provider.get_wk(layer))
    for layer in range(p.n_layers): _process_and_write(provider.get_wv(layer))
    for layer in range(p.n_layers): _process_and_write(provider.get_wo(layer))
    for layer in range(p.n_layers): _process_and_write(provider.get_w1(layer))
    for layer in range(p.n_layers): _process_and_write(provider.get_w2(layer))
    for layer in range(p.n_layers): _process_and_write(provider.get_w3(layer))
    
    if not provider.shared_classifier:
        _process_and_write(provider.get_output())

    print(f"max quantization group error across all weights: {max_err}")
    out_file.close()
    print(f"wrote {filepath}")


def hf_export(provider, filepath, dtype=torch.float32):
    """Generate the pytorch_model.bin state_dict and config.json for HuggingFace"""
    try:
        from transformers.models.llama.configuration_llama import LlamaConfig
    except ImportError:
        print("Error: transformers package is required to load huggingface models")
        return None

    p = provider.params
    hf_state_dict = {}

    def permute_original(w, n_heads, dim1, dim2):
        return (
            w.view(dim1, dim2)
            .reshape(n_heads, dim1 // n_heads // 2, 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
        )

    hf_state_dict["model.embed_tokens.weight"] = provider.get_tok_embeddings().to(dtype)
    hf_state_dict["model.norm.weight"] = provider.get_norm().to(dtype)

    dim = p.dim
    num_key_value_heads = p.n_kv_heads if p.n_kv_heads is not None else p.n_heads
    n_rep = p.n_heads // num_key_value_heads
    key_value_dim = dim // n_rep

    for i in range(p.n_layers):
        hf_state_dict[f"model.layers.{i}.input_layernorm.weight"] = provider.get_attention_norm(i).to(dtype)
        hf_state_dict[f"model.layers.{i}.self_attn.q_proj.weight"] = permute_original(provider.get_wq(i), p.n_heads, dim, dim).to(dtype)
        hf_state_dict[f"model.layers.{i}.self_attn.k_proj.weight"] = permute_original(provider.get_wk(i), num_key_value_heads, key_value_dim, dim).to(dtype)
        hf_state_dict[f"model.layers.{i}.self_attn.v_proj.weight"] = provider.get_wv(i).to(dtype)
        hf_state_dict[f"model.layers.{i}.self_attn.o_proj.weight"] = provider.get_wo(i).to(dtype)
        hf_state_dict[f"model.layers.{i}.post_attention_layernorm.weight"] = provider.get_ffn_norm(i).to(dtype)
        hf_state_dict[f"model.layers.{i}.mlp.gate_proj.weight"] = provider.get_w1(i).to(dtype)
        hf_state_dict[f"model.layers.{i}.mlp.down_proj.weight"] = provider.get_w2(i).to(dtype)
        hf_state_dict[f"model.layers.{i}.mlp.up_proj.weight"] = provider.get_w3(i).to(dtype)

    if provider.shared_classifier:
        hf_state_dict["lm_head.weight"] = hf_state_dict["model.embed_tokens.weight"]
    else:
        hf_state_dict["lm_head.weight"] = provider.get_output().to(dtype)

    config = LlamaConfig(
        vocab_size=p.vocab_size,
        hidden_size=p.dim,
        intermediate_size=get_hidden_dim(p),
        num_hidden_layers=p.n_layers,
        num_attention_heads=p.n_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=p.max_seq_len,
        rms_norm_eps=p.norm_eps,
        tie_word_embeddings=provider.shared_classifier,
        architectures=["LlamaForCausalLM"],
        hidden_act="silu",
    )

    os.makedirs(filepath, exist_ok=True)
    torch.save(hf_state_dict, os.path.join(filepath, "pytorch_model.bin"))
    config.save_pretrained(filepath)


# -----------------------------------------------------------------------------
# Streaming Model Providers (Fixes the OOM Issue)


class CheckpointProvider:
    def __init__(self, checkpoint_path):
        # Using mmap avoids loading all tensors to RAM. They remain on disk until explicitly sliced.
        checkpoint_dict = torch.load(checkpoint_path, map_location="cpu", mmap=True)
        self.params = ModelArgs(**checkpoint_dict["model_args"])
        self.state_dict = checkpoint_dict["model"]
        
        unwanted_prefix = "_orig_mod."
        for k, v in list(self.state_dict.items()):
            if k.startswith(unwanted_prefix):
                self.state_dict[k[len(unwanted_prefix) :]] = self.state_dict.pop(k)
                
        self.shared_classifier = torch.equal(
            self.state_dict["tok_embeddings.weight"], 
            self.state_dict.get("output.weight", self.state_dict["tok_embeddings.weight"])
        )

    def pop(self, key):
        # Remove reference from dictionary immediately after fetch to GC it
        return self.state_dict.pop(key)

    def get_tok_embeddings(self): return self.pop("tok_embeddings.weight")
    def get_attention_norm(self, i): return self.pop(f"layers.{i}.attention_norm.weight")
    def get_wq(self, i): return self.pop(f"layers.{i}.attention.wq.weight")
    def get_wk(self, i): return self.pop(f"layers.{i}.attention.wk.weight")
    def get_wv(self, i): return self.pop(f"layers.{i}.attention.wv.weight")
    def get_wo(self, i): return self.pop(f"layers.{i}.attention.wo.weight")
    def get_ffn_norm(self, i): return self.pop(f"layers.{i}.ffn_norm.weight")
    def get_w1(self, i): return self.pop(f"layers.{i}.feed_forward.w1.weight")
    def get_w2(self, i): return self.pop(f"layers.{i}.feed_forward.w2.weight")
    def get_w3(self, i): return self.pop(f"layers.{i}.feed_forward.w3.weight")
    def get_norm(self): return self.pop("norm.weight")
    def get_output(self): return self.state_dict.pop("output.weight", None)


class MetaProvider:
    def __init__(self, model_path):
        params_path = os.path.join(model_path, "params.json")
        with open(params_path) as f:
            params = json.load(f)

        self.models = [torch.load(p, map_location="cpu", mmap=True) for p in sorted(list(Path(model_path).glob("consolidated.*.pth")))]

        self.params = ModelArgs()
        self.params.dim = params["dim"]
        self.params.n_layers = params["n_layers"]
        self.params.n_heads = params["n_heads"]
        self.params.n_kv_heads = params.get("n_kv_heads") or params["n_heads"]
        self.params.multiple_of = params["multiple_of"]
        self.params.norm_eps = params["norm_eps"]
        self.params.max_seq_len = 2048
        self.shared_classifier = False
        
        tok_shape = self.get_tok_embeddings().shape
        self.params.vocab_size = tok_shape[0]

    def get_tensor(self, name):
        tensors = [model.get(name) for model in self.models]
        tensors = [t for t in tensors if t is not None]
        if len(tensors) == 0:
            return None
        if len(tensors) == 1 or len(tensors[0].shape) == 1:
            return tensors[0]
            
        is_axis_1 = (
            name.startswith("tok_embeddings.")
            or name.endswith(".attention.wo.weight")
            or name.endswith(".feed_forward.w2.weight")
        )
        return torch.cat(tensors, dim=1 if is_axis_1 else 0)

    def get_tok_embeddings(self): return self.get_tensor("tok_embeddings.weight")
    def get_attention_norm(self, i): return self.get_tensor(f"layers.{i}.attention_norm.weight")
    def get_wq(self, i): return self.get_tensor(f"layers.{i}.attention.wq.weight")
    def get_wk(self, i): return self.get_tensor(f"layers.{i}.attention.wk.weight")
    def get_wv(self, i): return self.get_tensor(f"layers.{i}.attention.wv.weight")
    def get_wo(self, i): return self.get_tensor(f"layers.{i}.attention.wo.weight")
    def get_ffn_norm(self, i): return self.get_tensor(f"layers.{i}.ffn_norm.weight")
    def get_w1(self, i): return self.get_tensor(f"layers.{i}.feed_forward.w1.weight")
    def get_w2(self, i): return self.get_tensor(f"layers.{i}.feed_forward.w2.weight")
    def get_w3(self, i): return self.get_tensor(f"layers.{i}.feed_forward.w3.weight")
    def get_norm(self): return self.get_tensor("norm.weight")
    def get_output(self): return self.get_tensor("output.weight")


class HFProvider:
    def __init__(self, model_path):
        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            pass

        # Load weights lazily and immediately dereference the original model structure.
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True)
        self.hf_dict = model.state_dict()
        
        config = model.config
        self.params = ModelArgs()
        self.params.dim = config.hidden_size
        self.params.n_layers = config.num_hidden_layers
        self.params.n_heads = config.num_attention_heads
        self.params.n_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        self.params.vocab_size = config.vocab_size
        self.params.hidden_dim = config.intermediate_size
        self.params.norm_eps = config.rms_norm_eps
        self.params.max_seq_len = getattr(config, "max_position_embeddings", 2048)

        self.shared_classifier = torch.equal(
            self.hf_dict["model.embed_tokens.weight"], 
            self.hf_dict.get("lm_head.weight", self.hf_dict["model.embed_tokens.weight"])
        )
        
        del model
        gc.collect()

    def pop(self, key):
        return self.hf_dict.pop(key)

    def _permute_reverse(self, w):
        n_heads = self.params.n_heads
        dim1 = self.params.dim
        dim2 = self.params.dim
        return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    def _permute_reverse_kv(self, w):
        n_kv_heads = self.params.n_kv_heads
        dim1 = w.shape[0] 
        dim2 = w.shape[1] 
        return w.view(n_kv_heads, 2, dim1 // n_kv_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    def get_tok_embeddings(self): return self.pop("model.embed_tokens.weight")
    def get_attention_norm(self, i): return self.pop(f"model.layers.{i}.input_layernorm.weight")
    def get_wq(self, i): return self._permute_reverse(self.pop(f"model.layers.{i}.self_attn.q_proj.weight"))
    def get_wk(self, i): return self._permute_reverse_kv(self.pop(f"model.layers.{i}.self_attn.k_proj.weight"))
    def get_wv(self, i): return self.pop(f"model.layers.{i}.self_attn.v_proj.weight")
    def get_wo(self, i): return self.pop(f"model.layers.{i}.self_attn.o_proj.weight")
    def get_ffn_norm(self, i): return self.pop(f"model.layers.{i}.post_attention_layernorm.weight")
    def get_w1(self, i): return self.pop(f"model.layers.{i}.mlp.gate_proj.weight")
    def get_w2(self, i): return self.pop(f"model.layers.{i}.mlp.down_proj.weight")
    def get_w3(self, i): return self.pop(f"model.layers.{i}.mlp.up_proj.weight")
    def get_norm(self): return self.pop("model.norm.weight")
    def get_output(self): return self.hf_dict.pop("lm_head.weight", None)


# -----------------------------------------------------------------------------
# API entrypoint

def model_export(provider, filepath, version, dtype=torch.float32):
    """
    Routes to correct export protocol.
    """
    if version == 0:
        legacy_export(provider, filepath)
    elif version == 1:
        version1_export(provider, filepath)
    elif version == 2:
        version2_export(provider, filepath)
    elif version == -1:
        hf_export(provider, filepath, dtype)
    else:
        raise ValueError(f"unknown version {version}")


def torchscript_export(model, filepath, zero_params=False, gzip_output=False):
    """
    Saves the model as a TorchScript.
    (Requires full Model instantiation. Not optimized via streaming providers).
    """
    if zero_params:
        for p in model.parameters():
            p.detach().zero_()

    torch.jit.save(torch.jit.script(model), filepath)

    if gzip_output:
        with open(filepath, "rb") as f_in:
            with gzip.open(f"{filepath}.gz", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.unlink(filepath)


# -----------------------------------------------------------------------------
# CLI entrypoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("filepath", type=str, help="the output filepath")
    parser.add_argument(
        "--version", default=0, type=int, help="the version to export with"
    )
    parser.add_argument(
        "--dtype", type=str, help="dtype of the model (fp16, fp32)", default="fp32"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="model checkpoint, .pt file")
    group.add_argument("--meta-llama", type=str, help="meta llama model path")
    group.add_argument("--hf", type=str, help="huggingface model path")
    args = parser.parse_args()
    
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    if args.checkpoint or args.meta_llama:
        print("Error: For now, only Hugging Face models are supported")
        exit(1)
    
    # create "$HOME/.dvllm/{args.hf}" if it doesn't exist
    if not os.path.exists(os.path.expanduser(f"~/.dvllm/{args.hf}")):
        os.makedirs(os.path.expanduser(f"~/.dvllm/{args.hf}"))
    output_path = os.path.expanduser(f"~/.dvllm/{args.hf}/model.bin")

    provider = None
    if args.checkpoint:
        provider = CheckpointProvider(args.checkpoint)
    elif args.meta_llama:
        provider = MetaProvider(args.meta_llama)
    elif args.hf:
        provider = HFProvider(args.hf)

    if provider is None:
        parser.error("Can't load input model!")

    model_export(provider, output_path, args.version, dtype)