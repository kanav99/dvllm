import argparse
import torch
from download import CheckpointProvider, MetaProvider, HFProvider

def main():
    parser = argparse.ArgumentParser(description="Print the largest elements in each weight vector.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="model checkpoint, .pt file")
    group.add_argument("--meta-llama", type=str, help="meta llama model path")
    group.add_argument("--hf", type=str, help="huggingface model path")
    args = parser.parse_args()

    print("=== Model Weights Maximum Values ===")
    # Reuse the streaming providers from download.py
    if args.checkpoint:
        provider = CheckpointProvider(args.checkpoint)
        print(f"Model: {args.checkpoint}")
    elif args.meta_llama:
        provider = MetaProvider(args.meta_llama)
        print(f"Model: {args.meta_llama}")
    elif args.hf:
        provider = HFProvider(args.hf)
        print(f"Model: {args.hf}")

    p = provider.params

    def get_max(tensor):
        if tensor is None: 
            return "N/A"
        # Extract both raw max and absolute max (useful for integer quantization bounds)
        raw_max = tensor.max().item()
        abs_max = tensor.abs().max().item()
        return f"max: {raw_max:8.4f}  |  abs max: {abs_max:8.4f}"

    print(f"Token Embeddings -> {get_max(provider.get_tok_embeddings())}")

    for i in range(p.n_layers):
        print(f"\n--- Layer {i} ---")
        print(f"  attention_norm : {get_max(provider.get_attention_norm(i))}")
        print(f"  wq             : {get_max(provider.get_wq(i))}")
        print(f"  wk             : {get_max(provider.get_wk(i))}")
        print(f"  wv             : {get_max(provider.get_wv(i))}")
        print(f"  wo             : {get_max(provider.get_wo(i))}")
        print(f"  ffn_norm       : {get_max(provider.get_ffn_norm(i))}")
        print(f"  w1             : {get_max(provider.get_w1(i))}")
        print(f"  w2             : {get_max(provider.get_w2(i))}")
        print(f"  w3             : {get_max(provider.get_w3(i))}")

    print(f"\nFinal Norm       -> {get_max(provider.get_norm())}")
    
    if not provider.shared_classifier:
        print(f"Output wcls      -> {get_max(provider.get_output())}")
    else:
        print("Output wcls      -> [Shared with Embeddings]")

if __name__ == "__main__":
    main()