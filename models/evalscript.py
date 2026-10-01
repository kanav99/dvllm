import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import math
import gc

# -----------------------------------------------------------------------------
# 1. Fixed Point Simulation Layer
# -----------------------------------------------------------------------------
class FixedPointLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, scale: float = 2048.0):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.scale = scale

        # Preserve original weights in float32; set requires_grad=False to prevent graph creation
        self.weight = nn.Parameter(
            torch.round(linear_layer.weight.data.clone().to(torch.float32) * scale),
            requires_grad=False
        )
        if linear_layer.bias is not None:
            self.bias = nn.Parameter(
                linear_layer.bias.data.clone().to(torch.float32),
                requires_grad=False
            )
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # 1. Quantize input and weights (mimicking your integer conversion)
        # Using torch.round to snap to the nearest integer
        x_int = torch.round(x * self.scale)

        # 2. Integer Matmul
        # We perform this in float32 to simulate int32/int64 accumulation capacities
        # without triggering PyTorch's strict int8 overflow limits.
        y_int = torch.matmul(x_int, self.weight.t())

        # 3. Dequantize (mimicking your: float(out) / (FP_SCALE * FP_W_SCALE))
        y = y_int / (self.scale * self.scale)

        # 4. Add floating-point bias if it exists
        if self.bias is not None:
            y = y + self.bias

        return y

@torch.no_grad()
def swap_linear_layers_with_fixed_point(model, scale=2048.0):
    """Recursively replaces all nn.Linear layers with FixedPointLinear"""
    replaced_count = 0
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Exclude the lm_head if you want to keep the final classifier in float
            # if name == "lm_head": continue

            setattr(model, name, FixedPointLinear(module, scale))
            del module  # Remove the original linear layer
            replaced_count += 1
        else:
            # Recurse down the tree
            replaced_count += swap_linear_layers_with_fixed_point(module, scale)
    gc.collect()
    return replaced_count

# -----------------------------------------------------------------------------
# 2. Evaluation (Perplexity & Accuracy)
# -----------------------------------------------------------------------------
def evaluate_model(model, tokenizer, dataset_name="Salesforce/wikitext", dataset_config="wikitext-2-raw-v1", device="cuda"):
    """Calculates perplexity and next-token accuracy."""
    print(f"Loading {dataset_name} dataset for evaluation...")
    test = load_dataset(dataset_name, dataset_config, split="test")

    # Concatenate all text
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    max_length = 2048 #getattr(model.config, "sliding_window", model.config.max_position_embeddings)
    # Cap stride to something reasonable for speed, e.g., 512
    stride = 512
    seq_len = encodings.input_ids.size(1)

    # Restrict to first 25,000 tokens for a quick benchmark (remove min() for full benchmark)
    eval_tokens = seq_len # min(seq_len, 25000)

    nlls = []
    total_correct = 0
    total_active_elements = 0
    prev_end_loc = 0

    model.eval()
    print(f"Evaluating (max length: {max_length})...")
    for begin_loc in tqdm(range(0, eval_tokens, stride)):
        end_loc = min(begin_loc + max_length, eval_tokens)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100 # Ignore loss for context tokens

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # Loss is calculated as cross-entropy, which is negative log likelihood
            neg_log_likelihood = outputs.loss

            # Calculate Accuracy
            logits = outputs.logits
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()

            # Only consider non-ignored indices
            mask = shift_labels != -100
            preds = torch.argmax(shift_logits, dim=-1)

            correct = (preds == shift_labels) & mask
            total_correct += correct.sum().item()
            total_active_elements += mask.sum().item()

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc

        if end_loc == eval_tokens:
            break

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    accuracy = (total_correct / total_active_elements) * 100 if total_active_elements > 0 else 0
    return ppl, accuracy

import sys
# -----------------------------------------------------------------------------
# 3. Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use a small test model like Llama-3.2-1B or a tiny proxy like SmolLM
    model_id = sys.argv[1]

    print(f"Loading original model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(device)

    # 1. Baseline Evaluation
    baseline_ppl, baseline_acc = evaluate_model(model, tokenizer, device=device)
    print(f"--> Baseline: PPL={baseline_ppl:.4f}, Accuracy={baseline_acc:.2f}%\n")

    # 2. Swap to Fixed Point
    scale_factor = 2048.0
    print(f"Swapping linear layers to Fixed Point (scale={scale_factor})...")
    num_replaced = swap_linear_layers_with_fixed_point(model, scale=scale_factor)
    print(f"Replaced {num_replaced} linear layers.")

    # 3. Fixed Point Evaluation
    fixed_ppl, fixed_acc = evaluate_model(model, tokenizer, device=device)
    print(f"--> Fixed Point: PPL={fixed_ppl:.4f}, Accuracy={fixed_acc:.2f}%\n")

    # 4. Report
    print(f"=== SUMMARY ===")
    print(f"PPL Degradation: {((fixed_ppl - baseline_ppl) / baseline_ppl * 100):.2f}%")
    print(f"Acc Change:      {(fixed_acc - baseline_acc):.2f} percentage points")