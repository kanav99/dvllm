"""
Tokenizer export for Gemma 3.

Gemma 3 ships a HF "fast tokenizer" tokenizer.json (BPE: a vocab dict plus
a merges list, rank = list index) rather than a raw SentencePiece .model
proto, so we parse the JSON directly (stdlib only, no sentencepiece/
tokenizers dependency needed) and emit the exact same tokenizer.bin format
that llama2_tok.py/llama3.2_tok.py already produce: a uint32
max_token_length, then per token (float score, uint32 len, bytes).

The client (gemma3q.cpp) merges by *ascending* score, same convention as
llama3.2_tok.py, since merge rank (lower = merge first) plays the role
tiktoken's rank plays there.
"""
import os
import json
import struct
import argparse

SPIECE_UNDERLINE = "▁"  # sentencepiece's '▁' space marker


def export(tokenizer_json_path, output_path):
    with open(tokenizer_json_path) as f:
        data = json.load(f)

    model = data["model"]
    assert model["type"] == "BPE", f"expected a BPE tokenizer.json, got {model['type']}"

    vocab = model["vocab"]  # piece (str) -> id
    merges = model["merges"]  # list of [a, b] pairs (or "a b" strings), rank = index

    vocab_size = max(vocab.values()) + 1
    id_to_piece = [None] * vocab_size
    for piece, idx in vocab.items():
        id_to_piece[idx] = piece
    assert all(p is not None for p in id_to_piece), "tokenizer.json vocab has gaps in token ids"

    rank_of = {}
    for i, m in enumerate(merges):
        pair = tuple(m) if isinstance(m, list) else tuple(m.split(" ", 1))
        merged = pair[0] + pair[1]
        # a piece can in principle be produced by more than one merge path;
        # keep the lowest (highest-priority) rank, matching greedy BPE
        if merged not in rank_of or i < rank_of[merged]:
            rank_of[merged] = i

    tokens = []
    scores = []
    for piece in id_to_piece:
        t = piece.replace(SPIECE_UNDERLINE, " ")
        b = t.encode("utf-8")
        tokens.append(b)
        # pieces with no known merge rank (single-byte-fallback/added/special
        # tokens) get a very high score so they're never preferred as a merge
        # result over an actual ranked merge, but can still be looked up directly
        scores.append(float(rank_of.get(piece, len(merges) + idx_for(piece, vocab))))

    max_token_length = max(len(t) for t in tokens)

    with open(output_path, "wb") as f:
        f.write(struct.pack("I", max_token_length))
        for b, s in zip(tokens, scores):
            f.write(struct.pack("fI", s, len(b)))
            f.write(b)
    print(f"wrote {output_path} ({vocab_size} tokens)")


def idx_for(piece, vocab):
    # stable, deterministic tiebreak among unranked pieces
    return vocab[piece]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tokenizer-json", type=str, required=True,
                         help="path to tokenizer.json, or a directory containing it")
    parser.add_argument("-o", "--output", type=str, default=None,
                         help="output path for tokenizer.bin (default: alongside the input)")
    args = parser.parse_args()

    src = args.tokenizer_json
    if os.path.isdir(src):
        src = os.path.join(src, "tokenizer.json")

    output_path = args.output or os.path.join(os.path.dirname(os.path.abspath(src)), "tokenizer.bin")
    export(src, output_path)
