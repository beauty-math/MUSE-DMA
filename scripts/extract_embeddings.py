from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from muse_dma.data import load_config, read_names


MODELS = {
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "biobert": "dmis-lab/biobert-base-cased-v1.1",
    "biolinkbert": "michiyasunaga/BioLinkBERT-base",
    "biomedlm": "stanford-crfm/BioMedLM",
}


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.unsqueeze(-1).float()
    return (hidden * expanded).sum(1) / expanded.sum(1).clamp(min=1e-9)


def load_model(model_id: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    try:
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        mode = "encoder"
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        mode = "causal_lm"
    model.to(device).eval()
    return tokenizer, model, mode


def encode(names, prefix, tokenizer, model, mode, device, batch_size, max_length):
    prompts = [f"{prefix}: {name}" for name in names]
    blocks = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            tokens = tokenizer(
                prompts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            if mode == "causal_lm":
                output = model(**tokens, output_hidden_states=True, use_cache=False)
                hidden = output.hidden_states[-1]
            else:
                hidden = model(**tokens).last_hidden_state
            pooled = mean_pool(hidden, tokens["attention_mask"])
            blocks.append(F.normalize(pooled, p=2, dim=1).cpu().numpy())
    return np.vstack(blocks).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen biomedical semantic embeddings")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()
    config = load_config(args.config)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    dataset = args.dataset_dir.resolve()
    output = dataset / config["embedding_dir"]
    output.mkdir(parents=True, exist_ok=True)
    drug_names = read_names(dataset / config["drug_names"])
    microbe_names = read_names(dataset / config["microbe_names"])
    if len(drug_names) != config["n_drugs"] or len(microbe_names) != config["n_microbes"]:
        raise ValueError("Entity-name counts do not match the configuration")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for key in args.models:
        tokenizer, model, mode = load_model(MODELS[key], device)
        drug = encode(drug_names, "drug compound", tokenizer, model, mode, device, args.batch_size, args.max_length)
        microbe = encode(microbe_names, "microbial species", tokenizer, model, mode, device, args.batch_size, args.max_length)
        np.save(output / f"{key}_drug.npy", drug)
        np.save(output / f"{key}_microbe.npy", microbe)
        metadata = {
            "model_id": MODELS[key],
            "resolved_commit": getattr(model.config, "_commit_hash", None),
            "mode": mode,
            "pooling": "attention-mask-aware mean pooling followed by L2 normalization",
            "drug_prompt": "drug compound: <name>",
            "microbe_prompt": "microbial species: <name>",
            "max_length": args.max_length,
            "drug_shape": list(drug.shape),
            "microbe_shape": list(microbe.shape),
        }
        (output / f"{key}_meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
