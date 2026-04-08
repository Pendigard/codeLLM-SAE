import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

model = HookedTransformer.from_pretrained(
    "qwen2.5-7b-instruct",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

prompt = "Write a Python function that computes factorial."
tokens = model.to_tokens(prompt)

logits, cache = model.run_with_cache(tokens)

layer = 15
acts = cache[f"blocks.{layer}.hook_resid_post"]   # [batch, seq, d_model]

x = acts[0]   # [seq, d_model]

