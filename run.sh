python -m src.extract_sae_token_features \
  --snippets-dir /Vrac/renton/code_snippets \
  --output-path outputs/code_sae_token_features.pkl \
  --model-name google/gemma-2-2b \
  --sae-release gemma-scope-2b-pt-res-canonical \
  --sae-id layer_20/width_16k/canonical \
  --hook-name blocks.20.hook_resid_post \
  --top-k 100 \
  --max-length 256 \
  --device cuda

python -m src.extract_sae_text_token_features \
  --output-path text_sae_token_features.pkl \
  --model-name google/gemma-2-2b \
  --sae-release gemma-scope-2b-pt-res-canonical \
  --sae-id layer_20/width_16k/canonical \
  --hook-name blocks.20.hook_resid_post \
  --top-k 100 \
  --max-length 256 \
  --device cuda \
  --target-rows-like outputs/code_sae_token_features.pkl