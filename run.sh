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

# Build the token annotation dataset, aligning LLM tokens with code tokens and annotating them with various metadata

python -m build_token_annotation_dataset \
  --snippets-dir /Vrac/renton/code_snippets \
  --tokenizer-name google/gemma-2-2b \
  --output-path outputs/code_token_annotations_3.parquet

# Extract the activations for the LLM tokens

python -m extract_llm_token_activations \
  --snippets-dir /Vrac/renton/code_snippets \
  --model-name google/gemma-2-2b \
  --layer 20 \
  --activation-kind resid_post \
  --output-path outputs/layers/code_token_activations_layer_20.parquet \
  --device cuda

# Extract the SAE features for the LLM tokens (joinable with the token annotations))

python -m extract_joinable_sae_token_features \
  --activations-path outputs/layers/code_token_activations_layer_20.parquet \
  --sae-release gemma-scope-2b-pt-res-canonical \
  --sae-id layer_20/width_16k/canonical \
  --output-path outputs/SAE/code_sae_token_features_layer_20.parquet \
  --top-k 100

