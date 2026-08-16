"""Resume the ALLaM-7B weight download. snapshot_download picks up the partial
.incomplete shards automatically and only fetches the model files we need."""
from huggingface_hub import snapshot_download

path = snapshot_download(
    "ALLaM-AI/ALLaM-7B-Instruct-preview",
    allow_patterns=[
        "model-*.safetensors",
        "model.safetensors.index.json",
        "config.json",
        "generation_config.json",
        "tokenizer*",
        "special_tokens_map.json",
    ],
    max_workers=4,
)
print(f"[ok] ALLaM snapshot complete: {path}")
