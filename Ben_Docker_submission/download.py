from huggingface_hub import snapshot_download


# Downloads everything under ./roberta-large-local
path = snapshot_download(
    repo_id="roberta-large",
    local_dir="./roberta-large-local",
    local_dir_use_symlinks=False
)

print(f"Model downloaded to: {path}")