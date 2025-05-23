from huggingface_hub import snapshot_download
#snapshot_download("gpt2", local_dir="./gpt2", local_dir_use_symlinks=False)
#snapshot_download("bert-base-uncased", local_dir="./bert-base-uncased", local_dir_use_symlinks=False)


import nltk
nltk.download("punkt", download_dir="./nltk_data")
nltk.download("punkt_tab", download_dir="./nltk_data")
