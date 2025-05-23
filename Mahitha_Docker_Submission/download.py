import nltk
nltk.download("punkt", download_dir="./nltk_data")
nltk.download("averaged_perceptron_tagger", download_dir="./nltk_data")
nltk.download("stopwords", download_dir="./nltk_data")
nltk.download("sentiwordnet", download_dir="./nltk_data")
nltk.download("wordnet", download_dir="./nltk_data")

from huggingface_hub import snapshot_download

snapshot_download("gpt2", local_dir="./gpt2", local_dir_use_symlinks=False)
snapshot_download("roberta-base", local_dir="./roberta-base", local_dir_use_symlinks=False)
snapshot_download("facebook/bart-base", local_dir="./bart-base", local_dir_use_symlinks=False)
