# -*- coding: utf-8 -*-
"""main - gpt2_bert_from_hub.py"""

import argparse
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import nltk
import textstat
from nltk.tokenize import word_tokenize
import pickle
import os



# ==========================
# Configuration
# ==========================
OBSERVER_MODEL_HUB_ID = "./gpt2"
PERFORMER_MODEL_HUB_ID = "./gpt2"
BERT_MODEL_HUB_ID = './bert-base-uncased' # BERT will also be loaded from Hub

# Paths to LOCALLY SAVED models (XGB, RF, TFIDF). BERT paths removed.
SAVED_XGB_MODEL_PATH = "./xgb_model.pkl"
SAVED_RF_MODEL_PATH = "./rf_model.pkl"
SAVED_TFIDF_VECTORIZER_PATH = "./tfidf_vectorizer.pkl"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# Helper Functions & Classes (Identical to previous version)
# ==========================
def calculate_perplexity(model, tokenizer, text, device="cuda", median=False):
    text = text.strip();
    if not text: return float('inf')
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length, padding=True).to(device)
    if inputs.input_ids.nelement() == 0 or inputs.input_ids.shape[1] == 0: return float('inf')
    with torch.no_grad(): outputs = model(**inputs); logits = outputs.logits
    if logits.shape[1] <= 1: return float('inf')
    shifted_logits = logits[:, :-1, :]; labels = inputs['input_ids'][:, 1:]; attention_mask = inputs['attention_mask'][:, 1:]
    if labels.shape[1] == 0 or shifted_logits.shape[2] == 0: return float('inf')
    log_probs = F.log_softmax(shifted_logits, dim=-1)
    loss = F.nll_loss(log_probs.transpose(1, 2), labels, reduction='none') * attention_mask
    if median:
        loss_for_median = loss.clone(); loss_for_median[attention_mask == 0] = float('nan')
        if torch.all(torch.isnan(loss_for_median)): return float('inf')
        ppl = np.nanmedian(loss_for_median.cpu().numpy(), axis=1)
    else:
        loss_sum = loss.sum(dim=1); valid_tokens = attention_mask.sum(dim=1); ppl_list = []
        for i in range(valid_tokens.size(0)):
            if valid_tokens[i].item() == 0: ppl_list.append(float('inf'))
            else: ppl_list.append(torch.exp(loss_sum[i] / valid_tokens[i]).cpu().item())
        ppl = np.array(ppl_list)
    return ppl[0] if ppl.shape[0] == 1 else ppl

def calculate_cross_perplexity_with_logits(
    observer_model, performer_model, tokenizer, text, device="cuda", max_length=None
):
    text = text.strip();
    if not text: return float('inf')
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    effective_max_length = max_length if max_length is not None else tokenizer.model_max_length
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=effective_max_length, padding=True).to(device)
    if inputs.input_ids.nelement() == 0 or inputs.input_ids.shape[1] == 0: return float('inf')
    with torch.no_grad(): p_outputs = performer_model(**inputs); q_outputs = observer_model(**inputs)
    p_logits = p_outputs.logits; q_logits = q_outputs.logits
    if p_logits.shape[1] == 0 or q_logits.shape[1] == 0: return float('inf')
    batch_size, seq_len, vocab_size = q_logits.shape
    q_log_probs = F.log_softmax(q_logits.view(-1, vocab_size), dim=-1)
    p_probs = F.softmax(p_logits.view(-1, vocab_size), dim=-1)
    cross_entropy = F.kl_div(q_log_probs, p_probs, reduction='mean', log_target=False)
    return torch.exp(cross_entropy).item()

def calculate_perplexity_ratio(observer_model, performer_model, tokenizer, text, device):
    text = text.strip();
    if not text: return 0.0
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    perplexity_val = calculate_perplexity(observer_model, tokenizer, text, device=device)
    cross_perplexity_val = calculate_cross_perplexity_with_logits(observer_model, performer_model, tokenizer, text, device=device)
    if cross_perplexity_val == 0 or np.isinf(perplexity_val) or np.isinf(cross_perplexity_val) or np.isnan(perplexity_val) or np.isnan(cross_perplexity_val): return 0.0
    ratio = perplexity_val / cross_perplexity_val
    return ratio if np.isfinite(ratio) else 0.0

class BERTEmbedderInfer: # MODIFIED: Now loads from Hugging Face Hub
    def __init__(self, model_hub_id, device=None): # Takes Hub ID
        print(f"Loading BERT tokenizer and model ({model_hub_id}) from Hugging Face Hub...")
        self.tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased", local_files_only=True)
        self.model = BertModel.from_pretrained("./bert-base-uncased", local_files_only=True)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device); self.model.eval()
        print("BERT embedder initialized from Hub.")


    def encode(self, texts, batch_size=1):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded_input = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            with torch.no_grad(): output = self.model(**encoded_input)
            all_embeddings.append(output.last_hidden_state.mean(dim=1).cpu())
        return torch.cat(all_embeddings).numpy()

def feature_extractors_predict(df_single_row, bert_embedder, observer_model, performer_model, observer_tokenizer, tfidf_vectorizer_fitted, device):
    text_list = df_single_row['text'].tolist()
    binocular_feature = np.array(
        [[calculate_perplexity_ratio(observer_model, performer_model, observer_tokenizer, t, device) for t in text_list]]
    ).reshape(-1, 1)
    bert_features = bert_embedder.encode(text_list)
    length_feature = np.array([[len(word_tokenize(t))] for t in text_list])
    readability_feature = np.array([[textstat.flesch_reading_ease(t)] for t in text_list])
    tfidf_matrix = tfidf_vectorizer_fitted.transform(text_list)
    return tfidf_matrix, bert_features, length_feature, readability_feature, binocular_feature

class Predictor:
    def __init__(self, device_to_use):
        self.device = device_to_use
        print(f"Predictor initializing on device: {self.device}")

        required_local_files = [ # Only XGB, RF, TFIDF are now local
            SAVED_XGB_MODEL_PATH, SAVED_RF_MODEL_PATH, SAVED_TFIDF_VECTORIZER_PATH
        ]
        missing_files = [f for f in required_local_files if not os.path.isfile(f)] # Check only files now
        if missing_files:
            print("CRITICAL Error: The following required model files are missing for Predictor initialization:")
            for f_path in missing_files:
                print(f"- /app/{f_path}")
            print("Please ensure these files were copied into the Docker image during build or mounted correctly.")
            raise FileNotFoundError("Essential model components (XGB, RF, TFIDF) missing.")

        print(f"Loading Observer GPT-2 ({OBSERVER_MODEL_HUB_ID}) from Hugging Face Hub...")
        self.observer_tokenizer = AutoTokenizer.from_pretrained("./gpt2", local_files_only=True)
        self.observer_model = AutoModelForCausalLM.from_pretrained("./gpt2", local_files_only=True)
        self.observer_model.to(self.device); self.observer_model.eval()
        if self.observer_tokenizer.pad_token_id is None:
            self.observer_tokenizer.pad_token_id = self.observer_tokenizer.eos_token_id
            self.observer_model.config.pad_token_id = self.observer_tokenizer.pad_token_id

        if OBSERVER_MODEL_HUB_ID == PERFORMER_MODEL_HUB_ID:
            self.performer_tokenizer = self.observer_tokenizer
            self.performer_model = self.observer_model
            print("Performer GPT-2 is same as observer (loaded from Hub).")
        else:
            print(f"Loading Performer GPT-2 ({PERFORMER_MODEL_HUB_ID}) from Hugging Face Hub...")
            self.performer_tokenizer = AutoTokenizer.from_pretrained(PERFORMER_MODEL_HUB_ID,local_files_only = True)
            self.performer_model = AutoModelForCausalLM.from_pretrained(PERFORMER_MODEL_HUB_ID,local_files_only = True)
            self.performer_model.to(self.device); self.performer_model.eval()
            if self.performer_tokenizer.pad_token_id is None:
                self.performer_tokenizer.pad_token_id = self.performer_tokenizer.eos_token_id
                self.performer_model.config.pad_token_id = self.performer_tokenizer.pad_token_id
        
        # MODIFIED: Initialize BERTEmbedderInfer with Hub ID
        self.bert_embedder = BERTEmbedderInfer(model_hub_id=BERT_MODEL_HUB_ID, device=self.device)
        
        print("Loading TF-IDF vectorizer FROM LOCAL FILE...")
        with open(SAVED_TFIDF_VECTORIZER_PATH, 'rb') as f: self.tfidf_vectorizer = pickle.load(f)
        print("Loading XGBoost model FROM LOCAL FILE...")
        with open(SAVED_XGB_MODEL_PATH, 'rb') as f: self.xgb_model = pickle.load(f)
        print("Loading Random Forest model FROM LOCAL FILE...")
        with open(SAVED_RF_MODEL_PATH, 'rb') as f: self.rf_model = pickle.load(f)
        print("All models and components (GPT-2 & BERT from Hub, others local) loaded for inference.")

    def predict_single(self, text_to_predict):
        df_single = pd.DataFrame([{"text": text_to_predict}])
        tfidf_matrix, bert_features, length_feature, readability_feature, binocular_feature = \
            feature_extractors_predict(
                df_single, self.bert_embedder, self.observer_model, self.performer_model,
                self.observer_tokenizer, self.tfidf_vectorizer, self.device
            )
        X_combined = np.hstack([
            binocular_feature, tfidf_matrix.toarray(), bert_features, length_feature, readability_feature
        ])
        xgb_preds_proba = self.xgb_model.predict_proba(X_combined)
        final_prediction = self.rf_model.predict(xgb_preds_proba)
        return final_prediction[0]

# ==========================
# Main Execution Function
# ==========================
def run_predictions(args):
    try:
        predictor = Predictor(device_to_use=DEVICE)
    except FileNotFoundError as e:
        print(f"Failed to initialize Predictor: {e}")
        return

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        print(f"Please ensure this path is correct and accessible inside the container.")
        return

    print(f"Processing input file: {args.input_file}")
    predictions_for_output = []
    try:
        with open(args.input_file, 'r') as fin:
            for line_num, line in enumerate(fin):
                try:
                    entry = json.loads(line)
                    if "text" not in entry or "id" not in entry:
                        print(f"Warning: Skipping line {line_num+1} missing 'text' or 'id': {line.strip()}")
                        continue
                    prediction_label = predictor.predict_single(entry["text"])
                    predictions_for_output.append({"id": entry["id"], "label": int(prediction_label)})
                except json.JSONDecodeError: 
                    print(f"Warning: Skipping line {line_num+1} JSON error: {line.strip()}")
                except Exception as e: 
                    print(f"Warning: Error on line {line_num+1} (id: {entry.get('id', 'N/A')}): {e}. Skipping.")
    except Exception as e:
        print(f"Error reading input file {args.input_file}: {e}")
        return

    output_file_path = os.path.join(args.output_dir, "predictions.jsonl")
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
            print(f"Created output directory: {args.output_dir}")
        except OSError as e:
            print(f"Error creating output directory {args.output_dir}: {e}")
            return
    

    print(f"Writing predictions to: {args.output_file}")
    try:
        with open(output_file_path, 'w') as fout:
            for pred_entry in predictions_for_output:
                fout.write(json.dumps(pred_entry) + "\n")
        print(f"Predictions written successfully.")
    except Exception as e:
        print(f"Error writing output file {args.output_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict human vs AI text (GPT-2 & BERT from Hub, others local).")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path to the input JSONL file.")
    parser.add_argument("--output_dir", type=str, required = True, 
                        help="Path to the output JSONL file.")
    
    args = parser.parse_args()
    
    print(f"Running with arguments: Input='{args.input_file}', Output='{args.output_file}'")
    run_predictions(args)