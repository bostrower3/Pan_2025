import pandas as pd
import numpy as np
import random
import tqdm
import math
import re
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from typing import List
import traceback

from transformers import AutoTokenizer
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import RobertaTokenizer, RobertaModel


import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')
import spacy

nlp = spacy.load('en_core_web_sm')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')
stopwords_english = set(stopwords.words('english'))
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn

nltk.download('wordnet')
nltk.download('sentiwordnet')
from skdim.id import MLE
from sklearn.feature_extraction.text import TfidfVectorizer


#Code reference :https://github.com/Shixuan-Ma/TOCSIN
class BARTScorer:
    def __init__(self, device='cuda:0', max_length=1024, checkpoint='facebook/bart-base'):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path=None):
        """ Load model from paraphrase finetuning """
        if path is None:
            path = 'models/bart.pth'
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def score(self, srcs, tgts, batch_size=4):
        """ Score a batch of examples """
        score_list = []
        # print(len(srcs))
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            # print(src_list)
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)
                    # print(tgt_tokens)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    # print(logits.shape)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    # print(loss, tgt_len)
                    loss = loss.sum(dim=1) / tgt_len
                    # print(loss)
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list


def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    return log_likelihood.mean(dim=1)


def get_logrank(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    # get rank of each label token in the model's likelihood ordering
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]

    # make sure we got exactly one match for each timestep in the sequence
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1  # convert to 1-indexed rank
    ranks = torch.log(ranks)
    return -ranks.mean().item()


pct = 0.015


def fill_and_mask(text, pct=pct):
    tokens = text.split(' ')
    # print(len(tokens))
    n_spans = pct * len(tokens)
    n_spans = int(n_spans)
    # print(n_spans)
    repeated_random_numbers = np.random.choice(range(len(tokens)), size=n_spans)

    return repeated_random_numbers.tolist()


def apply_extracted_fills(texts, indices_list=[]):
    tokens = [x.split(' ') for x in texts]

    for idx, (text, indices) in enumerate(zip(tokens, indices_list)):
        for idx in indices:
            text[idx] = ""

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def perturb_texts_(texts, pct=pct):
    indices_list = [fill_and_mask(x, pct) for x in texts]
    perturbed_texts = apply_extracted_fills(texts, indices_list)

    return perturbed_texts


def perturb_texts(texts, pct=pct):
    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), 50), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(texts[i:i + 50], pct))
    return outputs


def get_samples(logits, labels):
    nsamples = 10000
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples_2 = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples_2


def get_score(logits_score, labels, source_texts, perturbed_texts):
    log_likelihood_x = get_likelihood(logits_score, labels)
    log_rank_x = get_logrank(logits_score, labels)
    source_texts_list = [source_texts] * 10
    values = bart_scorer.score(perturbed_texts, source_texts_list, batch_size=10)
    mean_values = np.mean(values)
    lrr_score = (log_likelihood_x.squeeze(-1).item() / log_rank_x) * math.pow(math.e, -mean_values)
    loglikelihood_score = log_likelihood_x.squeeze(-1).item() * math.pow(math.e, mean_values)
    bart_score = -mean_values
    samples_2 = get_samples(logits_score, labels)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples_2)
    miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)
    fast_score = ((log_likelihood_x.squeeze(-1).item() - miu_tilde.item()) / (sigma_tilde.item())) * math.pow(math.e,
                                                                                                              mean_values)
    return lrr_score, loglikelihood_score, bart_score, fast_score


def generate_scores(original_text, tokenizer, model):
    perturbed_original_texts = perturb_texts([x for x in [original_text] for _ in range(10)])
    tokenized = tokenizer(original_text, truncation=True, return_tensors="pt", return_token_type_ids=False).to(device)
    labels = tokenized.input_ids[:, 1:]
    logits_score = model(**tokenized).logits[:, :-1]
    lrr_score, loglikelihood_score, bart_srore, fast_score = get_score(logits_score, labels, original_text,
                                                                       perturbed_original_texts)
    return [lrr_score, loglikelihood_score, bart_srore, fast_score]


def find_pos_tags_spacy(sentence):
    doc = nlp(sentence)
    verbs = [token for token in doc if token.tag_ in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']]

    # Filter nouns, pronouns, and verbs
    proper_nouns = [token for token in doc if token.tag_ in ['NNP', 'NNPS']]
    nouns = [token for token in doc if token.tag_ in ['NNP', 'NNPS', 'NN', 'NNS']]  # Nouns

    past_tense = [token for token in doc if token.tag_ in ['VBD', 'VBN']]
    intrg_words = [token for token in doc if token.tag_ in ['WRB', 'WDT', 'WP']]

    return len(nouns), len(past_tense), len(verbs)


def remove_stopwords(text, stopword_list):
    filtered_tokens = ' '.join(word.lower() for word in text.split() if word.lower() not in stopword_list)
    # print(filtered_tokens)
    return filtered_tokens


def stopwords_count(text, stopword_list):
    stop_tokens = [token for token in text.split() if token.lower() in stopword_list]
    return len(stop_tokens)


def remove_punctuation(text):
    text = re.sub(r'[\.]', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    return text


def out_of_vocab_count(text):
    filtered = remove_stopwords(text, stopwords_english)
    words = re.findall(r'\w+', filtered.lower())
    oov_words = [word for word in words if not list(swn.senti_synsets(word))]
    return len(oov_words)


def find_word_stats(text):
    clean_txt = remove_punctuation(text)
    words = clean_txt.split()
    len_words = len(words)
    ttr = len(set(words)) / len(words)
    stopwords = stopwords_count(clean_txt, stopwords_english)
    out_of_vocab = out_of_vocab_count(clean_txt)
    return len_words, ttr, stopwords, out_of_vocab


def preprocess_text(text):
    return text.replace('\n', ' ').replace('  ', ' ')


def get_mle_single(text, solver):
    inputs = tokenizer_rob(preprocess_text(text), truncation=True, max_length=512, return_tensors="pt").to(
        device)  # Move inputs to device
    with torch.no_grad():
        outp = model_rob(**inputs)
    return solver.fit_transform(outp[0][0].cpu().numpy()[1:-1])


def tfid_vectors(df):
    tfidf_vectorizer = TfidfVectorizer(max_features=200)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])
    return tfidf_matrix.toarray()


def get_features(df, tokenizer, model):
    length_feature = np.array([[len(word_tokenize(t))] for t in df['text']])
    pos_tags = np.array([find_pos_tags_spacy(t) for t in df['text']])
    word_stats = np.array([find_word_stats(t) for t in df['text']])
    mle_score = np.array([[get_mle_single(t, MLE_solver)] for t in df['text']])
    tocsin = np.array([generate_scores(t, tokenizer, model) for t in df['text']])
    tfid_vals = tfid_vectors(df)
    return length_feature, pos_tags, word_stats, mle_score, tocsin, tfid_vals


device = "cuda" if torch.cuda.is_available() else "cpu"
MLE_solver = MLE()
tokenizer_path = model_path = "roberta-base"
tokenizer_rob = RobertaTokenizer.from_pretrained(tokenizer_path)
model_rob = RobertaModel.from_pretrained(model_path)
model_rob.to(device)
bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-base')


def get_x_features(df, tokenizer, model):
    # print("device",device)
    length_feature, pos_tags, word_stats, mle_score, tocsin, tfid = get_features(df, tokenizer, model)
    X_features = np.hstack([
        length_feature,
        pos_tags,
        word_stats,
        mle_score,
        tocsin,
        tfid
    ])
    return X_features
