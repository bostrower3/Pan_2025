import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import aidetector
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import argparse
import json
import os
import sys
def load_tokenizer():
    return AutoTokenizer.from_pretrained("./gpt2",local_files_only = True)

def load_model():
    model = AutoModelForCausalLM.from_pretrained("./gpt2",local_files_only = True)
    return model


def predict(text, model, tokenizer, xgb):
    df = pd.DataFrame({'text': [text]})
    X_features = aidetector.get_x_features(df, tokenizer, model)
    xgb_pred = xgb.predict(X_features)
    return xgb_pred[0]

def load_xgboost_model():
    model_xgb = xgb.XGBClassifier(use_label_encoder=False,eval_metric='logloss',random_state=42)
    current_folder = os.getcwd()
    model_file = os.path.join(current_folder, "xgboost_model.json")
    model_xgb.load_model(model_file)
    return model_xgb

def main(output_folder,input_file):
    model = load_model()
    tokenizer = load_tokenizer()
    model_xgb = load_xgboost_model()
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder,"predictions.jsonl")
    

    with open(input_file, 'r') as fin, open(output_file, 'w') as json_file:
        for line in fin:
            entry = json.loads(line)
            pred = predict(entry["text"], model, tokenizer,model_xgb)
            # print(type(pred))
            json.dump({"id": entry["id"], "label": pred.item()}, json_file)
            json_file.write("\n")



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Process input file and save as a JSONL file.")
    # parser.add_argument("input_file", type=str, help="Path to the input file")
    # parser.add_argument("output_folder", type=str, help="Path to the output folder")
    #
    # args = parser.parse_args()
    output_folder = sys.argv[2]  # First argument: folder path
    input_file = sys.argv[1]
    main(output_folder, input_file)
