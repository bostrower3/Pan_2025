import argparse
import json
import torch
from transformers import RobertaForSequenceClassification
from transformers import RobertaTokenizer
import os

def load_tokenizer():
    return RobertaTokenizer.from_pretrained("./roberta-large-local", local_files_only=True)

def load_model(device):
    #checkpoint = torch.load("detector_model.pth", map_location="cpu")
    model = RobertaForSequenceClassification.from_pretrained("./roberta-large-local", num_labels=2, local_files_only=True).to(device)
    model.load_state_dict(torch.load("detector_model.pth", map_location=device))
    model.eval()
    return model


def predict(text, model, tokenizer,device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        return torch.argmax(logits, dim=-1).item()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    model = load_model(device)
    tokenizer = load_tokenizer()
    print("yeehaw")
    input_path = args.input_file
    output_path = os.path.join(args.output_dir, "predictions.jsonl")

    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            entry = json.loads(line)
            print(entry)
            prob = predict(entry["text"], model, tokenizer, device)
            fout.write(json.dumps({"id": entry["id"], "label": prob}) + "\n")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required =True)
    args = parser.parse_args()
    main(args)
