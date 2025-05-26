

Submit via:
```
tira-cli code-submission \
	--path . \
	--task generative-ai-authorship-verification-panclef-2025 \
	--dataset pan25-generative-ai-detection-smoke-test-20250428-training \
	--mount-hf-model bert-base-uncased gpt2 \
	--command 'python3 main.py --input_file $inputDataset/dataset.jsonl --output_file $outputDir'
```
